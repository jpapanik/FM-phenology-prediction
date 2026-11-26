"""Foundation model embedding extractors: PRESTO, TESSERA, AlphaEarth, and OlmoEarth v1.3."""

from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd


def dequantize_olmo(q: np.ndarray) -> np.ndarray:
    """Dequantize int8 OlmoEarth Zarr embeddings (-128 -> NaN, quadratic signed scaling)."""
    q_arr = np.asarray(q)
    v = np.where(q_arr == -128, np.nan, q_arr).astype("float32")
    return (np.abs(v) / 127.5) ** 2.0 * np.sign(v)


def lonlat_to_utm_zone(lon: float, lat: float) -> int:
    """Map (lon, lat) in WGS84 to UTM zone 1..60 (including Norway/Svalbard exceptions)."""
    if 56.0 <= lat < 64.0 and 3.0 <= lon < 12.0:
        return 32
    if 72.0 <= lat <= 84.0:
        if 0.0 <= lon < 9.0:
            return 31
        if 9.0 <= lon < 21.0:
            return 33
        if 21.0 <= lon < 33.0:
            return 35
        if 33.0 <= lon < 42.0:
            return 37
    return int(np.floor((lon + 180.0) / 6.0)) + 1


def extract_presto_raw_inputs(
    study_folder: Path,
    modeling_csv_path: Path,
    gee_project: str = "thesis-phenology",
    min_year: int = 2017,
    gee_batch_size: int = 500,
) -> Path:
    """Extract 12-month x 15-channel (S2 SR + S1 GRD + ERA5 monthly + NASADEM) GEE inputs to Parquet."""
    import ee

    try:
        ee.Initialize()
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=gee_project)

    df_target = pd.read_csv(modeling_csv_path, sep=";")
    df_target["Year"] = df_target["Year"].astype(int)
    available_years = [y for y in sorted(df_target["Year"].unique()) if y >= min_year]
    print(f"Extracting PRESTO raw inputs for years: {available_years}")

    s2_bands = ["B2", "B3", "B4", "B8", "B5", "B6", "B7", "B8A", "B11", "B12"]
    s1_bands = ["VV", "VH"]
    dem = ee.Image("NASA/NASADEM_HGT/001").select("elevation")

    def _get_monthly_composite(year: int, month: int):
        m_str = f"{month:02d}"
        start = f"{year}-{m_str}-01"
        end = f"{year + 1}-01-01" if month == 12 else f"{year}-{month + 1:02d}-01"

        s2 = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterDate(start, end)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 50))
            .select(s2_bands)
            .median()
        )
        s1 = (
            ee.ImageCollection("COPERNICUS/S1_GRD")
            .filterDate(start, end)
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
            .filter(ee.Filter.eq("instrumentMode", "IW"))
            .select(s1_bands)
            .median()
        )
        era5 = (
            ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY_AGGR")
            .filterDate(start, end)
            .select(["temperature_2m", "total_precipitation_sum"], ["era5_temp", "era5_precip"])
            .mean()
        )
        return s2.addBands(s1).addBands(era5).addBands(dem.rename("elevation"))

    all_year_inputs: List[pd.DataFrame] = []
    for year in available_years:
        df_year = df_target[df_target["Year"] == year].copy()
        if df_year.empty:
            continue
        print(f"\nGathering monthly satellite rasters for {year}...")
        monthly_bands = []
        band_names: List[str] = []
        for m in range(1, 13):
            comp = _get_monthly_composite(year, m)
            prefix = f"m{m:02d}_"
            channels = [f"s2_{b}" for b in s2_bands] + ["s1_VV", "s1_VH"] + ["temp", "precip", "elev"]
            monthly_bands.append(comp)
            band_names.extend([f"{prefix}{c}" for c in channels])

        year_stack = ee.Image.cat(monthly_bands).rename(band_names)
        features = [
            ee.Feature(ee.Geometry.Point([row["Longitude"], row["Latitude"]]), {"row_id": int(idx)})
            for idx, row in df_year.iterrows()
        ]
        pts_fc = ee.FeatureCollection(features)
        feature_list = pts_fc.toList(pts_fc.size())
        num_batches = int(np.ceil(len(features) / gee_batch_size))

        sampled_records = []
        for b in range(num_batches):
            sub_fc = ee.FeatureCollection(feature_list.slice(b * gee_batch_size, (b + 1) * gee_batch_size))
            sampled = year_stack.reduceRegions(collection=sub_fc, reducer=ee.Reducer.first(), scale=10)
            for feat in sampled.getInfo()["features"]:
                sampled_records.append(feat["properties"])

        df_sampled = pd.DataFrame(sampled_records).set_index("row_id")
        all_year_inputs.append(df_year.join(df_sampled))

    df_raw_satellite_inputs = pd.concat(all_year_inputs, axis=0)
    raw_input_path = study_folder / "presto_raw_inputs_temp.parquet"
    df_raw_satellite_inputs.to_parquet(raw_input_path)
    return raw_input_path


def extract_presto_embeddings(
    study_folder: Path,
    modeling_csv_path: Optional[Path] = None,
    raw_input_path: Optional[Path] = None,
    gee_project: str = "thesis-phenology",
    batch_size: int = 256,
    gee_batch_size: int = 500,
) -> Path:
    """Compute 128-dim PRESTO encoder embeddings and save curated_for_modeling_with_presto_embeddings.csv."""
    import presto
    import torch

    parquet_path = raw_input_path or (study_folder / "presto_raw_inputs_temp.parquet")
    if not parquet_path.exists():
        if modeling_csv_path is None:
            raise FileNotFoundError(f"Missing {parquet_path} and no modeling_csv_path provided.")
        parquet_path = extract_presto_raw_inputs(
            study_folder=study_folder,
            modeling_csv_path=modeling_csv_path,
            gee_project=gee_project,
            gee_batch_size=gee_batch_size,
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = presto.Presto.load_pretrained().to(device)
    model.eval()

    df_inputs = pd.read_parquet(parquet_path)
    n_obs = len(df_inputs)

    s2_band_keys = ["B2", "B3", "B4", "B8", "B5", "B6", "B7", "B8A", "B11", "B12"]
    s1_band_keys = ["VV", "VH"]
    era5_band_keys = ["temperature_2m", "total_precipitation"]
    srtm_band_keys = ["elevation", "slope"]

    all_x, all_mask, all_dw = [], [], []
    for _, row in df_inputs.iterrows():
        s2_vals = np.zeros((12, 10), dtype=np.float32)
        s1_vals = np.zeros((12, 2), dtype=np.float32)
        era5_vals = np.zeros((12, 2), dtype=np.float32)
        for m in range(1, 13):
            prefix = f"m{m:02d}_"
            for b_idx, b in enumerate(s2_band_keys):
                val = row.get(f"{prefix}s2_{b}", 0.0)
                s2_vals[m - 1, b_idx] = 0.0 if pd.isna(val) else float(val)
            for b_idx, b in enumerate(s1_band_keys):
                val = row.get(f"{prefix}s1_{b}", -15.0)
                s1_vals[m - 1, b_idx] = -15.0 if pd.isna(val) else float(val)
            t_val = row.get(f"{prefix}temp", 283.15)
            p_val = row.get(f"{prefix}precip", 0.0)
            era5_vals[m - 1, 0] = 283.15 if pd.isna(t_val) else float(t_val)
            era5_vals[m - 1, 1] = 0.0 if pd.isna(p_val) else float(p_val)

        elev_val = row.get("m01_elev", 100.0)
        elev = 100.0 if pd.isna(elev_val) else float(elev_val)
        srtm_vals = np.tile(np.array([elev, 0.0], dtype=np.float32), (12, 1))

        x, mask, dw = presto.construct_single_presto_input(
            s2=torch.tensor(s2_vals, dtype=torch.float32),
            s2_bands=s2_band_keys,
            s1=torch.tensor(s1_vals, dtype=torch.float32),
            s1_bands=s1_band_keys,
            era5=torch.tensor(era5_vals, dtype=torch.float32),
            era5_bands=era5_band_keys,
            srtm=torch.tensor(srtm_vals, dtype=torch.float32),
            srtm_bands=srtm_band_keys,
        )
        all_x.append(x)
        all_mask.append(mask)
        all_dw.append(dw)

    x_tensor = torch.stack(all_x)
    mask_tensor = torch.stack(all_mask)
    dw_tensor = torch.stack(all_dw).long()
    latlon_tensor = torch.tensor(
        df_inputs[["Latitude", "Longitude"]].values.copy(), dtype=torch.float32
    )

    num_batches = int(np.ceil(n_obs / batch_size))
    all_embeddings = []
    with torch.no_grad():
        for b in range(num_batches):
            start = b * batch_size
            end = min((b + 1) * batch_size, n_obs)
            embeddings = model.encoder(
                x_tensor[start:end].float().to(device),
                dynamic_world=dw_tensor[start:end].long().to(device),
                mask=mask_tensor[start:end].to(device),
                latlons=latlon_tensor[start:end].float().to(device),
                month=0,
                eval_task=True,
            )
            if isinstance(embeddings, tuple):
                embeddings = embeddings[0]
            all_embeddings.append(embeddings.cpu().numpy())

    embedding_matrix = np.concatenate(all_embeddings, axis=0)
    embedding_cols = [f"presto_{i}" for i in range(128)]
    df_presto_features = pd.DataFrame(embedding_matrix, columns=embedding_cols, index=df_inputs.index)
    base_cols = [c for c in df_inputs.columns if not (c.startswith("m0") or c.startswith("m1"))]
    df_final = pd.concat([df_inputs[base_cols], df_presto_features], axis=1)

    out_csv_path = study_folder / "curated_for_modeling_with_presto_embeddings.csv"
    df_final.to_csv(out_csv_path, sep=";", index=False)
    return out_csv_path


def extract_tessera_embeddings(
    study_folder: Path,
    modeling_csv_path: Path,
    min_year: int = 2017,
    batch_size: int = 50,
) -> Path:
    """Stream 128-dim TESSERA embeddings from GeoTesseraZarr and save enriched CSV."""
    from geotessera import GeoTesseraZarr

    df_master = pd.read_csv(modeling_csv_path, sep=";")
    df_master["Year"] = df_master["Year"].astype(int)
    df_tessera_base = df_master[df_master["Year"] >= min_year].copy().reset_index(drop=True)

    gt = GeoTesseraZarr()
    tessera_cols = [f"tessera_{i}" for i in range(128)]
    extracted_year_dfs = []

    for y in sorted(df_tessera_base["Year"].unique()):
        df_year = df_tessera_base[df_tessera_base["Year"] == y].copy().reset_index(drop=True)
        coords = list(zip(df_year["Longitude"], df_year["Latitude"]))
        batch_embs = []
        for start_idx in range(0, len(coords), batch_size):
            batch_coords = coords[start_idx : start_idx + batch_size]
            arr = gt.sample_points(batch_coords, year=int(y))
            batch_embs.append(arr)
        year_arr = np.vstack(batch_embs)
        df_emb = pd.DataFrame(year_arr, columns=tessera_cols)
        extracted_year_dfs.append(pd.concat([df_year, df_emb], axis=1))

    df_with_tessera = pd.concat(extracted_year_dfs, ignore_index=True)
    enriched_file = study_folder / f"{modeling_csv_path.stem}_with_tessera_embeddings.csv"
    df_with_tessera.to_csv(enriched_file, sep=";", index=False)
    return enriched_file


def extract_alphaearth_embeddings(
    study_folder: Path,
    modeling_csv_path: Path,
    min_year: int = 2017,
    gee_batch_size: int = 500,
) -> Path:
    """Extract 64-dim AlphaEarth annual embeddings via GEE and save enriched CSV."""
    import ee

    try:
        ee.Initialize()
    except Exception:
        ee.Authenticate()
        ee.Initialize()

    df_master = pd.read_csv(modeling_csv_path, sep=";")
    df_master["Year"] = df_master["Year"].astype(int)
    df_alpha_base = df_master[df_master["Year"] >= min_year].copy().reset_index(drop=True)

    alphaearth_ic = ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
    alphaearth_bands = [f"A{i:02d}" for i in range(64)]
    extracted_dfs = []

    for year in sorted(df_alpha_base["Year"].unique()):
        df_year = df_master[df_master["Year"] == year].copy()
        if df_year.empty:
            continue
        annual_mosaic = (
            alphaearth_ic.filterDate(f"{year}-01-01", f"{year + 1}-01-01")
            .select(alphaearth_bands)
            .mosaic()
        )
        features = [
            ee.Feature(ee.Geometry.Point([row["Longitude"], row["Latitude"]]), {"row_id": int(idx)})
            for idx, row in df_year.iterrows()
        ]
        pts_fc = ee.FeatureCollection(features)
        feature_list = pts_fc.toList(pts_fc.size())
        num_batches = int(np.ceil(len(features) / gee_batch_size))

        sampled_records = []
        for b in range(num_batches):
            sub_fc = ee.FeatureCollection(feature_list.slice(b * gee_batch_size, (b + 1) * gee_batch_size))
            sampled_sub = annual_mosaic.reduceRegions(collection=sub_fc, reducer=ee.Reducer.first(), scale=10)
            for item in sampled_sub.getInfo()["features"]:
                sampled_records.append(item["properties"])

        df_sampled = pd.DataFrame(sampled_records)
        if df_sampled.empty:
            continue
        df_sampled = df_sampled.set_index("row_id").rename(
            columns={f"A{i:02d}": f"alphaearth_{i}" for i in range(64)}
        )
        alpha_cols = [f"alphaearth_{i}" for i in range(64)]
        valid_alpha_cols = [c for c in alpha_cols if c in df_sampled.columns]
        extracted_dfs.append(df_year.join(df_sampled[valid_alpha_cols]))

    if not extracted_dfs:
        raise RuntimeError("AlphaEarth extraction returned no records.")

    df_alphaearth_final = pd.concat(extracted_dfs, axis=0).sort_values(["Year", "DOY"])
    save_path = study_folder / "curated_for_modeling_with_alphaearth_embeddings.csv"
    df_alphaearth_final.to_csv(save_path, sep=";", index=False)
    return save_path


def extract_olmoearth_embeddings(
    study_folder: Path,
    modeling_csv_path: Path,
    store_url: str = "s3://olmoearth/olmoearth-v1-3-embeddings/embeddings.zarr",
    n_dims: int = 128,
) -> Path:
    """Sample and dequantize OlmoEarth v1.3 UTM Zarr embeddings and save enriched CSV."""
    from pyproj import Transformer
    import xarray as xr
    import zarr

    zarr.config.set({"array.sharding_coalesce_max_gap_bytes": 0})

    df_base = pd.read_csv(modeling_csv_path, sep=";")
    utm_zones = [
        lonlat_to_utm_zone(float(lon), float(lat))
        for lon, lat in zip(df_base["Longitude"], df_base["Latitude"])
    ]
    df_work = df_base.copy()
    df_work["_utm_zone"] = utm_zones
    df_work["_year_str"] = df_work["Year"].astype(str)

    emb_matrix = np.full((len(df_work), n_dims), np.nan, dtype="float32")
    for zone_num, df_zone in df_work.groupby("_utm_zone"):
        group_path = f"utm{zone_num:02d}"
        epsg_code = f"EPSG:326{zone_num:02d}"
        transformer = Transformer.from_crs("EPSG:4326", epsg_code, always_xy=True)
        utm_x, utm_y = transformer.transform(
            df_zone["Longitude"].to_numpy(),
            df_zone["Latitude"].to_numpy(),
        )

        ds = xr.open_zarr(store_url, group=group_path)
        arr = ds["embeddings"].isel(band=slice(0, n_dims))

        for yr_str, idxs in df_zone.groupby("_year_str").groups.items():
            pos_in_zone = df_zone.index.get_indexer(idxs)
            xs = xr.DataArray(utm_x[pos_in_zone], dims="points")
            ys = xr.DataArray(utm_y[pos_in_zone], dims="points")
            sampled_q = arr.sel(time=yr_str).sel(x=xs, y=ys, method="nearest").values
            if sampled_q.ndim == 2 and sampled_q.shape[0] == n_dims and sampled_q.shape[1] == len(idxs):
                sampled_q = sampled_q.T
            emb_matrix[idxs, :] = dequantize_olmo(sampled_q)

    emb_cols = [f"olmo_{i}" for i in range(n_dims)]
    df_emb = pd.DataFrame(emb_matrix, columns=emb_cols, index=df_base.index)
    df_out = pd.concat([df_base, df_emb], axis=1)
    out_csv = study_folder / "curated_for_modeling_with_olmo_embeddings.csv"
    df_out.to_csv(out_csv, sep=";", index=False)
    return out_csv
