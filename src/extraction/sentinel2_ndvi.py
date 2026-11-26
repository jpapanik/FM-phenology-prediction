"""Sentinel-2 L2A (SR) NDVI extraction via GEE, SCL cloud masking, and 365-day daily interpolation."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Sequence
import numpy as np
import pandas as pd


def extract_sentinel2_ndvi(
    df_observations: pd.DataFrame,
    s2_raw_dir: Path,
    s2_collection: str = "COPERNICUS/S2_SR_HARMONIZED",
    bands: Sequence[str] = ("B4", "B8", "SCL"),
    scale_meters: int = 10,
    max_workers: int = 8,
    gee_project: str = "thesis-phenology",
    min_year: int = 2017,
) -> Path:
    """Query Sentinel-2 pixel time series per unique station-year (2017+) via GEE getRegion with disk cache."""
    import ee
    try:
        ee.Initialize(project=gee_project)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=gee_project)
    s2_raw_dir.mkdir(parents=True, exist_ok=True)
    df_work = df_observations.copy()
    df_work["Year"] = df_work["Year"].astype(int)

    s2_targets = (
        df_work[df_work["Year"] >= min_year][["Latitude", "Longitude", "Year"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    pending = []
    for idx, row in s2_targets.iterrows():
        lat = float(row["Latitude"])
        lon = float(row["Longitude"])
        year = int(row["Year"])
        out_csv = s2_raw_dir / f"s2_{idx:04d}_{year}_{lat:.4f}_{lon:.4f}.csv"
        if not (out_csv.exists() and out_csv.stat().st_size > 50):
            pending.append((idx, lat, lon, year, out_csv))

    print(
        f"Total targets: {len(s2_targets)} | Already cached: {len(s2_targets) - len(pending)} | Remaining: {len(pending)}"
    )

    executor = ThreadPoolExecutor(max_workers=max_workers)
    future_to_meta = {}
    for idx, lat, lon, year, out_csv in pending:
        pt = ee.Geometry.Point([lon, lat])
        ic = (
            ee.ImageCollection(s2_collection)
            .filterBounds(pt)
            .filterDate(f"{year}-01-01", f"{year}-12-31")
            .select(list(bands))
        )
        region_call = ic.getRegion(pt, scale_meters)
        future = executor.submit(region_call.getInfo)
        future_to_meta[future] = (idx, lat, lon, year, out_csv)

    completed_count = 0
    for future in as_completed(future_to_meta):
        idx, lat, lon, year, out_csv = future_to_meta[future]
        completed_count += 1
        try:
            rows = future.result()
            if rows and len(rows) > 1:
                part_df = pd.DataFrame(rows[1:], columns=rows[0])
                part_df["Station_Latitude"] = lat
                part_df["Station_Longitude"] = lon
                part_df["Year"] = year
                part_df.to_csv(out_csv, sep=";", index=False)
            if completed_count % 50 == 0 or completed_count == len(pending):
                print(f"Completed {completed_count}/{len(pending)} GEE extractions...")
        except Exception as exc:
            print(f"Error on ({lat:.4f}, {lon:.4f}) in {year}: {exc}")

    executor.shutdown(wait=True)
    return s2_raw_dir


def compute_daily_ndvi_pivots(
    s2_raw_dir: Path,
    scl_keep_codes: Sequence[int] = (4, 5, 6, 7),
    ndvi_min: float = -0.2,
    ndvi_max: float = 1.0,
) -> pd.DataFrame:
    """Cloud-mask via SCL, compute clipped NDVI, pivot DOY 1..365, and linearly interpolate gaps."""
    s2_files = sorted(s2_raw_dir.glob("s2_*.csv"))
    if not s2_files:
        raise FileNotFoundError(f"No Sentinel-2 CSVs found in {s2_raw_dir.resolve()}.")

    s2_df = pd.concat([pd.read_csv(f, sep=";") for f in s2_files], ignore_index=True)
    s2_df["valid_time"] = pd.to_datetime(s2_df["time"], unit="ms", errors="coerce")
    s2_df["S2_DOY"] = s2_df["valid_time"].dt.dayofyear

    s2_clean = s2_df[s2_df["SCL"].isin(list(scl_keep_codes))].copy()
    b8 = pd.to_numeric(s2_clean["B8"], errors="coerce")
    b4 = pd.to_numeric(s2_clean["B4"], errors="coerce")
    s2_clean["NDVI"] = ((b8 - b4) / (b8 + b4)).clip(lower=ndvi_min, upper=ndvi_max)

    s2_clean = s2_clean.dropna(subset=["NDVI", "S2_DOY"])
    s2_clean = s2_clean[s2_clean["S2_DOY"] <= 365].copy()

    ndvi_wide = s2_clean.pivot_table(
        index=["Station_Latitude", "Station_Longitude", "Year"],
        columns="S2_DOY",
        values="NDVI",
        aggfunc="mean",
    )
    ndvi_wide = ndvi_wide.reindex(columns=range(1, 366))
    ndvi_wide = (
        ndvi_wide.interpolate(method="linear", axis=1)
        .bfill(axis=1)
        .ffill(axis=1)
    )
    ndvi_wide.columns = [f"NDVI_doy_{col}" for col in range(1, 366)]
    return ndvi_wide.reset_index().rename(
        columns={"Station_Latitude": "Latitude", "Station_Longitude": "Longitude"}
    )


def merge_ndvi_into_modeling_csv(
    modeling_csv_path: Path,
    s2_raw_dir: Path,
    scl_keep_codes: Sequence[int] = (4, 5, 6, 7),
    ndvi_min: float = -0.2,
    ndvi_max: float = 1.0,
    output_csv_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Merge 365-day interpolated NDVI_doy_1..NDVI_doy_365 columns into the modeling CSV."""
    df_master = pd.read_csv(modeling_csv_path, sep=";")
    df_master["Year"] = df_master["Year"].astype(int)

    ndvi_wide = compute_daily_ndvi_pivots(
        s2_raw_dir=s2_raw_dir,
        scl_keep_codes=scl_keep_codes,
        ndvi_min=ndvi_min,
        ndvi_max=ndvi_max,
    )
    existing_ndvi_cols = [c for c in df_master.columns if c.startswith("NDVI_doy_")]
    if existing_ndvi_cols:
        df_master = df_master.drop(columns=existing_ndvi_cols)

    df_merged = df_master.merge(ndvi_wide, on=["Latitude", "Longitude", "Year"], how="left")
    target_path = output_csv_path or modeling_csv_path
    df_merged.to_csv(target_path, sep=";", index=False, encoding="utf-8")
    return df_merged
