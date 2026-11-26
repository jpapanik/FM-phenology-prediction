"""PEP725 phenology quality control, AgERA5 timeseries download, weather merge, and GDD/AGDD calculation."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple
from urllib.request import urlretrieve
from zipfile import ZipFile

import geopandas as gpd
import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# 1. Crop Code & Common Name Lookup: (scientific_name, cult_season) -> (code, name)
# -----------------------------------------------------------------------------
crop_lookup: Dict[Tuple[str, int], Tuple[str, str]] = {
    # Barley
    ("Hordeum vulgare", 1): ("BARS", "Spring barley"),
    ("Hordeum vulgare", 2): ("BARW", "Winter barley"),
    ("Hordeum vulgare", 0): ("BAR", "Barley"),
    # Bread wheat
    ("Triticum aestivum", 1): ("WHBS", "Spring bread wheat"),
    ("Triticum aestivum", 2): ("WHBW", "Winter bread wheat"),
    ("Triticum aestivum", 0): ("WHB", "Bread wheat"),
    # Durum wheat
    ("Triticum turgidum durum", 1): ("WHDS", "Spring durum wheat"),
    ("Triticum turgidum durum", 2): ("WHDW", "Winter durum wheat"),
    ("Triticum turgidum durum", 0): ("WHD", "Durum wheat"),
    # Rapeseed
    ("Brassica napus", 1): ("RAPS", "Spring rapeseed"),
    ("Brassica napus", 2): ("RAPW", "Winter rapeseed"),
    ("Brassica napus", 0): ("RAP", "Rapeseed"),
    # Rye
    ("Secale cereale", 1): ("RYES", "Spring rye"),
    ("Secale cereale", 2): ("RYEW", "Winter rye"),
    ("Secale cereale", 0): ("RYE", "Rye"),
    # Non-seasonal / Single-season crops
    ("Zea mays", 0): ("MAZ", "Maize"),
    ("Solanum tuberosum", 0): ("POT", "Potato"),
    ("Beta vulgaris", 0): ("SBT", "Sugar beet"),
    ("Helianthus annuus", 0): ("SUN", "Sunflower"),
    ("Vicia faba", 0): ("FAB", "Faba bean"),
}

# -----------------------------------------------------------------------------
# 2. Crop Base Temperature (T_base in °C) for GDD / AGDD Calculation
# -----------------------------------------------------------------------------
CROP_TBASE_MAP: Dict[str, float] = {
    "BAR": 0.0,
    "BARS": 0.0,
    "BARW": 0.0,
    "WHB": 0.0,
    "WHBS": 0.0,
    "WHBW": 0.0,
    "WHD": 0.0,
    "WHDS": 0.0,
    "WHDW": 0.0,
    "RYE": 0.0,
    "RYES": 0.0,
    "RYEW": 0.0,
    "RAP": 0.0,
    "RAPS": 0.0,
    "RAPW": 0.0,
    "POT": 2.0,
    "SBT": 3.0,
    "SUGB": 3.0,
    "FAB": 3.0,
    "SUN": 5.0,
    "MAZ": 6.0,
}

NATURAL_EARTH_COUNTRIES_URL = (
    "https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip"
)


def ensure_utf8_cdsapirc(config_path: Optional[Path] = None) -> Path:
    """Normalize ~/.cdsapirc encoding to UTF-8 without BOM so cdsapi parses it reliably."""
    path = config_path or (Path.home() / ".cdsapirc")
    if not path.exists():
        raise FileNotFoundError(f"CDS API config not found at {path}.")
    raw = path.read_bytes()
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        text = raw.decode("utf-16")
    elif raw.startswith(b"\xef\xbb\xbf"):
        text = raw.decode("utf-8-sig")
    else:
        text = raw.decode("utf-8")
    path.write_text(text, encoding="utf-8")
    return path


def assign_countries_natural_earth(
    df: pd.DataFrame,
    data_dir: Path = Path("data"),
    shapefile_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Spatial join unique (Longitude, Latitude) points against Natural Earth countries."""
    shp = shapefile_path or (data_dir / "ne_110m_admin_0_countries.shp")
    if not shp.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
        zip_path = data_dir / "ne_110m_admin_0_countries.zip"
        urlretrieve(NATURAL_EARTH_COUNTRIES_URL, zip_path)
        with ZipFile(zip_path) as archive:
            archive.extractall(data_dir)

    unique_coords = df[["Latitude", "Longitude"]].drop_duplicates().copy()
    unique_gdf = gpd.GeoDataFrame(
        unique_coords,
        geometry=gpd.points_from_xy(unique_coords["Longitude"], unique_coords["Latitude"]),
        crs="EPSG:4326",
    )
    countries = gpd.read_file(shp).to_crs("EPSG:4326")
    country_lookup = gpd.sjoin(
        unique_gdf,
        countries[["ADMIN", "geometry"]],
        how="left",
        predicate="within",
    )
    country_lookup = (
        pd.DataFrame(country_lookup)[["Latitude", "Longitude", "ADMIN"]]
        .rename(columns={"ADMIN": "Country"})
        .drop_duplicates(subset=["Latitude", "Longitude"])
    )
    return df.merge(
        country_lookup,
        on=["Latitude", "Longitude"],
        how="left",
        validate="many_to_one",
    )


def curate_pep725_observations(
    pep_csv_path: Path,
    country: str,
    crop_code: str,
    target_bbch: float,
    start_date: str,
    end_date: str,
    data_dir: Path = Path("data"),
    shapefile_path: Optional[Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load PEP725 CSV, map crop codes, run stage & monotonicity QC, and aggregate to 0.1° AgERA5 grid."""
    df_dataset = pd.read_csv(pep_csv_path, sep=";", low_memory=False)
    df_dataset = df_dataset[
        ["lon", "lat", "species", "phase_id", "year", "day", "date", "cult_season"]
    ].rename(
        columns={
            "lon": "Longitude",
            "lat": "Latitude",
            "species": "CropName",
            "phase_id": "PhenoStageCode",
            "year": "Year",
            "day": "DOY",
            "date": "ObservationDate",
            "cult_season": "CultivationSeason",
        }
    )

    clean_species = df_dataset["CropName"].astype(str).str.strip()
    clean_season = (
        pd.to_numeric(df_dataset["CultivationSeason"], errors="coerce")
        .fillna(0)
        .astype(int)
    )
    index_pairs = pd.MultiIndex.from_arrays([clean_species, clean_season])
    mapped_records = index_pairs.map(crop_lookup)
    df_dataset["CropCode"] = [r[0] if isinstance(r, tuple) else np.nan for r in mapped_records]
    df_dataset["CommonName"] = [r[1] if isinstance(r, tuple) else np.nan for r in mapped_records]

    df_dataset["ObservationDate"] = pd.to_datetime(
        df_dataset["ObservationDate"], format="%d/%m/%Y", errors="coerce"
    )
    df_dataset["PhenoStageCode"] = pd.to_numeric(df_dataset["PhenoStageCode"], errors="coerce")
    df_dataset["DOY"] = pd.to_numeric(df_dataset["DOY"], errors="coerce")

    df_with_country = assign_countries_natural_earth(
        df_dataset, data_dir=data_dir, shapefile_path=shapefile_path
    )

    df_crop = df_with_country[
        (df_with_country["Country"] == country) & (df_with_country["CropCode"] == crop_code)
    ].copy()

    clean_obs = df_crop.dropna(
        subset=["Latitude", "Longitude", "ObservationDate", "PhenoStageCode"]
    ).drop_duplicates(subset=["Latitude", "Longitude", "ObservationDate", "PhenoStageCode"])

    clean_obs = clean_obs[
        (clean_obs["PhenoStageCode"] >= 0) & (clean_obs["PhenoStageCode"] < 100)
    ].copy()
    clean_obs["Year"] = clean_obs["ObservationDate"].dt.year

    clean_obs = clean_obs.sort_values(
        ["Latitude", "Longitude", "Year", "ObservationDate", "PhenoStageCode"]
    ).reset_index(drop=True)
    clean_obs["prev_stage"] = clean_obs.groupby(["Latitude", "Longitude", "Year"])[
        "PhenoStageCode"
    ].shift(1)
    is_seasonal_reset = (clean_obs["prev_stage"] >= 80) & (clean_obs["PhenoStageCode"] <= 10)
    clean_obs["stage_drop"] = (
        clean_obs["PhenoStageCode"] < clean_obs["prev_stage"]
    ) & (~is_seasonal_reset)

    clean_observations = (
        clean_obs[~clean_obs["stage_drop"]]
        .drop(columns=["prev_stage", "stage_drop"])
        .copy()
    )

    df_target = (
        clean_observations[clean_observations["PhenoStageCode"] == float(target_bbch)]
        .drop_duplicates(subset=["Latitude", "Longitude", "Year"])
        .reset_index(drop=True)
    )

    start_ts = pd.to_datetime(start_date)
    end_ts = pd.to_datetime(end_date)
    filtered_obs = df_target[
        (df_target["ObservationDate"] >= start_ts) & (df_target["ObservationDate"] <= end_ts)
    ].copy()

    filtered_obs["AgERA5_Latitude"] = filtered_obs["Latitude"].round(1)
    filtered_obs["AgERA5_Longitude"] = filtered_obs["Longitude"].round(1)

    grid_cells = (
        filtered_obs[["AgERA5_Longitude", "AgERA5_Latitude"]]
        .drop_duplicates()
        .rename(columns={"AgERA5_Longitude": "Longitude", "AgERA5_Latitude": "Latitude"})
        .reset_index(drop=True)
    )
    return filtered_obs, grid_cells


def download_agera5_temperature(
    grid_cells: pd.DataFrame,
    start_date: str,
    end_date: str,
    download_dir: Path,
    max_workers: int = 2,
    max_attempts: int = 5,
    client_factory: Optional[object] = None,
) -> Path:
    """Download AgERA5 timeseries temperature CSVs (max/mean/min 24h) per 0.1° grid cell via CDS API."""
    import cdsapi

    dataset = "sis-agrometeorological-indicators-timeseries"
    variables = [
        "2m_temperature_24_hour_maximum",
        "2m_temperature_24_hour_mean",
        "2m_temperature_24_hour_minimum",
    ]
    date_query = f"{start_date}/{end_date}"
    download_dir.mkdir(parents=True, exist_ok=True)

    pending_locations: List[Tuple[int, float, float, Path]] = []
    for i, loc in grid_cells.iterrows():
        lon = float(loc["Longitude"])
        lat = float(loc["Latitude"])
        filename = download_dir / f"temperature_{i:04d}_{lat:.1f}_{lon:.1f}.csv"
        if not (filename.exists() and filename.stat().st_size > 500):
            pending_locations.append((i, lat, lon, filename))

    print(
        f"Total locations: {len(grid_cells)} | "
        f"Already downloaded: {len(grid_cells) - len(pending_locations)} | "
        f"Remaining: {len(pending_locations)}"
    )

    def _download_location(item: Tuple[int, float, float, Path]) -> Tuple[int, bool, Optional[str]]:
        idx, lat, lon, filename = item
        request = {
            "variable": variables,
            "location": {"longitude": lon, "latitude": lat},
            "date": [date_query],
            "data_format": "csv",
        }
        client = client_factory() if callable(client_factory) else cdsapi.Client()

        for attempt in range(1, max_attempts + 1):
            try:
                print(
                    f"[{idx + 1}/{len(grid_cells)}] "
                    f"Requesting ({lat:.1f}N, {lon:.1f}E), "
                    f"attempt {attempt}/{max_attempts}"
                )
                client.retrieve(dataset, request, str(filename))
                if not filename.exists() or filename.stat().st_size == 0:
                    raise RuntimeError("Downloaded file is missing or empty")
                time.sleep(2)
                return idx, True, None
            except Exception as exc:
                error_msg = str(exc).lower()
                if attempt == max_attempts:
                    return idx, False, str(exc)
                if (
                    "temporarily limited" in error_msg
                    or "400" in error_msg
                    or "queue" in error_msg
                    or "429" in error_msg
                ):
                    wait_seconds = 60
                else:
                    wait_seconds = 15
                time.sleep(wait_seconds)
        return idx, False, "Maximum attempts exceeded"

    failed = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_download_location, item) for item in pending_locations]
        for future in as_completed(futures):
            idx, success, error = future.result()
            if success:
                print(f"[{idx + 1}] Download completed.")
            else:
                failed.append((idx, error))
                print(f"[{idx + 1}] FAILED: {error}")

    if failed:
        raise RuntimeError(f"{len(failed)} AgERA5 location downloads failed: {failed[:5]}")
    return download_dir


def combine_raw_agera5_weather(
    raw_dir: Path,
    combined_weather_file: Path,
) -> pd.DataFrame:
    """Combine downloaded temperature_*.csv files, convert Kelvin to Celsius, and cache combined CSV."""
    raw_files = sorted(raw_dir.glob("temperature_*.csv")) if raw_dir.exists() else []

    if raw_files:
        weather_parts = [pd.read_csv(f) for f in raw_files]
        weather_df = pd.concat(weather_parts, ignore_index=True)
        weather_df["valid_time"] = pd.to_datetime(weather_df["valid_time"], errors="coerce")
        weather_df = weather_df.rename(
            columns={
                "Temperature_Air_2m_Max_24h": "Tmax_C",
                "Temperature_Air_2m_Mean_24h": "Tmean_C",
                "Temperature_Air_2m_Min_24h": "Tmin_C",
                "latitude": "AgERA5_Latitude",
                "longitude": "AgERA5_Longitude",
            }
        )
        for col in ["Tmax_C", "Tmean_C", "Tmin_C"]:
            if col in weather_df.columns:
                weather_df[col] = pd.to_numeric(weather_df[col], errors="coerce") - 273.15

        weather_df["AgERA5_Latitude"] = weather_df["AgERA5_Latitude"].round(1)
        weather_df["AgERA5_Longitude"] = weather_df["AgERA5_Longitude"].round(1)
        weather_df = (
            weather_df.drop_duplicates(
                subset=["valid_time", "AgERA5_Latitude", "AgERA5_Longitude"]
            )
            .sort_values(["AgERA5_Latitude", "AgERA5_Longitude", "valid_time"])
            .reset_index(drop=True)
        )
        combined_weather_file.parent.mkdir(parents=True, exist_ok=True)
        weather_df.to_csv(combined_weather_file, sep=";", index=False, encoding="utf-8")
        return weather_df

    if combined_weather_file.exists():
        weather_df = pd.read_csv(combined_weather_file, sep=";")
        weather_df["valid_time"] = pd.to_datetime(weather_df["valid_time"], errors="coerce")
        weather_df["AgERA5_Latitude"] = weather_df["AgERA5_Latitude"].round(1)
        weather_df["AgERA5_Longitude"] = weather_df["AgERA5_Longitude"].round(1)
        return weather_df

    raise FileNotFoundError(
        f"Neither raw files in {raw_dir} nor {combined_weather_file} were found."
    )


def compute_daily_gdd_agdd_pivots(
    weather_df: pd.DataFrame,
    crop_code: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate daily GDD (clipped at 0) and cumulative AGDD (reset Jan 1) pivoted to DOY 1..365."""
    t_base = CROP_TBASE_MAP.get(crop_code.upper(), 0.0)
    df = weather_df.copy()
    df["valid_time"] = pd.to_datetime(df["valid_time"], errors="coerce")
    df["daily_gdd"] = (pd.to_numeric(df["Tmean_C"], errors="coerce") - t_base).clip(lower=0.0)

    df["Year"] = df["valid_time"].dt.year
    df["Weather_DOY"] = df["valid_time"].dt.dayofyear
    df = df.sort_values(["AgERA5_Latitude", "AgERA5_Longitude", "valid_time"])

    df["agdd"] = df.groupby(["AgERA5_Latitude", "AgERA5_Longitude", "Year"])[
        "daily_gdd"
    ].cumsum()

    weather_365 = df[df["Weather_DOY"] <= 365].copy()

    gdd_wide = weather_365.pivot_table(
        index=["AgERA5_Latitude", "AgERA5_Longitude", "Year"],
        columns="Weather_DOY",
        values="daily_gdd",
    )
    gdd_wide = gdd_wide.reindex(columns=range(1, 366))
    gdd_wide.columns = [f"GDD_doy_{int(col)}" for col in gdd_wide.columns]
    gdd_wide = gdd_wide.reset_index()

    agdd_wide = weather_365.pivot_table(
        index=["AgERA5_Latitude", "AgERA5_Longitude", "Year"],
        columns="Weather_DOY",
        values="agdd",
    )
    agdd_wide = agdd_wide.reindex(columns=range(1, 366))
    agdd_wide.columns = [f"AGDD_doy_{int(col)}" for col in agdd_wide.columns]
    agdd_wide = agdd_wide.reset_index()

    return gdd_wide, agdd_wide


def build_curated_modeling_table(
    observations_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    crop_code: str,
    output_csv_path: Path,
) -> pd.DataFrame:
    """Merge GDD_doy_1..365 and AGDD_doy_1..365 onto filtered phenology observations and save CSV."""
    gdd_wide, agdd_wide = compute_daily_gdd_agdd_pivots(weather_df, crop_code=crop_code)

    obs = observations_df.copy()
    obs["ObservationDate"] = pd.to_datetime(obs["ObservationDate"], errors="coerce")
    obs["Year"] = obs["ObservationDate"].dt.year
    obs["DOY"] = obs["ObservationDate"].dt.dayofyear
    if "AgERA5_Latitude" not in obs.columns:
        obs["AgERA5_Latitude"] = obs["Latitude"].round(1)
    if "AgERA5_Longitude" not in obs.columns:
        obs["AgERA5_Longitude"] = obs["Longitude"].round(1)

    obs = (
        obs.sort_values(["Latitude", "Longitude", "Year", "ObservationDate"])
        .drop_duplicates(subset=["Latitude", "Longitude", "Year"], keep="first")
        .reset_index(drop=True)
    )

    df_final = obs.merge(
        gdd_wide,
        on=["AgERA5_Latitude", "AgERA5_Longitude", "Year"],
        how="left",
    ).merge(
        agdd_wide,
        on=["AgERA5_Latitude", "AgERA5_Longitude", "Year"],
        how="left",
    )

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(output_csv_path, sep=";", index=False, encoding="utf-8")
    return df_final
