from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd


def load_phenology_dataset(
    csv_path: Path,
    embedding_prefix: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """Load curated phenology CSV and return cleaned DataFrame and channel column groups."""
    df_raw = pd.read_csv(csv_path, sep=";")
    df_raw["Year"] = df_raw["Year"].astype(int)

    gdd_cols = [c for c in df_raw.columns if c.startswith("GDD_doy_")]
    agdd_cols = [c for c in df_raw.columns if c.startswith("AGDD_doy_")]
    ndvi_cols = [c for c in df_raw.columns if c.startswith("NDVI_doy_")]

    if not gdd_cols or not agdd_cols or not ndvi_cols:
        raise ValueError(
            f"Missing daily baseline columns ('GDD_doy_', 'AGDD_doy_', or 'NDVI_doy_') in {csv_path}."
        )

    embedding_cols: List[str] = []
    if embedding_prefix:
        embedding_cols = sorted([c for c in df_raw.columns if c.startswith(f"{embedding_prefix}_")])
        if not embedding_cols:
            raise ValueError(f"No embedding columns starting with '{embedding_prefix}_' found in {csv_path}.")

    required_cols = ["Year", "Latitude", "Longitude", "DOY"] + embedding_cols + gdd_cols + agdd_cols + ndvi_cols
    df_clean = (
        df_raw.dropna(subset=required_cols)
        .sort_values(["Year", "DOY"])
        .reset_index(drop=True)
    )

    channels = {
        "gdd": gdd_cols,
        "agdd": agdd_cols,
        "ndvi": ndvi_cols,
        "embedding": embedding_cols,
    }
    return df_clean, channels


def build_baseline_feature_sets(
    df: pd.DataFrame,
    mode: str = "unreduced",
) -> Dict[str, List[str]]:
    """Construct B1-B5 feature dictionaries for unreduced or domain-reduced baselines."""
    if mode == "unreduced":
        gdd_cols = [c for c in df.columns if c.startswith("GDD_doy_")]
        agdd_cols = [c for c in df.columns if c.startswith("AGDD_doy_")]
        ndvi_cols = [c for c in df.columns if c.startswith("NDVI_doy_")]
        return {
            "B1": ["Latitude", "Longitude"],
            "B2": gdd_cols,
            "B3": agdd_cols,
            "B4": ndvi_cols,
            "B5": ["Latitude", "Longitude"] + gdd_cols + agdd_cols + ndvi_cols,
        }
    elif mode == "domain_reduced":
        gdd_cols = [c for c in df.columns if c.startswith("GDD_")]
        agdd_cols = [c for c in df.columns if c.startswith("AGDD_")]
        ndvi_cols = [c for c in df.columns if c.startswith("NDVI_") or c.startswith("ANDVI_")]
        return {
            "B1": ["Latitude", "Longitude"],
            "B2": gdd_cols,
            "B3": agdd_cols,
            "B4": ndvi_cols,
            "B5": ["Latitude", "Longitude"] + gdd_cols + agdd_cols + ndvi_cols,
        }
    raise ValueError(f"Unsupported baseline feature mode: {mode}")


def build_fm_feature_sets(
    df: pd.DataFrame,
    embedding_cols: List[str],
    mode: str = "unreduced",
) -> Dict[str, List[str]]:
    """Construct M1-M6 foundation model + baseline fusion feature dictionaries."""
    if mode == "unreduced":
        gdd_cols = [c for c in df.columns if c.startswith("GDD_doy_")]
        agdd_cols = [c for c in df.columns if c.startswith("AGDD_doy_")]
        ndvi_cols = [c for c in df.columns if c.startswith("NDVI_doy_")]
    elif mode in {"domain_reduced", "pca_reduced"}:
        gdd_cols = [c for c in df.columns if c.startswith("GDD_")]
        agdd_cols = [c for c in df.columns if c.startswith("AGDD_")]
        ndvi_cols = [c for c in df.columns if c.startswith("NDVI_") or c.startswith("ANDVI_")]
    else:
        raise ValueError(f"Unsupported FM feature mode: {mode}")

    return {
        "M1": embedding_cols,
        "M2": ["Latitude", "Longitude"] + embedding_cols,
        "M3": embedding_cols + gdd_cols,
        "M4": embedding_cols + agdd_cols,
        "M5": embedding_cols + ndvi_cols,
        "M6": embedding_cols + ["Latitude", "Longitude"] + gdd_cols + agdd_cols + ndvi_cols,
    }
