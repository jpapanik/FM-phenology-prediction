from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd


def _numeric_suffix_sort_key(col_name: str) -> Tuple[int, str]:
    """Sort columns like 'GDD_doy_2' before 'GDD_doy_10' (and 'olmo_2' before 'olmo_10')."""
    tail = col_name.rsplit("_", 1)[-1]
    return (int(tail), col_name) if tail.isdigit() else (10**9, col_name)


def _load_valid_embedding_keys(
    reference_csv: Path,
    embedding_prefix: str,
) -> pd.DataFrame:
    """Read key + embedding columns from a reference FM CSV and return finite valid keys."""
    key_cols = {"Year", "Latitude", "Longitude", "DOY"}
    prefix = f"{embedding_prefix}_"

    df_ref = pd.read_csv(
        reference_csv,
        sep=";",
        usecols=lambda c: c in key_cols or c.startswith(prefix),
    )
    emb_cols = sorted(
        [c for c in df_ref.columns if c.startswith(prefix)],
        key=_numeric_suffix_sort_key,
    )
    if not emb_cols:
        raise ValueError(
            f"No embedding columns starting with '{prefix}' found in {reference_csv}."
        )

    req_cols = ["Year", "Latitude", "Longitude", "DOY"] + emb_cols
    df_valid = df_ref.dropna(subset=req_cols).copy()
    vals = df_valid[req_cols].to_numpy(dtype=float)
    df_valid = df_valid.loc[np.isfinite(vals).all(axis=1)]

    return pd.DataFrame({
        "_year_key": df_valid["Year"].astype(int),
        "_lat_key": df_valid["Latitude"].astype(float).round(6),
        "_lon_key": df_valid["Longitude"].astype(float).round(6),
        "_doy_key": df_valid["DOY"].astype(float).round(4),
    }).drop_duplicates()


def load_phenology_dataset(
    csv_path: Path,
    embedding_prefix: Optional[str] = None,
    alignment_refs: Optional[Sequence[Tuple[Path, str]]] = None,
    strict_alignment: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """
    Load curated phenology CSV, enforce finite values, optionally intersect with
    shared valid rows across reference FM CSVs, and return sorted DataFrame + channel groups.
    """
    csv_path = Path(csv_path)
    df_raw = pd.read_csv(csv_path, sep=";")
    df_raw["Year"] = df_raw["Year"].astype(int)

    gdd_cols = sorted(
        [c for c in df_raw.columns if c.startswith("GDD_doy_")],
        key=_numeric_suffix_sort_key,
    )
    agdd_cols = sorted(
        [c for c in df_raw.columns if c.startswith("AGDD_doy_")],
        key=_numeric_suffix_sort_key,
    )
    ndvi_cols = sorted(
        [c for c in df_raw.columns if c.startswith("NDVI_doy_")],
        key=_numeric_suffix_sort_key,
    )

    if not gdd_cols or not agdd_cols or not ndvi_cols:
        raise ValueError(
            f"Missing daily baseline columns ('GDD_doy_', 'AGDD_doy_', or 'NDVI_doy_') in {csv_path}."
        )

    embedding_cols: List[str] = []
    if embedding_prefix:
        embedding_cols = sorted(
            [c for c in df_raw.columns if c.startswith(f"{embedding_prefix}_")],
            key=_numeric_suffix_sort_key,
        )
        if not embedding_cols:
            raise ValueError(
                f"No embedding columns starting with '{embedding_prefix}_' found in {csv_path}."
            )

    required_cols = (
        ["Year", "Latitude", "Longitude", "DOY"]
        + embedding_cols
        + gdd_cols
        + agdd_cols
        + ndvi_cols
    )
    df_clean = df_raw.dropna(subset=required_cols).copy()
    finite_mask = np.isfinite(df_clean[required_cols].to_numpy(dtype=float)).all(axis=1)
    df_clean = df_clean.loc[finite_mask].copy()

    # Intersect with all configured reference FM CSVs so every model shares identical rows
    if alignment_refs:
        df_clean["_year_key"] = df_clean["Year"].astype(int)
        df_clean["_lat_key"] = df_clean["Latitude"].astype(float).round(6)
        df_clean["_lon_key"] = df_clean["Longitude"].astype(float).round(6)
        df_clean["_doy_key"] = df_clean["DOY"].astype(float).round(4)

        n_start = len(df_clean)
        for ref_csv, ref_prefix in alignment_refs:
            ref_path = Path(ref_csv)
            if not ref_path.exists():
                if strict_alignment:
                    raise FileNotFoundError(f"Alignment reference CSV not found: {ref_path}")
                print(
                    f"[Dataset] WARNING: Skipping missing alignment file {ref_path.name}",
                    flush=True,
                )
                continue

            if ref_path.resolve() == csv_path.resolve() and embedding_prefix == ref_prefix:
                continue

            n_before = len(df_clean)
            valid_keys = _load_valid_embedding_keys(ref_path, embedding_prefix=ref_prefix)
            df_clean = df_clean.merge(
                valid_keys,
                on=["_year_key", "_lat_key", "_lon_key", "_doy_key"],
                how="inner",
            )
            dropped = n_before - len(df_clean)
            if dropped > 0:
                print(
                    f"[Dataset] Aligned {csv_path.name} against '{ref_prefix}': "
                    f"{n_before} -> {len(df_clean)} ({dropped} dropped)",
                    flush=True,
                )

        df_clean = df_clean.drop(columns=["_year_key", "_lat_key", "_lon_key", "_doy_key"])
        print(
            f"[Dataset] Shared-sample alignment complete for {csv_path.name}: "
            f"{n_start} -> {len(df_clean)} rows",
            flush=True,
        )

    df_clean = df_clean.sort_values(
        ["Year", "DOY", "Latitude", "Longitude"]
    ).reset_index(drop=True)

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
        gdd_cols = sorted(
            [c for c in df.columns if c.startswith("GDD_doy_")],
            key=_numeric_suffix_sort_key,
        )
        agdd_cols = sorted(
            [c for c in df.columns if c.startswith("AGDD_doy_")],
            key=_numeric_suffix_sort_key,
        )
        ndvi_cols = sorted(
            [c for c in df.columns if c.startswith("NDVI_doy_")],
            key=_numeric_suffix_sort_key,
        )
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
        gdd_cols = sorted(
            [c for c in df.columns if c.startswith("GDD_doy_")],
            key=_numeric_suffix_sort_key,
        )
        agdd_cols = sorted(
            [c for c in df.columns if c.startswith("AGDD_doy_")],
            key=_numeric_suffix_sort_key,
        )
        ndvi_cols = sorted(
            [c for c in df.columns if c.startswith("NDVI_doy_")],
            key=_numeric_suffix_sort_key,
        )
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