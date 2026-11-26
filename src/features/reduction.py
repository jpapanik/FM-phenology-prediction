from typing import List, Optional
import numpy as np
import pandas as pd

AGDD_CHECKPOINTS_DOY = [
    7, 14, 21, 28, 42, 56, 70, 84, 98, 112, 126, 140, 154, 168, 182,
    196, 210, 224, 238, 252, 266, 280, 294, 308, 322, 336, 350, 365,
]

MONTH_SLICES = {
    "Jan": (0, 31),
    "Feb": (31, 59),
    "Mar": (59, 90),
    "Apr": (90, 120),
    "May": (120, 151),
    "Jun": (151, 181),
    "Jul": (181, 212),
    "Aug": (212, 243),
    "Sep": (243, 273),
    "Oct": (273, 304),
    "Nov": (304, 334),
    "Dec": (334, 365),
}


def extract_domain_reduced_features(
    df: pd.DataFrame,
    embedding_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Compress 365-day GDD, AGDD, and NDVI time series into domain-informed phenological metrics."""
    keep_cols = ["Year", "Latitude", "Longitude", "DOY"]
    if "PEP_ID" in df.columns:
        keep_cols.append("PEP_ID")
    if embedding_cols:
        keep_cols.extend(embedding_cols)

    df_reduced = df[keep_cols].copy()

    # 1. GDD monthly integrals + dormancy/summer stress indicators
    gdd_cols = [c for c in df.columns if c.startswith("GDD_doy_")]
    if gdd_cols:
        gdd_mat = df[gdd_cols].values
        for month_name, (start_idx, end_idx) in MONTH_SLICES.items():
            df_reduced[f"GDD_sum_{month_name}"] = gdd_mat[:, start_idx:end_idx].sum(axis=1)
        df_reduced["GDD_cold_days_winter_spring"] = (gdd_mat[:, 0:120] <= 0.5).sum(axis=1)
        df_reduced["GDD_sum_summer_peak"] = gdd_mat[:, 151:243].sum(axis=1)

    # 2. AGDD bi-weekly checkpoints
    for doy_val in AGDD_CHECKPOINTS_DOY:
        col_name = f"AGDD_doy_{doy_val}"
        if col_name in df.columns:
            df_reduced[col_name] = df[col_name]

    # 3. NDVI Land Surface Phenology (LSP) curve metrics + bi-weekly ANDVI
    ndvi_cols = [c for c in df.columns if c.startswith("NDVI_doy_")]
    if ndvi_cols:
        ndvi_mat = df[ndvi_cols].values
        df_reduced["NDVI_base"] = np.min(ndvi_mat[:, 0:75], axis=1)

        ndvi_growth_window = ndvi_mat[:, 59:300]
        df_reduced["NDVI_max"] = np.max(ndvi_growth_window, axis=1)
        df_reduced["NDVI_peak_doy"] = np.argmax(ndvi_growth_window, axis=1) + 60
        df_reduced["NDVI_amplitude"] = df_reduced["NDVI_max"] - df_reduced["NDVI_base"]

        ndvi_7d_diff = ndvi_growth_window[:, 7:] - ndvi_growth_window[:, :-7]
        df_reduced["NDVI_max_greenup_slope"] = np.max(ndvi_7d_diff, axis=1) / 7.0
        df_reduced["NDVI_AUC_seasonal"] = np.sum(np.clip(ndvi_mat - 0.20, 0, None), axis=1)

        andvi_daily_integral = np.cumsum(np.clip(ndvi_mat - 0.15, 0, None), axis=1)
        for doy_val in AGDD_CHECKPOINTS_DOY:
            df_reduced[f"ANDVI_biweekly_doy_{doy_val}"] = andvi_daily_integral[:, doy_val - 1]

    return df_reduced
