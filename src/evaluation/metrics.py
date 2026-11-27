from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def get_station_ids(df: pd.DataFrame) -> np.ndarray:
    if "PEP_ID" in df.columns:
        return df["PEP_ID"].values
    return (
        df["Latitude"].round(4).astype(str)
        + "_"
        + df["Longitude"].round(4).astype(str)
    ).values


def compute_station_climatology(
    df_ref: pd.DataFrame,
    target_col: str = "DOY",
) -> pd.Series:
    """Compute per-station historical mean DOY from reference (training/CV) years."""
    ref_stations = get_station_ids(df_ref)
    return (
        pd.DataFrame({
            "Station": ref_stations,
            "DOY": df_ref[target_col].to_numpy(dtype=float),
        })
        .groupby("Station")["DOY"]
        .mean()
    )


def compute_r2_decomposition(
    actual_doy: np.ndarray,
    preds: np.ndarray,
    station_ids: np.ndarray,
    station_climatology: Optional[pd.Series] = None,
) -> Tuple[float, float, float]:
    """
    Decompose R² into (Global Pooled R², Spatial Station-Mean R², Temporal Anomaly R²).

    When `station_climatology` (computed strictly on training/CV years) is provided,
    temporal anomalies are measured relative to each station's historical baseline
    (excluding stations never seen in training). Otherwise, within-evaluation station
    means are used strictly on stations with >= 2 observations to avoid artificial zero
    anomalies from single-year stations.
    """
    actual_arr = np.asarray(actual_doy, dtype=float)
    preds_arr = np.asarray(preds, dtype=float)

    global_r2 = float(r2_score(actual_arr, preds_arr))

    df_tmp = pd.DataFrame({
        "Station": station_ids,
        "Actual": actual_arr,
        "Pred": preds_arr,
    })
    st_means = df_tmp.groupby("Station")[["Actual", "Pred"]].mean()
    spatial_r2 = (
        float(r2_score(st_means["Actual"], st_means["Pred"]))
        if len(st_means) >= 2 and st_means["Actual"].var() > 0
        else float("nan")
    )

    if station_climatology is not None:
        df_tmp["Ref_Mean"] = df_tmp["Station"].map(station_climatology)
        df_valid = df_tmp.dropna(subset=["Ref_Mean"])
        obs_anom = df_valid["Actual"] - df_valid["Ref_Mean"]
        pred_anom = df_valid["Pred"] - df_valid["Ref_Mean"]
        temporal_r2 = (
            float(r2_score(obs_anom, pred_anom))
            if len(df_valid) >= 2 and float(np.var(obs_anom)) > 0
            else float("nan")
        )
    else:
        obs_counts = df_tmp.groupby("Station")["Actual"].transform("count")
        df_multi = df_tmp.loc[obs_counts >= 2]
        if len(df_multi) >= 2:
            multi_means = df_multi.groupby("Station")[["Actual", "Pred"]].mean()
            df_merged = df_multi.merge(multi_means, on="Station", suffixes=("", "_mean"))
            obs_anom = df_merged["Actual"] - df_merged["Actual_mean"]
            pred_anom = df_merged["Pred"] - df_merged["Pred_mean"]
            temporal_r2 = (
                float(r2_score(obs_anom, pred_anom))
                if float(np.var(obs_anom)) > 0
                else float("nan")
            )
        else:
            temporal_r2 = float("nan")

    return global_r2, spatial_r2, temporal_r2


def save_fold_summary_and_r2(
    model_entries: List[Tuple[str, str, np.ndarray]],
    actual_doy: np.ndarray,
    years: np.ndarray,
    station_ids: np.ndarray,
    out_dir: Path,
    prefix: str,
    station_climatology: Optional[pd.Series] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Save fold-level MAE/RMSE/Bias summary, R² decomposition CSV, and per-year R² CSV."""
    out_dir.mkdir(parents=True, exist_ok=True)
    actual_arr = np.asarray(actual_doy, dtype=float)
    years_arr = np.asarray(years).astype(int)
    unique_years = sorted(np.unique(years_arr))
    if len(station_ids) != len(actual_arr):
        raise ValueError(
            f"Length mismatch: station_ids ({len(station_ids)}) != actual_arr ({len(actual_arr)})."
        )

    fold_rows = []
    for yr in unique_years:
        mask = years_arr == yr
        row = {"Test_Year": int(yr), "N_Samples": int(mask.sum())}
        for key, _, preds in model_entries:
            p_yr = np.asarray(preds, dtype=float)[mask]
            y_yr = actual_arr[mask]
            row[f"{key}_MAE"] = round(float(mean_absolute_error(y_yr, p_yr)), 2)
            row[f"{key}_RMSE"] = round(float(np.sqrt(mean_squared_error(y_yr, p_yr))), 2)
            row[f"{key}_Bias"] = round(float(np.mean(p_yr - y_yr)), 2)
        fold_rows.append(row)

    df_folds = pd.DataFrame(fold_rows)
    df_folds.to_csv(out_dir / f"fold_summary_{prefix}.csv", sep=";", index=False)

    decomp_rows = []
    df_per_year = pd.DataFrame({"Test_Year": unique_years})

    for key, label, preds in model_entries:
        preds_arr = np.asarray(preds, dtype=float)
        g_r2, s_r2, t_r2 = compute_r2_decomposition(
            actual_arr,
            preds_arr,
            station_ids,
            station_climatology=station_climatology,
        )
        decomp_rows.append({
            "Model_Key": key,
            "Model_Label": label,
            "Pooled_MAE": round(float(mean_absolute_error(actual_arr, preds_arr)), 3),
            "Pooled_RMSE": round(float(np.sqrt(mean_squared_error(actual_arr, preds_arr))), 3),
            "Mean_Bias": round(float(np.mean(preds_arr - actual_arr)), 3),
            "Global_Pooled_R2": round(g_r2, 4) if np.isfinite(g_r2) else np.nan,
            "Spatial_Station_Mean_R2": round(s_r2, 4) if np.isfinite(s_r2) else np.nan,
            "Temporal_Anomaly_R2": round(t_r2, 4) if np.isfinite(t_r2) else np.nan,
        })

        yearly_r2 = [
            round(float(r2_score(actual_arr[years_arr == yr], preds_arr[years_arr == yr])), 4)
            for yr in unique_years
        ]
        df_per_year[f"{key}_R2"] = yearly_r2

    df_decomp = pd.DataFrame(decomp_rows)
    df_decomp.to_csv(out_dir / f"r2_decomposition_{prefix}.csv", sep=";", index=False)
    df_per_year.to_csv(out_dir / f"r2_per_year_{prefix}.csv", sep=";", index=False)

    return df_folds, df_decomp, df_per_year