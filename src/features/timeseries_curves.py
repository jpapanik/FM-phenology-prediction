from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from tqdm import tqdm


def _double_logistic_fn(
    t: np.ndarray,
    wb: float,
    wa: float,
    sos: float,
    r_up: float,
    eos: float,
    r_dn: float,
) -> np.ndarray:
    return wb + wa * (
        1.0 / (1.0 + np.exp(-np.clip(r_up * (t - sos), -50, 50)))
        - 1.0 / (1.0 + np.exp(-np.clip(r_dn * (t - eos), -50, 50)))
    )


def fit_double_logistic_ndvi(ndvi_matrix: np.ndarray, index: pd.Index) -> Tuple[pd.DataFrame, List[str]]:
    """Fit 6-parameter double logistic curves to daily NDVI trajectories and derive POS DOY."""
    t_days = np.arange(1, ndvi_matrix.shape[1] + 1, dtype=float)
    dl_rows = []

    for row_ndvi in tqdm(ndvi_matrix, desc="Double Logistic NDVI"):
        w_base_init = float(np.percentile(row_ndvi, 10))
        w_amp_init = max(float(np.percentile(row_ndvi, 95) - w_base_init), 0.05)
        peak_day = float(np.argmax(row_ndvi) + 1)
        sos_init = max(40.0, min(peak_day - 35.0, 140.0))
        eos_init = max(peak_day + 35.0, min(peak_day + 70.0, 290.0))

        p0 = [w_base_init, w_amp_init, sos_init, 0.08, eos_init, 0.08]
        lower_bounds = [-0.2, 0.01, 15.0, 0.005, 110.0, 0.005]
        upper_bounds = [0.8, 1.2, 200.0, 0.5, 340.0, 0.5]

        try:
            popt, _ = curve_fit(
                _double_logistic_fn,
                t_days,
                row_ndvi,
                p0=p0,
                bounds=(lower_bounds, upper_bounds),
                maxfev=2500,
            )
            wb, wa, sos, r_up, eos, r_dn = popt
            fitted_curve = _double_logistic_fn(t_days, wb, wa, sos, r_up, eos, r_dn)
            pos_doy = float(np.argmax(fitted_curve) + 1)
            dl_rows.append([wb, wa, sos, r_up, eos, r_dn, pos_doy])
        except Exception:
            dl_rows.append([w_base_init, w_amp_init, sos_init, 0.08, eos_init, 0.08, peak_day])

    ndvi_dl_cols = [
        "ndvi_dl_base",
        "ndvi_dl_amp",
        "ndvi_dl_sos_doy",
        "ndvi_dl_greenup_rate",
        "ndvi_dl_eos_doy",
        "ndvi_dl_senescence_rate",
        "ndvi_dl_pos_doy",
    ]
    return pd.DataFrame(dl_rows, columns=ndvi_dl_cols, index=index), ndvi_dl_cols


def extract_catch22_and_dl_features(
    df: pd.DataFrame,
    gdd_cols: List[str],
    agdd_cols: List[str],
    ndvi_cols: List[str],
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """Extract catch22 features for GDD, AGDD, NDVI plus Double Logistic curve parameters."""
    import pycatch22

    gdd_matrix = df[gdd_cols].values.astype(float)
    agdd_matrix = df[agdd_cols].values.astype(float)
    ndvi_matrix = df[ndvi_cols].values.astype(float)

    c22_names = pycatch22.catch22_all(gdd_matrix[0])["names"]

    def _run_c22(mat: np.ndarray, prefix: str) -> Tuple[pd.DataFrame, List[str]]:
        vals = np.array([pycatch22.catch22_all(row)["values"] for row in tqdm(mat, desc=f"{prefix} catch22")])
        cols = [f"{prefix}_c22_{n}" for n in c22_names]
        out = pd.DataFrame(vals, columns=cols, index=df.index)
        out.replace([np.inf, -np.inf], np.nan, inplace=True)
        return out, cols

    df_gdd_c22, gdd_c22_cols = _run_c22(gdd_matrix, "gdd")
    df_agdd_c22, agdd_c22_cols = _run_c22(agdd_matrix, "agdd")
    df_ndvi_c22, ndvi_c22_cols = _run_c22(ndvi_matrix, "ndvi")
    df_ndvi_dl, ndvi_dl_cols = fit_double_logistic_ndvi(ndvi_matrix, df.index)

    df_out = pd.concat([df, df_gdd_c22, df_agdd_c22, df_ndvi_c22, df_ndvi_dl], axis=1)
    groups = {
        "gdd_c22": gdd_c22_cols,
        "agdd_c22": agdd_c22_cols,
        "ndvi_c22": ndvi_c22_cols,
        "ndvi_dl": ndvi_dl_cols,
    }
    return df_out, groups