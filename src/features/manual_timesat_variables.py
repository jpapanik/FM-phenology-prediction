"""Standalone pure-Python implementation of TIMESAT-style NDVI seasonality metrics."""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter


def _double_logistic_curve(
    t: np.ndarray,
    base: float,
    amp: float,
    sos: float,
    r_up: float,
    eos: float,
    r_dn: float,
) -> np.ndarray:
    """
    6-parameter double-logistic model of a single annual crop growing season.

    Parameters
    ----------
    t     : Day-of-year array (1..365).
    base  : Winter/dormant background NDVI level.
    amp   : Seasonal amplitude (height above base).
    sos   : Inflection day of the spring green-up curve.
    r_up  : Steepness (growth rate) of spring green-up.
    eos   : Inflection day of the summer/autumn senescence curve.
    r_dn  : Steepness (decline rate) of senescence.
    """
    # Clip exponents to [-50, 50] so np.exp() never overflows during optimization
    rising_limb = 1.0 / (1.0 + np.exp(-np.clip(r_up * (t - sos), -50.0, 50.0)))
    falling_limb = 1.0 / (1.0 + np.exp(-np.clip(r_dn * (t - eos), -50.0, 50.0)))
    return base + amp * (rising_limb - falling_limb)


def _fit_smooth_ndvi_curve(t: np.ndarray, raw_ndvi: np.ndarray) -> Optional[np.ndarray]:
    """
    Clean missing values and fit a smooth annual NDVI curve before computing thresholds.
    Step 1: Linearly interpolate any isolated NaNs.
    Step 2: Fit a double-logistic curve (TIMESAT-like smooth season model).
    Step 3: If curve_fit fails to converge, fall back to a Savitzky-Golay filter.
    """
    y = np.asarray(raw_ndvi, dtype=float)
    valid = np.isfinite(y)

    # If the entire 365-day trajectory is missing, we cannot compute metrics
    if not valid.any():
        return None

    # Fill isolated NaNs via linear interpolation along DOY
    y_clean = np.interp(t, t[valid], y[valid]) if not valid.all() else y

    # Build data-driven initial guesses (p0) to help curve_fit converge reliably
    base_init = float(np.percentile(y_clean, 10))
    amp_init = max(float(np.percentile(y_clean, 95) - base_init), 0.05)
    peak_day = float(t[int(np.argmax(y_clean))])
    sos_init = max(40.0, min(peak_day - 35.0, 140.0))
    eos_init = max(peak_day + 35.0, min(peak_day + 70.0, 290.0))

    p0 = [base_init, amp_init, sos_init, 0.08, eos_init, 0.08]
    lower_bounds = [-0.2, 0.01, 15.0, 0.005, 110.0, 0.005]
    upper_bounds = [0.8, 1.2, 200.0, 0.5, 340.0, 0.5]

    try:
        popt, _ = curve_fit(
            _double_logistic_curve,
            t,
            y_clean,
            p0=p0,
            bounds=(lower_bounds, upper_bounds),
            maxfev=2500,
        )
        return _double_logistic_curve(t, *popt)
    except Exception:
        # Fallback: smooth with a 31-day cubic Savitzky-Golay filter if non-linear fit fails
        win = min(31, len(y_clean) if len(y_clean) % 2 == 1 else len(y_clean) - 1)
        return savgol_filter(y_clean, window_length=max(win, 5), polyorder=3)


def compute_manual_season_metrics(
    t: np.ndarray,
    y_fit: np.ndarray,
    sos_frac: float = 0.20,
    eos_frac: float = 0.20,
) -> Dict[str, float]:
    """
    Derive 11 TIMESAT-style seasonality metrics from a single fitted 1D curve.

    Parameters
    ----------
    t        : 1D array of days (e.g., 1..365).
    y_fit    : 1D array of smoothed/fitted NDVI values matching `t`.
    sos_frac : Fraction of seasonal amplitude defining Start of Season (default 20%).
    eos_frac : Fraction of seasonal amplitude defining End of Season (default 20%).
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y_fit, dtype=float)

    # 1. Peak of Season (POS): highest point on the smoothed curve
    peak_idx = int(np.argmax(y))
    peak_doy = float(t[peak_idx])
    peak_val = float(y[peak_idx])

    # 2. Base value & Amplitude:
    #    TIMESAT defines the base level as the average of the left (pre-peak)
    #    and right (post-peak) minimum values. Amplitude is peak minus base.
    left_min = float(np.min(y[: peak_idx + 1])) if peak_idx > 0 else float(np.min(y))
    right_min = float(np.min(y[peak_idx:])) if peak_idx < len(y) - 1 else float(np.min(y))
    base_val = 0.5 * (left_min + right_min)
    amp = max(peak_val - base_val, 1e-6)  # Guard against division by zero on flat curves

    # 3. Dynamic amplitude thresholds for Start and End of Season
    sos_level = base_val + sos_frac * amp
    eos_level = base_val + eos_frac * amp

    # 4. Start of Season (SOS):
    #    Scan the rising limb (from day 1 up to the peak) and find the first day
    #    where NDVI crosses above `sos_level`.
    up_idx = np.where(y[: peak_idx + 1] >= sos_level)[0]
    sos_doy = float(t[up_idx[0]]) if len(up_idx) > 0 else float(t[0])

    # 5. End of Season (EOS):
    #    Scan the falling limb (from the peak to day 365) and find the first day
    #    where NDVI drops to or below `eos_level`.
    dn_idx = np.where(y[peak_idx:] <= eos_level)[0]
    eos_doy = float(t[peak_idx + dn_idx[0]]) if len(dn_idx) > 0 else float(t[-1])

    # 6. Length of Season (LOS): duration in days between SOS and EOS
    los_days = max(eos_doy - sos_doy, 0.0)

    # 7. Green-up rate (left derivative) & Senescence rate (right derivative):
    #    Following TIMESAT's convention, compute the slope between the 20% and 80%
    #    amplitude points so flat winter/post-harvest tails do not distort the slope.
    y20 = base_val + 0.20 * amp
    y80 = base_val + 0.80 * amp

    # Rising slope: NDVI gain per day from 20% to 80% on the left side of the peak
    up_20 = np.where(y[: peak_idx + 1] >= y20)[0]
    up_80 = np.where(y[: peak_idx + 1] >= y80)[0]
    if len(up_20) > 0 and len(up_80) > 0 and t[up_80[0]] > t[up_20[0]]:
        rate_increase = float((y80 - y20) / (t[up_80[0]] - t[up_20[0]]))
    else:
        rate_increase = 0.0  # Safe finite fallback if no rising interval is found

    # Falling slope: positive magnitude of NDVI drop per day from 80% to 20% after peak
    dn_80 = np.where(y[peak_idx:] <= y80)[0]
    dn_20 = np.where(y[peak_idx:] <= y20)[0]
    if len(dn_80) > 0 and len(dn_20) > 0 and t[peak_idx + dn_20[0]] > t[peak_idx + dn_80[0]]:
        rate_decrease = float((y80 - y20) / (t[peak_idx + dn_20[0]] - t[peak_idx + dn_80[0]]))
    else:
        rate_decrease = 0.0

    # 8. Seasonal Integrals (trapezoidal area under the curve from SOS to EOS):
    #    - large_integral: total area down to NDVI = 0 (total seasonal productivity)
    #    - small_integral: area above `base_val` only (seasonally active growth)
    season_mask = (t >= sos_doy) & (t <= eos_doy)
    if np.sum(season_mask) >= 2:
        t_seg = t[season_mask]
        y_seg = y[season_mask]
        large_integral = float(np.trapezoid(y_seg, t_seg))
        small_integral = float(np.trapezoid(np.maximum(y_seg - base_val, 0.0), t_seg))
    else:
        large_integral = 0.0
        small_integral = 0.0

    return {
        "sos_doy": sos_doy,
        "eos_doy": eos_doy,
        "los_days": los_days,
        "base_val": base_val,
        "peak_doy": peak_doy,
        "peak_val": peak_val,
        "amplitude": amp,
        "rate_increase": rate_increase,
        "rate_decrease": rate_decrease,
        "large_integral": large_integral,
        "small_integral": small_integral,
    }


def extract_manual_timesat_features(
    df: pd.DataFrame,
    ndvi_cols: List[str],
    sos_frac: float = 0.20,
    eos_frac: float = 0.20,
    prefix: str = "ndvi_tsat",
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Compute the 11 manual TIMESAT-style NDVI metrics for every observation row in `df`.

    Returns
    -------
    df_out    : Copy of `df` with the 11 new columns appended (index preserved).
    tsat_cols : Ordered list of the new column names for feature-set wiring.
    """
    ndvi_matrix = df[ndvi_cols].values.astype(float)
    n_samples, n_days = ndvi_matrix.shape
    t_days = np.arange(1, n_days + 1, dtype=float)

    metric_keys = [
        "sos_doy",
        "eos_doy",
        "los_days",
        "base_val",
        "peak_doy",
        "peak_val",
        "amplitude",
        "rate_increase",
        "rate_decrease",
        "large_integral",
        "small_integral",
    ]
    tsat_cols = [f"{prefix}_{k}" for k in metric_keys]

    rows: List[List[float]] = []
    for i in range(n_samples):
        # 1. Smooth/fit the 365-day NDVI curve for sample i
        y_fit = _fit_smooth_ndvi_curve(t_days, ndvi_matrix[i])

        # 2. Extract the 11 threshold-based metrics (or NaNs if curve was empty)
        if y_fit is None:
            rows.append([np.nan] * len(metric_keys))
        else:
            metrics = compute_manual_season_metrics(
                t=t_days,
                y_fit=y_fit,
                sos_frac=sos_frac,
                eos_frac=eos_frac,
            )
            rows.append([metrics[k] for k in metric_keys])

        # Print clean progress every 500 samples (PowerShell-log friendly, no tqdm \r spam)
        if (i + 1) % 500 == 0 or (i + 1) == n_samples:
            print(f"  [Manual TIMESAT] Processed {i + 1}/{n_samples} NDVI curves...", flush=True)

    df_tsat = pd.DataFrame(rows, columns=tsat_cols, index=df.index)
    df_tsat.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Concatenate onto `df` so Year, Latitude, Longitude, DOY, and embeddings remain intact
    df_out = pd.concat([df, df_tsat], axis=1)
    return df_out, tsat_cols