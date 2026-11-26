import numpy as np
import pandas as pd

from src.load_data.load_dataset import build_baseline_feature_sets, build_fm_feature_sets
from src.evaluation.metrics import compute_r2_decomposition, get_station_ids, save_fold_summary_and_r2
from src.features.reduction import extract_domain_reduced_features
from src.models.tuners import run_loyocv_experiment, run_pca_fm_loyocv_experiment


def _make_synthetic_df(n_per_year: int = 12) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    years = [2018, 2019, 2020, 2021]
    rows = []
    for yr in years:
        for i in range(n_per_year):
            row = {
                "PEP_ID": f"ST_{i % 4}",
                "Year": yr,
                "Latitude": 48.0 + (i % 4) * 0.5,
                "Longitude": 8.0 + (i % 4) * 0.5,
                "DOY": 120 + (i % 4) * 3 + (yr - 2019) * 2,
            }
            for d in range(1, 366):
                row[f"GDD_doy_{d}"] = max(0.0, rng.normal(5.0, 1.0))
                row[f"AGDD_doy_{d}"] = d * 4.5
                row[f"NDVI_doy_{d}"] = 0.2 + 0.5 * np.exp(-((d - 160) / 50.0) ** 2)
            for c in range(32):
                row[f"olmo_{c}"] = rng.normal(0.0, 1.0)
            rows.append(row)
    return pd.DataFrame(rows)


def test_domain_reduction_and_r2_decomposition(tmp_path):
    df = _make_synthetic_df()
    emb_cols = [f"olmo_{c}" for c in range(32)]
    df_red = extract_domain_reduced_features(df, embedding_cols=emb_cols)

    b_sets = build_baseline_feature_sets(df_red, mode="domain_reduced")
    assert set(b_sets.keys()) == {"B1", "B2", "B3", "B4", "B5"}

    fm_sets = build_fm_feature_sets(df_red, emb_cols, mode="domain_reduced")
    assert set(fm_sets.keys()) == {"M1", "M2", "M3", "M4", "M5", "M6"}

    cv_yrs, te_yrs = [2018, 2019], [2020, 2021]
    preds, actual, yrs = run_loyocv_experiment(
        df_red,
        {"B1": b_sets["B1"], "B3": b_sets["B3"]},
        model_name="rf",
        n_iter=2,
        seed=42,
        cv_years=cv_yrs,
        test_years=te_yrs,
    )
    eval_stations = get_station_ids(df_red[df_red["Year"].isin(te_yrs)])
    g_r2, s_r2, t_r2 = compute_r2_decomposition(actual, preds["B3"], eval_stations)
    assert np.isfinite(g_r2) and np.isfinite(s_r2) and np.isfinite(t_r2)

    save_fold_summary_and_r2(
        [("B1", "B1: Spatial", preds["B1"]), ("B3", "B3: AGDD", preds["B3"])],
        actual,
        yrs,
        eval_stations,
        tmp_path,
        "smoke_test",
    )
    assert (tmp_path / "r2_decomposition_smoke_test.csv").exists()
    assert (tmp_path / "r2_per_year_smoke_test.csv").exists()


def test_pca_fm_pipeline():
    df = _make_synthetic_df()
    emb_cols = [f"olmo_{c}" for c in range(32)]
    preds, actual, _ = run_pca_fm_loyocv_experiment(
        df,
        emb_cols,
        {"M1": [], "M2": ["Latitude", "Longitude"]},
        model_name="rf",
        n_pca_components=8,
        n_iter=2,
        cv_years=[2018, 2019],
        test_years=[2020, 2021],
    )
    assert len(preds["M1"]) == len(actual)