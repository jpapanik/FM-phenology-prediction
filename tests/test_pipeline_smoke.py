import numpy as np
import pandas as pd

from src.load_data.load_dataset import build_baseline_feature_sets, build_fm_feature_sets
from src.evaluation.metrics import compute_r2_decomposition, get_station_ids, save_fold_summary_and_r2
from src.features.reduction import extract_domain_reduced_features
from src.models.tuners import run_loyocv_experiment, run_pca_fm_loyocv_experiment


def _make_synthetic_df(n_per_year: int = 12) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    years = [2018, 2019, 2020]
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

    preds, actual, yrs = run_loyocv_experiment(
        df_red, {"B1": b_sets["B1"], "B3": b_sets["B3"]}, model_name="rf", n_iter=2, seed=42
    )
    g_r2, s_r2, t_r2 = compute_r2_decomposition(actual, preds["B3"], get_station_ids(df_red))
    assert np.isfinite(g_r2) and np.isfinite(s_r2) and np.isfinite(t_r2)

    save_fold_summary_and_r2(
        [("B1", "B1: Spatial", preds["B1"]), ("B3", "B3: AGDD", preds["B3"])],
        actual,
        yrs,
        get_station_ids(df_red),
        tmp_path,
        "smoke_test",
    )
    assert (tmp_path / "r2_decomposition_smoke_test.csv").exists()
    assert (tmp_path / "r2_per_year_smoke_test.csv").exists()


def test_pca_fm_pipeline():
    df = _make_synthetic_df()
    emb_cols = [f"olmo_{c}" for c in range(32)]
    preds, actual, _ = run_pca_fm_loyocv_experiment(
        df, emb_cols, {"M1": [], "M2": ["Latitude", "Longitude"]}, model_name="svr", n_pca_components=8, n_iter=2
    )
    assert len(preds["M1"]) == len(actual)


def test_extraction_helpers(tmp_path):
    from src.extraction.embeddings import dequantize_olmo, lonlat_to_utm_zone
    from src.extraction.pep725_weather import (
        CROP_TBASE_MAP,
        build_curated_modeling_table,
        combine_raw_agera5_weather,
        compute_daily_gdd_agdd_pivots,
    )
    from src.extraction.sentinel2_ndvi import compute_daily_ndvi_pivots

    assert CROP_TBASE_MAP["BARW"] == 0.0
    assert CROP_TBASE_MAP["MAZ"] == 6.0
    assert lonlat_to_utm_zone(5.66, 51.97) == 31

    q = np.array([-128, 0, 127], dtype=np.int8)
    deq = dequantize_olmo(q)
    assert np.isnan(deq[0]) and deq[1] == 0.0 and np.isclose(deq[2], (127.0 / 127.5) ** 2.0)

    # Test raw AgERA5 combine + Kelvin->C + GDD/AGDD pivot + merge
    raw_dir = tmp_path / "agera5_raw"
    raw_dir.mkdir()
    dates = pd.date_range("2020-01-01", "2020-01-05", freq="D")
    pd.DataFrame({
        "valid_time": dates.strftime("%Y-%m-%d"),
        "latitude": [52.0] * len(dates),
        "longitude": [5.7] * len(dates),
        "Temperature_Air_2m_Max_24h": [288.15] * len(dates),
        "Temperature_Air_2m_Mean_24h": [283.15] * len(dates),
        "Temperature_Air_2m_Min_24h": [278.15] * len(dates),
    }).to_csv(raw_dir / "temperature_0000_52.0_5.7.csv", index=False)

    combined_csv = tmp_path / "AgERA5_observations_combined_with_weather_data.csv"
    weather_df = combine_raw_agera5_weather(raw_dir, combined_csv)
    assert np.isclose(weather_df.loc[0, "Tmean_C"], 10.0)

    gdd_wide, agdd_wide = compute_daily_gdd_agdd_pivots(weather_df, crop_code="BARW")
    assert "GDD_doy_1" in gdd_wide.columns and "AGDD_doy_5" in agdd_wide.columns
    assert np.isclose(agdd_wide.loc[0, "AGDD_doy_5"], 50.0)

    obs_df = pd.DataFrame({
        "Latitude": [52.01],
        "Longitude": [5.72],
        "ObservationDate": ["2020-05-15"],
        "Year": [2020],
        "DOY": [136],
    })
    out_model_csv = tmp_path / "curated_for_modeling.csv"
    curated = build_curated_modeling_table(obs_df, weather_df, crop_code="BARW", output_csv_path=out_model_csv)
    assert out_model_csv.exists() and np.isclose(curated.loc[0, "AGDD_doy_5"], 50.0)

    s2_raw = tmp_path / "s2_raw"
    s2_raw.mkdir()
    ms = int(pd.Timestamp("2020-06-10").timestamp() * 1000)
    pd.DataFrame({
        "time": [ms],
        "B4": [100.0],
        "B8": [500.0],
        "SCL": [4],
        "Station_Latitude": [52.0],
        "Station_Longitude": [5.7],
        "Year": [2020],
    }).to_csv(s2_raw / "s2_0000_2020_52.0000_5.7000.csv", sep=";", index=False)
    ndvi_piv = compute_daily_ndvi_pivots(s2_raw)
    assert "NDVI_doy_162" in ndvi_piv.columns and np.isclose(ndvi_piv.loc[0, "NDVI_doy_162"], 0.6666667)
