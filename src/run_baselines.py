from pathlib import Path
from typing import List, Optional, Tuple

import hydra
import numpy as np
from omegaconf import DictConfig
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.evaluation.metrics import (
    compute_station_climatology,
    get_station_ids,
    save_fold_summary_and_r2,
)
from src.evaluation.plots import plot_r2_decomposition_and_trajectory, plot_scatter_grid
from src.features.manual_timesat_variables import extract_manual_timesat_features
from src.features.reduction import extract_domain_reduced_features
from src.features.timeseries_curves import extract_catch22_and_dl_features
from src.load_data.load_dataset import build_baseline_feature_sets, load_phenology_dataset
from src.models.tuners import run_loyocv_experiment, run_temporal_6_1_2_experiment


def _build_alignment_refs(cfg: DictConfig, study_dir: Path) -> Optional[List[Tuple[Path, str]]]:
    raw_refs = cfg.get("sample_alignment_refs", None)
    if not raw_refs:
        return None
    return [(study_dir / r["file"], str(r["prefix"])) for r in raw_refs]


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    study_dir = Path(cfg.data_root) / cfg.dataset.folder
    csv_path = study_dir / cfg.dataset.baseline_file
    alignment_refs = _build_alignment_refs(cfg, study_dir)
    strict_alignment = bool(cfg.get("strict_alignment", True))

    df, channels = load_phenology_dataset(
        csv_path,
        embedding_prefix=None,
        alignment_refs=alignment_refs,
        strict_alignment=strict_alignment,
    )

    feat_mode = cfg.features.name
    if feat_mode == "unreduced":
        df_model = df
        feature_sets = build_baseline_feature_sets(df_model, mode="unreduced")
        labels = {
            "B1": "B1: Spatial",
            "B2": "B2: GDD",
            "B3": "B3: AGDD",
            "B4": "B4: NDVI",
            "B5": "B5: Combined",
        }
    elif feat_mode == "domain_reduced":
        df_model = extract_domain_reduced_features(df)
        feature_sets = build_baseline_feature_sets(df_model, mode="domain_reduced")
        labels = {
            "B1": "B1: Spatial",
            "B2": "B2: GDD",
            "B3": "B3: AGDD",
            "B4": "B4: NDVI",
            "B5": "B5: Combined",
        }
    elif feat_mode == "ts_curves_c22":
        df_model, groups = extract_catch22_and_dl_features(
            df, channels["gdd"], channels["agdd"], channels["ndvi"]
        )
        all_c22 = groups["gdd_c22"] + groups["agdd_c22"] + groups["ndvi_c22"]
        feature_sets = {
            "B1_GDD_c22": groups["gdd_c22"],
            "B2_AGDD_c22": groups["agdd_c22"],
            "B3_NDVI_c22": groups["ndvi_c22"],
            "B4_NDVI_DoubleLogistic": groups["ndvi_dl"],
            "B5_All_c22_plus_DL": all_c22 + groups["ndvi_dl"],
        }
        labels = {k: k for k in feature_sets}
    elif feat_mode == "manual_timesat_variables":
        df_model, tsat_cols = extract_manual_timesat_features(
            df,
            channels["ndvi"],
            sos_frac=float(cfg.features.get("sos_frac", 0.20)),
            eos_frac=float(cfg.features.get("eos_frac", 0.20)),
        )
        feature_sets = {
            "B1_NDVI_TIMESAT": tsat_cols,
        }
        labels = {
            "B1_NDVI_TIMESAT": "B1: NDVI TIMESAT",
        }
    else:
        raise ValueError(f"Unsupported features.name='{feat_mode}' for run_baselines.py")

    all_station_ids = get_station_ids(df_model)
    cfg_cv_years = list(cfg.cv.cv_years) if cfg.cv.get("cv_years") else None
    cfg_test_years = list(cfg.cv.test_years) if cfg.cv.get("test_years") else None
    n_jobs = int(cfg.model.get("n_jobs", -1))

    if cfg.cv.name == "temporal_6_1_2":
        train_years = list(cfg.cv.train_years)
        val_years = list(cfg.cv.val_years)
        preds_dict, actual_doy, test_years = run_temporal_6_1_2_experiment(
            df=df_model,
            feature_sets=feature_sets,
            model_name=cfg.model.name,
            train_years=train_years,
            val_years=val_years,
            test_years=cfg_test_years,
            n_iter=cfg.model.n_iter,
            seed=cfg.seed,
            n_jobs=n_jobs,
        )
        ref_years = train_years + val_years
        station_climatology = compute_station_climatology(
            df_model.loc[df_model["Year"].astype(int).isin(ref_years)]
        )
    else:
        preds_dict, actual_doy, test_years = run_loyocv_experiment(
            df=df_model,
            feature_sets=feature_sets,
            model_name=cfg.model.name,
            n_iter=cfg.model.n_iter,
            seed=cfg.seed,
            cv_years=cfg_cv_years,
            test_years=cfg_test_years,
            n_jobs=n_jobs,
        )
        if cfg_cv_years is not None and cfg_test_years is not None:
            station_climatology = compute_station_climatology(
                df_model.loc[df_model["Year"].astype(int).isin(cfg_cv_years)]
            )
        else:
            station_climatology = None

    if cfg_test_years is not None:
        te_mask = df_model["Year"].astype(int).isin(cfg_test_years).values
        eval_station_ids = all_station_ids[te_mask]
    else:
        eval_station_ids = all_station_ids

    model_entries = [(k, labels[k], preds_dict[k]) for k in feature_sets]
    print(
        f"\n=== BASELINES BENCHMARK ({cfg.dataset.display_name} | "
        f"{cfg.model.name} | {feat_mode} | {cfg.cv.name}) ===",
        flush=True,
    )
    for _, label, p in model_entries:
        mae = mean_absolute_error(actual_doy, p)
        rmse = np.sqrt(mean_squared_error(actual_doy, p))
        r2 = r2_score(actual_doy, p)
        print(
            f"  {label:28s} -> MAE: {mae:5.2f} d | RMSE: {rmse:5.2f} d | "
            f"R²: {r2:6.3f}",
            flush=True,
        )

    cv_upper = str(cfg.cv.name).upper()
    cv_tag = f"_{cfg.cv.name}" if cfg.cv.name != "loyocv" else ""
    run_folder = f"{cfg.model.name}_{feat_mode}"
    out_dir = Path(cfg.results_root) / cfg.dataset.folder / f"{cv_upper}_BASELINES" / run_folder
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"baselines_{cfg.model.name}_{feat_mode}{cv_tag}"

    _, df_decomp, df_per_year = save_fold_summary_and_r2(
        model_entries=model_entries,
        actual_doy=actual_doy,
        years=test_years,
        station_ids=eval_station_ids,
        out_dir=out_dir,
        prefix=suffix,
        station_climatology=station_climatology,
    )

    title = (
        f"Baselines ({cfg.model.name.upper()} | {feat_mode} | {cfg.cv.name}) — N={len(actual_doy)}\n"
        f"{cfg.dataset.display_name}"
    )
    plot_scatter_grid(model_entries, actual_doy, title, out_dir / f"scatter_{suffix}.png")
    plot_r2_decomposition_and_trajectory(df_decomp, df_per_year, title, out_dir, suffix)
    print(f"Saved results and figures to: {out_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()