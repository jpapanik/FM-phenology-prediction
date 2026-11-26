from pathlib import Path
import hydra
import numpy as np
from omegaconf import DictConfig
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.features.manual_timesat_variables import extract_manual_timesat_features
from src.load_data.load_dataset import build_fm_feature_sets, load_phenology_dataset
from src.evaluation.metrics import get_station_ids, save_fold_summary_and_r2
from src.evaluation.plots import plot_r2_decomposition_and_trajectory, plot_scatter_grid
from src.features.reduction import extract_domain_reduced_features
from src.features.timeseries_curves import extract_catch22_and_dl_features
from src.models.tuners import (
    run_loyocv_experiment,
    run_pca_fm_loyocv_experiment,
    run_pca_fm_temporal_6_1_2_experiment,
    run_temporal_6_1_2_experiment,
)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    if not cfg.embedding.file or not cfg.embedding.prefix:
        raise ValueError("Specify an embedding config, e.g., `embedding=olmoearth` or `embedding=presto`.")

    csv_path = Path(cfg.data_root) / cfg.dataset.folder / cfg.embedding.file
    df, channels = load_phenology_dataset(csv_path, embedding_prefix=cfg.embedding.prefix)
    emb_cols = channels["embedding"]
    emb_upper = cfg.embedding.prefix.upper()
    feat_mode = cfg.features.name
    cfg_cv_years = list(cfg.cv.cv_years) if cfg.cv.get("cv_years") else None
    cfg_test_years = list(cfg.cv.test_years) if cfg.cv.get("test_years") else None

    if feat_mode == "pca_reduced":
        df_reduced = extract_domain_reduced_features(df, embedding_cols=emb_cols)
        gdd_cols = [c for c in df_reduced.columns if c.startswith("GDD_")]
        agdd_cols = [c for c in df_reduced.columns if c.startswith("AGDD_")]
        ndvi_cols = [c for c in df_reduced.columns if c.startswith("NDVI_") or c.startswith("ANDVI_")]
        extra_sets = {
            "M1": [],
            "M2": ["Latitude", "Longitude"],
            "M3": gdd_cols,
            "M4": agdd_cols,
            "M5": ndvi_cols,
            "M6": ["Latitude", "Longitude"] + gdd_cols + agdd_cols + ndvi_cols,
        }
        if cfg.cv.name == "temporal_6_1_2":
            preds_dict, actual_doy, test_years = run_pca_fm_temporal_6_1_2_experiment(
                df=df_reduced,
                embedding_cols=emb_cols,
                extra_feature_sets=extra_sets,
                model_name=cfg.model.name,
                train_years=list(cfg.cv.train_years),
                val_years=list(cfg.cv.val_years),
                test_years=list(cfg.cv.test_years),
                n_pca_components=cfg.features.n_pca_components,
                n_iter=cfg.model.n_iter,
                seed=cfg.seed,
            )
        else:
            preds_dict, actual_doy, test_years = run_pca_fm_loyocv_experiment(
                df=df_reduced,
                embedding_cols=emb_cols,
                extra_feature_sets=extra_sets,
                model_name=cfg.model.name,
                n_pca_components=cfg.features.n_pca_components,
                n_iter=cfg.model.n_iter,
                seed=cfg.seed,
                cv_years=cfg_cv_years,
                test_years=cfg_test_years,
            )
        df_eval = df_reduced
        labels = {
            "M1": f"M1: {emb_upper} Only",
            "M2": f"M2: {emb_upper} + Spatial",
            "M3": f"M3: {emb_upper} + GDD",
            "M4": f"M4: {emb_upper} + AGDD",
            "M5": f"M5: {emb_upper} + NDVI",
            "M6": f"M6: {emb_upper} + ALL",
        }
    elif feat_mode == "ts_curves_c22":
        df_eval, groups = extract_catch22_and_dl_features(
            df, channels["gdd"], channels["agdd"], channels["ndvi"]
        )
        all_c22_dl = groups["gdd_c22"] + groups["agdd_c22"] + groups["ndvi_c22"] + groups["ndvi_dl"]
        feature_sets = {
            "M1_FM_Only": emb_cols,
            "M2_FM_plus_GDD_c22": emb_cols + groups["gdd_c22"],
            "M3_FM_plus_AGDD_c22": emb_cols + groups["agdd_c22"],
            "M4_FM_plus_NDVI_c22": emb_cols + groups["ndvi_c22"],
            "M5_FM_plus_NDVI_DL": emb_cols + groups["ndvi_dl"],
            "M6_FM_plus_All_c22_DL": emb_cols + all_c22_dl,
        }
        labels = {k: k for k in feature_sets}
        if cfg.cv.name == "temporal_6_1_2":
            preds_dict, actual_doy, test_years = run_temporal_6_1_2_experiment(
                df=df_eval,
                feature_sets=feature_sets,
                model_name=cfg.model.name,
                train_years=list(cfg.cv.train_years),
                val_years=list(cfg.cv.val_years),
                test_years=list(cfg.cv.test_years),
                n_iter=cfg.model.n_iter,
                seed=cfg.seed,
            )
        else:
            preds_dict, actual_doy, test_years = run_loyocv_experiment(
                df=df_eval,
                feature_sets=feature_sets,
                model_name=cfg.model.name,
                n_iter=cfg.model.n_iter,
                seed=cfg.seed,
                cv_years=cfg_cv_years,
                test_years=cfg_test_years,
            )
    elif feat_mode == "manual_timesat_variables":
        # Extract 11 standalone TIMESAT-style seasonality metrics from daily NDVI
        df_eval, tsat_cols = extract_manual_timesat_features(
            df,
            channels["ndvi"],
            sos_frac=float(cfg.features.get("sos_frac", 0.20)),
            eos_frac=float(cfg.features.get("eos_frac", 0.20)),
        )
        # Evaluate FM + TIMESAT
        feature_sets = {
            "M1_FM_plus_NDVI_TIMESAT": emb_cols + tsat_cols,
        }
        labels = {
            "M1_FM_plus_NDVI_TIMESAT": f"M3: {emb_upper} + NDVI TIMESAT",
        }
        if cfg.cv.name == "temporal_6_1_2":
            preds_dict, actual_doy, test_years = run_temporal_6_1_2_experiment(
                df=df_eval,
                feature_sets=feature_sets,
                model_name=cfg.model.name,
                train_years=list(cfg.cv.train_years),
                val_years=list(cfg.cv.val_years),
                test_years=list(cfg.cv.test_years),
                n_iter=cfg.model.n_iter,
                seed=cfg.seed,
            )
        else:
            preds_dict, actual_doy, test_years = run_loyocv_experiment(
                df=df_eval,
                feature_sets=feature_sets,
                model_name=cfg.model.name,
                n_iter=cfg.model.n_iter,
                seed=cfg.seed,
                cv_years=cfg_cv_years,
                test_years=cfg_test_years,
            )
    else:
        if feat_mode == "unreduced":
            df_eval = df
            feature_sets = build_fm_feature_sets(df_eval, emb_cols, mode="unreduced")
        elif feat_mode == "domain_reduced":
            df_eval = extract_domain_reduced_features(df, embedding_cols=emb_cols)
            feature_sets = build_fm_feature_sets(df_eval, emb_cols, mode="domain_reduced")
        else:
            raise ValueError(f"Unsupported features.name='{feat_mode}' for run_fm_fusion.py")

        labels = {
            "M1": f"M1: {emb_upper} Only",
            "M2": f"M2: {emb_upper} + Spatial",
            "M3": f"M3: {emb_upper} + GDD",
            "M4": f"M4: {emb_upper} + AGDD",
            "M5": f"M5: {emb_upper} + NDVI",
            "M6": f"M6: {emb_upper} + ALL",
        }
        if cfg.cv.name == "temporal_6_1_2":
            preds_dict, actual_doy, test_years = run_temporal_6_1_2_experiment(
                df=df_eval,
                feature_sets=feature_sets,
                model_name=cfg.model.name,
                train_years=list(cfg.cv.train_years),
                val_years=list(cfg.cv.val_years),
                test_years=list(cfg.cv.test_years),
                n_iter=cfg.model.n_iter,
                seed=cfg.seed,
            )
        else:
            preds_dict, actual_doy, test_years = run_loyocv_experiment(
            df=df_eval,
            feature_sets=feature_sets,
            model_name=cfg.model.name,
            n_iter=cfg.model.n_iter,
            seed=cfg.seed,
            cv_years=cfg_cv_years,
            test_years=cfg_test_years,
            )
    model_entries = [(k, labels[k], preds_dict[k]) for k in labels]

    print(f"\n=== {emb_upper} FUSION BENCHMARK ({cfg.dataset.display_name} | {cfg.model.name} | {feat_mode} | {cfg.cv.name}) ===")
    for k, label, p in model_entries:
        mae = mean_absolute_error(actual_doy, p)
        rmse = np.sqrt(mean_squared_error(actual_doy, p))
        print(f"  {label:26s} -> MAE: {mae:5.2f} d | RMSE: {rmse:5.2f} d")

    all_station_ids = get_station_ids(df_eval)
    if cfg_test_years is not None:
        te_mask = df_eval["Year"].astype(int).isin(cfg_test_years).values
        eval_station_ids = all_station_ids[te_mask]
    else:
        eval_station_ids = all_station_ids

    cv_upper = str(cfg.cv.name).upper()
    cv_tag = f"_{cfg.cv.name}" if cfg.cv.name != "loyocv" else ""
    run_folder = f"{cfg.model.name}_{feat_mode}"
    out_dir = Path(cfg.results_root) / cfg.dataset.folder / f"{cv_upper}_{emb_upper}" / run_folder
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"{emb_upper.lower()}_{cfg.model.name}_{feat_mode}{cv_tag}"
    _, df_decomp, df_per_year = save_fold_summary_and_r2(
        model_entries=model_entries,
        actual_doy=actual_doy,
        years=test_years,
        station_ids=eval_station_ids,
        out_dir=out_dir,
        prefix=suffix,
    )

    title = f"{emb_upper} Fusion ({cfg.model.name.upper()} | {feat_mode} | {cfg.cv.name}) — N={len(actual_doy)}\n{cfg.dataset.display_name}"
    plot_scatter_grid(model_entries, actual_doy, title, out_dir / f"scatter_{suffix}.png")
    plot_r2_decomposition_and_trajectory(df_decomp, df_per_year, title, out_dir, suffix)
    print(f"Saved results and figures to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()