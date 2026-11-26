from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import time
from scipy.stats import loguniform, randint, uniform
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneGroupOut, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def get_estimator_and_search_space(
    model_name: str,
    n_features: int,
    seed: int = 42,
    use_pipeline_prefix: bool = False,
) -> Tuple[Any, Dict[str, Any]]:
    """Return base regressor and hyperparameter distribution for RandomizedSearchCV."""
    p = "model__" if use_pipeline_prefix else ""

    if model_name == "rf":
        estimator = RandomForestRegressor(random_state=seed, n_jobs=-1)
        max_feat = [1.0] if n_features <= 3 else uniform(0.25, 0.50)
        param_dist = {
            f"{p}n_estimators": randint(150, 350),
            f"{p}max_depth": [None, 12, 18, 25],
            f"{p}min_samples_leaf": randint(1, 5),
            f"{p}min_samples_split": randint(2, 6),
            f"{p}max_features": max_feat,
        }
        return estimator, param_dist

    if model_name == "xgboost":
        from xgboost import XGBRegressor

        estimator = XGBRegressor(
            random_state=seed,
            n_jobs=1,
            tree_method="hist",
            objective="reg:squarederror",
        )
        colsample = [1.0] if n_features <= 3 else uniform(0.3, 0.7)
        param_dist = {
            f"{p}n_estimators": randint(150, 450),
            f"{p}max_depth": randint(3, 10),
            f"{p}learning_rate": loguniform(0.01, 0.2),
            f"{p}subsample": uniform(0.6, 0.4),
            f"{p}colsample_bytree": colsample,
            f"{p}min_child_weight": randint(1, 8),
            f"{p}reg_alpha": loguniform(1e-3, 10.0),
            f"{p}reg_lambda": loguniform(1e-2, 10.0),
        }
        return estimator, param_dist

    raise ValueError(f"Unknown model_name '{model_name}'. Expected 'rf', 'xgboost'.")


def run_loyocv_experiment(
    df: pd.DataFrame,
    feature_sets: Dict[str, List[str]],
    model_name: str,
    n_iter: int = 25,
    seed: int = 42,
    cv_years: Optional[List[int]] = None,
    test_years: Optional[List[int]] = None,
) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Run Leave-One-Year-Out Cross-Validation & tune via LOYOCV on cv_years and test on test_years."""
    total_sets = len(feature_sets)
    years_all = df["Year"].astype(int).values
    y_all = df["DOY"].values
    logo = LeaveOneGroupOut()

    overlap = set(cv_years).intersection(test_years)
    if overlap:
        raise ValueError(f"cv_years and test_years overlap on {sorted(overlap)}.")

    cv_mask = np.isin(years_all, cv_years)
    te_mask = np.isin(years_all, test_years)
    if not (cv_mask.any() and te_mask.any()):
        raise ValueError(
            f"Empty split for cv_years={cv_years}, test_years={test_years} "
            f"in available years {sorted(np.unique(years_all))}."
        )

    y_cv = y_all[cv_mask]
    years_cv = years_all[cv_mask]
    actual_test_doy = y_all[te_mask]
    out_test_years = years_all[te_mask]
    predictions: Dict[str, np.ndarray] = {}

    for idx, (model_key, feat_cols) in enumerate(feature_sets.items(), start=1):
            t0 = time.perf_counter()
            print(
            f"[{idx}/{total_sets}] {model_key} ({len(feat_cols)} features) | "
            f"LOYOCV across {len(cv_years)} years...",
            flush=True,
            )
            X_cv = df.loc[cv_mask, feat_cols]
            X_te = df.loc[te_mask, feat_cols]

            base_est, param_dist = get_estimator_and_search_space(
                model_name, len(feat_cols), seed=seed
            )
            search = RandomizedSearchCV(
                estimator=base_est,
                param_distributions=param_dist,
                n_iter=n_iter,
                scoring="neg_mean_absolute_error",
                cv=logo,
                random_state=seed,
                n_jobs=-1,
                refit=True,  # Automatically refits best_estimator_ on all of cv_years (2017-2023)
                verbose=0,
            )
            search.fit(X_cv, y_cv, groups=years_cv)
            preds = search.best_estimator_.predict(X_te)
            predictions[model_key] = preds

            test_mae = float(mean_absolute_error(actual_test_doy, preds))
            print(
                f"  [{idx}/{total_sets}] {model_key} ({len(feat_cols)} features) finished in "
                f"{time.perf_counter() - t0:.1f}s | CV MAE: {-search.best_score_:.2f} d | "
                f"Test MAE: {test_mae:.2f} d",
                flush=True,
            )

    return predictions, actual_test_doy, out_test_years


def run_pca_fm_loyocv_experiment(
    df: pd.DataFrame,
    embedding_cols: List[str],
    extra_feature_sets: Dict[str, List[str]],
    model_name: str,
    n_pca_components: int = 24,
    n_iter: int = 25,
    seed: int = 42,
    cv_years: Optional[List[int]] = None,
    test_years: Optional[List[int]] = None,
) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Leak-free PCA + FM LOYOCV: supports held-out test_years after LOYOCV tuning on cv_years."""
    import time
    from sklearn.metrics import mean_absolute_error

    y_all = df["DOY"].values
    years_all = df["Year"].astype(int).values
    logo = LeaveOneGroupOut()

    if cv_years is not None and test_years is not None:
        overlap = set(cv_years).intersection(test_years)
        if overlap:
            raise ValueError(f"cv_years and test_years overlap on {sorted(overlap)}.")

        cv_mask = np.isin(years_all, cv_years)
        te_mask = np.isin(years_all, test_years)
        if not (cv_mask.any() and te_mask.any()):
            raise ValueError(
                f"Empty split for cv_years={cv_years}, test_years={test_years} "
                f"in available years {sorted(np.unique(years_all))}."
            )

        y_cv = y_all[cv_mask]
        years_cv = years_all[cv_mask]
        actual_test_doy = y_all[te_mask]
        out_test_years = years_all[te_mask]

        # Tune hyperparameters on M1 via 7-fold LOYOCV inside cv_years only
        t0 = time.perf_counter()
        m1_pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=n_pca_components, random_state=seed)),
            ("model", get_estimator_and_search_space(model_name, n_pca_components, seed=seed)[0]),
        ])
        _, raw_param_dist = get_estimator_and_search_space(
            model_name, n_pca_components, seed=seed, use_pipeline_prefix=True
        )
        search = RandomizedSearchCV(
            estimator=m1_pipe,
            param_distributions=raw_param_dist,
            n_iter=n_iter,
            scoring="neg_mean_absolute_error",
            cv=logo,
            random_state=seed,
            n_jobs=-1,
            verbose=0,
        )
        search.fit(df.loc[cv_mask, embedding_cols], y_cv, groups=years_cv)
        best_model_template = search.best_estimator_.named_steps["model"]
        print(
            f"  [PCA-LOYOCV] M1 CV finished in {time.perf_counter() - t0:.1f}s | "
            f"best CV MAE: {-search.best_score_:.2f} d",
            flush=True,
        )

        # Fit scaler + PCA on cv_years (2017-2023) only; transform test_years (2024-2025)
        scaler = StandardScaler()
        pca = PCA(n_components=n_pca_components, random_state=seed)
        emb_cv_pca = pca.fit_transform(scaler.fit_transform(df.loc[cv_mask, embedding_cols]))
        emb_te_pca = pca.transform(scaler.transform(df.loc[te_mask, embedding_cols]))

        predictions: Dict[str, np.ndarray] = {}
        total_sets = len(extra_feature_sets)
        for idx, (model_key, extra_cols) in enumerate(extra_feature_sets.items(), start=1):
            t_set = time.perf_counter()
            if extra_cols:
                X_cv = np.hstack([emb_cv_pca, df.loc[cv_mask, extra_cols].values])
                X_te = np.hstack([emb_te_pca, df.loc[te_mask, extra_cols].values])
            else:
                X_cv = emb_cv_pca
                X_te = emb_te_pca

            final_model = clone(best_model_template)
            final_model.fit(X_cv, y_cv)
            preds = final_model.predict(X_te)
            predictions[model_key] = preds

            test_mae = float(mean_absolute_error(actual_test_doy, preds))
            print(
                f"  [PCA-LOYOCV] [{idx}/{total_sets}] {model_key} finished in "
                f"{time.perf_counter() - t_set:.1f}s | test MAE: {test_mae:.2f} d",
                flush=True,
            )

        return predictions, actual_test_doy, out_test_years

def run_temporal_6_1_2_experiment(
    df: pd.DataFrame,
    feature_sets: Dict[str, List[str]],
    model_name: str,
    train_years: List[int],
    val_years: List[int],
    test_years: List[int],
    n_iter: int = 25,
    seed: int = 42,
) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Chronological split: fit on 6 train years, select on 1 val year, evaluate on 2 test years."""
    from sklearn.metrics import mean_absolute_error
    from sklearn.model_selection import ParameterSampler
    t0 = time.perf_counter()
    total_sets = len(feature_sets)
    years_all = df["Year"].astype(int).values
    y_all = df["DOY"].values
    unique_years = sorted(np.unique(years_all))

    for idx, (model_key, cols) in enumerate(feature_sets.items(), start=1):
        print(
            f"[{idx}/{total_sets}] {model_key} ({len(cols)} features) | "
            f"Validation across {len(val_years)} years...",
            flush=True,
        )

    tr_mask = np.isin(years_all, train_years)
    val_mask = np.isin(years_all, val_years)
    tr_val_mask = np.isin(years_all, list(train_years) + list(val_years))
    te_mask = np.isin(years_all, test_years)

    if not (tr_mask.any() and val_mask.any() and te_mask.any()):
        raise ValueError(
            f"Empty split for train={train_years}, val={val_years}, test={test_years} "
            f"in available years {sorted(np.unique(years_all))}."
        )

    predictions: Dict[str, np.ndarray] = {}
    actual_test_doy = y_all[te_mask]
    out_test_years = years_all[te_mask]

    for idx, (model_key, feat_cols) in enumerate(feature_sets.items(), start=1):
        t0 = time.perf_counter()
        X_tr, y_tr = df.loc[tr_mask, feat_cols], y_all[tr_mask]
        X_val, y_val = df.loc[val_mask, feat_cols], y_all[val_mask]
        X_tr_val, y_tr_val = df.loc[tr_val_mask, feat_cols], y_all[tr_val_mask]
        X_te = df.loc[te_mask, feat_cols]

        base_est, param_dist = get_estimator_and_search_space(model_name, len(feat_cols), seed=seed)
        sampled_params = list(ParameterSampler(param_dist, n_iter=n_iter, random_state=seed))

        best_val_mae = np.inf
        best_params: Dict[str, Any] = {}
        for params in sampled_params:
            est = clone(base_est).set_params(**params)
            est.fit(X_tr, y_tr)
            val_preds = est.predict(X_val)
            val_mae = float(mean_absolute_error(y_val, val_preds))
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_params = params
        print(
            f"  [{idx}/{total_sets}] {model_key} validation finished in {time.perf_counter() - t0:.1f}s | "
            f"best val MAE: {best_val_mae:.2f} d",
            flush=True,
        )
        final_est = clone(base_est).set_params(**best_params)
        final_est.fit(X_tr_val, y_tr_val)
        predictions[model_key] = final_est.predict(X_te)

        test_mae = float(mean_absolute_error(actual_test_doy, predictions[model_key]))
        print(
            f"  [{idx}/{total_sets}] {model_key} test finished in {time.perf_counter() - t0:.1f}s | "
            f"test MAE: {test_mae:.2f} d",
            flush=True,
        )

    return predictions, actual_test_doy, out_test_years

def run_pca_fm_temporal_6_1_2_experiment(
    df: pd.DataFrame,
    embedding_cols: List[str],
    extra_feature_sets: Dict[str, List[str]],
    model_name: str,
    train_years: List[int],
    val_years: List[int],
    test_years: List[int],
    n_pca_components: int = 24,
    n_iter: int = 25,
    seed: int = 42,
) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Chronological 6/1/2 split with StandardScaler + PCA fit only on train/val years."""
    from sklearn.metrics import mean_absolute_error
    from sklearn.model_selection import ParameterSampler

    years_all = df["Year"].astype(int).values
    y_all = df["DOY"].values

    tr_mask = np.isin(years_all, train_years)
    val_mask = np.isin(years_all, val_years)
    tr_val_mask = np.isin(years_all, list(train_years) + list(val_years))
    te_mask = np.isin(years_all, test_years)

    if not (tr_mask.any() and val_mask.any() and te_mask.any()):
        raise ValueError(
            f"Empty split for train={train_years}, val={val_years}, test={test_years} "
            f"in available years {sorted(np.unique(years_all))}."
        )

    # 1. Fit scaler + PCA on train_years only for validation tuning on M1
    t0 = time.perf_counter()
    scaler_tr = StandardScaler()
    pca_tr = PCA(n_components=n_pca_components, random_state=seed)
    emb_tr_pca = pca_tr.fit_transform(scaler_tr.fit_transform(df.loc[tr_mask, embedding_cols]))
    emb_val_pca = pca_tr.transform(scaler_tr.transform(df.loc[val_mask, embedding_cols]))

    y_tr = y_all[tr_mask]
    y_val = y_all[val_mask]

    base_est_m1, param_dist_m1 = get_estimator_and_search_space(
        model_name, n_pca_components, seed=seed
    )
    sampled_params = list(ParameterSampler(param_dist_m1, n_iter=n_iter, random_state=seed))

    best_val_mae = np.inf
    best_params: Dict[str, Any] = {}
    for params in sampled_params:
        est = clone(base_est_m1).set_params(**params)
        est.fit(emb_tr_pca, y_tr)
        val_preds = est.predict(emb_val_pca)
        val_mae = float(mean_absolute_error(y_val, val_preds))
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_params = params

    print(
        f"  [PCA-6/1/2] M1 validation finished in {time.perf_counter() - t0:.1f}s | "
        f"best val MAE: {best_val_mae:.2f} d",
        flush=True,
    )

    # 2. Refit scaler + PCA on train + val years; transform test years
    scaler_tr_val = StandardScaler()
    pca_tr_val = PCA(n_components=n_pca_components, random_state=seed)
    emb_tr_val_pca = pca_tr_val.fit_transform(
        scaler_tr_val.fit_transform(df.loc[tr_val_mask, embedding_cols])
    )
    emb_te_pca = pca_tr_val.transform(
        scaler_tr_val.transform(df.loc[te_mask, embedding_cols])
    )

    y_tr_val = y_all[tr_val_mask]
    actual_test_doy = y_all[te_mask]
    out_test_years = years_all[te_mask]

    predictions: Dict[str, np.ndarray] = {}
    total_sets = len(extra_feature_sets)
    for idx, (model_key, extra_cols) in enumerate(extra_feature_sets.items(), start=1):
        t_set = time.perf_counter()
        if extra_cols:
            X_tr_val = np.hstack([emb_tr_val_pca, df.loc[tr_val_mask, extra_cols].values])
            X_te = np.hstack([emb_te_pca, df.loc[te_mask, extra_cols].values])
            n_total_feats = n_pca_components + len(extra_cols)
        else:
            X_tr_val = emb_tr_val_pca
            X_te = emb_te_pca
            n_total_feats = n_pca_components

        est_template, _ = get_estimator_and_search_space(model_name, n_total_feats, seed=seed)
        final_est = clone(est_template).set_params(**best_params)
        final_est.fit(X_tr_val, y_tr_val)
        predictions[model_key] = final_est.predict(X_te)
        test_mae = float(mean_absolute_error(actual_test_doy, predictions[model_key]))
        print(
            f"  [PCA-6/1/2] [{idx}/{total_sets}] {model_key} finished in {time.perf_counter() - t_set:.1f}s | "
            f"test MAE: {test_mae:.2f} d",
            flush=True,
        )
    return predictions, actual_test_doy, out_test_years