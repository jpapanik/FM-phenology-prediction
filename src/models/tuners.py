import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import LeaveOneGroupOut, ParameterSampler, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def get_estimator_and_search_space(
    model_name: str,
    n_features: int,
    seed: int = 42,
    use_pipeline_prefix: bool = False,
    estimator_n_jobs: int = 1,
) -> Tuple[Any, Dict[str, Any]]:
    """Return base regressor ('rf' or 'xgboost') and regularized hyperparameter search space."""
    p = "model__" if use_pipeline_prefix else ""

    if model_name == "rf":
        estimator = RandomForestRegressor(random_state=seed, n_jobs=estimator_n_jobs)
        max_feat = [1.0] if n_features <= 3 else uniform(0.15, 0.65)
        param_dist = {
            f"{p}n_estimators": randint(150, 400),
            f"{p}max_depth": [4, 6, 10, 15, 22, None],
            f"{p}min_samples_leaf": randint(2, 20),
            f"{p}min_samples_split": randint(4, 25),
            f"{p}max_features": max_feat,
        }
        return estimator, param_dist

    if model_name == "xgboost":
        from xgboost import XGBRegressor

        estimator = XGBRegressor(
            random_state=seed,
            n_jobs=estimator_n_jobs,
            tree_method="hist",
            objective="reg:squarederror",
        )
        colsample = [1.0] if n_features <= 3 else uniform(0.20, 0.70)
        param_dist = {
            f"{p}n_estimators": randint(100, 450),
            f"{p}max_depth": randint(2, 8),
            f"{p}learning_rate": loguniform(0.01, 0.15),
            f"{p}subsample": uniform(0.55, 0.45),
            f"{p}colsample_bytree": colsample,
            f"{p}min_child_weight": randint(2, 25),
            f"{p}reg_alpha": loguniform(1e-3, 30.0),
            f"{p}reg_lambda": loguniform(1e-1, 50.0),
        }
        return estimator, param_dist

    raise ValueError(f"Unknown model_name '{model_name}'. Expected 'rf' or 'xgboost'.")


def _build_pca_preprocessor(
    embedding_cols: List[str],
    extra_cols: List[str],
    n_pca_components: int,
    seed: int,
) -> Any:
    """Build leak-free StandardScaler + PCA transformer for embeddings plus passthrough extra features."""
    emb_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=n_pca_components, random_state=seed)),
    ])
    if not extra_cols:
        return emb_pipe

    return ColumnTransformer(
        transformers=[
            ("emb_pca", emb_pipe, embedding_cols),
            ("extra", SimpleImputer(strategy="median", keep_empty_features=True), extra_cols),
        ],
        remainder="drop",
    )


def run_loyocv_experiment(
    df: pd.DataFrame,
    feature_sets: Dict[str, List[str]],
    model_name: str,
    n_iter: int = 25,
    seed: int = 42,
    cv_years: Optional[List[int]] = None,
    test_years: Optional[List[int]] = None,
    n_jobs: int = -1,
) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """
    Run LOYOCV hyperparameter tuning and evaluation.

    - If `cv_years` and `test_years` are provided: tunes via LOYOCV strictly on `cv_years`,
      refits on all of `cv_years`, and evaluates on held-out `test_years`.
    - Otherwise: falls back to full-panel LOYOCV across all unique years (preserving row order).
    """
    total_sets = len(feature_sets)
    years_all = df["Year"].astype(int).values
    y_all = df["DOY"].to_numpy(dtype=float)
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
                model_name,
                len(feat_cols),
                seed=seed,
                use_pipeline_prefix=True,
                estimator_n_jobs=1,
            )
            pipe = Pipeline([
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("model", base_est),
            ])
            search = RandomizedSearchCV(
                estimator=pipe,
                param_distributions=param_dist,
                n_iter=n_iter,
                scoring="neg_mean_absolute_error",
                cv=logo,
                random_state=seed,
                n_jobs=n_jobs,
                refit=True,
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

    # Fallback: full-panel LOYOCV across all available years (preserves df row order)
    unique_years = sorted(np.unique(years_all))
    predictions = {}

    for idx, (model_key, feat_cols) in enumerate(feature_sets.items(), start=1):
        t0 = time.perf_counter()
        X_all = df[feat_cols]
        base_est, param_dist = get_estimator_and_search_space(
            model_name,
            len(feat_cols),
            seed=seed,
            use_pipeline_prefix=True,
            estimator_n_jobs=1,
        )
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("model", base_est),
        ])
        search = RandomizedSearchCV(
            estimator=pipe,
            param_distributions=param_dist,
            n_iter=n_iter,
            scoring="neg_mean_absolute_error",
            cv=logo,
            random_state=seed,
            n_jobs=n_jobs,
            refit=False,
            verbose=0,
        )
        search.fit(X_all, y_all, groups=years_all)

        oof_preds = np.empty(len(df), dtype=float)
        for yr in unique_years:
            tr_mask = years_all != yr
            te_mask = years_all == yr
            fold_pipe = clone(pipe).set_params(**search.best_params_)
            fold_pipe.fit(X_all.loc[tr_mask], y_all[tr_mask])
            oof_preds[te_mask] = fold_pipe.predict(X_all.loc[te_mask])

        predictions[model_key] = oof_preds
        oof_mae = float(mean_absolute_error(y_all, oof_preds))
        print(
            f"  [{idx}/{total_sets}] {model_key} (full LOYOCV) finished in "
            f"{time.perf_counter() - t0:.1f}s | OOF MAE: {oof_mae:.2f} d",
            flush=True,
        )

    return predictions, y_all.copy(), years_all.copy()


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
    n_jobs: int = -1,
) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """
    Leak-free PCA + FM LOYOCV tuning each fusion subset (M1-M6) independently.

    StandardScaler and PCA are fit strictly inside each training fold via Pipeline.
    Supports held-out `test_years` or full-panel LOYOCV fallback when `cv_years`/`test_years` are None.
    """
    y_all = df["DOY"].to_numpy(dtype=float)
    years_all = df["Year"].astype(int).values
    logo = LeaveOneGroupOut()
    total_sets = len(extra_feature_sets)

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
        predictions: Dict[str, np.ndarray] = {}

        for idx, (model_key, extra_cols) in enumerate(extra_feature_sets.items(), start=1):
            t0 = time.perf_counter()
            all_cols = embedding_cols + extra_cols
            n_effective_feats = n_pca_components + len(extra_cols)
            print(
                f"  [PCA-LOYOCV] [{idx}/{total_sets}] {model_key} "
                f"({n_pca_components} PCs + {len(extra_cols)} extra = {n_effective_feats} features)...",
                flush=True,
            )

            prep = _build_pca_preprocessor(embedding_cols, extra_cols, n_pca_components, seed)
            base_est, param_dist = get_estimator_and_search_space(
                model_name,
                n_effective_feats,
                seed=seed,
                use_pipeline_prefix=True,
                estimator_n_jobs=1,
            )
            pipe = Pipeline([("prep", prep), ("model", base_est)])
            search = RandomizedSearchCV(
                estimator=pipe,
                param_distributions=param_dist,
                n_iter=n_iter,
                scoring="neg_mean_absolute_error",
                cv=logo,
                random_state=seed,
                n_jobs=n_jobs,
                refit=True,
                verbose=0,
            )
            search.fit(df.loc[cv_mask, all_cols], y_cv, groups=years_cv)
            preds = search.best_estimator_.predict(df.loc[te_mask, all_cols])
            predictions[model_key] = preds

            test_mae = float(mean_absolute_error(actual_test_doy, preds))
            print(
                f"  [PCA-LOYOCV] [{idx}/{total_sets}] {model_key} finished in "
                f"{time.perf_counter() - t0:.1f}s | CV MAE: {-search.best_score_:.2f} d | "
                f"Test MAE: {test_mae:.2f} d",
                flush=True,
            )

        return predictions, actual_test_doy, out_test_years

    # Fallback: full-panel LOYOCV across all available years (preserves df row order)
    unique_years = sorted(np.unique(years_all))
    predictions = {}

    for idx, (model_key, extra_cols) in enumerate(extra_feature_sets.items(), start=1):
        t0 = time.perf_counter()
        all_cols = embedding_cols + extra_cols
        n_effective_feats = n_pca_components + len(extra_cols)

        prep = _build_pca_preprocessor(embedding_cols, extra_cols, n_pca_components, seed)
        base_est, param_dist = get_estimator_and_search_space(
            model_name,
            n_effective_feats,
            seed=seed,
            use_pipeline_prefix=True,
            estimator_n_jobs=1,
        )
        pipe = Pipeline([("prep", prep), ("model", base_est)])
        search = RandomizedSearchCV(
            estimator=pipe,
            param_distributions=param_dist,
            n_iter=n_iter,
            scoring="neg_mean_absolute_error",
            cv=logo,
            random_state=seed,
            n_jobs=n_jobs,
            refit=False,
            verbose=0,
        )
        search.fit(df[all_cols], y_all, groups=years_all)

        oof_preds = np.empty(len(df), dtype=float)
        for yr in unique_years:
            tr_mask = years_all != yr
            te_mask = years_all == yr
            fold_pipe = clone(pipe).set_params(**search.best_params_)
            fold_pipe.fit(df.loc[tr_mask, all_cols], y_all[tr_mask])
            oof_preds[te_mask] = fold_pipe.predict(df.loc[te_mask, all_cols])

        predictions[model_key] = oof_preds
        oof_mae = float(mean_absolute_error(y_all, oof_preds))
        print(
            f"  [PCA-LOYOCV] [{idx}/{total_sets}] {model_key} (full LOYOCV) finished in "
            f"{time.perf_counter() - t0:.1f}s | OOF MAE: {oof_mae:.2f} d",
            flush=True,
        )

    return predictions, y_all.copy(), years_all.copy()


def run_temporal_6_1_2_experiment(
    df: pd.DataFrame,
    feature_sets: Dict[str, List[str]],
    model_name: str,
    train_years: List[int],
    val_years: List[int],
    test_years: List[int],
    n_iter: int = 25,
    seed: int = 42,
    n_jobs: int = -1,
) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Chronological split: fit on 6 train years, select on 1 val year, evaluate on 2 test years."""
    total_sets = len(feature_sets)
    years_all = df["Year"].astype(int).values
    y_all = df["DOY"].to_numpy(dtype=float)

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
        print(
            f"[{idx}/{total_sets}] {model_key} ({len(feat_cols)} features) | "
            f"Temporal 6/1/2 tuning...",
            flush=True,
        )

        # Leak-free median imputation fit strictly on train (for val) and train+val (for test)
        imp_tr = SimpleImputer(strategy="median", keep_empty_features=True)
        X_tr = imp_tr.fit_transform(df.loc[tr_mask, feat_cols])
        X_val = imp_tr.transform(df.loc[val_mask, feat_cols])
        y_tr, y_val = y_all[tr_mask], y_all[val_mask]

        base_est, param_dist = get_estimator_and_search_space(
            model_name, len(feat_cols), seed=seed, estimator_n_jobs=n_jobs
        )
        sampled_params = list(ParameterSampler(param_dist, n_iter=n_iter, random_state=seed))

        best_val_mae = np.inf
        best_params: Dict[str, Any] = {}
        for params in sampled_params:
            est = clone(base_est).set_params(**params)
            est.fit(X_tr, y_tr)
            val_mae = float(mean_absolute_error(y_val, est.predict(X_val)))
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_params = params

        imp_tr_val = SimpleImputer(strategy="median", keep_empty_features=True)
        X_tr_val = imp_tr_val.fit_transform(df.loc[tr_val_mask, feat_cols])
        X_te = imp_tr_val.transform(df.loc[te_mask, feat_cols])
        y_tr_val = y_all[tr_val_mask]

        final_est = clone(base_est).set_params(**best_params)
        final_est.fit(X_tr_val, y_tr_val)
        preds = final_est.predict(X_te)
        predictions[model_key] = preds

        test_mae = float(mean_absolute_error(actual_test_doy, preds))
        print(
            f"  [{idx}/{total_sets}] {model_key} finished in {time.perf_counter() - t0:.1f}s | "
            f"Val MAE: {best_val_mae:.2f} d | Test MAE: {test_mae:.2f} d",
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
    n_jobs: int = -1,
) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Chronological 6/1/2 split with leak-free PCA and per-subset hyperparameter tuning."""
    years_all = df["Year"].astype(int).values
    y_all = df["DOY"].to_numpy(dtype=float)

    tr_mask = np.isin(years_all, train_years)
    val_mask = np.isin(years_all, val_years)
    tr_val_mask = np.isin(years_all, list(train_years) + list(val_years))
    te_mask = np.isin(years_all, test_years)

    if not (tr_mask.any() and val_mask.any() and te_mask.any()):
        raise ValueError(
            f"Empty split for train={train_years}, val={val_years}, test={test_years} "
            f"in available years {sorted(np.unique(years_all))}."
        )

    # 1. Fit imputer + scaler + PCA strictly on train_years (for validation tuning)
    emb_pipe_tr = Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=n_pca_components, random_state=seed)),
    ])
    emb_tr_pca = emb_pipe_tr.fit_transform(df.loc[tr_mask, embedding_cols])
    emb_val_pca = emb_pipe_tr.transform(df.loc[val_mask, embedding_cols])

    # 2. Fit imputer + scaler + PCA strictly on train + val years (for final test evaluation)
    emb_pipe_tr_val = Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=n_pca_components, random_state=seed)),
    ])
    emb_tr_val_pca = emb_pipe_tr_val.fit_transform(df.loc[tr_val_mask, embedding_cols])
    emb_te_pca = emb_pipe_tr_val.transform(df.loc[te_mask, embedding_cols])

    y_tr, y_val = y_all[tr_mask], y_all[val_mask]
    y_tr_val = y_all[tr_val_mask]
    actual_test_doy = y_all[te_mask]
    out_test_years = years_all[te_mask]

    predictions: Dict[str, np.ndarray] = {}
    total_sets = len(extra_feature_sets)

    for idx, (model_key, extra_cols) in enumerate(extra_feature_sets.items(), start=1):
        t0 = time.perf_counter()
        if extra_cols:
            imp_tr = SimpleImputer(strategy="median", keep_empty_features=True)
            extra_tr = imp_tr.fit_transform(df.loc[tr_mask, extra_cols])
            extra_val = imp_tr.transform(df.loc[val_mask, extra_cols])
            X_tr = np.hstack([emb_tr_pca, extra_tr])
            X_val = np.hstack([emb_val_pca, extra_val])

            imp_tr_val = SimpleImputer(strategy="median", keep_empty_features=True)
            extra_tr_val = imp_tr_val.fit_transform(df.loc[tr_val_mask, extra_cols])
            extra_te = imp_tr_val.transform(df.loc[te_mask, extra_cols])
            X_tr_val = np.hstack([emb_tr_val_pca, extra_tr_val])
            X_te = np.hstack([emb_te_pca, extra_te])
            n_total_feats = n_pca_components + len(extra_cols)
        else:
            X_tr, X_val = emb_tr_pca, emb_val_pca
            X_tr_val, X_te = emb_tr_val_pca, emb_te_pca
            n_total_feats = n_pca_components

        base_est, param_dist = get_estimator_and_search_space(
            model_name, n_total_feats, seed=seed, estimator_n_jobs=n_jobs
        )
        sampled_params = list(ParameterSampler(param_dist, n_iter=n_iter, random_state=seed))

        best_val_mae = np.inf
        best_params: Dict[str, Any] = {}
        for params in sampled_params:
            est = clone(base_est).set_params(**params)
            est.fit(X_tr, y_tr)
            val_mae = float(mean_absolute_error(y_val, est.predict(X_val)))
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_params = params

        final_est = clone(base_est).set_params(**best_params)
        final_est.fit(X_tr_val, y_tr_val)
        preds = final_est.predict(X_te)
        predictions[model_key] = preds

        test_mae = float(mean_absolute_error(actual_test_doy, preds))
        print(
            f"  [PCA-6/1/2] [{idx}/{total_sets}] {model_key} finished in "
            f"{time.perf_counter() - t0:.1f}s | Val MAE: {best_val_mae:.2f} d | "
            f"Test MAE: {test_mae:.2f} d",
            flush=True,
        )

    return predictions, actual_test_doy, out_test_years