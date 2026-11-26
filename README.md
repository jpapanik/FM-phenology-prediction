# Crop Phenology Prediction with Earth Observation Foundation Models

MSc Thesis repository (**`FM-phenology-prediction`**) benchmarking Sentinel-era Earth Observation Foundation Model embeddings (**PRESTO**, **TESSERA**, **AlphaEarth**, **OlmoEarth**) standalone and fused with meteorological (**GDD**, **AGDD**) and spectral (**NDVI**) time-series predictors to forecast crop phenology Day of Year (DOY) across principal BBCH stages (primary benchmark: German winter barley heading, **BBCH 51**, 2017–2025).

---

## Repository Structure

```text
FM-phenology-prediction/
├── configs/                        # Hydra YAML configuration groups
│   ├── config.yaml                 # Default composition, roots & optional TESSERA alignment filter
│   ├── cv/                         # loyocv (2017–2023 CV -> 2024–2025 test), temporal_6_1_2
│   ├── dataset/                    # Study regions & crop/BBCH targets (e.g., germany_barw_bbch51)
│   ├── embedding/                  # none, presto, tessera, alphaearth, olmoearth
│   ├── extraction/                 # pep725_agera5, sentinel2_ndvi, embeddings
│   ├── features/                   # unreduced, domain_reduced, pca_reduced, ts_curves_c22, manual_timesat_variables
│   └── model/                      # rf, xgboost
├── data/                           # Raw downloads (gitignored)
├── modeling_data/                  # Curated study folders (gitignored)
├── logs/                           # Timestamped PowerShell batch execution logs (gitignored)
├── notebooks/                      # Exploratory Jupyter notebooks
├── results/                        # Structured CSV metrics & diagnostic figures (gitignored)
├── src/
│   ├── data/
│   │   └── load_dataset.py         # Dataset loading, B1–B5 / M1–M6 feature builders & TESSERA row alignment
│   ├── extraction/
│   │   ├── pep725_weather.py       # PEP725 QC, AgERA5 temperature retrieval, GDD/AGDD calculation
│   │   ├── sentinel2_ndvi.py       # Sentinel-2 L2A cloud-masked daily NDVI interpolation
│   │   └── embeddings.py           # PRESTO, TESSERA, AlphaEarth & OlmoEarth extractors
│   ├── features/
│   │   ├── reduction.py            # Domain-reduced thermal & NDVI summary features
│   │   ├── timeseries_curves.py    # Automated catch22 & double-logistic curve parameters
│   │   └── manual_timesat_variables.py  # Standalone pure-Python TIMESAT-style NDVI seasonality metrics
│   ├── models/
│   │   └── tuners.py               # LOYOCV, temporal 6/1/2 & leak-free PCA-FM hyperparameter tuners
│   ├── evaluation/
│   │   ├── metrics.py              # Fold MAE/RMSE, R² decomposition & array-length guards
│   │   └── plots.py                # Scatter grids, R² decomposition & per-year trajectory figures
│   ├── run_extraction.py           # CLI entrypoint for data curation & FM embedding extraction
│   ├── run_baselines.py            # Hydra entrypoint for standalone climate/spectral baselines
│   └── run_fm_fusion.py            # Hydra entrypoint for FM standalone & multimodal fusion benchmarks
└── tests/                          # Unit & leak-free pipeline smoke tests
```

---

## Installation & Setup

### 1. Create Environment & Install Dependencies

**Option A — Conda (recommended, Python 3.12):**
```bash
conda create -n phenology-fm python=3.12 -y
conda activate phenology-fm
pip install -r requirements.txt
pip install -e .
```

**Option B — Standard `venv`:**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

### 2. External Credentials (Only Needed for Raw Extraction)
* **Copernicus Climate Data Store (AgERA5):** Configure `~/.cdsapirc` with your CDS API URL and key.
* **Google Earth Engine (Sentinel-2 NDVI, PRESTO inputs, AlphaEarth):** Run `earthengine authenticate` once and configure your GEE Cloud project (default: `thesis-phenology`).
* **PEP725 Phenology Data:** Place `pep725_data.csv` in the repository root (or pass `--pep-csv /path/to/pep725_data.csv`).

---

## Step 1: Data Curation & Embedding Extraction (`src/run_extraction.py`)

Extraction is modularized into stages controlled by `--study-folder` and `--raw-data-dir` (default: `data/`):

* **Curated outputs saved inside `--study-folder` (e.g., `data/germany_barw_bbch51/`):**
  * `AgERA5_all_observations.csv` — QC-filtered PEP725 station-year observations mapped to the 0.1° AgERA5 grid.
  * `AgERA5_observations_combined_with_weather_data.csv` — combined daily 2m air temperature series (°C).
  * `curated_for_modeling.csv` — semicolon-delimited (`;`) modeling table containing `GDD_doy_1..365`, `AGDD_doy_1..365`, and `NDVI_doy_1..365`.
  * `curated_for_modeling_with_<fm>_embeddings.csv` — enriched tables for `presto`, `tessera`, `alphaearth`, and `olmo`.
* **Raw/intermediate caches saved inside `--raw-data-dir` (default: `data/`):**
  * `data/agERA5_<study_folder>_raw/temperature_*.csv` — raw per-grid-cell CDS downloads.
  * `data/s2_ndvi_raw_<study_folder>/` — raw Sentinel-2 NDVI downloads before merging.
  * `data/ne_110m_admin_0_countries.*` — Natural Earth country boundaries.

> **Time-series convention & leap years:** Daily `GDD`, `AGDD`, `NDVI`, and target `DOY` share a consistent 1–365 day-of-year index (elapsed days since Jan 1). In leap years (`2020`, `2024`), Feb 29 is preserved inside the thermal accumulation and NDVI trajectories while Dec 31 (DOY 366) is truncated.

```bash
STUDY_DIR="data/germany_barw_bbch51"

# 1. Filter PEP725 observations & map stations to the 0.1° AgERA5 grid
python src/run_extraction.py \
  --stage pep725 \
  --study-folder "$STUDY_DIR" \
  --pep-csv pep725_data.csv \
  --country Germany \
  --crop-code BARW \
  --target-bbch 51 \
  --start-date 2017-01-01 \
  --end-date 2025-12-31

# 2. Download AgERA5 temperatures, compute GDD/AGDD (DOY 1..365) & build curated_for_modeling.csv
python src/run_extraction.py \
  --stage agera5 \
  --study-folder "$STUDY_DIR" \
  --crop-code BARW \
  --start-date 2017-01-01 \
  --end-date 2025-12-31

# (Optional) Recompute GDD/AGDD from cached AgERA5 CSVs
python src/run_extraction.py --stage gdd_agdd --study-folder "$STUDY_DIR" --crop-code BARW

# 3. Extract cloud-masked Sentinel-2 L2A NDVI (DOY 1..365) & merge into curated_for_modeling.csv
python src/run_extraction.py --stage s2_ndvi --study-folder "$STUDY_DIR"

# 4. Extract Foundation Model embeddings into separate enriched CSVs
python src/run_extraction.py --stage presto --study-folder "$STUDY_DIR"
python src/run_extraction.py --stage tessera --study-folder "$STUDY_DIR"
python src/run_extraction.py --stage alphaearth --study-folder "$STUDY_DIR"
python src/run_extraction.py --stage olmoearth --study-folder "$STUDY_DIR"
```

### Cross-FM Sample Alignment
To guarantee fair head-to-head comparisons across baselines and all four Foundation Models, `load_phenology_dataset()` supports filtering every dataset to the exact `(Year, Latitude, Longitude, DOY)` observations that contain finite **TESSERA** embeddings (controlled via `tessera_filter_file` in `configs/config.yaml`; override with `tessera_filter_file=null` to disable).

---

## Step 2: Feature Representations & Validation Splits

### Hydra Configuration Groups

| Group | Available Options |
| :--- | :--- |
| `dataset` | `germany_barw_bbch51`, `germany_whbw_bbch0`, `france_sbt_bbch39` |
| `embedding` | `none`, `presto`, `tessera`, `alphaearth`, `olmoearth` |
| `features` | `unreduced`, `domain_reduced`, `pca_reduced`, `ts_curves_c22`, `manual_timesat_variables` |
| `model` | `rf`, `xgboost` (and `svr`) |
| `cv` | `loyocv`, `temporal_6_1_2` |

### Feature Modes
* **`unreduced`**: Full 365-day daily `GDD`, `AGDD`, and `NDVI` trajectories.
* **`domain_reduced`**: Agronomic summary features (monthly GDD + thermal stress, AGDD checkpoints, NDVI LSP + ANDVI).
* **`pca_reduced`** *(FM fusion)*: Leak-free `StandardScaler` + `PCA` compression of FM embeddings combined with domain-reduced meteorological/spectral features.
* **`ts_curves_c22`**: Automated `catch22` time-series descriptors plus double-logistic curve parameters.
* **`manual_timesat_variables`**: Standalone, pure-Python implementation (`src/features/manual_timesat_variables.py`) independent of `reduction.py` and `timeseries_curves.py`. Fits a 6-parameter double-logistic curve (with Savitzky-Golay fallback) to each 365-day NDVI trajectory and extracts **11 TIMESAT-style seasonality metrics**:
  * `sos_doy`, `eos_doy`, `los_days`, `base_val`, `peak_doy`, `peak_val`, `amplitude`, `rate_increase`, `rate_decrease`, `large_integral`, and `small_integral` (with CLI-configurable `features.sos_frac` and `features.eos_frac`, defaulting to `0.20`).

### Validation Strategies (`cv`)
* **`temporal_6_1_2`** — Chronological forward split:
  * **Train:** 6 years (`2017–2022`)
  * **Validation (hyperparameter selection):** 1 year (`2023`)
  * **Held-out Test:** 2 years (`2024–2025`), evaluated after refitting best hyperparameters on `2017–2023`.
  * Under `features=pca_reduced`, `StandardScaler` and `PCA` are fit strictly on `train_years` during validation and refit on `train_years + val_years` before transforming `test_years` (zero test leakage).
* **`loyocv`** — 7-fold Leave-One-Year-Out CV with held-out temporal test:
  * **Hyperparameter Search & Refit (`cv_years`):** 7-fold LOYOCV across `2017–2023` (`RandomizedSearchCV(..., cv=LeaveOneGroupOut(), refit=True)`).
  * **Held-out Test (`test_years`):** Evaluated on the exact same untouched `[2024, 2025]` test set as `temporal_6_1_2` for direct comparison. *(Omitting `cv_years`/`test_years` falls back to full-panel LOYOCV.)*

---

## Step 3: Running Experiments

All runners print real-time per-model validation/test MAE and elapsed durations (`flush=True`).

### A. Standalone Baselines (`src/run_baselines.py`)
Evaluates `B1` (Spatial), `B2` (GDD), `B3` (AGDD), `B4` (NDVI), and `B5` (Combined) — or standalone NDVI TIMESAT metrics under `manual_timesat_variables`:

```bash
# Unreduced 365-day baselines with XGBoost (chronological 6/1/2)
python -u src/run_baselines.py dataset=germany_barw_bbch51 features=unreduced model=xgboost cv=temporal_6_1_2

# Domain-reduced baselines with Random Forest (LOYOCV)
python -u src/run_baselines.py dataset=germany_barw_bbch51 features=domain_reduced model=rf cv=loyocv

# Standalone TIMESAT-style NDVI seasonality metrics (customizable SOS/EOS thresholds)
python -u src/run_baselines.py dataset=germany_barw_bbch51 features=manual_timesat_variables model=xgboost cv=temporal_6_1_2
python -u src/run_baselines.py dataset=germany_barw_bbch51 features=manual_timesat_variables features.sos_frac=0.15 features.eos_frac=0.15 model=rf cv=loyocv
```

### B. Foundation Model & Multimodal Fusion (`src/run_fm_fusion.py`)
Evaluates `M1` (FM Only), `M2` (FM + Spatial), `M3` (FM + GDD), `M4` (FM + AGDD), `M5` (FM + NDVI), and `M6` (FM + Spatial + ALL), plus dedicated fusion sets for `pca_reduced`, `ts_curves_c22`, and `manual_timesat_variables`:

```bash
# OlmoEarth + domain-reduced baselines with XGBoost (temporal 6/1/2)
python -u src/run_fm_fusion.py dataset=germany_barw_bbch51 embedding=olmoearth features=domain_reduced model=xgboost cv=temporal_6_1_2

# TESSERA + PCA-compressed embeddings with Random Forest (temporal 6/1/2)
python -u src/run_fm_fusion.py dataset=germany_barw_bbch51 embedding=tessera features=pca_reduced model=rf cv=temporal_6_1_2

# PRESTO + standalone TIMESAT NDVI features with XGBoost (LOYOCV)
python -u src/run_fm_fusion.py dataset=germany_barw_bbch51 embedding=presto features=manual_timesat_variables model=xgboost cv=loyocv
```

### C. Sequential Batch Execution in PowerShell (Windows)
For unattended multi-embedding / multi-feature sweeps with live console progress and persistent logs, set `$env:PYTHONUNBUFFERED = "1"`, pass `-u` to Python, pipe through `Tee-Object`, and check `$LASTEXITCODE`:

```powershell
$env:PYTHONUNBUFFERED = "1"
$py = "C:\Users\jimpa\anaconda3\envs\phenology-fm\python.exe"

:outer foreach ($emb in @("tessera", "alphaearth", "presto", "olmoearth")) {
    foreach ($model in @("rf", "xgboost")) {
        Write-Host "`n=== Running $emb | manual_timesat_variables | $model | temporal_6_1_2 ===" -ForegroundColor Cyan
        & $py -u src/run_fm_fusion.py dataset=germany_barw_bbch51 features=manual_timesat_variables model=$model cv=temporal_6_1_2 embedding=$emb
        if ($LASTEXITCODE -ne 0) {
            Write-Host "FAILED on $emb ($model) with exit code $LASTEXITCODE" -ForegroundColor Red
            break outer
        }
    }
}
```

### D. Custom Data / Results Directories
```bash
python -u src/run_baselines.py data_root=/path/to/data results_root=/path/to/results
```

---

## Step 4: Output Directory Structure

Outputs are isolated by dataset, validation split, experiment target (`BASELINES` or FM name), and `{model}_{features}` subfolder:

```text
results/
└── germany_barw_bbch51/
    ├── LOYOCV_BASELINES/
    │   ├── rf_unreduced/
    │   ├── rf_domain_reduced/
    │   ├── xgboost_ts_curves_c22/
    │   └── xgboost_manual_timesat_variables/
    ├── TEMPORAL_6_1_2_BASELINES/
    │   └── xgboost_domain_reduced/
    └── TEMPORAL_6_1_2_OLMOEARTH/
        ├── rf_domain_reduced/
        ├── rf_unreduced/
        ├── xgboost_domain_reduced/
        ├── xgboost_pca_reduced/
        └── xgboost_manual_timesat_variables/
            ├── fold_summary_<suffix>.csv
            ├── r2_decomposition_<suffix>.csv
            ├── r2_per_year_<suffix>.csv
            ├── scatter_<suffix>.png
            └── r2_decomp_and_trajectory_<suffix>.png
```

---

## Step 5: Running Tests

Verify feature engineering, leak-free splits, $R^2$ decomposition, and extraction helpers on synthetic data:

```bash
pytest -v
```
