<<<<<<< HEAD
# FM-phenology-predictions
This thesis will use several different FMs for phenology prediction tasks and compare their performance against one another. The repository will contain code for the baseline models, data retrieval, implementaion of the FMs and downstream task.
=======
# Crop Phenology Prediction: Satellite Foundation Model Embeddings & Climate/Spectral Baselines

MSc Thesis repository evaluating post-2017 Sentinel-era Earth Observation Foundation Model embeddings (**PRESTO**, **TESSERA**, **AlphaEarth**, **OlmoEarth**) individually and fused with meteorological (**GDD**, **AGDD**) and spectral (**NDVI**) time-series predictors to forecast crop phenology Day of Year (DOY) across principal BBCH stages under Leave-One-Year-Out Cross-Validation (LOYOCV).

---

## Repository Structure

```text
phenology-fm-benchmark/
├── configs/                 # Hydra YAML configuration groups
│   ├── config.yaml          # Default composition & data/results roots
│   ├── cv/                  # loyocv
│   ├── dataset/             # Study regions & crop/BBCH targets
│   ├── embedding/           # none, presto, tessera, alphaearth, olmoearth
│   ├── extraction/          # pep725_agera5, sentinel2_ndvi, embeddings
│   ├── features/            # unreduced, domain_reduced, pca_reduced, ts_curves_c22
│   └── model/               # rf, xgboost, svr
├── data/output/             # Curated study folders (gitignored)
├── notebooks/               # Original & exploratory Jupyter notebooks
├── results/                 # Output metrics, R² decomposition CSVs & figures (gitignored)
├── src/
│   ├── data/                # Dataset loading & B1–B5 / M1–M6 feature builders
│   ├── extraction/          # PEP725/AgERA5/GDD/AGDD, Sentinel-2 NDVI, PRESTO, TESSERA, AlphaEarth, OlmoEarth
│   ├── features/            # Domain reduction, catch22 & double logistic extraction
│   ├── models/              # LOYOCV / Nested LOYOCV tuners (RF, XGBoost, SVR)
│   ├── evaluation/          # MAE/RMSE, R² decomposition & diagnostic plots
│   ├── run_extraction.py    # CLI entrypoint for data curation & FM embedding extraction
│   ├── run_baselines.py     # Hydra entrypoint for standalone baselines (B1–B5)
│   └── run_fm_fusion.py     # Hydra entrypoint for FM + baseline fusion (M1–M6)
└── tests/                   # Unit & leak-free pipeline smoke tests
```

---

## Installation & Setup

### 1. Create Environment & Install Dependencies
```bash
python -m venv .venv
source .venv/bin/activate      # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

### 2. External Credentials (Only Needed for Raw Extraction)
* **Copernicus Climate Data Store (AgERA5):** Configure `~/.cdsapirc` with your CDS API URL and key.
* **Google Earth Engine (Sentinel-2 NDVI, PRESTO inputs, AlphaEarth):** Run `earthengine authenticate` once and ensure access to your GEE project (default: `thesis-phenology`).
* **PEP725 Phenology Data:** Place `pep725_data.csv` in the repository root (or pass `--pep-csv /path/to/pep725_data.csv`).

---

## Step 1: Data Curation & Embedding Extraction (`src/run_extraction.py`)

Each extraction stage writes intermediate and modeling-ready semicolon-delimited (`;`) CSVs into `data/output/<study_folder>/`:

```bash
STUDY_DIR="germany_barw_bbch51_2017-01-01_2025-12-31"

# 1. Quality-control PEP725 observations & map stations to the 0.1° AgERA5 grid
python src/run_extraction.py \
  --stage pep725 \
  --study-folder "$STUDY_DIR" \
  --pep-csv pep725_data.csv \
  --country Germany \
  --crop-code BARW \
  --target-bbch 51 \
  --start-date 2017-01-01 \
  --end-date 2025-12-31

# 2. Download AgERA5 daily 2m temperatures via CDS API, compute GDD & AGDD (DOY 1..365),
#    and generate curated_for_modeling.csv
python src/run_extraction.py \
  --stage agera5 \
  --study-folder "$STUDY_DIR" \
  --crop-code BARW \
  --start-date 2017-01-01 \
  --end-date 2025-12-31

# (Optional) Recompute GDD/AGDD & rebuild curated_for_modeling.csv from cached weather CSVs
python src/run_extraction.py \
  --stage gdd_agdd \
  --study-folder "$STUDY_DIR" \
  --crop-code BARW

# 3. Extract Sentinel-2 L2A (B4, B8, SCL) from GEE, cloud-mask, interpolate DOY 1..365,
#    and merge NDVI_doy_1..365 into curated_for_modeling.csv
python src/run_extraction.py \
  --stage s2_ndvi \
  --study-folder "$STUDY_DIR"

# 4. Extract foundation model embeddings into separate enriched CSVs
python src/run_extraction.py --stage presto --study-folder "$STUDY_DIR"
python src/run_extraction.py --stage tessera --study-folder "$STUDY_DIR"
python src/run_extraction.py --stage alphaearth --study-folder "$STUDY_DIR"
python src/run_extraction.py --stage olmoearth --study-folder "$STUDY_DIR"
```

---

## Step 2: Running LOYOCV Experiments (Hydra CLI)

The modeling scripts use [Hydra](https://hydra.cc/) so you can override any config group (`dataset`, `embedding`, `features`, `model`, `cv`) directly from the command line without editing Python files.

### Available Config Options
| Group | Options |
| :--- | :--- |
| `dataset` | `germany_barw_bbch51`, `germany_whbw_bbch0`, `france_sbt_bbch39` |
| `embedding` | `none`, `presto`, `tessera`, `alphaearth`, `olmoearth` |
| `features` | `unreduced`, `domain_reduced`, `pca_reduced`, `ts_curves_c22`, `minirocket` |
| `model` | `rf`, `xgboost`, `svr` |
| `cv` | `loyocv`, `nested_loyocv` |

### A. Standalone Baselines (`B1`–`B5`)
Evaluates spatial coordinates (`B1`), daily/reduced GDD (`B2`), AGDD (`B3`), NDVI (`B4`), and combined baselines (`B5`):

```bash
# Unreduced 365-day time series with Random Forest
python src/run_baselines.py dataset=germany_barw_bbch51 features=unreduced model=rf

# Domain-reduced features (monthly GDD + stress, AGDD checkpoints, NDVI LSP + ANDVI) with XGBoost
python src/run_baselines.py dataset=germany_barw_bbch51 features=domain_reduced model=xgboost

# Automated catch22 + double logistic curve parameters with SVR
python src/run_baselines.py dataset=germany_barw_bbch51 features=ts_curves_c22 model=svr

# Nested LOYOCV hyperparameter tuning
python src/run_baselines.py dataset=germany_barw_bbch51 features=domain_reduced model=rf cv=nested_loyocv
```

### B. Foundation Model & Multimodal Fusion (`M1`–`M6`)
Evaluates standalone FM embeddings (`M1`), FM + coordinates (`M2`), FM + GDD (`M3`), FM + AGDD (`M4`), FM + NDVI (`M5`), and FM + all baselines (`M6`):

```bash
# OlmoEarth + domain-reduced baselines with XGBoost
python src/run_fm_fusion.py dataset=germany_barw_bbch51 embedding=olmoearth features=domain_reduced model=xgboost

# PRESTO + unreduced 365-day baselines with Random Forest
python src/run_fm_fusion.py dataset=germany_barw_bbch51 embedding=presto features=unreduced model=rf

# TESSERA + PCA-compressed embeddings + domain-reduced baselines
python src/run_fm_fusion.py dataset=germany_barw_bbch51 embedding=tessera features=pca_reduced model=xgboost

# AlphaEarth on Sugar Beet (France, BBCH 39)
python src/run_fm_fusion.py dataset=france_sbt_bbch39 embedding=alphaearth features=domain_reduced model=rf
```

### Custom Data / Results Directories
Override `DATA_ROOT` or `RESULTS_ROOT` via environment variables or Hydra flags:
```bash
python src/run_baselines.py data_root=/path/to/output results_root=/path/to/results
```

---

## Step 3: Running Tests

Verify feature engineering, leak-free LOYOCV splits, $R^2$ decomposition, and extraction helpers on synthetic data:
```bash
pytest -v
```
>>>>>>> 042f4c8 (Initial commit)
