# Integrated Multimodal Machine Learning Framework for Short-Term Flash Flood Risk Prediction

This repository reproduces all empirical results, models, and figures of the research paper **"A Multimodal Machine Learning Framework for Short-Term Flash Flood Risk Prediction Using Satellite and Meteorological Data in Khyber Pakhtunkhwa and Balochistan, Pakistan"** (Shahan Anwar, The Superior University, Lahore; Supervisor: Dr. Saleem Mustafa). The framework fuses multi-source satellite observations (Sentinel-1 SAR, VIIRS, SRTM DEM), atmospheric reanalysis (ECMWF ERA5), and precipitation records (NASA GPM, CHIRPS) into two complementary predictive modules: a spatial district-year susceptibility module evaluated under strict spatial GroupKFold cross-validation (Module A), and a short-term temporal early-warning module predicting next-day ($H=1$) and multi-day lead time ($H=3$, $H=7$) flood occurrences (Module B).

## Companion Repositories

This framework synthesizes and benchmarks data across five dedicated companion datasets:
- **[Dataset 1: flood-d1-dadu-sindh](https://github.com/su92-msaiw-f25-006-boop/flood-d1-dadu-sindh):** Daily satellite, soil wetness, and weather observations for Dadu District, Sindh (2022–2024).
- **[Dataset 2: flood-d2-river-flow](https://github.com/su92-msaiw-f25-006-boop/flood-d2-river-flow):** Daily river inflow and dam reservoir monitoring across Pakistan (2021–2025).
- **[Dataset 3: flood-d3-tehsil-flood-extent](https://github.com/su92-msaiw-f25-006-boop/flood-d3-tehsil-flood-extent):** Tehsil-level monsoon flood inundation and Getis-Ord spatial clustering (2022–2025).
- **[Dataset 4: flood-d4-rainfall-kp-balochistan](https://github.com/su92-msaiw-f25-006-boop/flood-d4-rainfall-kp-balochistan):** Long-term decadal precipitation time-series for KP and Balochistan (1981–2025).
- **[Dataset 5: flood-d5-terrain-rain-gee](https://github.com/su92-msaiw-f25-006-boop/flood-d5-terrain-rain-gee):** Topographic, hydrological, and rainfall features built by the author with Google Earth Engine.

## Folder Structure

```
flood-d6-integrated-framework/
├── 1_code/
│   ├── common.py           # Shared metrics, leakage filters, model definitions, and torch LSTM
│   ├── data_a.py           # Feature builder for district-year spatial susceptibility (Module A)
│   ├── prep_b.py           # Fuses daily satellite and meteorology into fused_dadu_daily.csv (Module B)
│   ├── run_a.py            # Executes Module A models with spatial GroupKFold by district
│   ├── run_b.py            # Executes Module B next-day models (train 2022–23, test 2024)
│   └── run_h.py            # Executes Module B lead-time experiments (H=3 and H=7 days)
├── 2_kaggle_input/
│   ├── district_year_table.csv  # Derived district-year tabular dataset for Module A
│   └── fused_dadu_daily.csv     # Derived multimodal daily dataset for Module B (1,094 days)
├── 3_kaggle_output/
│   ├── figures/                 # High-resolution publication figures (Fig1_v4, Fig_resA_v3, etc.)
│   ├── oof_a.csv                # Out-of-fold spatial predictions (Module A)
│   ├── pred_b_test.csv          # Out-of-time test set predictions for H=1 (Module B)
│   ├── pred_b_h3.csv            # Lead-time predictions for H=3 days
│   ├── pred_b_h7.csv            # Lead-time predictions for H=7 days
│   ├── results_a.json           # Evaluation metrics for Module A
│   ├── results_b.json           # Evaluation metrics for Module B (H=1)
│   ├── results_b_h3.json        # Evaluation metrics for H=3 lead time
│   ├── results_b_h7.json        # Evaluation metrics for H=7 lead time
│   ├── results_clim.json        # Climatological baseline comparisons
│   └── results_paired.json      # Statistical significance tests
├── 4_doc/
│   └── README.md                # Package documentation and execution notes
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

## How to Run

### 1. Environment Setup
- Python 3.10 or higher.
- Install the required dependencies:
  ```bash
  pip install -r requirements.txt
  ```

### 2. Execution Pipeline
Run the scripts from within the `1_code/` directory:

1. **Prepare Module B dataset:**
   ```bash
   python prep_b.py
   ```
   Generates `fused_dadu_daily.csv` (1,094 daily records).

2. **Run next-day prediction ($H=1$):**
   ```bash
   python run_b.py
   ```
   Trains Logistic Regression, Random Forest, XGBoost, and PyTorch LSTM models on 2022–2023 data and evaluates out-of-time on 2024 records.

3. **Run multi-day lead-time benchmarks ($H=3, 7$):**
   - On Linux / macOS / Bash:
     ```bash
     H=3 python run_h.py
     H=7 python run_h.py
     ```
   - On Windows PowerShell:
     ```powershell
     $env:H=3; python run_h.py
     $env:H=7; python run_h.py
     ```

4. **Run Module A spatial cross-validation:**
   ```bash
   python run_a.py
   ```
   Evaluates spatial generalizability across unseen districts via GroupKFold cross-validation.

*Note: If reproducing across an external folder layout (e.g., Kaggle mounts), set the environment variable `FLOOD_PKG=/path/to/Flood_Datasets_Package`.*

## Data Source and Licence

- **Code Licence:** The complete codebase is released under the [MIT License](LICENSE).
- **Data Tables:** Derived features in `2_kaggle_input/` are synthesized from the open datasets acknowledged in the companion repositories (NASA GPM, ECMWF ERA5, Sentinel-1 SAR, SRTM DEM, MERIT Hydro, ESA WorldCover, and CHIRPS).

## Citation

```bibtex
@article{anwar2026flashflood,
  author    = {Shahan Anwar and Saleem Mustafa},
  title     = {A Multimodal Machine Learning Framework for Short-Term Flash Flood Risk Prediction Using Satellite and Meteorological Data in Khyber Pakhtunkhwa and Balochistan, Pakistan},
  journal   = {Research Article, Faculty of Computer Science \& IT, The Superior University, Lahore},
  year      = {2026}
}
```
