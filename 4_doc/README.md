# Dataset 6 – Integrated Multimodal Framework (Dadu daily + district-year)

Reproduces all results of the paper "A Multimodal Machine Learning Framework for Short-Term Flash Flood Risk Prediction Using Satellite and Meteorological Data in Khyber Pakhtunkhwa and Balochistan, Pakistan".

Inputs (Datasets 1–5 in the sibling folders):
D1 Dadu daily satellite/meteorology · D2 river flow · D3 tehsil/district flood extent · D4 rainfall 1981–2025 · D5 terrain + rain features (GEE, built by us).

Run order (from 1_code, Python 3.10+, pandas, numpy, scikit-learn, xgboost, torch, statsmodels):
1. `python prep_b.py`  -> fused_dadu_daily.csv (Module B table, 1094 days)
2. `python run_b.py`   -> results_b.json, pred_b_test.csv (next-day, train 2022–23, test 2024)
3. `H=3 python run_h.py`, `H=7 python run_h.py` -> lead-time results
4. `python run_a.py`   -> results_a.json, oof_a.csv (Module A, district-year, GroupKFold by district)

Set `FLOOD_PKG=/path/to/Flood_Datasets_Package` if the folder layout differs (Kaggle: point it to your input mount).
Outputs shipped in 3_kaggle_output (JSON metrics, predictions, figures). 2_kaggle_input holds the two derived tables.

Notes: D2 dates are reconstructed from day-month text; leakage columns (gi_z, hotspot, lisa, same-year flooded %) are excluded; scalers fit on training data only.
