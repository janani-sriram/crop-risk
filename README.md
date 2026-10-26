# Crop Yield Failure Risk Scoring

This version compares models with and without climate inputs, two failure-label definitions, and random, temporal, and country-held-out evaluation. The original `stage1_model.py` and `stage2_analysis.R` remain in the project.

## Run

Use **Python 3.12** for Streamlit Community Cloud (select it in the app's advanced
settings). `requirements.txt` pins the direct packages imported by the app and training
pipeline to the versions installed in the development `.venv` when this deployment setup
was prepared. Launch the commands from the project root; the app reads only relative
project paths. The website loads its saved model and result tables through Streamlit
caches and does not train models at startup.

```bash
pip install -r requirements.txt
python train.py
streamlit run app.py
```

For Community Cloud, commit the generated `results/crop_risk_models.joblib` and the
app-used tables/plots. Launch with `streamlit run app.py`; the app loads those artifacts
and does not need the raw yield datasets. `train.py` is a local/offline training step and
requires `yield.csv` and, optionally, `yield_df.csv` in the project root. Those raw files
are ignored by Git; check the source Kaggle licence before publishing or redistributing
either dataset.

`train.py` reads both files when available. It writes separate no-climate and climate metric tables, a matched-row climate comparison, paired tests, tuning results, predictions, model artifacts, and ROC/calibration PNGs under `results/`. The paper-ready combined table is `results/paper_comparison_summary.csv`; it includes test failure rate, prevalence-baseline Brier score, and Brier skill score. Existing result snapshots are preserved with `_v1` and `_v2` suffixes.

## Inputs and comparisons

The no-climate source is `yield.csv` (56,717 rows, 1961–2016). The climate source is `yield_df.csv` (28,242 rows, 1990–2013), mapping `average_rain_fall_mm_per_year` to rainfall, `pesticides_tonnes` to pesticides, and `avg_temp` to temperature. Since `yield_df.csv` is a subset with duplicate Area/Item/Year observations, the pipeline also builds a no-climate `yield.csv` subset with the same rows and duplicate weighting. Use this matched comparison to isolate the added climate features; the full-file comparison also reflects differences in coverage.

Rainfall is constant within country in `yield_df.csv`: each of its 101 countries has exactly one unique `average_rain_fall_mm_per_year` value. It therefore acts as a country-level climate proxy. The with-climate random split is optimistic because repeated country-year rows can cross the split. Temporal evaluation shares countries but not years. Unseen-country leave-one-country-out GroupKFold is the strictest test.

Two labels are evaluated: yield below the overall 20th percentile, and yield below the 20th percentile within its crop. The random split is stratified 70/30. Temporal evaluation trains on 1961–2005 and tests on 2006–2016 (the climate file's available temporal range is 1990–2005 / 2006–2013). Country evaluation uses leave-one-country-out GroupKFold.

Every scored comparison includes Single CART raw leaf frequencies; the original single entropy tree with Beta(2,8) leaf posterior; raw-frequency, fixed-prior, and crop-specific-prior random forests; isotonic-calibrated forest; and gradient boosting. Forest parameters are tuned by five-fold Brier CV on the relevant training partition. Isotonic calibration uses a separate validation slice of that training partition. Country-held-out folds use a prespecified compact forest for runtime, and fit isotonic calibration within each fold's training data.

The `yield.csv`, global-label, random-split original Bayesian tree reproduces the paper values: **Brier 0.118546 (0.1185 rounded), AUC 0.839736 (0.8397 rounded)**. This result is written to [`results/original_bayesian_reproduction.csv`](results/original_bayesian_reproduction.csv).

## Main climate comparison

Global-bottom-20 random-split results on the matched rows (ECE is expected
calibration error):

| Model | No-climate Brier | No-climate AUC | No-climate ECE | Climate Brier | Climate AUC | Climate ECE |
|---|---:|---:|---:|---:|---:|---:|
| Single CART (raw k/N) | 0.0966 | 0.8921 | 0.0062 | 0.0848 | 0.9085 | 0.0070 |
| Original Bayesian tree | 0.0967 | 0.8922 | 0.0076 | 0.0849 | 0.9085 | 0.0085 |
| RF raw-frequency | 0.0730 | 0.9479 | 0.0535 | 0.0420 | 0.9875 | 0.0626 |
| RF fixed Beta(2,8) | 0.0791 | 0.9480 | 0.0779 | 0.0489 | 0.9876 | 0.0819 |
| RF crop-specific prior | 0.0745 | 0.9450 | 0.0328 | 0.0424 | 0.9877 | 0.0613 |
| Gradient boosting | 0.1052 | 0.8919 | 0.0706 | 0.0900 | 0.9145 | 0.0634 |
| RF isotonic calibrated | 0.0686 | 0.9478 | 0.0072 | 0.0333 | 0.9873 | 0.0060 |

The test split is random by row, while the climate data contains repeated Area/Item/Year rows; duplicate observations can therefore appear on both sides of a random split. Temporal and country-held-out results provide additional generalization checks. See `results/matched_climate_effect.csv` for Brier, AUC, and ECE changes for every model, label, and split; see the separate `comparison_yield_no_climate.csv` and `comparison_yield_with_climate.csv` tables for the complete metrics.

Paired tests compare each model's squared errors with the original Bayesian tree's on the same evaluation rows. The reported difference is baseline error minus model error, so positive values favor that model. Each row includes a one-sided p-value, 95% CI, and Cohen's dz.

[`results/paired_tests_vs_original.csv`](results/paired_tests_vs_original.csv) reports paired one-sided t and Wilcoxon tests with Holm-adjusted p-values. Country GroupKFold fold-level Brier values and per-model mean/SD summaries are in `results/country_group_fold_brier.csv` and `results/country_group_fold_brier_summary.csv`.
[`results/country_level_tests.csv`](results/country_level_tests.csv) aggregates per-record squared errors within each held-out country, then reports one-sided Wilcoxon tests, median differences, and bootstrap confidence intervals across countries.

## Website

The app defaults to the temporal evaluation model and lets you choose dataset, failure definition, split model, and scoring model. Its headline table compares all models across splits, labels, and climate conditions, with selectable columns. It includes feature importance, downloadable single predictions, and batch CSV scoring. The batch CSV needs `Area`, `Item`, and `Year`; climate columns are optional for the no-climate model and use the source names above for the climate model. Year input is limited to 1961–2016.
