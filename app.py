from pathlib import Path
import json
import re

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from prediction import feature_row, rf_probabilities, score_one

ROOT = Path(".")
RESULTS = ROOT / "results"
MODEL_ORDER = ["Single CART (raw k/N)", "Original Bayesian tree (Beta(2,8))",
               "RF raw-frequency", "RF fixed Beta(2,8)", "RF crop-specific prior",
               "Gradient boosting", "RF isotonic calibrated"]
MODEL_FILE = RESULTS / "crop_risk_models.joblib"


@st.cache_resource
def load_model_artifact(path: str):
    """Load the saved model bundle once per Streamlit process."""
    return joblib.load(path)


@st.cache_data
def load_result_csv(path: str) -> pd.DataFrame:
    """Cache immutable result tables between reruns."""
    return pd.read_csv(path)

st.set_page_config(page_title="Crop Yield Failure Risk", page_icon="🌾", layout="wide")
st.markdown("""<style>
.block-container {max-width:1240px; padding-top:1.7rem}
div[data-testid="stMetric"] {background:#173b35!important; padding:1rem; border-radius:12px; border:1px solid #315c54}
div[data-testid="stMetric"] label,
div[data-testid="stMetric"] [data-testid="stMetricValue"],
div[data-testid="stMetric"] [data-testid="stMetricDelta"] {color:#f3f8f6!important}
</style>""", unsafe_allow_html=True)
st.title("🌾 Crop Yield Failure Risk Scoring")

if not MODEL_FILE.exists():
    st.warning("Model artifacts are missing. Run `python train.py` first.")
    st.stop()
artifact = load_model_artifact(str(MODEL_FILE))
models = artifact["models"]
dataset_options = artifact["dataset_names"]
label_options = artifact["labelings"]


def key_for(dataset, label, split):
    slug = "_".join("".join(ch.lower() if ch.isalnum() else " " for ch in dataset).split())
    return f"{slug}|{label}|{split}"


with st.sidebar:
    st.header("Scenario and model")
    dataset = st.selectbox("Dataset", dataset_options, index=0)
    labeling = st.selectbox("Failure definition", label_options,
                            format_func=lambda s: "Global bottom 20%" if s == "global_bottom_20" else "Bottom 20% within each crop")
    split = st.selectbox("Evaluation model", ["random", "temporal"], index=1,
                         format_func=lambda s: "Random split model" if s == "random" else "Temporal split model")
    selected_model = st.selectbox("Risk model", MODEL_ORDER, index=6)
    entry = models[key_for(dataset, labeling, split)]
    area_names = entry["encoders"]["Area"]
    crop_names = entry["encoders"]["Item"]
    covered_crops = sorted(entry["training_coverage"])
    crop = st.selectbox("Crop", covered_crops or crop_names)
    available_countries = sorted(entry["training_coverage"].get(crop, {}))
    if available_countries:
        country = st.selectbox("Country", available_countries)
    else:
        st.warning("No countries for this crop appear in the selected model's training data.")
        country = ""
    observed_years = entry["training_coverage"].get(crop, {}).get(country, [])
    default_year = max(observed_years) if observed_years else 2005
    year = st.number_input("Year", min_value=1961, max_value=2016, value=int(default_year), step=1)
    scenario = {"Year": year}
    climate_labels = {"rainfall": "Rainfall (mm/year)", "temperature": "Temperature (°C)", "pesticides": "Pesticides (tonnes)"}
    for feature, title in climate_labels.items():
        if feature in entry["numeric"]:
            scenario[feature] = st.number_input(title, value=0.0)


scenario_input = {"Area": country, "Item": crop, "Year": year, **scenario}
if dataset == "yield_df.csv (with climate)":
    for c in ("rainfall", "temperature", "pesticides"):
        scenario_input.setdefault(c, 0.0)
observed_years = entry["training_coverage"].get(crop, {}).get(country, [])
has_training_record = int(year) in observed_years
if not has_training_record:
    st.warning("No data for this crop-country-year combination in the selected model's training data; treat this prediction as uncertain.")
    risk, spread_low, spread_high = np.nan, None, None
else:
    risk, spread_low, spread_high = score_one(scenario_input, entry, selected_model)
if has_training_record and np.isfinite(risk):
    risk_level = "Low" if risk < .10 else ("Medium" if risk < .25 else "High")
    color = {"Low": "#22866b", "Medium": "#d18b20", "High": "#c84b45"}[risk_level]
    c1, c2, c3 = st.columns(3)
    c1.metric("Failure probability", f"{risk:.1%}")
    c1.caption("Estimated chance of yield falling below the selected failure threshold.")
    c2.metric("Risk level", risk_level)
    c2.caption("Low <10%, medium 10–25%, and high ≥25% estimated risk.")
    spread_text = f"{spread_low:.1%}–{spread_high:.1%}" if spread_low is not None else "N/A"
    c3.metric("Tree-ensemble spread", spread_text)
    c3.caption("90% posterior spread across tree leaves when supported by the selected model.")
    st.markdown(f"<div style='height:12px;background:#e7ece9;border-radius:9px'><div style='width:{min(risk*100,100):.1f}%;height:12px;background:{color};border-radius:9px'></div></div>", unsafe_allow_html=True)
    one = pd.DataFrame([{"Area": country, "Item": crop, "Year": year,
                         **{k: v for k, v in scenario_input.items() if k not in ("Area", "Item", "Year")},
                         "dataset": dataset,
                         "failure_definition": labeling, "model": selected_model,
                         "failure_probability": risk, "spread_90_low": spread_low, "spread_90_high": spread_high}])
    st.download_button("Download this prediction as CSV", one.to_csv(index=False), "crop_risk_prediction.csv", "text/csv")
else:
    c1, c2, c3 = st.columns(3)
    c1.metric("Failure probability", "No estimate")
    c1.caption("No training observation supports this crop-country-year selection.")
    c2.metric("Risk level", "No estimate")
    c2.caption("A risk category requires a supported probability estimate.")
    c3.metric("Tree-ensemble spread", "N/A")
    c3.caption("Spread is unavailable without a prediction estimate.")
tabs = st.tabs(["Model comparison", "Risk by crop", "Risk by country", "Batch upload", "About / Method"])
with tabs[0]:
    st.subheader("Key findings")
    st.info("The original Bayesian tree reproduces the reference result on full yield.csv: Brier 0.1185 and AUC 0.8397. On matched rows with the global-bottom-20 label, isotonic RF Brier is 0.0686 without climate and 0.0333 with climate, but duplicate country-year rows can cross the random split, making that result optimistic. Rainfall is constant within country; temporal evaluation shares countries but not years, while unseen-country GroupKFold is the strictest generalization check.")
    headline_path = RESULTS / "paper_comparison_summary.csv"
    if headline_path.exists():
        headline = load_result_csv(str(headline_path))
        filter_cols = st.columns(3)
        climate_filter = filter_cols[0].selectbox("Climate condition", ["without climate", "with climate"], index=0)
        label_filter = filter_cols[1].selectbox("Failure labeling", ["global_bottom_20", "within_crop_bottom_20"], index=0)
        split_filter = filter_cols[2].selectbox("Split", ["random", "temporal", "country_group_kfold"], index=1)
        headline = headline[(headline.climate_condition == climate_filter) &
                            (headline.labeling == label_filter) & (headline.split == split_filter)]
        headline = headline.sort_values("brier_score")
        if climate_filter == "with climate" and split_filter in ("random", "temporal"):
            st.warning("Rainfall is a country-level proxy. With-climate random scores may leak through repeated country-year rows; temporal splits share countries but not years. Unseen-country GroupKFold is stricter.")
        headline_columns = ["climate_condition", "labeling", "split", "model", "brier_score",
                            "brier_skill_score", "roc_auc", "ece", "test_failure_rate", "baseline_brier"]
        st.subheader("Headline results")
        shown_columns = st.multiselect("Headline table columns", headline_columns, default=headline_columns)
        st.dataframe(headline[shown_columns], width="stretch")
    effect_path = RESULTS / "matched_climate_effect.csv"
    if effect_path.exists():
        effect = load_result_csv(str(effect_path))
        effect = effect[effect.labeling == labeling]
        st.subheader("Matched-row effect of adding climate features")
        st.dataframe(effect.style.format({
            "brier_score_no_climate": "{:.4f}", "brier_score_with_climate": "{:.4f}",
            "brier_score_climate_minus_no_climate": "{:+.4f}",
            "roc_auc_no_climate": "{:.4f}", "roc_auc_with_climate": "{:.4f}",
            "roc_auc_climate_minus_no_climate": "{:+.4f}",
            "ece_no_climate": "{:.4f}", "ece_with_climate": "{:.4f}",
            "ece_climate_minus_no_climate": "{:+.4f}"}, na_rep="—"), width="stretch")
    metric_file = RESULTS / ("comparison_yield_with_climate.csv" if dataset == "yield_df.csv (with climate)"
                             else "comparison_yield_matched_no_climate.csv" if dataset == "yield.csv (matched rows; no climate)"
                             else "comparison_yield_no_climate.csv")
    if metric_file.exists():
        table = load_result_csv(str(metric_file))
        table = table[(table.labeling == labeling) & (table.split.isin(["random", "temporal", "country_group_kfold"]))]
        st.dataframe(table[["split", "model", "brier_score", "brier_skill_score", "test_failure_rate",
                            "baseline_brier", "roc_auc", "ece"]].style.format(
            {"brier_score": "{:.4f}", "brier_skill_score": "{:+.4f}", "test_failure_rate": "{:.4f}",
             "baseline_brier": "{:.4f}", "roc_auc": "{:.4f}", "ece": "{:.4f}"}, na_rep="—"), width="stretch")
    split_to_show = "random" if split == "random" else "temporal"
    plot_tag = "_".join("".join(ch.lower() if ch.isalnum() else " " for ch in dataset).split())
    stem = f"{plot_tag}__{labeling}__{split_to_show}"
    a, b = st.columns(2)
    for col, filename in ((a, f"roc__{stem}.png"), (b, f"calibration__{stem}.png")):
        if (RESULTS / filename).exists():
            col.image(str(RESULTS / filename), width="stretch")
    tune_path = RESULTS / "forest_tuning_summary.csv"
    if tune_path.exists():
        tune = load_result_csv(str(tune_path))
        tune = tune[(tune.dataset == dataset) & (tune.labeling == labeling) & (tune.split == split)]
        if not tune.empty:
            st.caption(f"Training-only 5-fold RF tuning: {tune.iloc[0].best_params}; mean CV Brier {tune.iloc[0].best_cv_brier:.4f}.")
    fold_summary_path = RESULTS / "country_group_fold_brier_summary.csv"
    if fold_summary_path.exists():
        fold_summary = load_result_csv(str(fold_summary_path))
        fold_summary = fold_summary[(fold_summary.dataset == dataset) & (fold_summary.labeling == labeling)]
        if not fold_summary.empty:
            st.subheader("Country-held-out Brier variability across folds")
            st.dataframe(fold_summary[["model", "folds", "mean_brier", "sd_brier"]].sort_values("mean_brier").style.format(
                {"mean_brier": "{:.4f}", "sd_brier": "{:.4f}"}), width="stretch")
    country_test_path = RESULTS / "country_level_tests.csv"
    if country_test_path.exists():
        country_tests = load_result_csv(str(country_test_path))
        country_tests = country_tests[(country_tests.dataset == dataset) & (country_tests.labeling == labeling)]
        if not country_tests.empty:
            st.subheader("Country-level paired tests vs Original Bayesian tree")
            st.dataframe(country_tests[["model_vs_original", "n_countries", "median_brier_difference",
                "bootstrap_ci95_low", "bootstrap_ci95_high", "wilcoxon_p_value_one_sided",
                "wilcoxon_p_value_holm"]].style.format({
                    "median_brier_difference": "{:+.5f}", "bootstrap_ci95_low": "{:+.5f}",
                    "bootstrap_ci95_high": "{:+.5f}", "wilcoxon_p_value_one_sided": "{:.3g}",
                    "wilcoxon_p_value_holm": "{:.3g}"}), width="stretch")
    if selected_model.startswith("RF"):
        importance = entry["forest"].feature_importances_
    elif selected_model == "Gradient boosting":
        importance = entry["boosting"].feature_importances_
    else:
        importance = entry["tree"].feature_importances_
    imp = pd.DataFrame({"feature": entry["feature_names"], "importance": importance})
    imp = imp[imp.feature != "domain_encoded"]
    imp = imp.sort_values("importance", ascending=False).head(15)
    st.subheader("Feature importance")
    chart = alt.Chart(imp).mark_bar(color="#397b72").encode(
        x=alt.X("feature:N", sort=alt.SortField(field="importance", order="descending"), title=None),
        y=alt.Y("importance:Q", scale=alt.Scale(domainMin=0), title="Importance"),
        tooltip=["feature", alt.Tooltip("importance:Q", format=".4f")],
    ).properties(height=330)
    st.altair_chart(chart, width="stretch")
with tabs[1]:
    group = load_result_csv(str(RESULTS / "comparison_predictions.csv"))
    group = group[(group.dataset == dataset) & (group.labeling == labeling) & (group.split == "random")]
    if not group.empty and "Item" in group:
        st.bar_chart(group.groupby("Item")["actual_failure"].mean())
with tabs[2]:
    group = load_result_csv(str(RESULTS / "comparison_predictions.csv"))
    group = group[(group.dataset == dataset) & (group.labeling == labeling) & (group.split == "random")]
    if not group.empty and "Area" in group:
        st.bar_chart(group.groupby("Area")["actual_failure"].mean())
with tabs[3]:
    uploaded = st.file_uploader("Upload a CSV with Area, Item, and Year columns", type=["csv"])
    if uploaded is not None:
        batch = pd.read_csv(uploaded)
        required = {"Area", "Item", "Year"}
        if not required.issubset(batch.columns):
            st.error("Upload must include Area, Item, and Year columns.")
        else:
            results = []
            for _, row in batch.iterrows():
                values = row.to_dict()
                raw_year = values.get("Year")
                covered = pd.notna(raw_year) and int(raw_year) in entry["training_coverage"].get(str(values.get("Item", "")), {}).get(str(values.get("Area", "")), [])
                if covered:
                    p, low, high = score_one(values, entry, selected_model)
                else:
                    p, low, high = np.nan, None, None
                results.append({**values, "dataset": dataset, "failure_definition": labeling,
                                "model": selected_model, "failure_probability": p,
                                "spread_90_low": low, "spread_90_high": high,
                                "prediction_status": "scored" if covered else "no training data; no estimate"})
            scored = pd.DataFrame(results)
            st.dataframe(scored.head(100), width="stretch")
            st.download_button("Download scored batch CSV", scored.to_csv(index=False), "crop_risk_batch.csv", "text/csv")
with tabs[4]:
    st.markdown("""### Method
Both global bottom-20% and within-crop bottom-20% failure labels are available. Temporal
models train on 1961–2005 and test on 2006–2016; random models use a stratified 70/30 split.
The country table uses leave-one-country-out GroupKFold. Original CART and Beta-tree baselines
match the stage1 entropy tree setup. Forest tuning uses five-fold CV within training data;
isotonic calibration uses a training-only validation subset. `yield_df.csv` maps rainfall from
`average_rain_fall_mm_per_year`, pesticides from `pesticides_tonnes`, and temperature from
`avg_temp`. The tree-ensemble spread summarizes posterior uncertainty across fitted forest
leaves, rather than a population-level guarantee.""")
    st.warning("In yield_df.csv, average_rain_fall_mm_per_year has one unique value within each of the 101 countries and acts as a country-level climate proxy. With-climate random split is optimistic because repeated country-year rows can cross the split. Temporal split shares countries but not years. Unseen-country leave-one-country-out GroupKFold is the strictest test.")
    cov = RESULTS / "dataset_coverage.csv"
    if cov.exists():
        st.dataframe(load_result_csv(str(cov)), width="stretch")
