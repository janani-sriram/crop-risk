"""Full comparison pipeline for the crop-yield risk project."""
from pathlib import Path
import json
import shutil
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import beta as beta_dist, t, wilcoxon
from sklearn.calibration import calibration_curve
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, GroupKFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)
RANDOM_STATE = 42
ALPHA, BETA = 2.0, 8.0
MODEL_ORDER = ["Single CART (raw k/N)", "Original Bayesian tree (Beta(2,8))",
               "RF raw-frequency", "RF fixed Beta(2,8)", "RF crop-specific prior",
               "Gradient boosting", "RF isotonic calibrated"]


def preserve_prior_outputs():
    """Snapshot existing unsuffixed outputs before this expanded run replaces them."""
    for path in OUT.iterdir():
        if path.is_file() and not any(path.stem.endswith(f"_{v1}") for v1 in ("v1", "v2")):
            target = path.with_name(f"{path.stem}_v2{path.suffix}")
            if not target.exists():
                shutil.copy2(path, target)


def read_canonical(path):
    raw = pd.read_csv(path)
    raw.columns = [str(c).strip() for c in raw.columns]
    target = next((c for c in ("Value", "hg/ha_yield", "yield", "Yield") if c in raw), None)
    if target is None or "Area" not in raw or "Item" not in raw or "Year" not in raw:
        raise ValueError(f"{path.name} needs Area, Item, Year, and a yield column")
    df = pd.DataFrame({"Area": raw["Area"].astype(str), "Item": raw["Item"].astype(str),
                       "Year": pd.to_numeric(raw["Year"], errors="coerce"),
                       "yield": pd.to_numeric(raw[target], errors="coerce")})
    climate_sources = {
        "rainfall": ("average_rain_fall_mm_per_year", "rainfall"),
        "pesticides": ("pesticides_tonnes", "pesticides"),
        "temperature": ("avg_temp", "temperature"),
    }
    for dest, candidates in climate_sources.items():
        source = next((c for c in candidates if c in raw), None)
        if source:
            df[dest] = pd.to_numeric(raw[source], errors="coerce")
    df = df.dropna(subset=["Area", "Item", "Year", "yield"]).reset_index(drop=True)
    df["Year"] = df.Year.astype(int)
    df["_country"] = df.Area
    df["_crop"] = df.Item
    climate = all(c in df for c in ("rainfall", "pesticides", "temperature"))
    return df, climate


def build_datasets():
    no_climate, _ = read_canonical(ROOT / "yield.csv")
    datasets = {"yield.csv (no climate; full)": no_climate}
    if (ROOT / "yield_df.csv").exists():
        climate, _ = read_canonical(ROOT / "yield_df.csv")
        datasets["yield_df.csv (with climate)"] = climate
        # Reweight the no-climate source to exactly the climate file's row keys and duplicates.
        key_index = no_climate.set_index(["Area", "Item", "Year"])
        matched = key_index.loc[pd.MultiIndex.from_frame(climate[["Area", "Item", "Year"]])].reset_index()
        matched["_country"] = matched.Area
        matched["_crop"] = matched.Item
        datasets["yield.csv (matched rows; no climate)"] = matched
    return datasets


def labels_for(df):
    y_global = (df["yield"] < df["yield"].quantile(.20)).astype(int).to_numpy()
    crop_cutoff = df.groupby("Item")["yield"].transform(lambda s: s.quantile(.20))
    y_crop = (df["yield"] < crop_cutoff).astype(int).to_numpy()
    return {"global_bottom_20": y_global, "within_crop_bottom_20": y_crop}


def features_for(df):
    # Retain the original stage1_model.py LabelEncoder representation for Item/Area.
    categorical = [c for c in ("Item", "Area") if c in df]
    X = pd.DataFrame(index=df.index)
    encoders = {}
    for col in categorical:
        enc = LabelEncoder().fit(df[col].astype(str))
        X[f"{col.lower()}_encoded"] = enc.transform(df[col].astype(str))
        encoders[col] = enc.classes_.tolist()
    numeric = [c for c in ("Year", "rainfall", "temperature", "pesticides") if c in df]
    for col in numeric:
        X[col] = pd.to_numeric(df[col], errors="coerce")
        X[col] = X[col].fillna(X[col].median()).fillna(0)
    return X.astype(float), encoders, categorical, numeric


def safe_auc(y, p):
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else np.nan


def ece_score(y, p, n_bins=10):
    table = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    table["bin"] = pd.qcut(table.p.rank(method="first"), q=min(n_bins, len(table)), labels=False)
    grouped = table.groupby("bin", observed=True).agg(mean_p=("p", "mean"), rate=("y", "mean"), n=("y", "size"))
    return float(np.sum(grouped.n / len(table) * np.abs(grouped.mean_p - grouped.rate)))


def score_row(dataset, labeling, split, model, y, p, status="ok"):
    failure_rate = float(np.mean(y))
    baseline_brier = failure_rate * (1.0 - failure_rate)
    if p is None:
        return {"dataset": dataset, "labeling": labeling, "split": split, "model": model,
                "brier_score": np.nan, "test_failure_rate": failure_rate,
                "baseline_brier": baseline_brier, "brier_skill_score": np.nan,
                "roc_auc": np.nan, "ece": np.nan, "status": status}
    brier = brier_score_loss(y, p)
    return {"dataset": dataset, "labeling": labeling, "split": split, "model": model,
            "brier_score": brier, "test_failure_rate": failure_rate,
            "baseline_brier": baseline_brier,
            "brier_skill_score": 1 - brier / baseline_brier if baseline_brier else np.nan,
            "roc_auc": safe_auc(y, p), "ece": ece_score(y, p), "status": status}


def tree_counts(tree, X_train, y_train):
    leaves = tree.apply(X_train.to_numpy())
    grouped = pd.DataFrame({"leaf": leaves, "y": np.asarray(y_train)}).groupby("leaf").y.agg(["sum", "count"])
    return {int(leaf): [float(row["sum"]), int(row["count"])] for leaf, row in grouped.iterrows()}


def forest_leaf_stats(forest, X_train, y_train, crops):
    stats, crop_stats = [], []
    for tree in forest.estimators_:
        leaves = tree.apply(X_train.to_numpy())
        tab = pd.DataFrame({"leaf": leaves, "y": np.asarray(y_train), "crop": np.asarray(crops)})
        cnt = tab.groupby("leaf").y.agg(["sum", "count"])
        stats.append({int(leaf): [float(row["sum"]), int(row["count"])] for leaf, row in cnt.iterrows()})
        by_crop = tab.groupby(["leaf", "crop"]).y.agg(["sum", "count"])
        crop_stats.append({f"{int(leaf)}\t{crop}": [float(row["sum"]), int(row["count"])]
                           for (leaf, crop), row in by_crop.iterrows()})
    return stats, crop_stats


def forest_predict(forest, stats, crop_stats, X_train, y_train, crops_train, X_eval, crops_eval,
                   posterior_draws=0):
    beta_means, raw_means, crop_means = [], [], []
    rng = np.random.default_rng(RANDOM_STATE)
    draws = np.zeros((posterior_draws, len(X_eval)), dtype=float) if posterior_draws else None
    crop_rates = pd.Series(np.asarray(y_train)).groupby(np.asarray(crops_train)).mean()
    for j, tree in enumerate(forest.estimators_):
        leaf_train = tree.apply(X_train.to_numpy())
        leaf_eval = tree.apply(X_eval.to_numpy())
        y_by_leaf = pd.DataFrame({"leaf": leaf_train, "y": np.asarray(y_train)}).groupby("leaf").y.agg(["sum", "count"])
        n = y_by_leaf["count"].reindex(leaf_eval).to_numpy(float)
        k = y_by_leaf["sum"].reindex(leaf_eval).to_numpy(float)
        raw_means.append(k / n)
        beta_means.append((k + ALPHA) / (n + ALPHA + BETA))
        priors = pd.Series(np.asarray(crops_eval)).map(crop_rates).fillna(float(np.mean(y_train))).to_numpy()
        pairs = pd.MultiIndex.from_arrays([leaf_eval, np.asarray(crops_eval)])
        y_by_crop = pd.DataFrame({"leaf": leaf_train, "y": np.asarray(y_train), "crop": np.asarray(crops_train)}).groupby(["leaf", "crop"]).y.agg(["sum", "count"])
        matched = y_by_crop.reindex(pairs)
        ck = matched["sum"].fillna(0).to_numpy(float)
        cn = matched["count"].fillna(0).to_numpy(float)
        crop_means.append((ck + priors * 10) / (cn + 10))
        if posterior_draws:
            draws += rng.beta(k[None, :] + ALPHA, n[None, :] - k[None, :] + BETA,
                              size=(posterior_draws, len(X_eval))) / len(forest.estimators_)
    prediction = {"RF raw-frequency": np.mean(raw_means, axis=0),
                  "RF fixed Beta(2,8)": np.mean(beta_means, axis=0),
                  "RF crop-specific prior": np.mean(crop_means, axis=0)}
    interval = np.quantile(draws, [.05, .95], axis=0).T if posterior_draws else None
    return prediction, interval


def paired_rows(dataset, labeling, split, y, predictions):
    base = predictions["Original Bayesian tree (Beta(2,8))"]
    rows = []
    for name, p in predictions.items():
        if name == "Original Bayesian tree (Beta(2,8))" or p is None:
            continue
        diff = (base - y) ** 2 - (p - y) ** 2  # positive favors the compared model
        n = len(diff); mean = float(np.mean(diff)); sd = float(np.std(diff, ddof=1))
        se = sd / np.sqrt(n); stat = mean / se if se else 0.0; crit = float(t.ppf(.975, n - 1))
        p_one_sided = float(t.sf(stat, n - 1))
        if p_one_sided == 0.0:
            p_one_sided = float(np.nextafter(0.0, 1.0))
        rows.append({"dataset": dataset, "labeling": labeling, "split": split,
                     "comparison": f"Original Bayesian tree minus {name}", "n": n,
                     "mean_squared_error_difference": mean, "cohens_dz": mean / sd if sd else 0.0,
                     "ci95_low": mean - crit * se, "ci95_high": mean + crit * se,
                     "t_statistic": stat, "p_value_one_sided_improvement": p_one_sided,
                     "direction": "positive favors compared model"})
    return rows


def holm_adjust(p_values):
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty(len(p), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(p) - rank) * p[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def paired_test_table(long_predictions):
    rows = []
    group_cols = ["dataset", "labeling", "split"]
    for (dataset, labeling, split), part in long_predictions.groupby(group_cols, sort=False):
        base = part[part.model == "Original Bayesian tree (Beta(2,8))"].set_index("row_index")
        for model_name, model_part in part.groupby("model", sort=False):
            if model_name == "Original Bayesian tree (Beta(2,8))":
                continue
            compared = model_part.set_index("row_index").reindex(base.index)
            valid = compared.probability.notna().to_numpy()
            y = base.actual_failure.to_numpy()[valid]
            baseline_prob = base.probability.to_numpy()[valid]
            candidate_prob = compared.probability.to_numpy()[valid]
            diff = (baseline_prob - y) ** 2 - (candidate_prob - y) ** 2
            n = len(diff); mean = float(np.mean(diff)); sd = float(np.std(diff, ddof=1))
            se = sd / np.sqrt(n); stat = mean / se if se else 0.0
            critical = float(t.ppf(.975, n - 1)) if n > 1 else np.nan
            try:
                p_w = float(wilcoxon(diff, alternative="greater", zero_method="wilcox", method="auto").pvalue)
            except ValueError:
                p_w = 1.0
            if p_w == 0.0:
                p_w = float(np.nextafter(0.0, 1.0))
            p_t = float(t.sf(stat, n - 1)) if se else 1.0
            if p_t == 0.0:
                p_t = float(np.nextafter(0.0, 1.0))
            climate_condition = "with climate" if dataset == "yield_df.csv (with climate)" else "without climate"
            rows.append({"dataset": dataset, "climate_condition": climate_condition,
                         "labeling": labeling, "split": split,
                         "model_vs_original": model_name, "n": n,
                         "mean_squared_error_difference": mean,
                         "ci95_low": mean - critical * se, "ci95_high": mean + critical * se,
                         "cohens_dz": mean / sd if sd else 0.0,
                         "t_statistic": stat,
                         "t_p_value_one_sided": p_t,
                         "wilcoxon_p_value_one_sided": p_w,
                         "direction": "positive favors compared model"})
    result = pd.DataFrame(rows)
    result["t_p_value_holm"] = holm_adjust(result.t_p_value_one_sided)
    result["wilcoxon_p_value_holm"] = holm_adjust(result.wilcoxon_p_value_one_sided)
    return result


def country_level_test_table(country_predictions, bootstrap_reps=10000):
    rows = []
    rng = np.random.default_rng(RANDOM_STATE)
    baseline = "Original Bayesian tree (Beta(2,8))"
    for (dataset, labeling), part in country_predictions.groupby(["dataset", "labeling"], sort=False):
        y = part.actual_failure.to_numpy()
        base_error = (part[baseline].to_numpy() - y) ** 2
        country_errors = pd.DataFrame({"Area": part.Area.to_numpy(), "baseline_error": base_error})
        for model_name in MODEL_ORDER:
            if model_name == baseline:
                continue
            errors = pd.DataFrame({"Area": part.Area.to_numpy(),
                                   "model_error": (part[model_name].to_numpy() - y) ** 2})
            by_country = country_errors.groupby("Area").baseline_error.mean().to_frame().join(
                errors.groupby("Area").model_error.mean(), how="inner")
            difference = (by_country.baseline_error - by_country.model_error).to_numpy()
            n = len(difference)
            try:
                p_value = float(wilcoxon(difference, alternative="greater", zero_method="wilcox", method="auto").pvalue)
            except ValueError:
                p_value = 1.0
            if p_value == 0.0:
                p_value = float(np.nextafter(0.0, 1.0))
            boot = np.empty(bootstrap_reps, dtype=float)
            chunk = 500
            for start in range(0, bootstrap_reps, chunk):
                size = min(chunk, bootstrap_reps - start)
                sample = rng.choice(difference, size=(size, n), replace=True)
                boot[start:start + size] = np.median(sample, axis=1)
            lo, hi = np.quantile(boot, [.025, .975])
            rows.append({"dataset": dataset,
                         "climate_condition": "with climate" if dataset == "yield_df.csv (with climate)" else "without climate",
                         "labeling": labeling, "split": "country_group_kfold",
                         "model_vs_original": model_name, "n_countries": n,
                         "median_brier_difference": float(np.median(difference)),
                         "bootstrap_ci95_low": float(lo), "bootstrap_ci95_high": float(hi),
                         "mean_brier_difference": float(np.mean(difference)),
                         "wilcoxon_p_value_one_sided": p_value,
                         "direction": "positive favors compared model"})
    result = pd.DataFrame(rows)
    result["wilcoxon_p_value_holm"] = holm_adjust(result.wilcoxon_p_value_one_sided)
    return result


def split_indices(df, split, y):
    idx = np.arange(len(df))
    if split == "random":
        return train_test_split(idx, test_size=.30, random_state=RANDOM_STATE, stratify=y)
    if split == "temporal":
        train = idx[(df.Year.to_numpy() >= 1961) & (df.Year.to_numpy() <= 2005)]
        test = idx[(df.Year.to_numpy() >= 2006) & (df.Year.to_numpy() <= 2016)]
        if not len(train) or not len(test):
            raise ValueError("Temporal split needs observations in years 1961–2005 and 2006–2016")
        return train, test
    raise ValueError(split)


def tune_forest(Xfit, yfit):
    rf = RandomForestClassifier(criterion="entropy", random_state=RANDOM_STATE, n_jobs=-1)
    search = GridSearchCV(rf, {"n_estimators": [50, 100], "max_depth": [4, 6, 8, 10, 12],
                               "min_samples_leaf": [15, 30]}, scoring="neg_brier_score",
                          cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE), n_jobs=1)
    search.fit(Xfit, yfit)
    return search


def fit_split(dataset_name, df, X, encoders, categorical, numeric, labeling, y, split):
    train, test = split_indices(df, split, y)
    Xtrain, Xtest = X.iloc[train], X.iloc[test]
    ytrain, ytest = y[train], y[test]
    crops = df.Item.to_numpy()
    ctrain, ctest = crops[train], crops[test]
    training_coverage = {}
    for (crop_name, country_name), rows in df.iloc[train].groupby(["Item", "Area"]):
        training_coverage.setdefault(str(crop_name), {})[str(country_name)] = sorted(rows.Year.astype(int).unique().tolist())
    fit_rel, val_rel = train_test_split(np.arange(len(train)), test_size=.20, random_state=43, stratify=ytrain)
    fit_idx, val_idx = train[fit_rel], train[val_rel]
    Xfit, Xval = X.iloc[fit_idx], X.iloc[val_idx]
    yfit, yval = y[fit_idx], y[val_idx]
    cfit = crops[fit_idx]

    # Exact original CART/Bayesian-tree setup, trained on the complete split training portion.
    original_tree = DecisionTreeClassifier(criterion="entropy", max_depth=4, min_samples_leaf=15,
                                           random_state=RANDOM_STATE).fit(Xtrain, ytrain)
    cart_raw = original_tree.predict_proba(Xtest)[:, 1]
    orig_counts = tree_counts(original_tree, Xtrain, ytrain)
    leaves = original_tree.apply(Xtest.to_numpy())
    orig_posterior = np.array([(orig_counts[int(leaf)][0] + 2) / (orig_counts[int(leaf)][1] + 10) for leaf in leaves])

    search = tune_forest(Xfit, yfit)
    forest = search.best_estimator_
    val_forest, _ = forest_predict(forest, [], [], Xfit, yfit, cfit, Xval, crops[val_idx])
    isotonic = IsotonicRegression(out_of_bounds="clip").fit(val_forest["RF fixed Beta(2,8)"], yval)
    forest_pred, spread = forest_predict(forest, [], [], Xfit, yfit, cfit, Xtest, ctest, posterior_draws=50)
    isotonic_pred = isotonic.predict(forest_pred["RF fixed Beta(2,8)"])
    boosting = GradientBoostingClassifier(n_estimators=100, learning_rate=.05, max_depth=2,
                                           random_state=RANDOM_STATE).fit(Xtrain, ytrain)
    boost_pred = boosting.predict_proba(Xtest)[:, 1]
    preds = {"Single CART (raw k/N)": cart_raw,
             "Original Bayesian tree (Beta(2,8))": orig_posterior,
             **forest_pred, "RF isotonic calibrated": isotonic_pred,
             "Gradient boosting": boost_pred}
    metrics = [score_row(dataset_name, labeling, split, name, ytest, p) for name, p in preds.items()]
    bundle = {
        "dataset": dataset_name, "labeling": labeling, "split": split,
        "feature_names": X.columns.tolist(), "encoders": encoders, "categorical": categorical,
        "numeric": numeric, "tree": original_tree, "tree_counts": orig_counts,
        "forest": forest, "forest_counts": forest_leaf_stats(forest, Xfit, yfit, cfit),
        "isotonic": isotonic, "boosting": boosting,
        "failure_threshold_global": float(df["yield"].quantile(.2)),
        "crop_failure_rates": pd.Series(ytrain).groupby(crops[train]).mean().to_dict(),
        "training_coverage": training_coverage,
        "best_params": search.best_params_, "brier_predictions": preds,
        "tree_spread_test": spread,
    }
    pred_table = pd.DataFrame({"row_index": test, "Area": df.Area.to_numpy()[test], "Item": ctest,
                               "Year": df.Year.to_numpy()[test], "actual_failure": ytest})
    for name, p in preds.items():
        if p is not None:
            pred_table[name] = p
    return metrics, paired_rows(dataset_name, labeling, split, ytest, preds), pred_table, bundle, search


def fit_country_gkf(dataset_name, df, X, labeling, y):
    groups = df._country.to_numpy()
    cv = GroupKFold(n_splits=len(np.unique(groups)))
    oof = {name: np.full(len(y), np.nan) for name in MODEL_ORDER}
    fold_rows = []
    fold_number = 0
    for train, test in cv.split(X, y, groups):
        fold_number += 1
        Xtrain, Xtest = X.iloc[train], X.iloc[test]
        ytrain, ytest = y[train], y[test]
        crops = df.Item.to_numpy()
        tree = DecisionTreeClassifier(criterion="entropy", max_depth=4, min_samples_leaf=15,
                                      random_state=RANDOM_STATE).fit(Xtrain, ytrain)
        raw_cart = tree.predict_proba(Xtest)[:, 1]
        cnt = tree_counts(tree, Xtrain, ytrain)
        leaf = tree.apply(Xtest.to_numpy())
        bayes = np.array([(cnt[int(k)][0] + 2) / (cnt[int(k)][1] + 10) for k in leaf])
        # Fixed, prespecified small forest keeps exact 212-country LOCO feasible; tuning is nested
        # separately on random/temporal training partitions above.
        fit_rel, val_rel = train_test_split(np.arange(len(train)), test_size=.20, random_state=43, stratify=ytrain)
        fit, val = train[fit_rel], train[val_rel]
        rf = RandomForestClassifier(n_estimators=10, criterion="entropy", max_depth=8,
                                    min_samples_leaf=15, random_state=RANDOM_STATE, n_jobs=-1).fit(X.iloc[fit], y[fit])
        vpred, _ = forest_predict(rf, [], [], X.iloc[fit], y[fit], crops[fit], X.iloc[val], crops[val])
        iso = IsotonicRegression(out_of_bounds="clip").fit(vpred["RF fixed Beta(2,8)"], y[val])
        rfpred, _ = forest_predict(rf, [], [], X.iloc[fit], y[fit], crops[fit], Xtest, crops[test])
        isotonic_pred = iso.predict(rfpred["RF fixed Beta(2,8)"])
        gb = GradientBoostingClassifier(n_estimators=30, learning_rate=.05, max_depth=2,
                                        random_state=RANDOM_STATE).fit(Xtrain, ytrain)
        values = {"Single CART (raw k/N)": raw_cart, "Original Bayesian tree (Beta(2,8))": bayes,
                  **rfpred, "RF isotonic calibrated": isotonic_pred,
                  "Gradient boosting": gb.predict_proba(Xtest)[:, 1]}
        for name, prob in values.items():
            oof[name][test] = prob
            fold_rows.append({"dataset": dataset_name, "labeling": labeling, "fold": fold_number,
                              "held_out_countries": ";".join(sorted(set(groups[test]))), "n": len(test),
                              "model": name, "brier_score": brier_score_loss(ytest, prob),
                              "roc_auc": safe_auc(ytest, prob)})
        if fold_number % 25 == 0:
            print(f"{dataset_name} / {labeling}: country GroupKFold {fold_number}/{cv.get_n_splits()}", flush=True)
    predictions = oof
    metrics = [score_row(dataset_name, labeling, "country_group_kfold", name, y, p)
               for name, p in predictions.items()]
    paired = paired_rows(dataset_name, labeling, "country_group_kfold", y, predictions)
    return metrics, paired, predictions, pd.DataFrame(fold_rows)


def slug(value):
    return "_".join("".join(ch.lower() if ch.isalnum() else " " for ch in value).split())


def plot_predictions(dataset, labeling, split, y, pred_map):
    usable = [(n, p) for n, p in pred_map.items() if p is not None]
    tag = f"{slug(dataset)}__{labeling}__{split}"
    fig, ax = plt.subplots(figsize=(9, 6))
    for name, p in usable:
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, label=f"{name} (AUC {safe_auc(y, p):.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=.5)
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title=f"ROC · {dataset} · {labeling} · {split}")
    ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(OUT / f"roc__{tag}.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 6))
    for name, p in usable:
        frac, avg = calibration_curve(y, p, n_bins=10, strategy="quantile")
        ax.plot(avg, frac, marker="o", label=name)
    ax.plot([0, 1], [0, 1], "k--", alpha=.5)
    ax.set(xlabel="Mean predicted probability", ylabel="Observed failure rate", title=f"Calibration · {dataset} · {labeling} · {split}")
    ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(OUT / f"calibration__{tag}.png", dpi=150); plt.close(fig)


def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    preserve_prior_outputs()
    datasets = build_datasets()
    all_metrics, all_paired, all_predictions, all_tuning, baseline_rows = [], [], [], [], []
    all_country_folds, all_country_predictions = [], []
    model_bundles = {}
    for dataset_name, df in datasets.items():
        print(f"\n=== {dataset_name}: {len(df)} rows, years {df.Year.min()}–{df.Year.max()} ===", flush=True)
        X, encoders, categorical, numeric = features_for(df)
        labels = labels_for(df)
        for labeling, y in labels.items():
            if len(np.unique(y)) < 2:
                print(f"Skipping {labeling}: target has one class", flush=True)
                continue
            for split in ("random", "temporal"):
                metrics, paired, pred_table, bundle, search = fit_split(
                    dataset_name, df, X, encoders, categorical, numeric, labeling, y, split)
                all_metrics.extend(metrics); all_paired.extend(paired)
                pred_table.insert(0, "dataset", dataset_name); pred_table.insert(1, "labeling", labeling); pred_table.insert(2, "split", split)
                all_predictions.append(pred_table)
                all_tuning.append({"dataset": dataset_name, "labeling": labeling, "split": split,
                                   "best_params": json.dumps(search.best_params_), "best_cv_brier": -search.best_score_})
                model_bundles[f"{slug(dataset_name)}|{labeling}|{split}"] = bundle
                train, test = split_indices(df, split, y)
                plot_predictions(dataset_name, labeling, split, y[test], bundle["brier_predictions"])
                # Exact paper reproduction check for the original dataset/global label/random split.
                if dataset_name == "yield.csv (no climate; full)" and labeling == "global_bottom_20" and split == "random":
                    bayes_metrics = next(r for r in metrics if r["model"] == "Original Bayesian tree (Beta(2,8))")
                    baseline_rows.append({"dataset": dataset_name, "labeling": labeling, "split": split,
                                          "brier_score": bayes_metrics["brier_score"], "roc_auc": bayes_metrics["roc_auc"],
                                          "expected_brier": .1185, "expected_auc": .8397})
            g_metrics, g_paired, g_preds, g_fold_rows = fit_country_gkf(dataset_name, df, X, labeling, labels[labeling])
            all_metrics.extend(g_metrics); all_paired.extend(g_paired)
            all_country_folds.append(g_fold_rows)
            group_pred = pd.DataFrame({"row_index": np.arange(len(df)), "dataset": dataset_name,
                                       "labeling": labeling, "split": "country_group_kfold",
                                       "Area": df.Area, "Item": df.Item, "Year": df.Year,
                                       "actual_failure": labels[labeling]})
            for name, prob in g_preds.items():
                group_pred[name] = prob
            all_country_predictions.append(group_pred)
            # Pooled country-held-out ROC and calibration plots keep both tree baselines visible.
            plot_predictions(dataset_name, labeling, "country_group_kfold", y, g_preds)
        # Fit final scenario bundles for both split definitions are retained individually above.
    metrics_df = pd.DataFrame(all_metrics)
    paired_df = pd.DataFrame(all_paired)
    metrics_df["model_order"] = metrics_df.model.map({m: i for i, m in enumerate(MODEL_ORDER)})
    metrics_df = metrics_df.sort_values(["dataset", "labeling", "split", "model_order"]).drop(columns="model_order")
    metrics_df.to_csv(OUT / "comparison_metrics.csv", index=False)
    paired_df.to_csv(OUT / "paired_squared_error_tests.csv", index=False)
    predictions_df = pd.concat(all_predictions, ignore_index=True, sort=False)
    predictions_df.to_csv(OUT / "comparison_predictions.csv", index=False)
    country_predictions_df = pd.concat(all_country_predictions, ignore_index=True, sort=False)
    country_predictions_df.to_csv(OUT / "country_group_oof_predictions.csv", index=False)
    country_tests = country_level_test_table(country_predictions_df)
    country_tests.to_csv(OUT / "country_level_tests.csv", index=False)
    all_test_predictions = pd.concat([predictions_df, country_predictions_df], ignore_index=True, sort=False)
    long_rows = []
    id_cols = ["dataset", "labeling", "split", "row_index", "actual_failure"]
    for model_name in MODEL_ORDER:
        if model_name in all_test_predictions:
            current = all_test_predictions[id_cols + [model_name]].rename(columns={model_name: "probability"})
            current["model"] = model_name
            long_rows.append(current)
    paired_tests = paired_test_table(pd.concat(long_rows, ignore_index=True))
    paired_tests.to_csv(OUT / "paired_tests_vs_original.csv", index=False)
    fold_brier = pd.concat(all_country_folds, ignore_index=True)
    fold_brier["climate_condition"] = fold_brier.dataset.map(
        lambda x: "with climate" if x == "yield_df.csv (with climate)" else "without climate")
    fold_brier.to_csv(OUT / "country_group_fold_brier.csv", index=False)
    fold_summary = fold_brier.groupby(["dataset", "climate_condition", "labeling", "model"], as_index=False).agg(
        folds=("fold", "nunique"), mean_brier=("brier_score", "mean"), sd_brier=("brier_score", "std"))
    fold_summary.to_csv(OUT / "country_group_fold_brier_summary.csv", index=False)
    tuning_df = pd.DataFrame(all_tuning); tuning_df.to_csv(OUT / "forest_tuning_summary.csv", index=False)
    pd.DataFrame(baseline_rows).to_csv(OUT / "original_bayesian_reproduction.csv", index=False)
    # Explicit standalone tables requested for showing the climate comparison.
    for dataset_name, filename in [("yield.csv (no climate; full)", "comparison_yield_no_climate.csv"),
                                   ("yield_df.csv (with climate)", "comparison_yield_with_climate.csv"),
                                   ("yield.csv (matched rows; no climate)", "comparison_yield_matched_no_climate.csv")]:
        metrics_df[metrics_df.dataset == dataset_name].to_csv(OUT / filename, index=False)
    paper = metrics_df[metrics_df.dataset.isin(["yield.csv (no climate; full)", "yield_df.csv (with climate)"])].copy()
    paper["climate_condition"] = paper.dataset.map({"yield.csv (no climate; full)": "without climate",
                                                     "yield_df.csv (with climate)": "with climate"})
    paper = paper[["climate_condition", "dataset", "labeling", "split", "model", "brier_score",
                   "test_failure_rate", "baseline_brier", "brier_skill_score", "roc_auc", "ece", "status"]]
    paper.to_csv(OUT / "paper_comparison_summary.csv", index=False)
    matched = metrics_df[metrics_df.dataset.isin(["yield.csv (matched rows; no climate)", "yield_df.csv (with climate)"])]
    compare = matched.pivot(index=["labeling", "split", "model"], columns="dataset",
                            values=["brier_score", "roc_auc", "ece"])
    flat = pd.DataFrame(index=compare.index).reset_index()
    for metric in ("brier_score", "roc_auc", "ece", "test_failure_rate", "baseline_brier", "brier_skill_score"):
        no_col = (metric, "yield.csv (matched rows; no climate)")
        yes_col = (metric, "yield_df.csv (with climate)")
        flat[f"{metric}_no_climate"] = compare[no_col].to_numpy()
        flat[f"{metric}_with_climate"] = compare[yes_col].to_numpy()
        flat[f"{metric}_climate_minus_no_climate"] = flat[f"{metric}_with_climate"] - flat[f"{metric}_no_climate"]
    flat.to_csv(OUT / "matched_climate_effect.csv", index=False)
    # Save scenario model artifacts; app defaults to the temporal-split model.
    joblib.dump({"models": model_bundles, "dataset_names": list(datasets), "labelings": list(labels_for(next(iter(datasets.values()))))},
                OUT / "crop_risk_models.joblib", compress=3)
    pd.DataFrame([{"dataset": name, "rows": len(df), "year_min": int(df.Year.min()), "year_max": int(df.Year.max()),
                   "climate_features": ",".join(c for c in ("rainfall", "temperature", "pesticides") if c in df)}
                  for name, df in datasets.items()]).to_csv(OUT / "dataset_coverage.csv", index=False)
    print("\nOriginal Bayesian tree reproduction:")
    print(pd.DataFrame(baseline_rows).to_string(index=False))
    print("\nResults written under", OUT)


if __name__ == "__main__":
    main()
