"""Prediction helpers shared by the Streamlit app and parity checks."""
import numpy as np
import pandas as pd


def feature_row(source, config):
    """Encode fields with the training encoders and preserve training feature order."""
    values = {}
    for category in ("Area", "Item", "Domain"):
        if category not in config["encoders"]:
            continue
        classes = config["encoders"][category]
        value = source.get(category, "")
        if category == "Domain" and not str(value):
            value = classes[0]
        value = str(value)
        if value not in classes:
            raise ValueError(f"Unknown {category}: {value!r}")
        values[f"{category.lower()}_encoded"] = classes.index(value)
    values["Year"] = float(source.get("Year", 2010))
    for feature, aliases in {
        "rainfall": ("rainfall", "average_rain_fall_mm_per_year"),
        "pesticides": ("pesticides", "pesticides_tonnes"),
        "temperature": ("temperature", "avg_temp"),
    }.items():
        if feature in config["numeric"]:
            found = next((source[a] for a in aliases if a in source and pd.notna(source[a])), np.nan)
            values[feature] = float(found) if pd.notna(found) else 0.0
    return pd.DataFrame([[values.get(col, 0.0) for col in config["feature_names"]]],
                        columns=config["feature_names"], dtype=float)


def rf_probabilities(config, X, item, spread_model=None):
    forest = config["forest"]
    leaf_stats, crop_stats = config["forest_counts"]
    raw, fixed, crop_prior, draws = [], [], [], []
    rate = float(config["crop_failure_rates"].get(item, np.mean(list(config["crop_failure_rates"].values()))))
    rng = np.random.default_rng(42)
    X_array = X.to_numpy()
    for j, tree in enumerate(forest.estimators_):
        leaf = int(tree.apply(X_array)[0])
        k, n = leaf_stats[j].get(leaf, [0, 0])
        ck, cn = crop_stats[j].get(f"{leaf}\t{item}", [0, 0])
        raw.append(k / n if n else np.nan)
        fixed.append((k + 2) / (n + 10))
        crop_prior.append((ck + rate * 10) / (cn + 10))
        if spread_model == "RF fixed Beta(2,8)":
            draws.append(rng.beta(k + 2, n - k + 8, size=1000))
        elif spread_model == "RF crop-specific prior":
            draws.append(rng.beta(ck + rate * 10, cn - ck + (1 - rate) * 10, size=1000))
    probs = {"RF raw-frequency": float(np.mean(raw)), "RF fixed Beta(2,8)": float(np.mean(fixed)),
             "RF crop-specific prior": float(np.mean(crop_prior))}
    interval = np.quantile(np.mean(np.asarray(draws), axis=0), [.05, .95]) if draws else None
    return probs, interval


def score_one(source, config, model_name):
    X = feature_row(source, config)
    item = str(source.get("Item", ""))
    model = config["tree"]
    if model_name == "Single CART (raw k/N)":
        return float(model.predict_proba(X)[0, 1]), None, None
    if model_name == "Original Bayesian tree (Beta(2,8))":
        leaf = int(model.apply(X)[0]); k, n = config["tree_counts"].get(leaf, [0, 0])
        return float((k + 2) / (n + 10)), None, None
    if model_name.startswith("RF"):
        selected = model_name if model_name in ("RF fixed Beta(2,8)", "RF crop-specific prior") else None
        probs, interval = rf_probabilities(config, X, item, spread_model=selected)
        if model_name == "RF isotonic calibrated":
            return float(config["isotonic"].predict([probs["RF fixed Beta(2,8)"]])[0]), None, None
        p = probs[model_name]
        return p, (float(interval[0]), float(interval[1])) if interval is not None else (None, None)
    if model_name == "Gradient boosting":
        return float(config["boosting"].predict_proba(X)[0, 1]), None, None
    raise ValueError(f"Unknown model: {model_name}")
