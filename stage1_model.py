import os
import glob
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.preprocessing import LabelEncoder

def load_dataset():
    candidates = ["yield_df.csv", "yield.csv", "crop_yield_data.csv", "crop_yield.csv"]
    csv_file = None
    for cand in candidates:
        if os.path.exists(cand):
            csv_file = cand
            break
            
    if not csv_file:
        all_csvs = glob.glob("*.csv")
        csv_file = all_csvs[0] if all_csvs else None

    if not csv_file:
        raise FileNotFoundError("No CSV file found.")

    print(f"[Python] Loading dataset from '{csv_file}'...")
    df = pd.read_csv(csv_file)

    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])

    # Target: yield value
    if 'Value' in df.columns:
        df['crop_yield'] = pd.to_numeric(df['Value'], errors='coerce')
    elif 'hg/ha_yield' in df.columns:
        df['crop_yield'] = pd.to_numeric(df['hg/ha_yield'], errors='coerce')
    else:
        num_cols = df.select_dtypes(include=[np.number]).columns
        df['crop_yield'] = pd.to_numeric(df[num_cols[-1]], errors='coerce')

    df = df.dropna(subset=['crop_yield'])

    features = []

    # Encode categorical features if present
    for col_name in ['Item', 'Area', 'Domain']:
        if col_name in df.columns:
            le = LabelEncoder()
            enc_name = f"{col_name.lower()}_encoded"
            df[enc_name] = le.fit_transform(df[col_name].astype(str))
            features.append(enc_name)

    # Add numeric features if present
    for col_name in ['Year', 'rainfall', 'temperature', 'pesticides']:
        if col_name in df.columns:
            df[col_name] = pd.to_numeric(df[col_name], errors='coerce')
            features.append(col_name)

    if not features:
        features = [c for c in df.select_dtypes(include=[np.number]).columns if c != 'crop_yield']

    df = df.dropna(subset=features)

    # 20th percentile threshold for yield failure
    threshold = np.percentile(df['crop_yield'], 20)
    df['yield_failure'] = (df['crop_yield'] < threshold).astype(int)

    print(f"[Python] Successfully processed {len(df)} records.")
    print(f"[Python] Features selected: {features}")
    print(f"[Python] Failure threshold: < {threshold:.2f}")

    return df, features

class BayesianProbabilisticTree:
    def __init__(self, max_depth=4, min_samples_leaf=15, alpha_0=2.0, beta_0=8.0):
        self.tree = DecisionTreeClassifier(criterion='entropy', max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=42)
        self.alpha_0 = alpha_0
        self.beta_0 = beta_0
        self.leaf_posteriors = {}

    def fit(self, X, y):
        X_mat = np.asarray(X)
        y_vec = np.asarray(y)
        self.tree.fit(X_mat, y_vec)
        leaf_indices = self.tree.apply(X_mat)
        for leaf in np.unique(leaf_indices):
            mask = (leaf_indices == leaf)
            N_m = np.sum(mask)
            k_m = np.sum(y_vec[mask] == 1)
            self.leaf_posteriors[leaf] = (k_m + self.alpha_0) / (N_m + self.alpha_0 + self.beta_0)
        return self

    def predict_risk(self, X):
        X_mat = np.asarray(X)
        leaf_indices = self.tree.apply(X_mat)
        prior = self.alpha_0 / (self.alpha_0 + self.beta_0)
        return np.array([self.leaf_posteriors.get(l, prior) for l in leaf_indices])

def main():
    df, features = load_dataset()
    X = df[features]
    y = df['yield_failure']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)

    # One tree, two leaf estimators (same partition for both)
    bayes_tree = BayesianProbabilisticTree(max_depth=4, min_samples_leaf=15, alpha_0=2.0, beta_0=8.0)
    bayes_tree.fit(X_train, y_train)

    bayes_probs = bayes_tree.predict_risk(X_test)                           # Beta-Binomial posterior mean
    cart_probs = bayes_tree.tree.predict_proba(np.asarray(X_test))[:, 1]    # raw k/N, same partition

    print("\n[Python] Evaluation Metrics:")
    print(f"CART Brier Score    : {brier_score_loss(y_test, cart_probs):.4f}")
    print(f"Bayesian Brier Score: {brier_score_loss(y_test, bayes_probs):.4f}")
    print(f"CART ROC-AUC        : {roc_auc_score(y_test, cart_probs):.4f}")
    print(f"Bayesian ROC-AUC    : {roc_auc_score(y_test, bayes_probs):.4f}")

    results_df = X_test.copy()
    results_df['actual_failure'] = y_test.values
    results_df['prob_cart'] = cart_probs
    results_df['prob_bayes'] = bayes_probs
    results_df.to_csv("model_predictions.csv", index=False)
    print("[Python] Predictions exported to 'model_predictions.csv' for R analysis.\n")

if __name__ == '__main__':
    main()