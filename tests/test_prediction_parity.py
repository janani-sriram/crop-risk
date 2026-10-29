"""App predictions must reproduce the saved holdout predictions."""
from pathlib import Path
import unittest

import joblib
import numpy as np
import pandas as pd

from prediction import score_one
from train import MODEL_ORDER, build_datasets, labels_for


ROOT = Path(__file__).resolve().parents[1]


class PredictionParityTest(unittest.TestCase):
    def test_random_test_rows_match_saved_predictions(self):
        artifact = joblib.load(ROOT / "results/crop_risk_models.joblib")
        saved = pd.read_csv(ROOT / "results/comparison_predictions.csv")
        datasets = build_datasets()
        rng = np.random.default_rng(20261029)
        mismatches = {}

        for dataset_name, df in datasets.items():
            for labeling, y in labels_for(df).items():
                for split in ("random", "temporal"):
                    key = "_".join("".join(c.lower() if c.isalnum() else " " for c in dataset_name).split())
                    key = f"{key}|{labeling}|{split}"
                    config = artifact["models"][key]
                    rows = saved[(saved.dataset == dataset_name) & (saved.labeling == labeling) &
                                 (saved.split == split)]
                    chosen = rows.iloc[rng.choice(len(rows), size=min(20, len(rows)), replace=False)]
                    for _, row in chosen.iterrows():
                        source = df.iloc[int(row.row_index)].to_dict()
                        for model_name in MODEL_ORDER:
                            actual = score_one(source, config, model_name)[0]
                            expected = float(row[model_name])
                            if not np.isclose(actual, expected, atol=1e-6, rtol=0):
                                mismatches.setdefault(model_name, []).append(
                                    (dataset_name, labeling, split, int(row.row_index), actual, expected))
        self.assertFalse(mismatches, f"App prediction mismatches by model: {mismatches}")


if __name__ == "__main__":
    unittest.main()
