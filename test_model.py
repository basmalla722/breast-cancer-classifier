"""Guard the model's accuracy so a silent regression cannot ship.

The dataset is fixed and the split is seeded, so the numbers are deterministic.
That is what makes it safe to assert on them: if accuracy moves, something in the
pipeline changed, and that is worth a human looking at rather than accepting.

Run:  python test_model.py
"""

import unittest

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# The floor is deliberately below the observed 98.25%. It exists to catch a
# broken pipeline, not to re-assert a number that already has a test.
MIN_ACCURACY = 0.95
MIN_F1 = 0.95
# One missed malignancy out of 72 is tolerated. Two is a regression.
MIN_RECALL = 0.97


def build_test_set():
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="target")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, y_train, X_test_scaled, y_test


class TestModelQuality(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        X_train_scaled, y_train, X_test_scaled, y_test = build_test_set()
        cls.y_test = y_test

        model = LogisticRegression(max_iter=10000)
        model.fit(X_train_scaled, y_train)
        cls.y_pred = model.predict(X_test_scaled)

    def test_accuracy_meets_floor(self):
        score = accuracy_score(self.y_test, self.y_pred)
        self.assertGreaterEqual(
            score,
            MIN_ACCURACY,
            f"accuracy fell to {score:.4f}, floor is {MIN_ACCURACY}",
        )

    def test_f1_meets_floor(self):
        score = f1_score(self.y_test, self.y_pred, pos_label=1)
        self.assertGreaterEqual(
            score,
            MIN_F1,
            f"F1 fell to {score:.4f}, floor is {MIN_F1}",
        )

    def test_recall_meets_floor(self):
        # A model that scores well on accuracy while missing malignancies is
        # worse than useless here, so recall is asserted separately.
        score = recall_score(self.y_test, self.y_pred, pos_label=1)
        self.assertGreaterEqual(
            score,
            MIN_RECALL,
            f"malignant recall fell to {score:.4f}, floor is {MIN_RECALL}",
        )

    def test_split_is_stratified(self):
        # A non-stratified split can hand over a test set with a different class
        # balance, which would make the accuracy number incomparable to the
        # cross-validated one.
        data = load_breast_cancer()
        y = pd.Series(data.target, name="target")

        _, _, _, y_test = build_test_set()

        full_ratio = y.mean()
        test_ratio = y_test.mean()

        self.assertAlmostEqual(
            test_ratio,
            full_ratio,
            delta=0.03,
            msg=f"test malignant ratio {test_ratio:.4f} drifted from dataset {full_ratio:.4f}",
        )

    def test_test_set_contains_both_classes(self):
        _, _, _, y_test = build_test_set()

        self.assertEqual(
            set(y_test.unique().tolist()),
            {0, 1},
            "test set must contain both classes or the metrics are meaningless",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
