"""
Unit tests for RetainIQ data ingestion, feature engineering, and model loading pipeline.
Runnable via:
    python -m unittest discover tests
or
    pytest tests/
"""

import unittest
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from src.data_loader import EXPECTED_COLUMNS, validate_data
from src.feature_engineering import (
    ALL_ENGINEERED_FEATURES,
    BASE_ENGINEERED_FEATURES,
    NEW_ENGINEERED_FEATURES,
    engineer_features,
)
from src.preprocessing import encode_categoricals, preprocess
from src.utils import PROJECT_ROOT, load_config


class TestRetainIQPipeline(unittest.TestCase):
    """Test suite covering pipeline components, transformations, and artifacts."""

    def setUp(self):
        """Build sample raw banking record DataFrame."""
        self.raw_sample = pd.DataFrame(
            {
                "id": [1, 2],
                "CustomerId": [15634602, 15647311],
                "Surname": ["Hargrave", "Hill"],
                "CreditScore": [619, 608],
                "Geography": ["France", "Spain"],
                "Gender": ["Female", "Female"],
                "Age": [42.0, 41.0],
                "Tenure": [2, 1],
                "Balance": [0.0, 83807.86],
                "NumOfProducts": [1, 1],
                "HasCrCard": [1, 0],
                "IsActiveMember": [1, 1],
                "EstimatedSalary": [101348.88, 112542.58],
                "Exited": [1, 0],
            }
        )

    def test_schema_columns_expectation(self):
        """Verify standard 14 schema attributes."""
        self.assertEqual(len(EXPECTED_COLUMNS), 14)
        for col in ["CreditScore", "Geography", "Gender", "Age", "Exited"]:
            self.assertIn(col, EXPECTED_COLUMNS)

    def test_data_validation_success(self):
        """Verify data validation passes on valid data without exceptions."""
        summary = validate_data(self.raw_sample)
        self.assertTrue(summary["is_valid"])
        self.assertEqual(summary["rows"], 2)
        self.assertEqual(summary["total_missing"], 0)

    def test_data_validation_empty_failure(self):
        """Verify validation raises ValueError on empty DataFrame."""
        with self.assertRaises(ValueError):
            validate_data(pd.DataFrame())

    def test_categorical_encoding(self):
        """Verify Geography and Gender are properly encoded into binary flags."""
        encoded = encode_categoricals(self.raw_sample)
        self.assertNotIn("Geography", encoded.columns)
        self.assertNotIn("Gender", encoded.columns)
        self.assertIn("Geography_Germany", encoded.columns)
        self.assertIn("Geography_Spain", encoded.columns)
        self.assertIn("Gender_Male", encoded.columns)

        # First row was France -> Geography_Germany=0, Geography_Spain=0
        self.assertEqual(encoded["Geography_Germany"].iloc[0], 0)
        self.assertEqual(encoded["Geography_Spain"].iloc[0], 0)
        # Second row was Spain -> Geography_Spain=1
        self.assertEqual(encoded["Geography_Spain"].iloc[1], 1)

    def test_feature_engineering_output(self):
        """Verify all baseline and Phase 3 engineered features are generated."""
        encoded = encode_categoricals(self.raw_sample)
        feat_df = engineer_features(encoded, include_all=True)

        for f in BASE_ENGINEERED_FEATURES:
            self.assertIn(f, feat_df.columns, f"Missing base feature: {f}")

        for f in NEW_ENGINEERED_FEATURES:
            self.assertIn(f, feat_df.columns, f"Missing new feature: {f}")

        # Check IsZeroBalance logic: first row Balance=0, second row Balance>0
        self.assertEqual(feat_df["IsZeroBalance"].iloc[0], 1)
        self.assertEqual(feat_df["IsZeroBalance"].iloc[1], 0)

        # Check AgeSquared
        self.assertEqual(feat_df["AgeSquared"].iloc[0], 42.0 ** 2)

    def test_preprocess_pipeline_end_to_end(self):
        """Verify full preprocess function generates clean 27-feature matrix."""
        # Drop non-feature columns
        clean_input = self.raw_sample.drop(
            columns=["id", "CustomerId", "Surname", "Exited"]
        )
        processed = preprocess(clean_input, include_all_features=True)

        feature_names_path = PROJECT_ROOT / "models" / "feature_names.pkl"
        if feature_names_path.exists():
            expected_names = joblib.load(feature_names_path)
            self.assertEqual(len(expected_names), 27)
            for col in expected_names:
                self.assertIn(col, processed.columns)

    def test_model_artifact_compatibility(self):
        """Verify final trained model artifact loads and scores properly."""
        model_path = PROJECT_ROOT / "models" / "final_model.pkl"
        feature_path = PROJECT_ROOT / "models" / "feature_names.pkl"

        if model_path.exists() and feature_path.exists():
            model = joblib.load(model_path)
            features = joblib.load(feature_path)

            clean_input = self.raw_sample.drop(
                columns=["id", "CustomerId", "Surname", "Exited"]
            )
            processed = preprocess(clean_input, include_all_features=True)
            aligned = processed[features].copy()

            preds = model.predict(aligned)
            self.assertEqual(len(preds), 2)
            self.assertTrue(all(p in [0, 1] for p in preds))

            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(aligned)
                self.assertEqual(probs.shape, (2, 2))
                self.assertTrue(np.all((probs >= 0.0) & (probs <= 1.0)))


if __name__ == "__main__":
    unittest.main()

