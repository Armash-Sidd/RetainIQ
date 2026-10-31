"""
Unit and integration tests for RetainIQ FastAPI serving layer.
Runnable via:
    python -m unittest discover tests
or
    pytest tests/
"""

import unittest
from datetime import datetime
from pydantic import ValidationError

from api.main import (
    CustomerData,
    BatchCustomerData,
    home,
    health_check,
    get_model_info,
    predict,
    predict_batch,
    determine_risk_tier,
)


class TestRetainIQAPI(unittest.TestCase):
    """Test suite covering API routes, schemas, inferences, and validations."""

    def setUp(self):
        """Set up standard valid test fixtures."""
        self.valid_customer_high_risk = CustomerData(
            CreditScore=650.0,
            Geography="Germany",
            Gender="Female",
            Age=52.0,
            Tenure=2.0,
            Balance=125000.0,
            NumOfProducts=1.0,
            HasCrCard=1,
            IsActiveMember=0,
            EstimatedSalary=75000.0,
        )

        self.valid_customer_low_risk = CustomerData(
            CreditScore=740.0,
            Geography="France",
            Gender="Male",
            Age=30.0,
            Tenure=6.0,
            Balance=0.0,
            NumOfProducts=2.0,
            HasCrCard=1,
            IsActiveMember=1,
            EstimatedSalary=65000.0,
        )

    def test_home_endpoint(self):
        """Verify root endpoint metadata and status."""
        resp = home()
        self.assertIsInstance(resp, dict)
        self.assertEqual(resp["service"], "RetainIQ API")
        self.assertEqual(resp["status"], "Online")
        self.assertIn("version", resp)
        self.assertIn("documentation", resp)

    def test_health_check_endpoint(self):
        """Verify health check returns healthy status and loaded model details."""
        resp = health_check()
        self.assertEqual(resp.status, "healthy")
        self.assertTrue(resp.model_loaded)
        self.assertEqual(resp.features_count, 27)
        self.assertIsNotNone(resp.model_name)
        self.assertIsInstance(resp.timestamp, str)

    def test_model_info_endpoint(self):
        """Verify model info returns valid test set metrics and feature schema."""
        info = get_model_info()
        self.assertIsNotNone(info.model_name)
        self.assertEqual(info.features_count, 27)
        self.assertEqual(len(info.features), 27)
        self.assertIn("test_accuracy", info.test_metrics)
        self.assertIn("test_roc_auc", info.test_metrics)
        self.assertGreater(info.test_metrics["test_roc_auc"], 0.90)

    def test_predict_single_high_risk_customer(self):
        """Verify single prediction outputs calibrated risk, probability, and SHAP factors."""
        res = predict(self.valid_customer_high_risk)
        self.assertIn(res.prediction, [0, 1])
        self.assertGreaterEqual(res.exit_probability, 0.0)
        self.assertLessEqual(res.exit_probability, 1.0)
        self.assertIn(res.status, ["Likely to Exit", "Likely to Stay"])
        self.assertIn(res.risk_tier, ["Low", "Moderate", "High"])
        self.assertGreater(res.confidence, 50.0)
        self.assertIsInstance(res.explanation, list)

        # High risk demographic profile in Germany with 1 product should have elevated probability
        self.assertGreater(res.exit_probability, 0.50)
        self.assertEqual(res.prediction, 1)
        self.assertEqual(res.status, "Likely to Exit")

    def test_predict_single_low_risk_customer(self):
        """Verify young active customer with 2 products has low churn risk."""
        res = predict(self.valid_customer_low_risk)
        self.assertIn(res.prediction, [0, 1])
        self.assertLess(res.exit_probability, 0.50)
        self.assertEqual(res.prediction, 0)
        self.assertEqual(res.status, "Likely to Stay")
        self.assertEqual(res.risk_tier, "Low")

    def test_predict_batch_customers(self):
        """Verify batch endpoint vectorization across multiple customer instances."""
        batch_input = BatchCustomerData(
            customers=[
                self.valid_customer_high_risk,
                self.valid_customer_low_risk,
            ]
        )
        batch_res = predict_batch(batch_input)
        self.assertEqual(batch_res.total_customers, 2)
        self.assertEqual(len(batch_res.predictions), 2)
        self.assertGreaterEqual(batch_res.predicted_exits, 0)
        self.assertGreaterEqual(batch_res.predicted_retained, 0)
        self.assertEqual(
            batch_res.predicted_exits + batch_res.predicted_retained, 2
        )
        self.assertGreaterEqual(batch_res.mean_exit_probability, 0.0)
        self.assertLessEqual(batch_res.mean_exit_probability, 1.0)

    def test_pydantic_validation_invalid_credit_score(self):
        """Verify validation failure when CreditScore is outside legitimate 300-850 range."""
        with self.assertRaises(ValidationError):
            CustomerData(
                CreditScore=200.0,  # Below 300
                Geography="France",
                Gender="Female",
                Age=40.0,
                Tenure=3.0,
                Balance=50000.0,
                NumOfProducts=1.0,
                HasCrCard=1,
                IsActiveMember=1,
                EstimatedSalary=50000.0,
            )

    def test_pydantic_validation_invalid_geography(self):
        """Verify validation failure when Geography is not France, Germany, or Spain."""
        with self.assertRaises(ValidationError):
            CustomerData(
                CreditScore=700.0,
                Geography="USA",  # Invalid
                Gender="Female",
                Age=40.0,
                Tenure=3.0,
                Balance=50000.0,
                NumOfProducts=1.0,
                HasCrCard=1,
                IsActiveMember=1,
                EstimatedSalary=50000.0,
            )

    def test_pydantic_validation_invalid_age(self):
        """Verify validation failure when Age is under 18."""
        with self.assertRaises(ValidationError):
            CustomerData(
                CreditScore=700.0,
                Geography="France",
                Gender="Male",
                Age=15.0,  # Below 18
                Tenure=1.0,
                Balance=1000.0,
                NumOfProducts=1.0,
                HasCrCard=1,
                IsActiveMember=1,
                EstimatedSalary=20000.0,
            )

    def test_pydantic_validation_invalid_num_of_products(self):
        """Verify validation failure when NumOfProducts is outside 1-4."""
        with self.assertRaises(ValidationError):
            CustomerData(
                CreditScore=700.0,
                Geography="France",
                Gender="Male",
                Age=35.0,
                Tenure=1.0,
                Balance=1000.0,
                NumOfProducts=5.0,  # Above 4
                HasCrCard=1,
                IsActiveMember=1,
                EstimatedSalary=20000.0,
            )

    def test_risk_tier_helper(self):
        """Verify standardized probability threshold cutoffs."""
        self.assertEqual(determine_risk_tier(0.15), "Low")
        self.assertEqual(determine_risk_tier(0.39), "Low")
        self.assertEqual(determine_risk_tier(0.40), "Moderate")
        self.assertEqual(determine_risk_tier(0.69), "Moderate")
        self.assertEqual(determine_risk_tier(0.70), "High")
        self.assertEqual(determine_risk_tier(0.95), "High")


if __name__ == "__main__":
    unittest.main()

