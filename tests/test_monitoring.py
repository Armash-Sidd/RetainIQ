"""
Unit tests for RetainIQ Phase 8 inference monitoring, audit trail, and drift detection.
Runnable via:
    python -m unittest discover tests
or
    pytest tests/
"""

import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

from src.monitoring import (
    TRAINING_BASELINE_STATS,
    detect_data_drift,
    get_monitoring_summary,
    get_recent_audit_logs,
    log_batch_predictions,
    log_prediction,
)


class TestRetainIQMonitoring(unittest.TestCase):
    """Test suite covering prediction audit logging, telemetry, and distribution drift checks."""

    def setUp(self):
        """Create temporary directory and log file for isolated tests."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_log_file = Path(self.temp_dir.name) / "test_audit.jsonl"

        self.sample_customer = {
            "CreditScore": 650.0,
            "Geography": "Germany",
            "Gender": "Female",
            "Age": 48.0,
            "Tenure": 3.0,
            "Balance": 112000.0,
            "NumOfProducts": 1.0,
            "HasCrCard": 1,
            "IsActiveMember": 0,
            "EstimatedSalary": 85000.0,
        }

    def tearDown(self):
        """Clean up temporary directory."""
        self.temp_dir.cleanup()

    def test_log_prediction_creates_valid_entry(self):
        """Verify log_prediction formats and appends record properly."""
        rec = log_prediction(
            customer_input=self.sample_customer,
            exit_probability=0.8142,
            prediction=1,
            risk_tier="High",
            latency_ms=12.5,
            log_file=self.test_log_file,
        )

        self.assertTrue(self.test_log_file.exists())
        self.assertEqual(rec["prediction"], 1)
        self.assertEqual(rec["status"], "Likely to Exit")
        self.assertEqual(rec["exit_probability"], 0.8142)
        self.assertEqual(rec["risk_tier"], "High")
        self.assertEqual(rec["latency_ms"], 12.5)

        # Verify file content
        with open(self.test_log_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        self.assertEqual(len(lines), 1)
        parsed = json.loads(lines[0])
        self.assertEqual(parsed["prediction"], 1)

    def test_get_recent_audit_logs_order(self):
        """Verify get_recent_audit_logs returns records in reverse chronological order."""
        for prob in [0.2, 0.5, 0.8]:
            log_prediction(
                customer_input=self.sample_customer,
                exit_probability=prob,
                prediction=1 if prob > 0.5 else 0,
                risk_tier="High" if prob > 0.7 else ("Moderate" if prob > 0.4 else "Low"),
                log_file=self.test_log_file,
            )

        logs = get_recent_audit_logs(limit=2, log_file=self.test_log_file)
        self.assertEqual(len(logs), 2)
        # Most recent should be prob 0.8
        self.assertEqual(logs[0]["exit_probability"], 0.8)
        self.assertEqual(logs[1]["exit_probability"], 0.5)

    def test_log_batch_predictions(self):
        """Verify vectorized batch audit logging."""
        records = [
            {
                "timestamp": "2026-10-31T12:00:00",
                "prediction": 0,
                "status": "Likely to Stay",
                "exit_probability": 0.12,
                "risk_tier": "Low",
                "customer_signals": self.sample_customer,
            },
            {
                "timestamp": "2026-10-31T12:00:01",
                "prediction": 1,
                "status": "Likely to Exit",
                "exit_probability": 0.92,
                "risk_tier": "High",
                "customer_signals": self.sample_customer,
            },
        ]
        written = log_batch_predictions(records, log_file=self.test_log_file)
        self.assertEqual(written, 2)

        summary = get_monitoring_summary(log_file=self.test_log_file)
        self.assertEqual(summary["total_logged_predictions"], 2)
        self.assertEqual(summary["risk_tier_counts"]["High"], 1)
        self.assertEqual(summary["risk_tier_counts"]["Low"], 1)

    def test_detect_data_drift_stable_distribution(self):
        """Verify detect_data_drift flags no drift for samples drawn near baseline means."""
        normal_data = pd.DataFrame(
            {
                "CreditScore": [655.0, 660.0, 658.0],
                "Age": [37.5, 38.0, 38.2],
                "Tenure": [5.0, 5.2, 4.8],
                "Balance": [76000.0, 77000.0, 76500.0],
                "NumOfProducts": [1.5, 1.6, 1.55],
                "EstimatedSalary": [118000.0, 119000.0, 117500.0],
            }
        )

        drift_res = detect_data_drift(normal_data, z_threshold=4.0)
        self.assertFalse(drift_res["drift_detected"])
        self.assertEqual(len(drift_res["drifted_features"]), 0)

    def test_detect_data_drift_anomalous_distribution(self):
        """Verify detect_data_drift flags significant covariate shifts."""
        drifted_data = pd.DataFrame(
            {
                "CreditScore": [350.0, 340.0, 360.0, 355.0],  # Massive drop vs 658.85 mean
                "Age": [72.0, 75.0, 70.0, 74.0],              # Massive increase vs 37.89 mean
                "Tenure": [5.0, 5.0, 5.0, 5.0],
                "Balance": [76485.0, 76485.0, 76485.0, 76485.0],
                "NumOfProducts": [1.6, 1.6, 1.6, 1.6],
                "EstimatedSalary": [118000.0, 118000.0, 118000.0, 118000.0],
            }
        )

        drift_res = detect_data_drift(drifted_data, z_threshold=3.0)
        self.assertTrue(drift_res["drift_detected"])
        self.assertIn("CreditScore", drift_res["drifted_features"])
        self.assertIn("Age", drift_res["drifted_features"])


if __name__ == "__main__":
    unittest.main()

