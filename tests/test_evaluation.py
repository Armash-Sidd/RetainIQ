"""
Unit tests for RetainIQ Phase 5 evaluation artifacts and threshold data.
Runnable via:
    python -m unittest discover tests
or
    pytest tests/
"""

import unittest
from pathlib import Path
import pandas as pd
from PIL import Image

from src.utils import PROJECT_ROOT


class TestRetainIQEvaluationArtifacts(unittest.TestCase):
    """Test suite ensuring all evaluation reports, plots, and CSV tables exist and are valid."""

    def test_evaluation_report_markdown_exists(self):
        """Verify reports/evaluation_report.md exists and contains the required 12 sections."""
        report_path = PROJECT_ROOT / "reports" / "evaluation_report.md"
        self.assertTrue(report_path.exists(), "evaluation_report.md does not exist")

        with open(report_path, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn("# RetainIQ — Model Evaluation Report", content)
        self.assertIn("## 1. Final Model", content)
        self.assertIn("## 2. Test Set Performance", content)
        self.assertIn("## 3. Classification Report", content)
        self.assertIn("## 4. Confusion Matrix", content)
        self.assertIn("## 5. ROC Curve", content)
        self.assertIn("## 6. Precision-Recall Curve", content)
        self.assertIn("## 7. Feature Importance", content)
        self.assertIn("## 8. SHAP Global Explanation", content)
        self.assertIn("## 9. Model Comparison", content)
        self.assertIn("## 10. Key Findings", content)
        self.assertIn("## 11. Interpretation", content)
        self.assertIn("## 12. Limitations", content)

    def test_threshold_analysis_csv_exists_and_valid(self):
        """Verify reports/threshold_analysis.csv contains required columns and ranges."""
        csv_path = PROJECT_ROOT / "reports" / "threshold_analysis.csv"
        self.assertTrue(csv_path.exists(), "threshold_analysis.csv does not exist")

        df = pd.read_csv(csv_path)
        expected_cols = ["threshold", "accuracy", "precision", "recall", "f1_score"]
        for c in expected_cols:
            self.assertIn(c, df.columns)

        self.assertGreaterEqual(len(df), 5)
        self.assertTrue((df["threshold"] >= 0.0).all() and (df["threshold"] <= 1.0).all())
        self.assertTrue((df["accuracy"] >= 0.0).all() and (df["accuracy"] <= 1.0).all())
        self.assertTrue((df["precision"] >= 0.0).all() and (df["precision"] <= 1.0).all())
        self.assertTrue((df["recall"] >= 0.0).all() and (df["recall"] <= 1.0).all())
        self.assertTrue((df["f1_score"] >= 0.0).all() and (df["f1_score"] <= 1.0).all())

    def test_evaluation_figures_exist_and_readable(self):
        """Verify all 9 generated publication-quality PNG figures exist and are non-empty."""
        figs_dir = PROJECT_ROOT / "reports" / "figures"
        required_figures = [
            "confusion_matrix.png",
            "roc_curve.png",
            "pr_curve.png",
            "feature_importance.png",
            "shap_summary.png",
            "shap_feature_importance.png",
            "shap_local_example.png",
            "model_comparison.png",
            "threshold_analysis.png",
        ]

        for fname in required_figures:
            fig_path = figs_dir / fname
            self.assertTrue(fig_path.exists(), f"Figure missing: {fname}")
            self.assertGreater(fig_path.stat().st_size, 1000, f"Figure too small: {fname}")

            # Verify image can be parsed without corruption
            with Image.open(fig_path) as img:
                w, h = img.size
                self.assertGreater(w, 200)
                self.assertGreater(h, 200)


if __name__ == "__main__":
    unittest.main()

