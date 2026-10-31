"""
RetainIQ — Academic Document Synchronization Script (Phase 9).
Synchronizes HCL_Synopsis_Report_RetainIQ_Updated.docx and RetainIQ_SRS_Updated.docx
with the exact verified Phase 5/6/7/8 results:
  - Final Model: XGBoost_Tuned (Feature-Engineered XGBoost)
  - Features: 27 aligned features (expanding from 10 raw attributes)
  - Test Accuracy: 87.37%
  - Test Precision: 65.19%
  - Test Recall: 81.70% (500 / 612 actual exiters captured)
  - Test F1 Score: 72.52%
  - Test ROC-AUC: 93.35%
  - Test Average Precision: 81.68%
  - Confusion Matrix: TN=2121, FP=267, FN=112, TP=500
  - Serving & Deployment: FastAPI with /predict, /predict/batch, telemetry & drift monitoring, Streamlit multi-tab app, Docker Compose, 26 unit & integration tests passing.
"""

import os
import docx

REPLACEMENTS = [
    # Metrics
    ("89.90%", "87.37%"),
    ("89.9%", "87.37%"),
    ("80.47%", "65.19%"),
    ("66.67%", "81.70%"),
    ("72.92%", "72.52%"),
    ("Accuracy of 89.90%", "Accuracy of 87.37%"),
    ("Precision of 80.47%", "Precision of 65.19%"),
    ("Recall of 66.67%", "Recall of 81.70%"),
    ("F1-score of 72.92%", "F1-score of 72.52%"),
    ("F1: 72.92%", "F1: 72.52%"),
    ("Recall: 66.67%", "Recall: 81.70%"),
    ("Precision: 80.47%", "Precision: 65.19%"),
    ("Accuracy: 89.90%", "Accuracy: 87.37%"),
    ("Accuracy 89.90%", "Accuracy 87.37%"),
    ("Precision 80.47%", "Precision 65.19%"),
    ("Recall 66.67%", "Recall 81.70%"),
    ("F1 72.92%", "F1 72.52%"),
    
    # Feature counts
    ("16 final features", "27 final aligned features"),
    ("16 features", "27 features"),
    ("16-feature", "27-feature"),
    ("16 model features", "27 aligned model features"),
    ("yielding a 16-feature vector", "yielding a 27-feature aligned vector"),
    ("16-feature data dictionary", "27-feature data dictionary"),
    ("five non-linear and interaction features (IsZeroBalance, BalancePerProduct, AgeSquared, ActiveProductInteraction, AgeActivityInteraction) yielding 16 aligned features",
     "domain-engineered features (financial ratios, non-linear age penalties, engagement cohorts, and behavioral interactions) yielding 27 aligned features"),
    ("five domain features (16 final features)", "13 domain-engineered features (27 final aligned features)"),
    ("generating 5 mathematical and behavioral features yielding a 16-feature vector",
     "generating 13 domain-engineered features yielding a 27-feature aligned vector"),
    ("computes 5 interaction metrics (BalancePerProduct, AgeSquared, IsZeroBalance, ActiveProductInteraction, AgeActivityInteraction) to form the 16-feature vector",
     "computes domain features (ratios, age penalties, cohorts, interaction terms) to form the 27-feature aligned vector"),
    
    # Artifact paths & Model Names
    ("model/xgb_model.pkl", "models/final_model.pkl"),
    ("model/feature_names.pkl", "models/feature_names.pkl"),
    ("Data/train (2).csv", "data/raw/bank_churn_raw.csv"),
    ("data/train (2).csv", "data/raw/bank_churn_raw.csv")
]


def update_runs_in_paragraph(p):
    # First attempt run-by-run replacement
    for run in p.runs:
        for old_val, new_val in REPLACEMENTS:
            if old_val in run.text:
                run.text = run.text.replace(old_val, new_val)
                
    # Check if any replacement spans multiple runs
    full_text = p.text
    for old_val, new_val in REPLACEMENTS:
        if old_val in full_text:
            # If still present, replace across runs
            full_text = full_text.replace(old_val, new_val)
            if len(p.runs) > 0:
                p.runs[0].text = full_text
                for r in p.runs[1:]:
                    r.text = ""


def sync_docx(file_path):
    print(f"\nProcessing: {file_path}")
    doc = docx.Document(file_path)
    
    # 1. Update Body Paragraphs
    for p in doc.paragraphs:
        update_runs_in_paragraph(p)
        
    # 2. Update Tables
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                for p in cell.paragraphs:
                    update_runs_in_paragraph(p)
                    
    # Save back
    doc.save(file_path)
    print(f"Successfully synchronized and saved: {file_path}")


def main():
    docs_to_sync = [
        "HCL_Synopsis_Report_RetainIQ_Updated.docx",
        "RetainIQ_SRS_Updated.docx"
    ]
    for d in docs_to_sync:
        if os.path.exists(d):
            sync_docx(d)
        else:
            print(f"File not found: {d}")

if __name__ == "__main__":
    main()

