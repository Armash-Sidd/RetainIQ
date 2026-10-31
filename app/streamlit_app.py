"""
RetainIQ Streamlit Analytics Dashboard
Phase 6 — Production Analytical UI & Serving Integration

Interactive decision-support platform for banking relationship managers and retention teams.
Features:
    1. Single Customer Evaluation with Real-time Scoring & SHAP Attribution
    2. Batch Account Scoring with CSV Upload & Downloadable Risk Tiers
    3. Model Scorecard & Phase 5 Evaluation Visualizations
    4. Resilient Local Inference Fallback when FastAPI is offline
"""

import json
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Set temporary Matplotlib cache directory before importing matplotlib to avoid permission warnings
os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st

# ============================================================
# PROJECT PATH & MODULE IMPORTS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import preprocess

API_URL = os.environ.get("RETAINIQ_API_URL", os.environ.get("API_URL", "http://127.0.0.1:8000"))

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="RetainIQ — Customer Exit Prediction",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CUSTOM CSS STYLING
# ============================================================

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 3rem;
        max-width: 1250px;
    }

    /* Header */
    .brand {
        font-size: 40px;
        font-weight: 800;
        color: #1e3a8a;
        margin-bottom: 0px;
    }

    .tagline {
        color: #6b7280;
        font-size: 16px;
        margin-top: 4px;
        margin-bottom: 12px;
    }

    /* Section headings */
    .section-title {
        font-size: 22px;
        font-weight: 700;
        color: #111827;
        margin-top: 10px;
        margin-bottom: 4px;
    }

    .section-description {
        color: #6b7280;
        font-size: 14px;
        margin-bottom: 18px;
    }

    /* Input sections */
    .input-section {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 22px 24px 10px 24px;
        margin-bottom: 24px;
    }

    /* Prediction result */
    .result-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 24px;
        margin-top: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }

    .result-status {
        font-size: 26px;
        font-weight: 700;
        margin: 10px 0 20px 0;
    }

    .probability-value {
        font-size: 34px;
        font-weight: 700;
        color: #1e3a8a;
        margin-top: 4px;
    }

    .risk-description {
        color: #6b7280;
        font-size: 13px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* SHAP explanation */
    .explanation-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 16px;
        margin-top: 10px;
    }

    .explanation-feature {
        font-size: 15px;
        font-weight: 600;
        color: #1f2937;
    }

    .explanation-impact {
        font-size: 13px;
        color: #4b5563;
    }

    /* Model information */
    .model-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 24px;
        margin-top: 28px;
    }

    .metric-box {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 18px 12px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
    }

    .metric-value {
        font-size: 24px;
        font-weight: 700;
        color: #1e3a8a;
    }

    .metric-label {
        color: #6b7280;
        font-size: 13px;
        margin-top: 4px;
        font-weight: 500;
    }

    /* Prediction button */
    div.stButton > button {
        height: 50px;
        border-radius: 10px;
        font-size: 16px;
        font-weight: 600;
        background-color: #2563eb;
        color: white;
        border: none;
        box-shadow: 0 4px 6px -1px rgba(37, 99, 235, 0.2);
    }

    div.stButton > button:hover {
        background-color: #1d4ed8;
        color: white;
    }

    /* Status badges */
    .badge-high {
        background-color: #fee2e2;
        color: #991b1b;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 12px;
    }

    .badge-moderate {
        background-color: #fef3c7;
        color: #92400e;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 12px;
    }

    .badge-low {
        background-color: #d1fae5;
        color: #065f46;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 12px;
    }

    /* Footer */
    .footer {
        color: #9ca3af;
        font-size: 13px;
        text-align: center;
        margin-top: 40px;
        padding-top: 20px;
        border-top: 1px solid #e5e7eb;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# CACHED ARTIFACT & METADATA LOADERS
# ============================================================

@st.cache_resource
def load_local_model_artifacts() -> Tuple[Optional[Any], Optional[List[str]], Dict[str, Any]]:
    """
    Load serialized model, feature names, and metadata from disk for resilient offline inference.

    Returns:
        Tuple[Optional[Any], Optional[List[str]], Dict[str, Any]]: (model, feature_names, metadata)
    """
    model_dir = PROJECT_ROOT / "models"
    final_path = model_dir / "final_model.pkl"
    xgb_path = model_dir / "xgb_model.pkl"
    features_path = model_dir / "feature_names.pkl"
    meta_path = model_dir / "model_metadata.json"

    model_obj = None
    if final_path.exists():
        model_obj = joblib.load(final_path)
    elif xgb_path.exists():
        model_obj = joblib.load(xgb_path)

    features_list = None
    if features_path.exists():
        features_list = joblib.load(features_path)

    metadata_dict = {}
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata_dict = json.load(f)

    return model_obj, features_list, metadata_dict


@st.cache_data(ttl=60)
def check_api_health() -> Tuple[bool, Optional[Dict[str, Any]]]:
    """
    Probe the FastAPI backend /health endpoint.

    Returns:
        Tuple[bool, Optional[Dict[str, Any]]]: (is_online, health_data)
    """
    try:
        resp = requests.get(f"{API_URL}/health", timeout=1.5)
        if resp.status_code == 200:
            return True, resp.json()
    except Exception:
        pass
    return False, None


# Feature display name mapping
FEATURE_DISPLAY_NAMES = {
    "Age": "Customer Age",
    "AgeSquared": "Age (Nonlinear Effect)",
    "CreditScore": "Credit Score",
    "NumOfProducts": "Number of Financial Products",
    "IsActiveMember": "Active Member Status",
    "Balance": "Account Balance",
    "BalancePerProduct": "Balance per Financial Product",
    "EstimatedSalary": "Estimated Annual Salary",
    "Tenure": "Relationship Tenure",
    "HasCrCard": "Credit Card Ownership",
    "Gender_Male": "Customer Gender",
    "Geography_Germany": "Resident in Germany",
    "Geography_Spain": "Resident in Spain",
    "IsZeroBalance": "Zero Balance Account",
    "ActiveProductInteraction": "Activity × Products Interaction",
    "AgeActivityInteraction": "Age × Active Membership",
    "BalanceToSalaryRatio": "Balance-to-Salary Ratio",
    "SalaryPerProduct": "Salary per Product Held",
    "AgeBin_Young": "Young Cohort (<30 yrs)",
    "AgeBin_Senior": "Senior Cohort (≥50 yrs)",
    "TenurePerAge": "Tenure / Age Ratio",
    "IsNewCustomer": "New Customer (≤1 yr)",
    "IsLoyalCustomer": "Loyal Customer (≥7 yrs)",
    "LowCreditScore": "Low Credit Score (<500)",
    "HighCreditScore": "High Credit Score (≥750)",
    "HasManyProducts": "Multi-Product Account (≥3)",
    "EngagementScore": "Composite Engagement Index",
}


def local_predict_customer(
    payload: Dict[str, Any],
    local_model: Any,
    expected_features: List[str],
) -> Dict[str, Any]:
    """
    Perform local inference when FastAPI is offline.

    Args:
        payload: Customer attributes dictionary.
        local_model: Loaded model estimator.
        expected_features: List of ordered feature column names.

    Returns:
        Dict[str, Any]: Prediction result dictionary.
    """
    raw_df = pd.DataFrame([payload])
    processed = preprocess(raw_df, include_all_features=True)
    aligned = processed[expected_features].copy()

    pred = int(local_model.predict(aligned)[0])

    if hasattr(local_model, "predict_proba"):
        prob = float(local_model.predict_proba(aligned)[0, 1])
    else:
        prob = float(pred)

    prob = round(max(0.0, min(1.0, prob)), 4)
    status_str = "Likely to Exit" if pred == 1 else "Likely to Stay"
    tier = "Low" if prob < 0.40 else ("Moderate" if prob < 0.70 else "High")
    conf = round(float(prob if pred == 1 else (1.0 - prob)) * 100.0, 2)

    return {
        "prediction": pred,
        "exit_probability": prob,
        "status": status_str,
        "risk_tier": tier,
        "confidence": conf,
        "explanation": [],
        "source": "Local Fallback Engine",
    }


# ============================================================
# HEADER & SERVICE STATUS
# ============================================================

col_header, col_status = st.columns([3, 1])

with col_header:
    st.markdown('<div class="brand">🏦 RetainIQ</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="tagline">Enterprise Customer Retention & Exit Prediction System</div>',
        unsafe_allow_html=True,
    )

local_model, local_features, model_meta = load_local_model_artifacts()
api_online, health_info = check_api_health()

with col_status:
    st.write("")
    if api_online:
        st.success("🟢 FastAPI Backend Live")
    else:
        st.info("⚡ Local Engine Active")

# ============================================================
# TABS NAVIGATION
# ============================================================

tab_single, tab_batch, tab_scorecard = st.tabs(
    [
        "👤 Single Customer Assessment",
        "📁 Batch Account Scoring",
        "📊 Model Scorecard & Evaluation",
    ]
)


# ============================================================
# TAB 1: SINGLE CUSTOMER ASSESSMENT
# ============================================================

with tab_single:
    st.markdown(
        '<div class="section-title">Customer Profile & Account Signals</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-description">'
        "Adjust customer demographic, relationship, and financial attributes to simulate exit risk."
        "</div>",
        unsafe_allow_html=True,
    )

    with st.container():
        st.markdown('<div class="input-section">', unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("##### 👤 Demographic Signals")
            geography = st.selectbox("Country of Residence", ["France", "Germany", "Spain"], index=1)
            gender = st.selectbox("Gender", ["Female", "Male"], index=0)
            age = st.slider("Age (Years)", min_value=18.0, max_value=90.0, value=48.0, step=1.0)

        with col2:
            st.markdown("##### 💳 Banking Relationship")
            tenure = st.slider("Tenure with Bank (Years)", min_value=0.0, max_value=10.0, value=3.0, step=1.0)
            credit_score = st.slider("Credit Rating Score", min_value=300.0, max_value=850.0, value=650.0, step=5.0)
            num_products = st.selectbox("Number of Products Held", [1.0, 2.0, 3.0, 4.0], index=0)

        with col3:
            st.markdown("##### 💰 Financial Position")
            balance = st.number_input("Current Account Balance ($/€)", min_value=0.0, max_value=250000.0, value=112000.0, step=2500.0)
            estimated_salary = st.number_input("Estimated Annual Salary ($/€)", min_value=0.0, max_value=250000.0, value=85000.0, step=5000.0)
            col_b1, col_b2 = st.columns(2)
            with col_b1:
                has_credit_card = st.selectbox("Credit Card", ["Yes", "No"], index=0)
            with col_b2:
                is_active_member = st.selectbox("Active Member", ["No", "Yes"], index=0)

        predict_button = st.button("🚀 Evaluate Customer Exit Risk", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    if predict_button:
        payload = {
            "CreditScore": float(credit_score),
            "Geography": geography,
            "Gender": gender,
            "Age": float(age),
            "Tenure": float(tenure),
            "Balance": float(balance),
            "NumOfProducts": float(num_products),
            "HasCrCard": 1 if has_credit_card == "Yes" else 0,
            "IsActiveMember": 1 if is_active_member == "Yes" else 0,
            "EstimatedSalary": float(estimated_salary),
        }

        result = None

        # Attempt FastAPI inference first
        if api_online:
            try:
                resp = requests.post(f"{API_URL}/predict", json=payload, timeout=8)
                if resp.status_code == 200:
                    result = resp.json()
                else:
                    st.warning(f"FastAPI returned error {resp.status_code}. Falling back to local model.")
            except Exception as e:
                logger.warning(f"FastAPI request failed: {e}")

        # Local fallback if API call failed or is offline
        if result is None and local_model is not None and local_features is not None:
            result = local_predict_customer(payload, local_model, local_features)

        if result:
            pred = result["prediction"]
            prob = result["exit_probability"]
            tier = result.get("risk_tier", "High" if prob >= 0.7 else ("Moderate" if prob >= 0.4 else "Low"))
            conf = result.get("confidence", round(float(prob if pred == 1 else (1.0 - prob)) * 100.0, 1))
            explanation = result.get("explanation", [])

            st.markdown('<div class="result-card">', unsafe_allow_html=True)

            res_col1, res_col2, res_col3, res_col4 = st.columns(4)

            with res_col1:
                st.markdown('<div class="risk-description">Prediction</div>', unsafe_allow_html=True)
                if pred == 1:
                    st.markdown('<div class="result-status" style="color: #dc2626;">🔴 Likely to Exit</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="result-status" style="color: #059669;">🟢 Likely to Stay</div>', unsafe_allow_html=True)

            with res_col2:
                st.markdown('<div class="risk-description">Exit Probability</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="probability-value">{prob * 100:.2f}%</div>', unsafe_allow_html=True)

            with res_col3:
                st.markdown('<div class="risk-description">Assigned Risk Tier</div>', unsafe_allow_html=True)
                if tier == "High":
                    st.markdown('<div class="probability-value" style="color: #dc2626;">High Risk</div>', unsafe_allow_html=True)
                elif tier == "Moderate":
                    st.markdown('<div class="probability-value" style="color: #d97706;">Moderate</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="probability-value" style="color: #059669;">Low Risk</div>', unsafe_allow_html=True)

            with res_col4:
                st.markdown('<div class="risk-description">Model Confidence</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="probability-value">{conf:.1f}%</div>', unsafe_allow_html=True)

            st.write("---")

            # Probability Gauge Progress
            st.markdown("##### 📈 Risk Spectrum")
            st.progress(float(prob))

            # Interpretation Note
            if prob >= 0.70:
                st.error(
                    f"⚠️ **High Churn Alert**: The customer has an estimated {prob * 100:.1f}% exit likelihood. "
                    "Immediate proactive intervention by a Relationship Manager is recommended to safeguard relationship balances."
                )
            elif prob >= 0.40:
                st.warning(
                    f"⚠️ **Moderate Attrition Risk**: The exit likelihood is {prob * 100:.1f}%. "
                    "Consider extending a tailored banking incentive, product consultation, or checking on client satisfaction."
                )
            else:
                st.success(
                    f"✅ **Account Stable**: The customer displays high retention indicators (exit probability {prob * 100:.1f}%). "
                    "Account is suitable for cross-selling and relationship expansion."
                )

            # Customer Snapshot Metrics
            st.markdown("##### 👤 Customer Relationship Snapshot")
            snap1, snap2, snap3, snap4 = st.columns(4)
            snap1.metric("Customer Age", f"{age:.0f} yrs")
            snap2.metric("Products Held", f"{num_products:.0f}")
            snap3.metric("Account Balance", f"${balance:,.0f}")
            snap4.metric("Annual Salary", f"${estimated_salary:,.0f}")

            # SHAP Explainability Breakdown
            if explanation:
                st.write("---")
                st.markdown('<div class="section-title">🔍 Local SHAP Factor Attribution</div>', unsafe_allow_html=True)
                st.markdown(
                    '<div class="section-description">'
                    "Quantifies the specific customer features that pushed the prediction higher (risk escalators) or lower (retention anchors)."
                    "</div>",
                    unsafe_allow_html=True,
                )

                max_val = max([abs(float(item["shap_value"])) for item in explanation]) if explanation else 1.0
                if max_val == 0:
                    max_val = 1.0

                for item in explanation:
                    f_key = item["feature"]
                    d_name = item.get("display_name", FEATURE_DISPLAY_NAMES.get(f_key, f_key))
                    s_val = float(item["shap_value"])
                    impact = item["impact"]

                    with st.container():
                        st.markdown('<div class="explanation-card">', unsafe_allow_html=True)
                        ecol1, ecol2, ecol3 = st.columns([3, 4, 2])

                        with ecol1:
                            st.markdown(f'<div class="explanation-feature">{d_name}</div>', unsafe_allow_html=True)
                            st.caption(f"Feature: `{f_key}`")

                        with ecol2:
                            st.progress(float(min(1.0, abs(s_val) / max_val)))
                            st.caption(f"Relative Impact: {abs(s_val):.4f}")

                        with ecol3:
                            if s_val > 0:
                                st.markdown('<span class="badge-high">▲ Pushes Toward Exit</span>', unsafe_allow_html=True)
                            else:
                                st.markdown('<span class="badge-low">▼ Anchors Retention</span>', unsafe_allow_html=True)

                        st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.error("Failed to generate prediction. Please check that model artifacts exist in `models/`.")


# ============================================================
# TAB 2: BATCH ACCOUNT SCORING
# ============================================================

with tab_batch:
    st.markdown('<div class="section-title">Batch Account Exit Scoring</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-description">'
        "Upload customer portfolio records (CSV) to perform high-throughput risk classification and priority ranking."
        "</div>",
        unsafe_allow_html=True,
    )

    upload_col, sample_col = st.columns([3, 1])

    with upload_col:
        uploaded_file = st.file_uploader(
            "Upload Customer Records CSV (must contain standard banking columns)",
            type=["csv"],
        )

    with sample_col:
        st.write("")
        st.write("")
        load_sample = st.button("📂 Load Benchmark Sample", help="Load sample records from bank_churn_raw.csv")

    batch_df: Optional[pd.DataFrame] = None

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.success(f"Successfully loaded {len(batch_df):,} accounts from uploaded file.")
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")

    elif load_sample:
        raw_csv_path = PROJECT_ROOT / "data" / "raw" / "bank_churn_raw.csv"
        if raw_csv_path.exists():
            full_sample = pd.read_csv(raw_csv_path)
            batch_df = full_sample.sample(n=min(50, len(full_sample)), random_state=42).copy()
            st.success(f"Loaded {len(batch_df)} representative accounts from benchmark dataset.")
        else:
            st.warning("Raw benchmark dataset not found.")

    if batch_df is not None:
        st.markdown("##### Preview of Input Accounts:")
        st.dataframe(batch_df.head(5), use_container_width=True)

        if st.button("⚡ Score Batch Portfolio", use_container_width=True):
            with st.spinner("Processing features and computing model inference across accounts..."):
                try:
                    # Clean input: ensure required columns
                    req_columns = [
                        "CreditScore", "Geography", "Gender", "Age",
                        "Tenure", "Balance", "NumOfProducts", "HasCrCard",
                        "IsActiveMember", "EstimatedSalary",
                    ]
                    missing = [c for c in req_columns if c not in batch_df.columns]
                    if missing:
                        st.error(f"Missing required columns in dataset: {missing}")
                    else:
                        # Prepare features
                        processed_batch = preprocess(batch_df, include_all_features=True)
                        if local_features is not None and local_model is not None:
                            aligned_batch = processed_batch[local_features].copy()
                            batch_preds = local_model.predict(aligned_batch)

                            if hasattr(local_model, "predict_proba"):
                                batch_probs = local_model.predict_proba(aligned_batch)[:, 1]
                            else:
                                batch_probs = batch_preds.astype(float)

                            # Append predictions to DataFrame
                            scored_df = batch_df.copy()
                            scored_df["Exit_Probability"] = np.round(batch_probs, 4)
                            scored_df["Prediction"] = batch_preds.astype(int)
                            scored_df["Status"] = np.where(batch_preds == 1, "Likely to Exit", "Likely to Stay")
                            scored_df["Risk_Tier"] = np.where(
                                batch_probs >= 0.70,
                                "High",
                                np.where(batch_probs >= 0.40, "Moderate", "Low"),
                            )

                            # Sort by exit probability descending (highest risk first)
                            scored_df = scored_df.sort_values(by="Exit_Probability", ascending=False).reset_index(drop=True)

                            # Batch Summary Cards
                            total_n = len(scored_df)
                            exits_n = int((scored_df["Prediction"] == 1).sum())
                            high_n = int((scored_df["Risk_Tier"] == "High").sum())
                            mod_n = int((scored_df["Risk_Tier"] == "Moderate").sum())
                            low_n = int((scored_df["Risk_Tier"] == "Low").sum())
                            avg_p = float(scored_df["Exit_Probability"].mean())

                            st.write("---")
                            st.markdown("##### 📊 Portfolio Risk Summary")
                            bcol1, bcol2, bcol3, bcol4, bcol5 = st.columns(5)
                            bcol1.metric("Total Accounts", f"{total_n:,}")
                            bcol2.metric("Flagged Exits", f"{exits_n:,}", f"{exits_n/total_n:.1%}")
                            bcol3.metric("High Risk Tier", f"{high_n:,}", f"{high_n/total_n:.1%}")
                            bcol4.metric("Moderate Risk", f"{mod_n:,}")
                            bcol5.metric("Mean Churn Rate", f"{avg_p:.1%}")

                            # Results Table
                            st.markdown("##### Prioritized Customer Account Actions:")
                            st.dataframe(
                                scored_df[
                                    [c for c in ["CustomerId", "Surname", "Geography", "Age", "Balance", "NumOfProducts", "Exit_Probability", "Risk_Tier", "Status"] if c in scored_df.columns]
                                ],
                                use_container_width=True,
                            )

                            # Download CSV
                            csv_data = scored_df.to_csv(index=False).encode("utf-8")
                            st.download_button(
                                label="📥 Download Complete Scored Portfolio CSV",
                                data=csv_data,
                                file_name=f"retainiq_scored_portfolio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv",
                            )
                except Exception as e:
                    st.error(f"Batch scoring encountered an error: {e}")


# ============================================================
# TAB 3: MODEL SCORECARD & EVALUATION ARTIFACTS
# ============================================================

with tab_scorecard:
    st.markdown('<div class="section-title">Verified Model Scorecard & Evaluation Artifacts</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-description">'
        "Official performance benchmarks and explainability visualizations computed on the held-out test dataset in Phase 5."
        "</div>",
        unsafe_allow_html=True,
    )

    # Live Metrics Display from metadata
    meta_name = model_meta.get("model_name", "XGBoost_Tuned")
    acc = model_meta.get("test_accuracy", 0.8737)
    prec = model_meta.get("test_precision", 0.6519)
    rec = model_meta.get("test_recall", 0.8170)
    f1 = model_meta.get("test_f1", 0.7252)
    roc = model_meta.get("test_roc_auc", 0.9335)

    st.markdown(f"**Final Serving Model:** `{meta_name}`")

    mcol1, mcol2, mcol3, mcol4, mcol5 = st.columns(5)
    with mcol1:
        st.markdown(f'<div class="metric-box"><div class="metric-value">{acc * 100:.2f}%</div><div class="metric-label">Accuracy</div></div>', unsafe_allow_html=True)
    with mcol2:
        st.markdown(f'<div class="metric-box"><div class="metric-value">{prec * 100:.2f}%</div><div class="metric-label">Precision</div></div>', unsafe_allow_html=True)
    with mcol3:
        st.markdown(f'<div class="metric-box"><div class="metric-value">{rec * 100:.2f}%</div><div class="metric-label">Recall</div></div>', unsafe_allow_html=True)
    with mcol4:
        st.markdown(f'<div class="metric-box"><div class="metric-value">{f1 * 100:.2f}%</div><div class="metric-label">F1 Score</div></div>', unsafe_allow_html=True)
    with mcol5:
        st.markdown(f'<div class="metric-box"><div class="metric-value">{roc * 100:.2f}%</div><div class="metric-label">ROC-AUC</div></div>', unsafe_allow_html=True)

    st.write("---")

    # Evaluation Figures from reports/figures/
    figures_dir = PROJECT_ROOT / "reports" / "figures"

    fig_col1, fig_col2 = st.columns(2)

    with fig_col1:
        cm_file = figures_dir / "confusion_matrix.png"
        if cm_file.exists():
            st.image(str(cm_file), caption="Confusion Matrix — Test Set Performance", use_container_width=True)

        pr_file = figures_dir / "pr_curve.png"
        if pr_file.exists():
            st.image(str(pr_file), caption="Precision-Recall Curve (Minority Class)", use_container_width=True)

        fi_file = figures_dir / "feature_importance.png"
        if fi_file.exists():
            st.image(str(fi_file), caption="Intrinsic Feature Importance Ranking", use_container_width=True)

    with fig_col2:
        roc_file = figures_dir / "roc_curve.png"
        if roc_file.exists():
            st.image(str(roc_file), caption="Receiver Operating Characteristic (ROC) Curve", use_container_width=True)

        thresh_file = figures_dir / "threshold_analysis.png"
        if thresh_file.exists():
            st.image(str(thresh_file), caption="Decision Threshold Sensitivity Analysis", use_container_width=True)

        shap_file = figures_dir / "shap_summary.png"
        if shap_file.exists():
            st.image(str(shap_file), caption="Global SHAP Summary Beeswarm Plot", use_container_width=True)

    comp_file = figures_dir / "model_comparison.png"
    if comp_file.exists():
        st.write("---")
        st.image(str(comp_file), caption="Multi-Model Cross-Validation Benchmark Comparison", use_container_width=True)


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        RetainIQ • Enterprise Customer Retention & Exit Prediction System • Phase 6 Production Release
    </div>
    """,
    unsafe_allow_html=True,
)
