import streamlit as st
import requests


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="RetainIQ",
    page_icon="🏦",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    /* Header */

    .brand {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 0;
    }

    .tagline {
        color: #6b7280;
        font-size: 17px;
        margin-top: 4px;
    }

    /* Section headings */

    .section-title {
        font-size: 23px;
        font-weight: 650;
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
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 22px 24px 10px 24px;
        margin-bottom: 28px;
    }

    /* Prediction result */

    .result-card {
        background-color: #f8fafc;
        border: 1px solid #e5e7eb;
        border-radius: 16px;
        padding: 28px;
        margin-top: 25px;
    }

    .result-status {
        font-size: 28px;
        font-weight: 700;
        margin: 12px 0 25px 0;
    }

    .probability-value {
        font-size: 38px;
        font-weight: 700;
        margin-top: 5px;
    }

    .risk-description {
        color: #6b7280;
        font-size: 14px;
        margin-top: 5px;
    }

    /* SHAP explanation */

    .explanation-card {
        background-color: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 18px;
        margin-top: 12px;
    }

    .explanation-feature {
        font-size: 16px;
        font-weight: 600;
    }

    .explanation-impact {
        font-size: 13px;
        color: #6b7280;
    }

    /* Model information */

    .model-card {
        background-color: #f8fafc;
        border: 1px solid #e5e7eb;
        border-radius: 16px;
        padding: 24px;
        margin-top: 28px;
    }

    .metric-box {
        background-color: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }

    .metric-value {
        font-size: 25px;
        font-weight: 700;
    }

    .metric-label {
        color: #6b7280;
        font-size: 13px;
        margin-top: 4px;
    }

    /* Prediction button */

    div.stButton > button {
        height: 52px;
        border-radius: 10px;
        font-size: 17px;
        font-weight: 600;
    }

    /* Footer */

    .footer {
        text-align: center;
        color: #9ca3af;
        font-size: 13px;
        margin-top: 40px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="brand">🏦 RetainIQ</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="tagline">'
    'Customer Retention Intelligence'
    '</div>',
    unsafe_allow_html=True
)

st.write(
    "Machine learning-powered customer exit prediction "
    "and banking analytics."
)

st.divider()


# ============================================================
# CUSTOMER PROFILE
# ============================================================

st.markdown(
    '<div class="input-section">',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-title">👤 Customer Profile</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-description">'
    'Enter the customer demographic and relationship information.'
    '</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:

    credit_score = st.number_input(
        "Credit Score",
        min_value=300,
        max_value=850,
        value=650,
        step=1
    )

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=40,
        step=1
    )

    geography = st.selectbox(
        "Geography",
        ["France", "Germany", "Spain"]
    )

with col2:

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    tenure = st.number_input(
        "Tenure (Years)",
        min_value=0,
        max_value=10,
        value=5,
        step=1
    )

    num_products = st.number_input(
        "Number of Products",
        min_value=1,
        max_value=4,
        value=2,
        step=1
    )

st.markdown(
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# ACCOUNT INFORMATION
# ============================================================

st.markdown(
    '<div class="input-section">',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-title">💳 Account Information</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-description">'
    'Enter the customer financial and account activity details.'
    '</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:

    balance = st.number_input(
        "Account Balance",
        min_value=0.0,
        value=80000.0,
        step=1000.0
    )

    estimated_salary = st.number_input(
        "Estimated Salary",
        min_value=0.0,
        value=100000.0,
        step=5000.0
    )

with col2:

    has_credit_card = st.selectbox(
        "Has Credit Card",
        ["Yes", "No"]
    )

    is_active_member = st.selectbox(
        "Active Member",
        ["Yes", "No"]
    )

st.markdown(
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# PREDICTION SECTION
# ============================================================

st.markdown(
    '<div class="section-title">🔮 Customer Risk Assessment</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-description">'
    'Use the trained machine learning model to estimate '
    'the likelihood of customer exit.'
    '</div>',
    unsafe_allow_html=True
)

predict_button = st.button(
    "🔮 Predict Closure",
    use_container_width=True
)


# ============================================================
# PREDICTION LOGIC
# ============================================================

if predict_button:

    payload = {

        "CreditScore": credit_score,

        "Geography": geography,

        "Gender": gender,

        "Age": age,

        "Tenure": tenure,

        "Balance": balance,

        "NumOfProducts": num_products,

        "HasCrCard": (
            1 if has_credit_card == "Yes" else 0
        ),

        "IsActiveMember": (
            1 if is_active_member == "Yes" else 0
        ),

        "EstimatedSalary": estimated_salary
    }

    try:

        response = requests.post(
            "http://127.0.0.1:8000/predict",
            json=payload,
            timeout=10
        )


        # ====================================================
        # SUCCESSFUL RESPONSE
        # ====================================================

        if response.status_code == 200:

            result = response.json()

            prediction = result["prediction"]

            probability = result["exit_probability"]

            explanation = result.get(
                "explanation",
                []
            )


            # =================================================
            # PREDICTION RESULT CARD
            # =================================================

            st.markdown(
                '<div class="result-card">',
                unsafe_allow_html=True
            )

            st.markdown(
                '<div class="section-title">'
                '📊 Prediction Result'
                '</div>',
                unsafe_allow_html=True
            )


            # Status

            if prediction == 1:

                st.markdown(
                    '<div class="result-status">'
                    '🔴 Likely to Exit'
                    '</div>',
                    unsafe_allow_html=True
                )

            else:

                st.markdown(
                    '<div class="result-status">'
                    '🟢 Likely to Stay'
                    '</div>',
                    unsafe_allow_html=True
                )


            # =================================================
            # RESULT METRICS
            # =================================================

            col1, col2, col3 = st.columns(3)


            # Exit probability

            with col1:

                st.markdown(
                    '<div class="risk-description">'
                    'EXIT PROBABILITY'
                    '</div>',
                    unsafe_allow_html=True
                )

                st.markdown(
                    f'<div class="probability-value">'
                    f'{probability * 100:.2f}%'
                    f'</div>',
                    unsafe_allow_html=True
                )


            # Prediction

            with col2:

                st.markdown(
                    '<div class="risk-description">'
                    'PREDICTION'
                    '</div>',
                    unsafe_allow_html=True
                )

                st.markdown(
                    f'<div class="probability-value" '
                    f'style="font-size: 25px;">'
                    f'{result["status"]}'
                    f'</div>',
                    unsafe_allow_html=True
                )


            # Risk level

            with col3:

                if probability < 0.40:

                    risk_level = "Low"

                elif probability < 0.70:

                    risk_level = "Moderate"

                else:

                    risk_level = "High"


                st.markdown(
                    '<div class="risk-description">'
                    'RISK LEVEL'
                    '</div>',
                    unsafe_allow_html=True
                )

                st.markdown(
                    f'<div class="probability-value" '
                    f'style="font-size: 25px;">'
                    f'{risk_level}'
                    f'</div>',
                    unsafe_allow_html=True
                )


            # =================================================
            # PROBABILITY BAR
            # =================================================

            st.write("### Exit Probability")

            st.progress(
                min(max(probability, 0.0), 1.0)
            )

            st.caption(
                f"The model estimates a "
                f"{probability * 100:.2f}% probability "
                f"of customer exit."
            )


            # =================================================
            # CUSTOMER SNAPSHOT
            # =================================================

            st.write("### 👤 Customer Snapshot")

            col1, col2, col3, col4 = st.columns(4)

            with col1:

                st.metric(
                    "Age",
                    f"{age:.0f}"
                )

            with col2:

                st.metric(
                    "Credit Score",
                    f"{credit_score:.0f}"
                )

            with col3:

                st.metric(
                    "Products",
                    f"{num_products:.0f}"
                )

            with col4:

                st.metric(
                    "Balance",
                    f"{balance:,.0f}"
                )


            # =================================================
            # INTERPRETATION
            # =================================================

            st.write("### 💡 Interpretation")

            if probability < 0.40:

                st.info(
                    "The model assigns a relatively low "
                    "probability of customer exit for "
                    "this profile."
                )

            elif probability < 0.70:

                st.warning(
                    "The model assigns a moderate probability "
                    "of customer exit for this profile."
                )

            else:

                st.warning(
                    "The model assigns a relatively high "
                    "probability of customer exit for "
                    "this profile."
                )


            st.markdown(
                '</div>',
                unsafe_allow_html=True
            )


            # =================================================
            # SHAP EXPLANATION
            # =================================================

            st.divider()

            st.markdown(
                '<div class="section-title">'
                '🔍 Why This Prediction?'
                '</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                '<div class="section-description">'
                'These are the customer features that had the '
                'largest influence on this individual prediction.'
                '</div>',
                unsafe_allow_html=True
            )


            # Feature display names

            feature_display_names = {

                "Age":
                    "Age",

                "AgeSquared":
                    "Age (Nonlinear Effect)",

                "CreditScore":
                    "Credit Score",

                "NumOfProducts":
                    "Number of Products",

                "IsActiveMember":
                    "Active Member",

                "Balance":
                    "Balance",

                "BalancePerProduct":
                    "Balance per Product",

                "EstimatedSalary":
                    "Estimated Salary",

                "Tenure":
                    "Tenure",

                "HasCrCard":
                    "Credit Card",

                "Gender_Male":
                    "Gender",

                "Geography_Germany":
                    "Germany",

                "Geography_Spain":
                    "Spain",

                "IsZeroBalance":
                    "Zero Balance",

                "ActiveProductInteraction":
                    "Activity × Products",

                "AgeActivityInteraction":
                    "Age × Activity"
            }


            # =================================================
            # DISPLAY SHAP FEATURES
            # =================================================

            if explanation:

                # Find maximum absolute SHAP value
                max_shap = max(
                    abs(float(item["shap_value"]))
                    for item in explanation
                )

                if max_shap == 0:
                    max_shap = 1


                for item in explanation:

                    feature = item["feature"]

                    shap_value = float(
                        item["shap_value"]
                    )

                    impact = item["impact"]


                    display_name = feature_display_names.get(
                        feature,
                        feature
                    )


                    # Positive / negative icon

                    if shap_value > 0:

                        icon = "🔴"

                    else:

                        icon = "🟢"


                    col1, col2 = st.columns([2, 3])


                    with col1:

                        st.markdown(
                            f"**{icon} {display_name}**"
                        )


                    with col2:

                        st.caption(
                            impact
                        )


                    # Relative SHAP magnitude

                    st.progress(
                        min(
                            abs(shap_value) / max_shap,
                            1.0
                        )
                    )


            else:

                st.info(
                    "No explanation data was returned "
                    "by the model."
                )


            # =================================================
            # CUSTOMER DETAILS
            # =================================================

            with st.expander(
                "View Customer Details"
            ):

                detail_col1, detail_col2 = st.columns(2)


                with detail_col1:

                    st.write(
                        f"**Geography:** {geography}"
                    )

                    st.write(
                        f"**Gender:** {gender}"
                    )

                    st.write(
                        f"**Age:** {age:.0f}"
                    )

                    st.write(
                        f"**Tenure:** {tenure:.0f} years"
                    )

                    st.write(
                        f"**Credit Score:** "
                        f"{credit_score:.0f}"
                    )


                with detail_col2:

                    st.write(
                        f"**Balance:** "
                        f"{balance:,.2f}"
                    )

                    st.write(
                        f"**Products:** "
                        f"{num_products:.0f}"
                    )

                    st.write(
                        f"**Credit Card:** "
                        f"{has_credit_card}"
                    )

                    st.write(
                        f"**Active Member:** "
                        f"{is_active_member}"
                    )

                    st.write(
                        f"**Estimated Salary:** "
                        f"{estimated_salary:,.2f}"
                    )


        # ====================================================
        # API ERROR
        # ====================================================

        else:

            st.error(
                f"❌ API Error: {response.status_code}"
            )

            try:

                st.json(
                    response.json()
                )

            except Exception:

                st.write(
                    response.text
                )


    # ========================================================
    # CONNECTION ERROR
    # ========================================================

    except requests.exceptions.ConnectionError:

        st.error(
            "❌ Could not connect to FastAPI."
        )

        st.info(
            "Make sure FastAPI is running on "
            "http://127.0.0.1:8000"
        )


    # ========================================================
    # TIMEOUT ERROR
    # ========================================================

    except requests.exceptions.Timeout:

        st.error(
            "❌ The prediction request timed out."
        )


    # ========================================================
    # OTHER ERROR
    # ========================================================

    except Exception as e:

        st.error(
            f"❌ Unexpected error: {str(e)}"
        )


# ============================================================
# MODEL INFORMATION
# ============================================================

st.divider()

st.markdown(
    '<div class="model-card">',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-title">'
    '🤖 Model Information'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-description">'
    'Performance of the final machine learning model '
    'evaluated on the held-out test dataset.'
    '</div>',
    unsafe_allow_html=True
)


# Model name

st.markdown(
    """
    <div style="margin-bottom: 18px;">
        <strong>Model:</strong>
        Feature-Engineered XGBoost
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# MODEL METRICS
# ============================================================

col1, col2, col3, col4, col5 = st.columns(5)


with col1:

    st.markdown(
        """
        <div class="metric-box">
            <div class="metric-value">89.90%</div>
            <div class="metric-label">Accuracy</div>
        </div>
        """,
        unsafe_allow_html=True
    )


with col2:

    st.markdown(
        """
        <div class="metric-box">
            <div class="metric-value">80.47%</div>
            <div class="metric-label">Precision</div>
        </div>
        """,
        unsafe_allow_html=True
    )


with col3:

    st.markdown(
        """
        <div class="metric-box">
            <div class="metric-value">66.67%</div>
            <div class="metric-label">Recall</div>
        </div>
        """,
        unsafe_allow_html=True
    )


with col4:

    st.markdown(
        """
        <div class="metric-box">
            <div class="metric-value">72.92%</div>
            <div class="metric-label">F1 Score</div>
        </div>
        """,
        unsafe_allow_html=True
    )


with col5:

    st.markdown(
        """
        <div class="metric-box">
            <div class="metric-value">93.35%</div>
            <div class="metric-label">ROC-AUC</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# Evaluation note

st.caption(
    "Evaluation results are based on the held-out test split "
    "used during model development. They represent model "
    "performance on this dataset and are not guarantees of "
    "future real-world performance."
)

st.markdown(
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        RetainIQ • Machine Learning Customer Analytics
    </div>
    """,
    unsafe_allow_html=True
)