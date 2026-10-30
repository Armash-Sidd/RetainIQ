from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import os
import sys
import shap


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.append(PROJECT_ROOT)


# ============================================================
# FEATURE ENGINEERING
# ============================================================

from src.feature_engineering import engineer_features


# ============================================================
# LOAD MODEL AND FEATURES
# ============================================================

MODEL_DIR = (
    "models"
    if os.path.exists(os.path.join(PROJECT_ROOT, "models"))
    else "model"
)

# Prefer final_model.pkl, fall back to xgb_model.pkl for backward compatibility
FINAL_MODEL_FILE = os.path.join(PROJECT_ROOT, MODEL_DIR, "final_model.pkl")
XGB_MODEL_FILE = os.path.join(PROJECT_ROOT, MODEL_DIR, "xgb_model.pkl")

MODEL_PATH = (
    FINAL_MODEL_FILE
    if os.path.exists(FINAL_MODEL_FILE)
    else XGB_MODEL_FILE
)

FEATURE_PATH = os.path.join(
    PROJECT_ROOT,
    MODEL_DIR,
    "feature_names.pkl"
)

model = joblib.load(MODEL_PATH)

feature_names = joblib.load(FEATURE_PATH)


# ============================================================
# SHAP EXPLAINER
# ============================================================

explainer = shap.TreeExplainer(model)


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="RetainIQ API",
    description="Savings Account Closure Prediction API",
    version="1.0.0"
)


# ============================================================
# CUSTOMER INPUT MODEL
# ============================================================

class CustomerData(BaseModel):

    CreditScore: float
    Geography: str
    Gender: str
    Age: float
    Tenure: float
    Balance: float
    NumOfProducts: float
    HasCrCard: int
    IsActiveMember: int
    EstimatedSalary: float


# ============================================================
# HOME ENDPOINT
# ============================================================

@app.get("/")
def home():

    return {
        "message": "RetainIQ API is running"
    }


# ============================================================
# PREDICTION ENDPOINT
# ============================================================

@app.post("/predict")
def predict(customer: CustomerData):

    # --------------------------------------------------------
    # Convert customer input to DataFrame
    # --------------------------------------------------------

    customer_df = pd.DataFrame(
        [customer.model_dump()]
    )


    # --------------------------------------------------------
    # Encode categorical variables
    # --------------------------------------------------------

    customer_df["Geography_Germany"] = (
        customer_df["Geography"] == "Germany"
    ).astype(int)

    customer_df["Geography_Spain"] = (
        customer_df["Geography"] == "Spain"
    ).astype(int)

    customer_df["Gender_Male"] = (
        customer_df["Gender"] == "Male"
    ).astype(int)


    # --------------------------------------------------------
    # Remove original categorical columns
    # --------------------------------------------------------

    customer_df = customer_df.drop(
        columns=["Geography", "Gender"]
    )


    # --------------------------------------------------------
    # Apply feature engineering
    # --------------------------------------------------------

    customer_df = engineer_features(
        customer_df
    )


    # --------------------------------------------------------
    # Match training feature order
    # --------------------------------------------------------

    customer_df = customer_df[
        feature_names
    ]


    # ========================================================
    # MODEL PREDICTION
    # ========================================================

    prediction = model.predict(
        customer_df
    )[0]

    probability = model.predict_proba(
        customer_df
    )[0][1]


    # ========================================================
    # SHAP EXPLANATION
    # ========================================================

    shap_values = explainer.shap_values(
        customer_df
    )


    # --------------------------------------------------------
    # Handle different SHAP output formats
    # --------------------------------------------------------

    if isinstance(shap_values, list):

        # Binary classification
        shap_values = shap_values[1]

    elif len(shap_values.shape) == 3:

        # Newer SHAP versions
        shap_values = shap_values[:, :, 1]


    # --------------------------------------------------------
    # SHAP values for this customer
    # --------------------------------------------------------

    customer_shap = shap_values[0]


    # ========================================================
    # CREATE EXPLANATION DATAFRAME
    # ========================================================

    explanation = pd.DataFrame(
        {
            "feature": feature_names,
            "shap_value": customer_shap,
            "absolute_value": abs(customer_shap)
        }
    )


    # --------------------------------------------------------
    # Sort by absolute SHAP impact
    # --------------------------------------------------------

    explanation = explanation.sort_values(
        by="absolute_value",
        ascending=False
    )


    # --------------------------------------------------------
    # Select top 6 features
    # --------------------------------------------------------

    top_features = explanation.head(6)


    # ========================================================
    # FORMAT EXPLANATION FOR API
    # ========================================================

    shap_explanation = []

    for _, row in top_features.iterrows():

        shap_explanation.append(
            {
                "feature": row["feature"],

                "shap_value": round(
                    float(row["shap_value"]),
                    4
                ),

                "impact": (
                    "Increases exit likelihood"
                    if row["shap_value"] > 0
                    else "Decreases exit likelihood"
                )
            }
        )


    # ========================================================
    # API RESPONSE
    # ========================================================

    return {

        "prediction": int(prediction),

        "exit_probability": round(
            float(probability),
            4
        ),

        "status": (
            "Likely to Exit"
            if prediction == 1
            else "Likely to Stay"
        ),

        "explanation": shap_explanation
    }