"""
RetainIQ Feature Engineering Module
Provides domain-specific feature engineering, polynomial transforms, interaction terms,
and a scikit-learn compatible transformer for customer churn prediction.
"""

from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif

from src.utils import get_logger

logger = get_logger(__name__)

# Base domain features from the initial prototype
BASE_ENGINEERED_FEATURES: List[str] = [
    "IsZeroBalance",
    "BalancePerProduct",
    "AgeSquared",
    "ActiveProductInteraction",
    "AgeActivityInteraction",
]

# Advanced banking and behavioral features introduced in Phase 3
NEW_ENGINEERED_FEATURES: List[str] = [
    "BalanceToSalaryRatio",
    "SalaryPerProduct",
    "AgeBin_Young",
    "AgeBin_Senior",
    "TenurePerAge",
    "IsNewCustomer",
    "IsLoyalCustomer",
    "LowCreditScore",
    "HighCreditScore",
    "HasManyProducts",
    "EngagementScore",
]

ALL_ENGINEERED_FEATURES: List[str] = BASE_ENGINEERED_FEATURES + NEW_ENGINEERED_FEATURES


def engineer_features(
    df: pd.DataFrame,
    include_all: bool = True,
) -> pd.DataFrame:
    """
    Apply domain-specific feature transformations to banking customer data.

    Calculates financial ratios, demographic polynomials, relationship depth indicators,
    and behavioral interactions. Preserves original features while appending new attributes.

    Args:
        df: Input DataFrame containing raw customer features. Expected columns include:
            'Balance', 'NumOfProducts', 'Age', 'IsActiveMember', 'EstimatedSalary',
            'Tenure', 'CreditScore', 'HasCrCard'.
        include_all: If True, computes both base prototype features and all Phase 3
                     enhanced features. If False, computes only the 5 base features.

    Returns:
        pd.DataFrame: A copy of the DataFrame with engineered feature columns appended.
    """
    df = df.copy()

    # Required columns for feature computation
    req_cols = ["Balance", "NumOfProducts", "Age", "IsActiveMember"]
    missing = [c for c in req_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Cannot engineer features: Missing required columns {missing} in DataFrame"
        )

    # Safe product count (avoid divide-by-zero if corrupted)
    products_safe = df["NumOfProducts"].replace(0, 1)

    # ============================================================
    # 1. BASELINE ENGINEERED FEATURES (Prototype Compatibility)
    # ============================================================

    # Binary indicator for dormant zero-balance accounts
    df["IsZeroBalance"] = (df["Balance"] == 0).astype(int)

    # Average balance per product held
    df["BalancePerProduct"] = df["Balance"] / products_safe

    # Quadratic age penalty capturing non-linear churn risk in older demographics
    df["AgeSquared"] = df["Age"] ** 2

    # Interaction between active engagement and portfolio breadth
    df["ActiveProductInteraction"] = df["IsActiveMember"] * df["NumOfProducts"]

    # Interaction between customer age and activity index
    df["AgeActivityInteraction"] = df["Age"] * df["IsActiveMember"]

    if not include_all:
        return df

    # ============================================================
    # 2. ADVANCED FINANCIAL & BALANCE RATIOS
    # ============================================================

    # Wealth-to-income ratio (salary normalized, +1 prevents division by zero)
    if "EstimatedSalary" in df.columns:
        df["BalanceToSalaryRatio"] = df["Balance"] / (df["EstimatedSalary"] + 1.0)
        df["SalaryPerProduct"] = df["EstimatedSalary"] / products_safe
    else:
        df["BalanceToSalaryRatio"] = 0.0
        df["SalaryPerProduct"] = 0.0

    # ============================================================
    # 3. DEMOGRAPHIC SEGMENTATION & LIFECYCLE
    # ============================================================

    # Age segment flags: high retention youth (<30) vs peak churn window (>=50)
    df["AgeBin_Young"] = (df["Age"] < 30).astype(int)
    df["AgeBin_Senior"] = (df["Age"] >= 50).astype(int)

    # Tenure as a fraction of customer life
    if "Tenure" in df.columns:
        df["TenurePerAge"] = df["Tenure"] / (df["Age"] + 1.0)
        # New customer flag (tenure <= 1 year has distinct onboarding churn patterns)
        df["IsNewCustomer"] = (df["Tenure"] <= 1).astype(int)
        # Long-term established relationship flag (tenure >= 7 years)
        df["IsLoyalCustomer"] = (df["Tenure"] >= 7).astype(int)
    else:
        df["TenurePerAge"] = 0.0
        df["IsNewCustomer"] = 0
        df["IsLoyalCustomer"] = 0

    # ============================================================
    # 4. CREDIT RISK & RELATIONSHIP DEPTH
    # ============================================================

    # Subprime credit indicator (<500) vs Prime credit tier (>=750)
    if "CreditScore" in df.columns:
        df["LowCreditScore"] = (df["CreditScore"] < 500).astype(int)
        df["HighCreditScore"] = (df["CreditScore"] >= 750).astype(int)
    else:
        df["LowCreditScore"] = 0
        df["HighCreditScore"] = 0

    # Non-linear portfolio churn: customers holding 3 or 4 products have elevated exit rates
    df["HasManyProducts"] = (df["NumOfProducts"] >= 3).astype(int)

    # Composite engagement score (0 to 3) combining membership, credit card, and multi-product
    has_card = df["HasCrCard"] if "HasCrCard" in df.columns else 0
    multi_prod = (df["NumOfProducts"] >= 2).astype(int)
    df["EngagementScore"] = df["IsActiveMember"] + has_card + multi_prod

    return df


class DomainFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible transformer for domain feature engineering.
    Can be seamlessly embedded into scikit-learn Pipelines.
    """

    def __init__(self, include_all: bool = True):
        self.include_all = include_all

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Fit transformer (stateless transformation)."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply feature engineering to X."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError("DomainFeatureEngineer expects a pandas DataFrame input.")
        return engineer_features(X, include_all=self.include_all)


def select_features(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: Optional[pd.DataFrame] = None,
    k: int = 15,
    method: str = "mutual_info",
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], List[str]]:
    """
    Select top-k features based on statistical association with target.

    Args:
        X_train: Training feature DataFrame.
        y_train: Training target Series.
        X_test: Optional test feature DataFrame.
        k: Number of top features to select.
        method: Scoring function ('mutual_info' or 'f_classif').

    Returns:
        Tuple[pd.DataFrame, Optional[pd.DataFrame], List[str]]:
            - X_train_selected (DataFrame with top k features)
            - X_test_selected (DataFrame with top k features, if X_test provided)
            - selected_feature_names (List of selected column names)
    """
    score_fn = mutual_info_classif if method == "mutual_info" else f_classif
    effective_k = min(k, X_train.shape[1])

    logger.info(
        f"Selecting top {effective_k} features from {X_train.shape[1]} using {method}..."
    )
    selector = SelectKBest(score_func=score_fn, k=effective_k)
    selector.fit(X_train, y_train)

    mask = selector.get_support()
    selected_cols = [col for col, selected in zip(X_train.columns, mask) if selected]

    X_train_selected = X_train[selected_cols].copy()
    X_test_selected = X_test[selected_cols].copy() if X_test is not None else None

    logger.info(f"Selected {len(selected_cols)} features: {selected_cols}")
    return X_train_selected, X_test_selected, selected_cols
