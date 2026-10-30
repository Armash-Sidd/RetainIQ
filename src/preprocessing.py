"""
RetainIQ Preprocessing Module
Provides a leakage-safe scikit-learn ColumnTransformer pipeline for categorical
encoding and numerical handling.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.utils import get_logger, load_config

logger = get_logger(__name__)


def get_feature_types(
    X: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[List[str], List[str]]:
    """
    Derive categorical and numerical column lists from input feature matrix X.

    Categorical columns are read from config; numerical columns are dynamically
    derived from remaining feature columns in X.

    Args:
        X: Input feature DataFrame.
        config: Optional configuration dictionary. If None, loaded via load_config().

    Returns:
        Tuple[List[str], List[str]]: (categorical_columns, numerical_columns)
    """
    if config is None:
        config = load_config()

    configured_cat = config.get("features", {}).get(
        "categorical_columns", ["Geography", "Gender"]
    )
    cat_columns = [col for col in configured_cat if col in X.columns]
    num_columns = [col for col in X.columns if col not in cat_columns]

    return cat_columns, num_columns


def create_preprocessor(
    X: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None,
    scale_numeric: bool = False,
) -> ColumnTransformer:
    """
    Construct a scikit-learn ColumnTransformer for leakage-safe preprocessing.

    Categorical columns are processed with OneHotEncoder(handle_unknown='ignore').
    Numerical columns use 'passthrough' by default, or an optional StandardScaler
    if scale_numeric=True (useful for linear baselines like Logistic Regression in Phase 4).

    Args:
        X: DataFrame to inspect feature names and determine column partitions.
        config: Optional configuration dictionary. If None, loaded via load_config().
        scale_numeric: Whether to apply StandardScaler to numerical features. Default: False.

    Returns:
        ColumnTransformer: Unfitted preprocessor.
    """
    if config is None:
        config = load_config()

    cat_columns, num_columns = get_feature_types(X, config)

    logger.info(
        f"Configuring preprocessor: categorical={cat_columns}, "
        f"numerical={num_columns}, scale_numeric={scale_numeric}"
    )

    cat_pipeline = Pipeline(
        steps=[
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            )
        ]
    )

    if scale_numeric:
        num_pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
            ]
        )
    else:
        num_pipeline = "passthrough"

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", cat_pipeline, cat_columns),
            ("num", num_pipeline, num_columns),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )

    return preprocessor


def fit_and_transform_pipeline(
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, ColumnTransformer]:
    """
    Leakage-safe helper: fit the preprocessor ONLY on X_train, then transform both
    X_train and X_test.

    Args:
        preprocessor: Unfitted ColumnTransformer.
        X_train: Training feature DataFrame.
        X_test: Testing feature DataFrame.

    Returns:
        Tuple[np.ndarray, np.ndarray, ColumnTransformer]:
            (X_train_transformed, X_test_transformed, fitted_preprocessor)
    """
    logger.info("Fitting preprocessor ONLY on X_train to prevent data leakage...")
    X_train_transformed = preprocessor.fit_transform(X_train)

    logger.info("Transforming X_test using preprocessor fitted on X_train...")
    X_test_transformed = preprocessor.transform(X_test)

    logger.info(
        f"Transformation complete: X_train_trans={X_train_transformed.shape}, "
        f"X_test_trans={X_test_transformed.shape}"
    )

    return X_train_transformed, X_test_transformed, preprocessor


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical columns (Geography, Gender) into binary indicators matching
    the project specification (France as baseline for Geography, Female for Gender).

    Args:
        df: Input DataFrame with raw categorical columns.

    Returns:
        pd.DataFrame: DataFrame with one-hot columns and raw categoricals dropped.
    """
    df = df.copy()
    if "Geography" in df.columns:
        df["Geography_Germany"] = (df["Geography"] == "Germany").astype(int)
        df["Geography_Spain"] = (df["Geography"] == "Spain").astype(int)
        df = df.drop(columns=["Geography"])
    if "Gender" in df.columns:
        df["Gender_Male"] = (df["Gender"] == "Male").astype(int)
        df = df.drop(columns=["Gender"])
    return df


def prepare_features(
    df: pd.DataFrame,
    include_all_features: bool = True,
) -> pd.DataFrame:
    """
    Prepare full feature set by combining categorical encoding and domain feature engineering.

    Args:
        df: Input DataFrame with raw features.
        include_all_features: If True, includes both baseline and Phase 3 engineered features.

    Returns:
        pd.DataFrame: Processed feature matrix ready for model consumption.
    """
    from src.feature_engineering import engineer_features

    encoded = encode_categoricals(df)
    features = engineer_features(encoded, include_all=include_all_features)
    return features


# Reusable alias for end-to-end feature preprocessing
preprocess = prepare_features

