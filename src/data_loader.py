"""
RetainIQ Data Loader Module
Handles data ingestion, validation, and stratified train/test splitting.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils import PROJECT_ROOT, get_logger, load_config

logger = get_logger(__name__)

# Canonical schema expectations
EXPECTED_COLUMNS: List[str] = [
    "id",
    "CustomerId",
    "Surname",
    "CreditScore",
    "Geography",
    "Gender",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
    "Exited",
]

NUMERIC_COLUMNS: List[str] = [
    "CreditScore",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
    "Exited",
]

CATEGORICAL_COLUMNS: List[str] = [
    "Geography",
    "Gender",
    "Surname",
]


def load_data(
    config: Optional[Dict[str, Any]] = None,
    data_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """
    Load the raw banking dataset from the configured path.

    Args:
        config: Optional configuration dictionary. If None, loaded via load_config().
        data_path: Optional direct path to CSV file. Overrides config if provided.

    Returns:
        pd.DataFrame: Loaded raw dataset.

    Raises:
        FileNotFoundError: If the specified raw data file cannot be found.
    """
    if config is None:
        config = load_config()

    if data_path is None:
        configured_path = config.get("data", {}).get(
            "raw_path", "data/raw/bank_churn_raw.csv"
        )
        resolved_path = PROJECT_ROOT / configured_path
    else:
        resolved_path = Path(data_path)
        if not resolved_path.is_absolute():
            resolved_path = (PROJECT_ROOT / resolved_path).resolve()

    if not resolved_path.exists():
        raise FileNotFoundError(
            f"Raw data file does not exist at: {resolved_path}"
        )

    logger.info(f"Loading raw dataset from: {resolved_path}")
    df = pd.read_csv(resolved_path)
    logger.info(
        f"Successfully loaded dataset with {len(df):,} rows and {len(df.columns)} columns"
    )

    return df


# Alias for backward compatibility and clean API design
load_raw_data = load_data


def validate_data(
    df: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Validate dataset against schema, data quality, and integrity requirements.

    Checks performed:
    A. Dataset is not empty.
    B. Required columns exist.
    C. Target column exists.
    D. Target values are valid binary indicators (0 or 1).
    E. Missing values detected and reported without silent modification.
    F. Duplicate rows detected and reported without silent deletion.
    G. Numeric and string/categorical data types conform to expectations.
    H. Target distribution calculated and logged.

    Args:
        df: Input DataFrame to validate.
        config: Optional configuration dictionary. If None, loaded via load_config().

    Returns:
        Dict[str, Any]: Summary dictionary containing validation metrics.

    Raises:
        ValueError: If dataset is empty, missing required columns, or contains invalid target values.
        TypeError: If numeric columns contain non-numeric data types.
    """
    if config is None:
        config = load_config()

    # A. Check not empty
    if df.empty or len(df) == 0:
        raise ValueError("Data validation failed: Dataset is empty (0 rows).")

    # B. Check required columns exist
    missing_columns = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Data validation failed: Missing required columns: {missing_columns}. "
            f"Expected schema: {EXPECTED_COLUMNS}"
        )

    # C. Check target column exists
    target_column = config.get("features", {}).get("target_column", "Exited")
    if target_column not in df.columns:
        raise ValueError(
            f"Data validation failed: Configured target column '{target_column}' "
            f"not found in dataset columns: {list(df.columns)}"
        )

    # D. Check target values are valid binary indicators (0, 1)
    unique_target_values = set(df[target_column].dropna().unique())
    valid_binary_set = {0, 1, 0.0, 1.0}
    invalid_targets = unique_target_values - valid_binary_set
    if invalid_targets:
        raise ValueError(
            f"Data validation failed: Target column '{target_column}' contains invalid values: "
            f"{invalid_targets}. Expected binary indicators in {valid_binary_set}."
        )

    # E. Missing values: detect and report without dropping
    null_counts = df.isnull().sum()
    total_missing = int(null_counts.sum())
    missing_by_column = {
        col: int(count) for col, count in null_counts.items() if count > 0
    }
    if total_missing > 0:
        logger.warning(
            f"Missing values detected: {total_missing} total across columns: {missing_by_column}"
        )
    else:
        logger.info("Missing value check: 0 missing values found.")

    # F. Duplicate rows: detect and report without deleting
    total_duplicates = int(df.duplicated().sum())
    if total_duplicates > 0:
        logger.warning(f"Duplicate rows detected: {total_duplicates} duplicate rows.")
    else:
        logger.info("Duplicate rows check: 0 duplicate rows found.")

    # G. Data type checks
    for num_col in NUMERIC_COLUMNS:
        if num_col in df.columns and not pd.api.types.is_numeric_dtype(df[num_col]):
            raise TypeError(
                f"Data validation failed: Column '{num_col}' expected numeric type, "
                f"found dtype: {df[num_col].dtype}"
            )

    for cat_col in CATEGORICAL_COLUMNS:
        if cat_col in df.columns and not (
            pd.api.types.is_string_dtype(df[cat_col])
            or pd.api.types.is_object_dtype(df[cat_col])
        ):
            logger.warning(
                f"Column '{cat_col}' has dtype '{df[cat_col].dtype}', expected string/object."
            )

    # H. Target distribution calculation & logging
    target_counts = {
        int(k): int(v) for k, v in df[target_column].value_counts().items()
    }
    target_proportions = {
        int(k): round(float(v), 4)
        for k, v in df[target_column].value_counts(normalize=True).items()
    }
    logger.info(
        f"Target distribution for '{target_column}': "
        f"counts={target_counts}, proportions={target_proportions}"
    )

    validation_summary = {
        "is_valid": True,
        "rows": len(df),
        "columns": len(df.columns),
        "total_missing": total_missing,
        "missing_by_column": missing_by_column,
        "total_duplicates": total_duplicates,
        "target_counts": target_counts,
        "target_proportions": target_proportions,
    }

    return validation_summary


def split_data(
    df: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Perform leakage-safe, stratified train/test split.

    Separates features X and target y, drops non-predictive identifier columns
    (id, CustomerId, Surname) as defined in config, and creates stratified splits.

    Args:
        df: Input DataFrame containing features and target.
        config: Optional configuration dictionary. If None, loaded via load_config().

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
            (X_train, X_test, y_train, y_test)
    """
    if config is None:
        config = load_config()

    target_column = config.get("features", {}).get("target_column", "Exited")
    drop_columns = config.get("features", {}).get(
        "drop_columns", ["id", "CustomerId", "Surname"]
    )
    test_size = config.get("data", {}).get("test_size", 0.20)
    random_state = config.get("data", {}).get("random_state", 42)

    # Separate target
    y = df[target_column].astype(int)

    # Drop identifiers and target to form feature matrix X
    cols_to_remove = [c for c in drop_columns if c in df.columns] + [target_column]
    X = df.drop(columns=cols_to_remove)

    logger.info(
        f"Splitting data: test_size={test_size}, random_state={random_state}, "
        f"stratify={target_column}, dropped_identifiers={drop_columns}"
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    train_pct = y_train.value_counts(normalize=True).to_dict()
    test_pct = y_test.value_counts(normalize=True).to_dict()

    logger.info(
        f"Split complete: Train set={len(X_train):,} samples, "
        f"Test set={len(X_test):,} samples"
    )
    logger.info(f"Train target proportions: {train_pct}")
    logger.info(f"Test target proportions: {test_pct}")

    return X_train, X_test, y_train, y_test
