"""
RetainIQ Hyperparameter Tuning Module
Implements Stratified K-Fold search optimization for gradient-boosted decision trees.
"""

from typing import Any, Dict, Optional, Tuple

import pandas as pd
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
)
from xgboost import XGBClassifier

from src.utils import get_logger

logger = get_logger(__name__)

# Default XGBoost search space specified in project requirements
DEFAULT_XGB_PARAM_GRID: Dict[str, list] = {
    "max_depth": [4, 6, 8],
    "learning_rate": [0.01, 0.05, 0.1],
    "n_estimators": [100, 200, 300],
    "scale_pos_weight": [1, 3, 4],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
}


def tune_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    param_grid: Optional[Dict[str, list]] = None,
    cv_folds: int = 5,
    random_state: int = 42,
    use_grid_search: bool = False,
    n_iter: int = 25,
) -> Tuple[XGBClassifier, Dict[str, Any], float]:
    """
    Tune XGBoost hyperparameters using Stratified K-Fold cross-validation.

    Evaluates candidate parameter configurations strictly on the training dataset
    to prevent information leakage from held-out evaluation splits.

    Args:
        X_train: Preprocessed training feature matrix.
        y_train: Training target labels (0 or 1).
        param_grid: Dictionary mapping parameter names to lists of candidate values.
        cv_folds: Number of stratified folds (default: 5).
        random_state: Seed for fold shuffling and search reproducibility.
        use_grid_search: If True, uses exhaustive GridSearchCV; if False (default),
                         uses RandomizedSearchCV for computational efficiency.
        n_iter: Number of parameter combinations to sample if using RandomizedSearchCV.

    Returns:
        Tuple[XGBClassifier, Dict[str, Any], float]:
            - best_model (fitted with best parameters on X_train)
            - best_params (dictionary of optimal hyperparameters)
            - best_cv_roc_auc (mean validation ROC-AUC across CV folds)
    """
    if param_grid is None:
        param_grid = DEFAULT_XGB_PARAM_GRID

    skf = StratifiedKFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=random_state,
    )

    base_xgb = XGBClassifier(
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=-1,
    )

    if use_grid_search:
        total_candidates = 1
        for v in param_grid.values():
            total_candidates *= len(v)
        logger.info(
            f"Starting GridSearchCV: {total_candidates} candidates across {cv_folds} folds "
            f"({total_candidates * cv_folds} total fits)..."
        )
        search = GridSearchCV(
            estimator=base_xgb,
            param_grid=param_grid,
            scoring="roc_auc",
            cv=skf,
            n_jobs=-1,
            verbose=1,
            refit=True,
        )
    else:
        logger.info(
            f"Starting RandomizedSearchCV: {n_iter} iterations across {cv_folds} folds "
            f"({n_iter * cv_folds} total fits)..."
        )
        search = RandomizedSearchCV(
            estimator=base_xgb,
            param_distributions=param_grid,
            n_iter=n_iter,
            scoring="roc_auc",
            cv=skf,
            random_state=random_state,
            n_jobs=-1,
            verbose=1,
            refit=True,
        )

    search.fit(X_train, y_train)

    best_model: XGBClassifier = search.best_estimator_
    best_params: Dict[str, Any] = search.best_params_
    best_cv_roc_auc: float = float(search.best_score_)

    logger.info("Hyperparameter tuning completed successfully.")
    logger.info(f"Best CV ROC-AUC: {best_cv_roc_auc:.4f}")
    logger.info(f"Optimal parameters: {best_params}")

    return best_model, best_params, best_cv_roc_auc
