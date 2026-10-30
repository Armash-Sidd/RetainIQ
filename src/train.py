"""
RetainIQ Model Training & Evaluation Pipeline
Orchestrates data ingestion, stratified splitting, feature engineering,
multi-model benchmarking, hyperparameter tuning, model selection,
and test-set evaluation.

Usage:
    python -m src.train
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.data_loader import load_raw_data, split_data, validate_data
from src.preprocessing import preprocess
from src.tuning import tune_xgboost
from src.utils import PROJECT_ROOT, get_logger, load_config, set_seed

logger = get_logger(__name__)


def build_benchmark_models(
    scale_pos_weight: float,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Construct a dictionary of candidate classification models.

    Args:
        scale_pos_weight: Imbalance ratio (negatives / positives) for XGBoost.
        random_state: Seed for reproducibility across estimators.

    Returns:
        Dict[str, Any]: Dictionary mapping model identifiers to estimator instances.
    """
    return {
        "LogisticRegression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "DecisionTree": DecisionTreeClassifier(
            max_depth=10,
            class_weight="balanced",
            random_state=random_state,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
        "XGBoost_Baseline": XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1,
        ),
        "XGBoost_Imbalanced": XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1,
        ),
    }


def cross_validate_benchmark(
    models: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_folds: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Evaluate multiple candidate models using Stratified K-Fold cross-validation.

    Args:
        models: Dictionary of model estimators.
        X_train: Training feature matrix.
        y_train: Training target labels.
        cv_folds: Number of cross-validation folds (default: 5).
        random_state: Seed for fold shuffling.

    Returns:
        pd.DataFrame: Summary table containing average performance metrics per model.
    """
    skf = StratifiedKFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=random_state,
    )
    scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    records: List[Dict[str, Any]] = []

    logger.info(
        f"Evaluating {len(models)} benchmark models using {cv_folds}-fold Stratified CV..."
    )

    for name, model in models.items():
        logger.info(f"Cross-validating {name}...")
        cv_res = cross_validate(
            estimator=model,
            X=X_train,
            y=y_train,
            cv=skf,
            scoring=scoring,
            n_jobs=-1,
            return_train_score=False,
        )

        acc = float(np.mean(cv_res["test_accuracy"]))
        prec = float(np.mean(cv_res["test_precision"]))
        rec = float(np.mean(cv_res["test_recall"]))
        f1 = float(np.mean(cv_res["test_f1"]))
        roc = float(np.mean(cv_res["test_roc_auc"]))

        logger.info(
            f"  {name:20s} -> Accuracy: {acc:.4f} | Recall: {rec:.4f} | "
            f"F1: {f1:.4f} | ROC-AUC: {roc:.4f}"
        )

        records.append(
            {
                "model": name,
                "cv_accuracy": round(acc, 4),
                "cv_precision": round(prec, 4),
                "cv_recall": round(rec, 4),
                "cv_f1": round(f1, 4),
                "cv_roc_auc": round(roc, 4),
            }
        )

    return pd.DataFrame(records)


def train_final_model(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Any:
    """
    Train the selected final model on the full training dataset.

    Args:
        model: Configured estimator instance.
        X_train: Complete training feature matrix.
        y_train: Complete training target labels.

    Returns:
        Fitted model instance.
    """
    logger.info("Fitting final selected model on full training set...")
    model.fit(X_train, y_train)
    return model


def evaluate_test_set(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, Any]:
    """
    Evaluate the final model once on the completely untouched test set.

    Args:
        model: Fitted estimator.
        X_test: Held-out test feature matrix.
        y_test: Ground truth test target labels.

    Returns:
        Dict[str, Any]: Test evaluation metrics and confusion matrix.
    """
    logger.info("Evaluating final model on untouched test set...")
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
        test_roc = float(roc_auc_score(y_test, y_prob))
    else:
        test_roc = float(roc_auc_score(y_test, y_pred))

    test_acc = float(accuracy_score(y_test, y_pred))
    test_prec = float(precision_score(y_test, y_pred, zero_division=0))
    test_rec = float(recall_score(y_test, y_pred, zero_division=0))
    test_f1 = float(f1_score(y_test, y_pred, zero_division=0))
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(
        y_test, y_pred, target_names=["Retained (0)", "Exited (1)"]
    )

    metrics = {
        "test_accuracy": round(test_acc, 4),
        "test_precision": round(test_prec, 4),
        "test_recall": round(test_rec, 4),
        "test_f1": round(test_f1, 4),
        "test_roc_auc": round(test_roc, 4),
        "confusion_matrix": cm.tolist(),
        "classification_report": cr,
    }

    return metrics


def save_experiment_log(
    benchmark_df: pd.DataFrame,
    final_model_name: str,
    test_metrics: Dict[str, Any],
    best_params: Dict[str, Any],
    config: Dict[str, Any],
) -> Path:
    """
    Append benchmark and final model evaluation results to experiment_log.csv.

    Args:
        benchmark_df: DataFrame of cross-validation results for candidate models.
        final_model_name: Name of the selected winning model.
        test_metrics: Metrics computed on the held-out test split.
        best_params: Parameters for the final model.
        config: Project configuration.

    Returns:
        Path: Location of updated experiment log.
    """
    model_dir = PROJECT_ROOT / config.get("paths", {}).get("model_dir", "models")
    model_dir.mkdir(parents=True, exist_ok=True)
    log_path = model_dir / "experiment_log.csv"

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows: List[Dict[str, Any]] = []

    for _, row in benchmark_df.iterrows():
        is_final = row["model"] == final_model_name
        rows.append(
            {
                "timestamp": timestamp,
                "model": row["model"],
                "cv_accuracy": row["cv_accuracy"],
                "cv_precision": row["cv_precision"],
                "cv_recall": row["cv_recall"],
                "cv_f1": row["cv_f1"],
                "cv_roc_auc": row["cv_roc_auc"],
                "test_accuracy": test_metrics["test_accuracy"] if is_final else "",
                "test_precision": test_metrics["test_precision"] if is_final else "",
                "test_recall": test_metrics["test_recall"] if is_final else "",
                "test_f1": test_metrics["test_f1"] if is_final else "",
                "test_roc_auc": test_metrics["test_roc_auc"] if is_final else "",
                "best_parameters": json.dumps(best_params) if is_final else "",
            }
        )

    new_log_df = pd.DataFrame(rows)

    if log_path.exists():
        existing_log = pd.read_csv(log_path)
        combined_log = pd.concat([existing_log, new_log_df], ignore_index=True)
    else:
        combined_log = new_log_df

    combined_log.to_csv(log_path, index=False)
    logger.info(f"Experiment log updated at: {log_path}")
    return log_path


def save_model_artifacts(
    final_model: Any,
    feature_names: List[str],
    final_model_name: str,
    cv_roc_auc: float,
    test_metrics: Dict[str, Any],
    best_params: Dict[str, Any],
    config: Dict[str, Any],
) -> Dict[str, Path]:
    """
    Serialize final model, feature list, and training metadata to disk.

    Saves to both models/final_model.pkl and models/xgb_model.pkl for seamless
    API backward compatibility.

    Args:
        final_model: Fitted model estimator.
        feature_names: Ordered list of feature names.
        final_model_name: Identifier of selected model.
        cv_roc_auc: Best cross-validation ROC-AUC score.
        test_metrics: Dictionary of test set metrics.
        best_params: Hyperparameters of the final model.
        config: Project configuration.

    Returns:
        Dict[str, Path]: Map of artifact names to file paths.
    """
    model_dir = PROJECT_ROOT / config.get("paths", {}).get("model_dir", "models")
    model_dir.mkdir(parents=True, exist_ok=True)

    paths: Dict[str, Path] = {
        "final_model": model_dir / "final_model.pkl",
        "xgb_model": model_dir / "xgb_model.pkl",
        "feature_names": model_dir / "feature_names.pkl",
        "metadata": model_dir / "model_metadata.json",
    }

    # Save primary model artifact
    joblib.dump(final_model, paths["final_model"])
    # Save xgb_model.pkl alias for backward compatibility
    joblib.dump(final_model, paths["xgb_model"])
    # Save feature names
    joblib.dump(feature_names, paths["feature_names"])

    # Extract serializable hyperparameters
    if hasattr(final_model, "get_params"):
        all_params = final_model.get_params()
        clean_params = {
            k: (
                v
                if isinstance(v, (int, float, str, bool, list, type(None)))
                else str(v)
            )
            for k, v in all_params.items()
        }
    else:
        clean_params = best_params

    # Save metadata
    metadata = {
        "model_name": final_model_name,
        "training_timestamp": datetime.now().isoformat(),
        "cv_roc_auc": cv_roc_auc,
        "test_accuracy": test_metrics["test_accuracy"],
        "test_precision": test_metrics["test_precision"],
        "test_recall": test_metrics["test_recall"],
        "test_f1": test_metrics["test_f1"],
        "test_roc_auc": test_metrics["test_roc_auc"],
        "confusion_matrix": test_metrics["confusion_matrix"],
        "features": feature_names,
        "features_count": len(feature_names),
        "hyperparameters": clean_params,
    }

    with open(paths["metadata"], "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Saved artifacts to {model_dir}:")
    for name, p in paths.items():
        logger.info(f"  - {name}: {p.name}")

    return paths


def run_pipeline() -> Dict[str, Any]:
    """
    Execute the end-to-end model training, tuning, selection, and evaluation pipeline.

    Returns:
        Dict[str, Any]: Execution summary dictionary.
    """
    logger.info("============================================================")
    logger.info("Starting RetainIQ Model Training & Evaluation Pipeline")
    logger.info("============================================================")

    # 1. Configuration & reproducibility
    config = load_config()
    random_state = config.get("data", {}).get("random_state", 42)
    cv_folds = config.get("training", {}).get("cv_folds", 5)
    set_seed(random_state)
    logger.info(f"Loaded config: random_state={random_state}, cv_folds={cv_folds}")

    # 2. Data loading and validation
    df = load_raw_data(config)
    validate_data(df, config)

    # 3. Stratified train/test split
    X_train_raw, X_test_raw, y_train, y_test = split_data(df, config)

    # 4. Feature engineering and encoding
    # Preprocessing strictly operates on each partition (leakage-safe)
    logger.info("Engineering domain features and encoding categoricals...")
    X_train = preprocess(X_train_raw, include_all_features=True)
    X_test = preprocess(X_test_raw, include_all_features=True)

    feature_names = list(X_train.columns)
    logger.info(f"Prepared feature matrices: {len(feature_names)} features.")

    # 5. Calculate class imbalance weight dynamically
    neg_count = int((y_train == 0).sum())
    pos_count = int((y_train == 1).sum())
    scale_pos_weight = round(neg_count / max(pos_count, 1), 2)
    logger.info(
        f"Training set class distribution: 0={neg_count}, 1={pos_count} "
        f"-> scale_pos_weight = {scale_pos_weight:.2f}"
    )

    # 6. Benchmark candidate models
    benchmark_models = build_benchmark_models(
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
    )
    benchmark_df = cross_validate_benchmark(
        models=benchmark_models,
        X_train=X_train,
        y_train=y_train,
        cv_folds=cv_folds,
        random_state=random_state,
    )

    print("\n" + "=" * 70)
    print("5-FOLD STRATIFIED CROSS-VALIDATION BENCHMARK RESULTS")
    print("=" * 70)
    print(benchmark_df.to_string(index=False))
    print("=" * 70 + "\n")

    # 7. Hyperparameter tuning on XGBoost
    logger.info("Starting hyperparameter tuning on XGBoost...")
    tuned_xgb_model, best_params, best_cv_roc_auc = tune_xgboost(
        X_train=X_train,
        y_train=y_train,
        cv_folds=cv_folds,
        random_state=random_state,
        n_iter=25,
    )

    # Evaluate tuned XGBoost across other metrics via 5-fold CV
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    tuned_cv_res = cross_validate(
        estimator=tuned_xgb_model,
        X=X_train,
        y=y_train,
        cv=skf,
        scoring=["accuracy", "precision", "recall", "f1", "roc_auc"],
        n_jobs=-1,
    )

    tuned_row = {
        "model": "XGBoost_Tuned",
        "cv_accuracy": round(float(np.mean(tuned_cv_res["test_accuracy"])), 4),
        "cv_precision": round(float(np.mean(tuned_cv_res["test_precision"])), 4),
        "cv_recall": round(float(np.mean(tuned_cv_res["test_recall"])), 4),
        "cv_f1": round(float(np.mean(tuned_cv_res["test_f1"])), 4),
        "cv_roc_auc": round(float(np.mean(tuned_cv_res["test_roc_auc"])), 4),
    }

    all_models_df = pd.concat(
        [benchmark_df, pd.DataFrame([tuned_row])],
        ignore_index=True,
    )

    print("\n" + "=" * 70)
    print("ALL MODELS COMPARISON (INCLUDING TUNED XGBOOST)")
    print("=" * 70)
    print(all_models_df.to_string(index=False))
    print("=" * 70 + "\n")

    # 8. Model selection based on CV ROC-AUC
    best_idx = all_models_df["cv_roc_auc"].idxmax()
    best_model_name = all_models_df.loc[best_idx, "model"]
    best_cv_score = all_models_df.loc[best_idx, "cv_roc_auc"]

    logger.info(f"Best CV Model: {best_model_name}")
    logger.info(f"Best CV ROC-AUC: {best_cv_score:.4f}")
    logger.info(f"Best Parameters: {best_params}")

    if best_model_name == "XGBoost_Tuned":
        final_estimator = tuned_xgb_model
        final_params = best_params
    else:
        final_estimator = benchmark_models[best_model_name]
        final_params = (
            final_estimator.get_params()
            if hasattr(final_estimator, "get_params")
            else {}
        )

    # 9. Train final model on complete training set
    final_model = train_final_model(
        model=final_estimator,
        X_train=X_train,
        y_train=y_train,
    )

    # 10. Single evaluation on the untouched test set
    test_metrics = evaluate_test_set(
        model=final_model,
        X_test=X_test,
        y_test=y_test,
    )

    print("\n" + "=" * 70)
    print(f"FINAL TEST SET EVALUATION ({best_model_name})")
    print("=" * 70)
    print(f"Test Accuracy : {test_metrics['test_accuracy'] * 100:.2f}%")
    print(f"Test Precision: {test_metrics['test_precision'] * 100:.2f}%")
    print(f"Test Recall   : {test_metrics['test_recall'] * 100:.2f}%")
    print(f"Test F1 Score : {test_metrics['test_f1'] * 100:.2f}%")
    print(f"Test ROC-AUC  : {test_metrics['test_roc_auc'] * 100:.2f}%")
    print("\nClassification Report:")
    print(test_metrics["classification_report"])
    print("Confusion Matrix:")
    print(np.array(test_metrics["confusion_matrix"]))
    print("=" * 70 + "\n")

    # 11. Save experiment log
    save_experiment_log(
        benchmark_df=all_models_df,
        final_model_name=best_model_name,
        test_metrics=test_metrics,
        best_params=final_params,
        config=config,
    )

    # 12. Save model artifacts
    saved_paths = save_model_artifacts(
        final_model=final_model,
        feature_names=feature_names,
        final_model_name=best_model_name,
        cv_roc_auc=best_cv_score,
        test_metrics=test_metrics,
        best_params=final_params,
        config=config,
    )

    logger.info("Pipeline executed successfully!")

    return {
        "best_model_name": best_model_name,
        "best_cv_roc_auc": best_cv_score,
        "test_metrics": test_metrics,
        "best_params": final_params,
        "saved_paths": saved_paths,
        "all_models_df": all_models_df,
    }


if __name__ == "__main__":
    run_pipeline()
