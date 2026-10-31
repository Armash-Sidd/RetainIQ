"""
RetainIQ Production Inference Monitoring & Audit Trail Module
Phase 8 — Runtime Telemetry, Audit Logging & Data Drift Detection

Tracks incoming scoring requests, logs predictions with calibrated risk tiers,
and monitors feature distributions against the 15,000-record training baseline
to alert on concept and covariate drift.
"""

import json
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.utils import PROJECT_ROOT, get_logger

logger = get_logger("RetainIQ.Monitoring")

LOGS_DIR = PROJECT_ROOT / "logs"
AUDIT_LOG_FILE = LOGS_DIR / "prediction_audit.jsonl"

_log_lock = threading.Lock()

# Verified training baseline statistics (from 15,000-record training dataset)
TRAINING_BASELINE_STATS: Dict[str, Dict[str, float]] = {
    "CreditScore": {"mean": 658.85, "std": 72.59, "min": 431.0, "max": 850.0},
    "Age": {"mean": 37.89, "std": 8.26, "min": 18.0, "max": 74.0},
    "Tenure": {"mean": 5.01, "std": 2.81, "min": 0.0, "max": 10.0},
    "Balance": {"mean": 76485.89, "std": 62397.41, "min": 0.0, "max": 250000.0},
    "NumOfProducts": {"mean": 1.59, "std": 0.54, "min": 1.0, "max": 4.0},
    "EstimatedSalary": {"mean": 118134.83, "std": 46803.11, "min": 11.58, "max": 200000.0},
}


def log_prediction(
    customer_input: Dict[str, Any],
    exit_probability: float,
    prediction: int,
    risk_tier: str,
    latency_ms: Optional[float] = None,
    log_file: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Persist an inference transaction to the production audit log.

    Args:
        customer_input: Raw customer features provided for scoring.
        exit_probability: Model-estimated exit probability score (0.0 to 1.0).
        prediction: Binary classification indicator (0 or 1).
        risk_tier: Categorical risk level ('Low', 'Moderate', or 'High').
        latency_ms: Optional model inference latency in milliseconds.
        log_file: Destination JSONL path (defaults to logs/prediction_audit.jsonl).

    Returns:
        Dict[str, Any]: Structured record that was appended to the log.
    """
    target_file = log_file or AUDIT_LOG_FILE
    target_file.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "timestamp": datetime.now().isoformat(),
        "prediction": int(prediction),
        "status": "Likely to Exit" if prediction == 1 else "Likely to Stay",
        "exit_probability": round(float(exit_probability), 4),
        "risk_tier": risk_tier,
        "latency_ms": round(float(latency_ms), 2) if latency_ms is not None else None,
        "customer_signals": {
            "CreditScore": customer_input.get("CreditScore"),
            "Geography": customer_input.get("Geography"),
            "Gender": customer_input.get("Gender"),
            "Age": customer_input.get("Age"),
            "Tenure": customer_input.get("Tenure"),
            "Balance": customer_input.get("Balance"),
            "NumOfProducts": customer_input.get("NumOfProducts"),
            "HasCrCard": customer_input.get("HasCrCard"),
            "IsActiveMember": customer_input.get("IsActiveMember"),
            "EstimatedSalary": customer_input.get("EstimatedSalary"),
        },
    }

    try:
        with _log_lock:
            with open(target_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
    except Exception as e:
        logger.warning(f"Could not persist prediction audit log: {e}")

    return record


def log_batch_predictions(
    records: List[Dict[str, Any]],
    log_file: Optional[Path] = None,
) -> int:
    """
    Persist multiple inference transactions to the audit log in a single operation.

    Args:
        records: List of structured inference records.
        log_file: Destination JSONL path.

    Returns:
        int: Number of records successfully written.
    """
    target_file = log_file or AUDIT_LOG_FILE
    target_file.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    try:
        with _log_lock:
            with open(target_file, "a", encoding="utf-8") as f:
                for rec in records:
                    f.write(json.dumps(rec) + "\n")
                    written += 1
    except Exception as e:
        logger.warning(f"Batch audit logging error: {e}")

    return written


def get_recent_audit_logs(
    limit: int = 50,
    log_file: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """
    Retrieve the most recent inference entries from the audit log.

    Args:
        limit: Maximum number of recent records to return (default: 50).
        log_file: Path to audit log file.

    Returns:
        List[Dict[str, Any]]: Chronologically descending list of prediction logs.
    """
    target_file = log_file or AUDIT_LOG_FILE
    if not target_file.exists():
        return []

    entries: List[Dict[str, Any]] = []
    try:
        with _log_lock:
            with open(target_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        entries.append(json.loads(line))
    except Exception as e:
        logger.warning(f"Failed to read audit log: {e}")
        return []

    # Return latest records first
    return entries[-limit:][::-1]


def detect_data_drift(
    batch_df: pd.DataFrame,
    baseline_stats: Optional[Dict[str, Dict[str, float]]] = None,
    z_threshold: float = 3.0,
) -> Dict[str, Any]:
    """
    Assess incoming customer batch for covariate distribution drift against baseline.

    Calculates standardized Z-score deviations between current batch means and baseline means.
    Flags features where |Z-score deviation| exceeds the configured threshold.

    Args:
        batch_df: DataFrame containing incoming customer records.
        baseline_stats: Dictionary of baseline mean and std for numerical features.
        z_threshold: Number of standard deviations indicating significant statistical shift (default: 3.0).

    Returns:
        Dict[str, Any]: Comprehensive drift report including status and per-feature metrics.
    """
    stats_ref = baseline_stats or TRAINING_BASELINE_STATS
    report: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "sample_size": len(batch_df),
        "drift_detected": False,
        "drifted_features": [],
        "feature_metrics": {},
    }

    if batch_df.empty:
        return report

    for feature, base in stats_ref.items():
        if feature in batch_df.columns:
            series = pd.to_numeric(batch_df[feature], errors="coerce").dropna()
            if len(series) == 0:
                continue

            current_mean = float(series.mean())
            current_std = float(series.std()) if len(series) > 1 else 0.0
            base_mean = base["mean"]
            base_std = base["std"]

            # Standard error of the mean
            sem = base_std / np.sqrt(len(series)) if len(series) > 0 else 1.0
            z_score = abs(current_mean - base_mean) / (sem if sem > 0 else 1.0)

            is_drift = bool(z_score > z_threshold)

            report["feature_metrics"][feature] = {
                "current_mean": round(current_mean, 2),
                "baseline_mean": round(base_mean, 2),
                "current_std": round(current_std, 2),
                "baseline_std": round(base_std, 2),
                "z_deviation": round(float(z_score), 2),
                "drift_flag": is_drift,
            }

            if is_drift:
                report["drifted_features"].append(feature)

    report["drift_detected"] = len(report["drifted_features"]) > 0
    return report


def get_monitoring_summary(log_file: Optional[Path] = None) -> Dict[str, Any]:
    """
    Compile runtime inference telemetry metrics from the audit log.

    Args:
        log_file: Path to audit log file.

    Returns:
        Dict[str, Any]: High-level telemetry summary.
    """
    logs = get_recent_audit_logs(limit=1000, log_file=log_file)
    if not logs:
        return {
            "total_logged_predictions": 0,
            "mean_exit_probability": 0.0,
            "risk_tier_counts": {"Low": 0, "Moderate": 0, "High": 0},
            "status_counts": {"Likely to Exit": 0, "Likely to Stay": 0},
        }

    probs = [l["exit_probability"] for l in logs if "exit_probability" in l]
    tiers = [l["risk_tier"] for l in logs if "risk_tier" in l]
    preds = [l["prediction"] for l in logs if "prediction" in l]

    return {
        "total_logged_predictions": len(logs),
        "mean_exit_probability": round(float(np.mean(probs)), 4) if probs else 0.0,
        "risk_tier_counts": {
            "Low": tiers.count("Low"),
            "Moderate": tiers.count("Moderate"),
            "High": tiers.count("High"),
        },
        "status_counts": {
            "Likely to Exit": preds.count(1),
            "Likely to Stay": preds.count(0),
        },
        "latest_inference_time": logs[0]["timestamp"] if logs else None,
    }

