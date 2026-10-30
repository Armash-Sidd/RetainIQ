"""
RetainIQ Utilities Module
Provides centralized configuration loading, reproducibility seed management,
and unified logging helpers.
"""

import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import yaml

# Resolve project root relative to this utility module (src/utils.py -> project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config(config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """
    Load project configuration from YAML file.

    Args:
        config_path: Optional path to YAML config file. If None, defaults to
                     <PROJECT_ROOT>/config/config.yaml.

    Returns:
        Dict[str, Any]: Configuration dictionary.

    Raises:
        FileNotFoundError: If the configuration file cannot be found.
        yaml.YAMLError: If parsing fails.
    """
    if config_path is None:
        target_path = PROJECT_ROOT / "config" / "config.yaml"
    else:
        target_path = Path(config_path)
        if not target_path.is_absolute():
            candidate = PROJECT_ROOT / target_path
            if candidate.exists():
                target_path = candidate
            else:
                target_path = target_path.resolve()

    if not target_path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {target_path}")

    with open(target_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config or {}


def set_seed(seed: int = 42) -> None:
    """
    Set random seed across Python standard random and NumPy for reproducibility.

    Args:
        seed: Integer random seed value (default: 42).
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Retrieve a configured logger with consistent formatting and duplicate handler prevention.

    Args:
        name: Name of the logger (defaults to root/calling module).

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name or "RetainIQ")

    # Set default level to INFO if not already configured
    if logger.level == logging.NOTSET:
        logger.setLevel(logging.INFO)

    # Avoid adding multiple StreamHandlers if get_logger is called repeatedly
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger
