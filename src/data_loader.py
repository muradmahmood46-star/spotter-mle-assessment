"""
Data loading and basic preparation utilities.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent


def load_train_data(path: str | Path | None = None) -> pd.DataFrame:
    """Load historical training and test dataset."""
    if path is None:
        path = BASE_DIR / "train_test.csv"
    else:
        path = Path(path)
        if not path.is_absolute():
            path = BASE_DIR / path
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_validation_data(path: str | Path | None = None) -> pd.DataFrame:
    """Load validation dataset requiring predictions."""
    if path is None:
        path = BASE_DIR / "validation.csv"
    else:
        path = Path(path)
        if not path.is_absolute():
            path = BASE_DIR / path
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_december_inputs(path: str | Path | None = None) -> pd.DataFrame:
    """Load December 2025 chart inputs."""
    if path is None:
        path = BASE_DIR / "december-chart-inputs.csv"
    else:
        path = Path(path)
        if not path.is_absolute():
            path = BASE_DIR / path
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df
