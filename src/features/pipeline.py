"""
Preprocessing pipeline construction — Step 6 of the Master Framework.

Rules:
  - Scalers are fit on TRAINING data only.
  - The same fitted pipeline is applied identically to val and test sets.
  - Pipeline is serialised with joblib for production use.
"""

import logging
from pathlib import Path

import joblib
import numpy as np
import yaml
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

logger = logging.getLogger(__name__)


def _load_config() -> dict:
    cfg_path = Path(__file__).parents[2] / "config.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


CFG = _load_config()
MODELS_DIR = Path(CFG["data"]["models_dir"])


# ---------------------------------------------------------------------------
# Pipeline factory
# ---------------------------------------------------------------------------

def build_preprocessing_pipeline(feature_columns: list[str]) -> Pipeline:
    """
    Build a preprocessing pipeline for the given feature columns.

    Steps:
      1. SimpleImputer (median) — handles any residual NaNs at inference time
      2. StandardScaler         — zero-mean, unit-variance

    All columns are numeric; no categorical encoding needed as time features
    are already cyclically encoded (sin/cos).
    """
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])

    preprocessor = ColumnTransformer(
        transformers=[("num", numeric_transformer, feature_columns)],
        remainder="drop",
    )

    pipeline = Pipeline(steps=[("preprocessor", preprocessor)])
    return pipeline


def fit_pipeline(pipeline: Pipeline, X_train) -> Pipeline:
    """Fit pipeline on training data only. Returns the fitted pipeline."""
    logger.info("Fitting preprocessing pipeline on %d training samples", len(X_train))
    pipeline.fit(X_train)
    return pipeline


def transform(pipeline: Pipeline, X):
    """Apply fitted pipeline to any split (train, val, test)."""
    return pipeline.transform(X)


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

def save_pipeline(pipeline: Pipeline, target: str) -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / f"pipeline_{target}.joblib"
    joblib.dump(pipeline, path)
    logger.info("Pipeline saved → %s", path)
    return path


def load_pipeline(target: str) -> Pipeline:
    path = MODELS_DIR / f"pipeline_{target}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"No pipeline found at {path}. Train the model first.")
    return joblib.load(path)


# ---------------------------------------------------------------------------
# Train / Val / Test split — TimeSeriesSplit-aware
# ---------------------------------------------------------------------------

def time_split(df, val_frac: float = 0.15, test_frac: float = 0.10):
    """
    Chronological split into train / val / test.
    No shuffling — preserves temporal order to prevent leakage.

    Returns
    -------
    (df_train, df_val, df_test)
    """
    n = len(df)
    n_test = int(n * test_frac)
    n_val  = int(n * val_frac)
    n_train = n - n_val - n_test

    df_train = df.iloc[:n_train]
    df_val   = df.iloc[n_train : n_train + n_val]
    df_test  = df.iloc[n_train + n_val:]

    logger.info(
        "Time split → train=%d  val=%d  test=%d  (total=%d)",
        len(df_train), len(df_val), len(df_test), n,
    )
    return df_train, df_val, df_test
