"""
Step 10 — Diagnostic evaluation, error analysis & interpretability.

Provides:
  - SHAP global + local feature importance
  - Residual analysis (distribution, autocorrelation, heteroskedasticity)
  - Error slicing by hour-of-day and day-of-week
  - Learning curve from MLflow history
  - Calibration / prediction interval coverage
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SHAP analysis
# ---------------------------------------------------------------------------

def compute_shap_values(model, X: np.ndarray, feature_names: list[str]):
    """
    Compute SHAP values using TreeExplainer.

    Returns (shap_values array, expected_value)
    """
    try:
        import shap
    except ImportError:
        logger.warning("shap not installed — skipping SHAP analysis")
        return None, None

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    return shap_values, explainer.expected_value


def shap_summary_df(shap_values: np.ndarray, feature_names: list[str]) -> pd.DataFrame:
    """Return DataFrame of mean(|SHAP|) per feature, sorted descending."""
    mean_abs = np.abs(shap_values).mean(axis=0)
    return (
        pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Residual analysis
# ---------------------------------------------------------------------------

def residuals(y_true, y_pred) -> pd.Series:
    return pd.Series(np.asarray(y_true) - np.asarray(y_pred), name="residual")


def residual_stats(res: pd.Series) -> dict:
    clean = res.dropna()
    return {
        "mean":     round(clean.mean(), 4),
        "std":      round(clean.std(),  4),
        "min":      round(clean.min(),  4),
        "max":      round(clean.max(),  4),
        "skewness": round(clean.skew(), 4),
        "kurtosis": round(clean.kurtosis(), 4),
    }


def error_by_hour(y_true: pd.Series, y_pred: np.ndarray) -> pd.DataFrame:
    """
    Slice MAE and RMSE by hour-of-day.

    y_true must have a DatetimeIndex.
    """
    df = pd.DataFrame({
        "actual": y_true.values,
        "pred":   y_pred,
        "hour":   y_true.index.hour,
    })
    def _metrics(g):
        return pd.Series({
            "MAE":  np.mean(np.abs(g["actual"] - g["pred"])),
            "RMSE": np.sqrt(np.mean((g["actual"] - g["pred"]) ** 2)),
            "n":    len(g),
        })
    return df.groupby("hour").apply(_metrics).reset_index()


def error_by_dow(y_true: pd.Series, y_pred: np.ndarray) -> pd.DataFrame:
    """Slice MAE and RMSE by day-of-week (0=Mon … 6=Sun)."""
    dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    df = pd.DataFrame({
        "actual": y_true.values,
        "pred":   y_pred,
        "dow":    y_true.index.dayofweek,
    })
    def _metrics(g):
        return pd.Series({
            "MAE":  np.mean(np.abs(g["actual"] - g["pred"])),
            "RMSE": np.sqrt(np.mean((g["actual"] - g["pred"]) ** 2)),
            "n":    len(g),
        })
    result = df.groupby("dow").apply(_metrics).reset_index()
    result["day"] = result["dow"].map(lambda x: dow_names[x])
    return result


# ---------------------------------------------------------------------------
# Prediction interval coverage
# ---------------------------------------------------------------------------

def coverage_score(y_true, lower, upper) -> float:
    """Fraction of actuals that fall within [lower, upper]."""
    y = np.asarray(y_true)
    l, u = np.asarray(lower), np.asarray(upper)
    mask = ~(np.isnan(y) | np.isnan(l) | np.isnan(u))
    return float(np.mean((y[mask] >= l[mask]) & (y[mask] <= u[mask])))


def naive_prediction_intervals(
    y_pred: np.ndarray,
    residuals_train: np.ndarray,
    alpha: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build (1-alpha) prediction intervals from training residual quantiles.

    Returns (lower, upper) arrays.
    """
    q_low  = np.nanquantile(residuals_train, alpha / 2)
    q_high = np.nanquantile(residuals_train, 1 - alpha / 2)
    return y_pred + q_low, y_pred + q_high
