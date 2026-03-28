"""
Step 10 — Evaluation metrics.

Standard regression metrics for time series forecasting:
  MAE   — Mean Absolute Error
  RMSE  — Root Mean Square Error
  MAPE  — Mean Absolute Percentage Error (masked for near-zero values)
  SMAPE — Symmetric MAPE (robust to zero values)
  R2    — Coefficient of determination
"""

import numpy as np
import pandas as pd


def mae(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask])))


def rmse(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    return float(np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2)))


def mape(y_true, y_pred, eps: float = 1.0) -> float:
    """MAPE with epsilon guard to avoid division by near-zero values."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    mask = (~np.isnan(y_true)) & (~np.isnan(y_pred)) & (np.abs(y_true) > eps)
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def smape(y_true, y_pred) -> float:
    """Symmetric MAPE — bounded [0, 200%], robust to zeros."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    denom = (np.abs(y_true[mask]) + np.abs(y_pred[mask])) / 2
    denom = np.where(denom == 0, 1e-9, denom)
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / denom) * 100)


def r2(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    ss_res = np.sum((y_true[mask] - y_pred[mask]) ** 2)
    ss_tot = np.sum((y_true[mask] - np.mean(y_true[mask])) ** 2)
    if ss_tot == 0:
        return float("nan")
    return float(1 - ss_res / ss_tot)


def evaluate_all(y_true, y_pred, model_name: str = "model") -> dict:
    """Return all metrics as a dict."""
    return {
        "model":  model_name,
        "MAE":    round(mae(y_true, y_pred),   3),
        "RMSE":   round(rmse(y_true, y_pred),  3),
        "MAPE":   round(mape(y_true, y_pred),  2),
        "SMAPE":  round(smape(y_true, y_pred), 2),
        "R2":     round(r2(y_true, y_pred),    4),
        "n":      int(np.sum(~np.isnan(np.asarray(y_true, float)))),
    }


def compare_models(results: list[dict]) -> pd.DataFrame:
    """Pretty-print comparison table sorted by RMSE."""
    df = pd.DataFrame(results)
    if "RMSE" in df.columns:
        df = df.sort_values("RMSE").reset_index(drop=True)
    return df
