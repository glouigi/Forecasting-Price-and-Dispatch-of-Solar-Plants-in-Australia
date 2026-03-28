"""
Step 7 — Baseline Modelling.

Two naive baselines that MUST be beaten before any ML model is accepted:
  1. Persistence      : ŷ(t) = y(t - 24h)   — "same hour yesterday"
  2. Seasonal Naive   : ŷ(t) = y(t - 168h)  — "same hour last week"

These establish the minimum performance bar.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class PersistenceModel:
    """Predict = value 24 hours ago (2 half-hours × 48 steps at 30-min)."""

    def __init__(self, lag_hours: int = 24, freq_minutes: int = 30):
        self.lag_hours = lag_hours
        self.steps = max(1, (lag_hours * 60) // freq_minutes)
        self.name = f"Persistence ({lag_hours}h)"

    def predict(self, series: pd.Series) -> pd.Series:
        """
        Parameters
        ----------
        series : full historical series (indexed by datetime)

        Returns aligned predictions (NaN for the first `steps` rows).
        """
        return series.shift(self.steps).rename(f"{series.name}_persistence")


class SeasonalNaiveModel:
    """Predict = value 168 hours ago (same hour last week)."""

    def __init__(self, lag_hours: int = 168, freq_minutes: int = 30):
        self.lag_hours = lag_hours
        self.steps = max(1, (lag_hours * 60) // freq_minutes)
        self.name = f"Seasonal Naive ({lag_hours}h)"

    def predict(self, series: pd.Series) -> pd.Series:
        return series.shift(self.steps).rename(f"{series.name}_seasonal_naive")


def evaluate_baselines(
    df: pd.DataFrame,
    target: str,
    test_start_idx: int,
    freq_minutes: int = 30,
) -> pd.DataFrame:
    """
    Run both baselines on the test portion and return predictions DataFrame.

    Parameters
    ----------
    df           : full feature DataFrame (train + val + test)
    target       : column name to forecast
    test_start_idx : integer iloc index where the test set begins
    """
    series = df[target]

    persistence = PersistenceModel(lag_hours=24, freq_minutes=freq_minutes)
    seasonal    = SeasonalNaiveModel(lag_hours=168, freq_minutes=freq_minutes)

    pred_p = persistence.predict(series).iloc[test_start_idx:]
    pred_s = seasonal.predict(series).iloc[test_start_idx:]
    actual = series.iloc[test_start_idx:]

    results = pd.DataFrame({
        "actual": actual,
        "persistence_pred": pred_p,
        "seasonal_naive_pred": pred_s,
    })

    logger.info(
        "Baselines evaluated on %d test samples for target '%s'",
        len(results.dropna()), target,
    )
    return results


def baseline_summary(baseline_results: pd.DataFrame) -> pd.DataFrame:
    """Compute MAE and RMSE for each baseline column."""
    from src.evaluation.metrics import mae, rmse  # avoid circular at module level

    actual = baseline_results["actual"].dropna()
    rows = []
    for col in ["persistence_pred", "seasonal_naive_pred"]:
        if col not in baseline_results.columns:
            continue
        pred = baseline_results[col].reindex(actual.index).dropna()
        common = actual.reindex(pred.index).dropna()
        pred = pred.reindex(common.index)
        rows.append({
            "model": col.replace("_pred", "").replace("_", " ").title(),
            "MAE":  round(mae(common, pred), 3),
            "RMSE": round(rmse(common, pred), 3),
            "n":    len(common),
        })
    return pd.DataFrame(rows)
