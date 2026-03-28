"""
Feature engineering for solar price & dispatch forecasting.

Builds a unified feature matrix from AEMO + weather DataFrames:
  - Cyclical time encoding  (hour, day-of-week, month)
  - Lag features            (configurable windows)
  - Rolling statistics      (mean, std, min, max)
  - Weather features        (solar irradiance, temp, cloud)
  - Business calendar flags (weekend, peak hours)
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


def _load_config() -> dict:
    cfg_path = Path(__file__).parents[2] / "config.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


CFG = _load_config()
FEAT_CFG = CFG["features"]
LAGS = FEAT_CFG["lags_hours"]
ROLLING = FEAT_CFG["rolling_windows"]
TARGETS = FEAT_CFG["targets"]


# ---------------------------------------------------------------------------
# Time encoding
# ---------------------------------------------------------------------------

def add_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode hour, day-of-week, and month as (sin, cos) pairs."""
    df = df.copy()
    idx = df.index if hasattr(df.index, "hour") else pd.DatetimeIndex(df["datetime"])

    df["hour_sin"] = np.sin(2 * np.pi * idx.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * idx.hour / 24)
    df["dow_sin"]  = np.sin(2 * np.pi * idx.dayofweek / 7)
    df["dow_cos"]  = np.cos(2 * np.pi * idx.dayofweek / 7)
    df["month_sin"] = np.sin(2 * np.pi * idx.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * idx.month / 12)

    # Raw values for filtering / display
    df["hour"]       = idx.hour
    df["day_of_week"] = idx.dayofweek          # 0=Mon … 6=Sun
    df["month"]      = idx.month

    return df


def add_calendar_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Weekend flag and Australian peak-hour windows."""
    df = df.copy()
    idx = df.index if hasattr(df.index, "hour") else pd.DatetimeIndex(df["datetime"])

    df["is_weekend"]   = (idx.dayofweek >= 5).astype(int)
    df["is_peak_am"]   = ((idx.hour >= 7)  & (idx.hour < 10)).astype(int)
    df["is_peak_pm"]   = ((idx.hour >= 17) & (idx.hour < 21)).astype(int)
    df["is_solar_hrs"] = ((idx.hour >= 6)  & (idx.hour <= 18)).astype(int)

    return df


# ---------------------------------------------------------------------------
# Lag & rolling features
# ---------------------------------------------------------------------------

def add_lag_features(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Add lag features at configured windows (in hours → rows at 30-min = 2×hours)."""
    df = df.copy()
    # Detect frequency; assume 30-min if not determinable
    try:
        freq_minutes = int(pd.infer_freq(df.index).replace("min", "").replace("T", ""))
    except Exception:
        freq_minutes = 30

    steps_per_hour = max(1, 60 // freq_minutes)

    for col in columns:
        if col not in df.columns:
            continue
        for lag_h in LAGS:
            lag_steps = lag_h * steps_per_hour
            df[f"{col}_lag_{lag_h}h"] = df[col].shift(lag_steps)

    return df


def add_rolling_features(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Add rolling mean, std, min, max for each window."""
    df = df.copy()
    try:
        freq_minutes = int(pd.infer_freq(df.index).replace("min", "").replace("T", ""))
    except Exception:
        freq_minutes = 30

    steps_per_hour = max(1, 60 // freq_minutes)

    for col in columns:
        if col not in df.columns:
            continue
        for win_h in ROLLING:
            win_steps = win_h * steps_per_hour
            roll = df[col].rolling(window=win_steps, min_periods=win_steps // 2)
            df[f"{col}_roll_mean_{win_h}h"] = roll.mean()
            df[f"{col}_roll_std_{win_h}h"]  = roll.std()
            df[f"{col}_roll_min_{win_h}h"]  = roll.min()
            df[f"{col}_roll_max_{win_h}h"]  = roll.max()

    return df


# ---------------------------------------------------------------------------
# Weather feature alignment
# ---------------------------------------------------------------------------

def merge_weather(aemo_df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
    """
    Left-join weather features onto the AEMO DataFrame.
    Weather is hourly; AEMO may be 30-min — weather is forward-filled.
    """
    weather_cols = [c for c in weather_df.columns if c != "region"]
    weather_clean = weather_df[weather_cols].copy()

    # Resample weather to match AEMO frequency
    aemo_freq = pd.infer_freq(aemo_df.index)
    if aemo_freq and aemo_freq != weather_clean.index.freq:
        weather_clean = weather_clean.resample(aemo_freq).ffill()

    merged = aemo_df.join(weather_clean, how="left")
    # Forward-fill any gaps from the join
    for col in weather_cols:
        if col in merged.columns:
            merged[col] = merged[col].ffill().bfill()

    return merged


# ---------------------------------------------------------------------------
# Master pipeline
# ---------------------------------------------------------------------------

def build_feature_matrix(
    aemo_df: pd.DataFrame,
    weather_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Full feature engineering pipeline.

    Parameters
    ----------
    aemo_df    : DataFrame from aemo_client (price + solar dispatch)
    weather_df : DataFrame from weather_client (optional but recommended)

    Returns
    -------
    Feature-rich DataFrame ready for train/val/test splitting.
    NaN rows from lagging are NOT dropped here — caller decides.
    """
    df = aemo_df.copy()

    # Merge weather first (before lags, so weather lags are also possible)
    if weather_df is not None:
        df = merge_weather(df, weather_df)

    df = add_cyclical_features(df)
    df = add_calendar_flags(df)
    df = add_lag_features(df, columns=TARGETS)
    df = add_rolling_features(df, columns=TARGETS)

    logger.info(
        "Feature matrix: %d rows × %d columns (before NaN drop)",
        len(df), df.shape[1],
    )
    return df


def get_feature_columns(df: pd.DataFrame, exclude: list[str] | None = None) -> list[str]:
    """Return all feature column names (excludes targets and region label)."""
    exclude = set(exclude or [])
    exclude.update(TARGETS)
    exclude.update(["region"])
    return [c for c in df.columns if c not in exclude]
