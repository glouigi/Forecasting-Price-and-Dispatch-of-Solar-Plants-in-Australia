"""
Solar Price & Dispatch Forecasting — Streamlit Web Application
Steps 11 & 12 of the Master Framework (Production + Deployment)

Features:
  - Live AEMO price and solar dispatch dashboard
  - Day-of-week forecast selector (pick any weekday → 24h forecast)
  - Model training with Optuna HPO directly from the UI
  - Model performance diagnostics (SHAP, residuals, error slices)
  - Weather conditions overlay
"""

import sys
import logging
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yaml

# Make src importable when app is run from project root
sys.path.insert(0, str(Path(__file__).parents[1]))

from src.data.aemo_client    import fetch_region_data, fetch_live_data
from src.data.weather_client import fetch_forecast_weather, fetch_weather_for_training
from src.features.engineering import build_feature_matrix, get_feature_columns
from src.features.pipeline    import (
    build_preprocessing_pipeline, fit_pipeline, transform,
    save_pipeline, load_pipeline, time_split,
)
from src.models.baseline  import evaluate_baselines, baseline_summary
from src.models.lgbm_model import (
    train, save_model, load_model, model_exists, predict, feature_importance,
)
from src.evaluation.metrics     import evaluate_all, compare_models
from src.evaluation.diagnostics import (
    compute_shap_values, shap_summary_df,
    error_by_hour, error_by_dow, residuals,
    naive_prediction_intervals,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Solar Forecast Australia",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded",
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _load_config() -> dict:
    cfg_path = Path(__file__).parents[1] / "config.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


CFG = _load_config()
REGIONS = list(CFG["aemo"]["regions"].keys())
TARGETS = CFG["features"]["targets"]
DOW_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# ---------------------------------------------------------------------------
# Cached data loaders
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300, show_spinner="Fetching live AEMO data…")
def _load_live(region: str) -> pd.DataFrame:
    try:
        return fetch_live_data(region)
    except Exception as exc:
        st.warning(f"Live data unavailable: {exc}")
        return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner="Fetching training data from AEMO + Open-Meteo…")
def _load_training_data(region: str) -> pd.DataFrame:
    aemo_df    = fetch_region_data(region, period=f"{CFG['aemo']['train_period_days']}d")
    weather_df = fetch_weather_for_training(region, CFG["aemo"]["train_period_days"])
    feature_df = build_feature_matrix(aemo_df, weather_df)
    return feature_df.dropna()


@st.cache_data(ttl=3600, show_spinner="Fetching weather forecast…")
def _load_weather_forecast(region: str) -> pd.DataFrame:
    try:
        return fetch_forecast_weather(region, forecast_days=7)
    except Exception as exc:
        st.warning(f"Weather forecast unavailable: {exc}")
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Training workflow
# ---------------------------------------------------------------------------

def _train_models(region: str):
    with st.status("Training models — this may take a few minutes…", expanded=True) as status:
        st.write("📥 Fetching AEMO + weather data…")
        feature_df = _load_training_data(region)

        st.write(f"📊 Dataset: {len(feature_df):,} rows × {feature_df.shape[1]} columns")
        df_train, df_val, df_test = time_split(feature_df)

        feat_cols = get_feature_columns(feature_df)

        for target in TARGETS:
            st.write(f"🏋️ Training model for **{target}**…")

            pipeline = build_preprocessing_pipeline(feat_cols)
            fit_pipeline(pipeline, df_train[feat_cols])
            save_pipeline(pipeline, target)

            X_train = transform(pipeline, df_train[feat_cols])
            X_val   = transform(pipeline, df_val[feat_cols])
            X_test  = transform(pipeline, df_test[feat_cols])
            y_train = df_train[target].values
            y_val   = df_val[target].values

            model = train(X_train, y_train, feat_cols, target, X_val, y_val)
            save_model(model, target)

            # Evaluate on locked test set (done ONCE here — Step 10)
            y_test  = df_test[target].values
            y_hat   = predict(model, X_test)
            metrics = evaluate_all(y_test, y_hat, model_name="LightGBM")
            st.write(f"✅ {target} — MAE={metrics['MAE']:.2f}  RMSE={metrics['RMSE']:.2f}  R²={metrics['R2']:.3f}")

        status.update(label="Training complete!", state="complete")

    st.cache_data.clear()
    st.rerun()


# ---------------------------------------------------------------------------
# Forecast generation for a target day-of-week
# ---------------------------------------------------------------------------

def _generate_dow_forecast(region: str, target_dow: int) -> pd.DataFrame | None:
    """
    Generate 24-hour forecast for the next occurrence of target_dow.

    target_dow: 0=Monday … 6=Sunday
    """
    for tgt in TARGETS:
        if not model_exists(tgt):
            return None

    # Build future feature frame
    weather_fc = _load_weather_forecast(region)
    if weather_fc.empty:
        st.error("Cannot generate forecast without weather data.")
        return None

    # Find next occurrence of target day-of-week in the forecast window
    fc_days = weather_fc.index.normalize().unique()
    target_dates = [d for d in fc_days if d.dayofweek == target_dow]
    if not target_dates:
        st.warning(f"No {DOW_NAMES[target_dow]} found in the 7-day forecast window.")
        return None
    target_date = target_dates[0]

    # Filter weather to that day (hourly)
    day_weather = weather_fc[
        (weather_fc.index >= target_date) &
        (weather_fc.index < target_date + timedelta(days=1))
    ].copy()

    # Build a minimal feature frame for that day
    day_weather = day_weather.resample("30min").ffill()
    day_feat = build_feature_matrix(
        pd.DataFrame(index=day_weather.index, data={"region": region}),
        day_weather,
    )

    # For lag/rolling features we need historical context — load recent actuals
    try:
        recent = fetch_region_data(region, period="14d")
        combined = pd.concat([recent, day_weather.rename(columns=lambda c: c)], axis=0)
    except Exception:
        combined = day_feat

    # Re-build features with historical context for lags
    try:
        hist_weather = fetch_weather_for_training(region, train_period_days=14)
        full_feat = build_feature_matrix(
            fetch_region_data(region, period="14d"),
            hist_weather,
        )
        # Append the future day rows
        future_idx = day_weather.index
        # Fill targets with NaN for future rows
        future_rows = day_feat.reindex(future_idx)
        for t in TARGETS:
            if t not in future_rows.columns:
                future_rows[t] = np.nan
        all_feat = pd.concat([full_feat, future_rows]).sort_index()
        # Re-compute rolling/lag features over the full range
        from src.features.engineering import add_lag_features, add_rolling_features
        all_feat = add_lag_features(all_feat, columns=TARGETS)
        all_feat = add_rolling_features(all_feat, columns=TARGETS)
        day_rows = all_feat.loc[future_idx]
    except Exception:
        day_rows = day_feat

    results = {"datetime": [], "hour": []}
    for tgt in TARGETS:
        results[f"{tgt}_pred"] = []
        results[f"{tgt}_lower"] = []
        results[f"{tgt}_upper"] = []

    feat_cols = get_feature_columns(day_rows, exclude=TARGETS)

    for tgt in TARGETS:
        try:
            pipeline = load_pipeline(tgt)
            model    = load_model(tgt)
        except FileNotFoundError:
            continue

        # Get training residuals for prediction intervals
        try:
            train_data = _load_training_data(region)
            df_tr, _, _ = time_split(train_data)
            feat_cols_avail = [c for c in feat_cols if c in df_tr.columns]
            X_tr = transform(pipeline, df_tr[feat_cols_avail])
            y_tr = df_tr[tgt].values
            tr_preds = predict(model, X_tr)
            tr_res   = y_tr - tr_preds
        except Exception:
            tr_res = np.array([0.0])

        try:
            feat_cols_day = [c for c in feat_cols if c in day_rows.columns]
            X_day = transform(pipeline, day_rows[feat_cols_day].fillna(0))
            preds = predict(model, X_day)
            preds = np.maximum(preds, 0)
            lower, upper = naive_prediction_intervals(preds, tr_res, alpha=0.1)
        except Exception as e:
            logger.warning("Forecast failed for %s: %s", tgt, e)
            preds = np.zeros(len(day_rows))
            lower, upper = preds.copy(), preds.copy()

        results[f"{tgt}_pred"]  = preds.tolist()
        results[f"{tgt}_lower"] = np.maximum(lower, 0).tolist()
        results[f"{tgt}_upper"] = np.maximum(upper, 0).tolist()

    results["datetime"] = day_rows.index.tolist()
    results["hour"]     = [t.hour for t in day_rows.index]

    return pd.DataFrame(results).set_index("datetime")


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _live_chart(df: pd.DataFrame, col: str, title: str, unit: str) -> go.Figure:
    fig = px.line(
        df.reset_index(), x="datetime", y=col,
        title=title,
        labels={col: unit, "datetime": ""},
        template="plotly_dark",
        color_discrete_sequence=["#FFA500"],
    )
    fig.update_layout(height=300, margin=dict(t=40, b=20))
    return fig


def _forecast_chart(fc: pd.DataFrame, target: str, title: str, unit: str) -> go.Figure:
    pred_col  = f"{target}_pred"
    lower_col = f"{target}_lower"
    upper_col = f"{target}_upper"

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fc.index, y=fc[upper_col],
        mode="lines", line=dict(width=0),
        name="90% upper", showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=fc.index, y=fc[lower_col],
        mode="lines", line=dict(width=0),
        fill="tonexty", fillcolor="rgba(255,165,0,0.15)",
        name="90% interval",
    ))
    fig.add_trace(go.Scatter(
        x=fc.index, y=fc[pred_col],
        mode="lines+markers",
        line=dict(color="#FFA500", width=2),
        marker=dict(size=4),
        name="Forecast",
    ))
    fig.update_layout(
        title=title,
        yaxis_title=unit,
        xaxis_title="Hour",
        template="plotly_dark",
        height=380,
        margin=dict(t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.image("https://flagcdn.com/w80/au.png", width=50)
    st.title("Solar Forecast AU")
    st.caption("AEMO · NEM · Real-Time")

    st.divider()
    region = st.selectbox("NEM Region", REGIONS, index=0)

    st.divider()
    st.subheader("Day-of-Week Forecast")
    selected_dow_name = st.selectbox("Forecast day", DOW_NAMES, index=0)
    selected_dow = DOW_NAMES.index(selected_dow_name)

    st.divider()
    models_ready = all(model_exists(t) for t in TARGETS)
    if models_ready:
        st.success("Models trained ✓")
        if st.button("Retrain models", use_container_width=True):
            _train_models(region)
    else:
        st.warning("No trained models found.")
        if st.button("Fetch data & train models", type="primary", use_container_width=True):
            _train_models(region)

    st.divider()
    st.caption("Data: OpenNEM · Open-Meteo\nModels: LightGBM + Optuna")


# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------

st.title(f"☀️ Solar Price & Dispatch Forecast — {region}")
st.caption(f"Region: {region} | NEM Australia | Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M AEST')}")

tab_live, tab_forecast, tab_performance, tab_data = st.tabs([
    "⚡ Live Dashboard",
    "📅 Day-Ahead Forecast",
    "📊 Model Performance",
    "🔍 Data Explorer",
])

# ── Tab 1: Live Dashboard ──────────────────────────────────────────────────
with tab_live:
    st.subheader(f"Live AEMO Data — {region}")
    live_df = _load_live(region)

    if live_df.empty:
        st.info("Live data unavailable. Check API connection or use cached data.")
    else:
        # KPI cards
        if "electricity_price_aud_mwh" in live_df.columns:
            latest_price   = live_df["electricity_price_aud_mwh"].dropna().iloc[-1]
            avg_price_24h  = live_df["electricity_price_aud_mwh"].dropna().mean()
        else:
            latest_price = avg_price_24h = None

        if "solar_dispatch_mw" in live_df.columns:
            latest_solar   = live_df["solar_dispatch_mw"].dropna().iloc[-1]
            avg_solar_24h  = live_df["solar_dispatch_mw"].dropna().mean()
        else:
            latest_solar = avg_solar_24h = None

        col1, col2, col3, col4 = st.columns(4)
        if latest_price is not None:
            col1.metric("Current Price", f"${latest_price:.2f}/MWh",
                        delta=f"{latest_price - avg_price_24h:.2f} vs 24h avg")
        if latest_solar is not None:
            col2.metric("Solar Dispatch", f"{latest_solar:.0f} MW",
                        delta=f"{latest_solar - avg_solar_24h:.0f} MW vs 24h avg")
        if "demand_mw" in live_df.columns:
            latest_demand = live_df["demand_mw"].dropna().iloc[-1]
            col3.metric("Demand", f"{latest_demand:.0f} MW")
        col4.metric("Data Points", f"{len(live_df):,}")

        st.divider()

        # Live charts
        if "electricity_price_aud_mwh" in live_df.columns:
            st.plotly_chart(
                _live_chart(live_df, "electricity_price_aud_mwh",
                            "Electricity Price (Last 24h)", "AUD/MWh"),
                use_container_width=True,
            )
        if "solar_dispatch_mw" in live_df.columns:
            st.plotly_chart(
                _live_chart(live_df, "solar_dispatch_mw",
                            "Solar Dispatch (Last 24h)", "MW"),
                use_container_width=True,
            )

# ── Tab 2: Day-Ahead Forecast ─────────────────────────────────────────────
with tab_forecast:
    st.subheader(f"24-Hour Forecast — {selected_dow_name}")

    if not models_ready:
        st.info("Train the models first using the sidebar button.")
    else:
        with st.spinner(f"Generating forecast for {selected_dow_name}…"):
            fc_df = _generate_dow_forecast(region, selected_dow)

        if fc_df is not None and not fc_df.empty:
            st.caption(
                f"Forecast date: {fc_df.index[0].strftime('%A %d %b %Y')}  |  "
                f"Horizon: 24h  |  90% prediction intervals shown"
            )

            col_a, col_b = st.columns(2)
            with col_a:
                if "electricity_price_aud_mwh_pred" in fc_df.columns:
                    st.plotly_chart(
                        _forecast_chart(
                            fc_df, "electricity_price_aud_mwh",
                            f"Price Forecast — {selected_dow_name}", "AUD/MWh",
                        ),
                        use_container_width=True,
                    )
                    avg_fc_price = fc_df["electricity_price_aud_mwh_pred"].mean()
                    peak_fc_price = fc_df["electricity_price_aud_mwh_pred"].max()
                    st.metric("Avg Forecast Price", f"${avg_fc_price:.2f}/MWh")
                    st.metric("Peak Forecast Price", f"${peak_fc_price:.2f}/MWh")

            with col_b:
                if "solar_dispatch_mw_pred" in fc_df.columns:
                    st.plotly_chart(
                        _forecast_chart(
                            fc_df, "solar_dispatch_mw",
                            f"Solar Dispatch Forecast — {selected_dow_name}", "MW",
                        ),
                        use_container_width=True,
                    )
                    peak_solar = fc_df["solar_dispatch_mw_pred"].max()
                    total_energy = fc_df["solar_dispatch_mw_pred"].sum() * 0.5  # 30-min → MWh
                    st.metric("Peak Solar Forecast", f"{peak_solar:.0f} MW")
                    st.metric("Est. Daily Solar Energy", f"{total_energy:.0f} MWh")

            with st.expander("Forecast table (hourly)"):
                disp = fc_df.copy()
                disp.index = disp.index.strftime("%H:%M")
                price_cols = [c for c in disp.columns if "price" in c]
                solar_cols = [c for c in disp.columns if "solar" in c]
                if price_cols:
                    st.dataframe(disp[price_cols].round(2), use_container_width=True)
                if solar_cols:
                    st.dataframe(disp[solar_cols].round(1), use_container_width=True)
        else:
            st.warning(
                f"Could not generate forecast for {selected_dow_name}. "
                "This day may not appear in the 7-day forecast window."
            )

# ── Tab 3: Model Performance ───────────────────────────────────────────────
with tab_performance:
    st.subheader("Model Performance — Test Set Evaluation")

    if not models_ready:
        st.info("Train models first.")
    else:
        with st.spinner("Loading evaluation data…"):
            try:
                feature_df = _load_training_data(region)
                df_train, df_val, df_test = time_split(feature_df)
                feat_cols = get_feature_columns(feature_df)
                n_train = len(df_train)

                all_metrics = []
                for target in TARGETS:
                    st.markdown(f"#### {target.replace('_', ' ').title()}")

                    pipeline = load_pipeline(target)
                    model    = load_model(target)

                    X_test = transform(pipeline, df_test[feat_cols])
                    y_test = df_test[target].values
                    y_hat  = predict(model, X_test)

                    # LightGBM metrics
                    lgbm_metrics = evaluate_all(y_test, y_hat, model_name="LightGBM")
                    all_metrics.append(lgbm_metrics)

                    # Baselines
                    bl_results = evaluate_baselines(
                        feature_df, target, test_start_idx=n_train + len(df_val)
                    )
                    bl_summary = baseline_summary(bl_results)
                    for _, row in bl_summary.iterrows():
                        all_metrics.append({
                            "model": row["model"], "target": target,
                            "MAE": row["MAE"], "RMSE": row["RMSE"],
                            "MAPE": None, "SMAPE": None, "R2": None, "n": row["n"],
                        })

                    comp = compare_models([lgbm_metrics] + [
                        {"model": r["model"], "MAE": r["MAE"], "RMSE": r["RMSE"]}
                        for _, r in bl_summary.iterrows()
                    ])
                    st.dataframe(comp, use_container_width=True, hide_index=True)

                    # Error slices
                    c1, c2 = st.columns(2)
                    y_test_series = df_test[target]

                    with c1:
                        err_h = error_by_hour(y_test_series, y_hat)
                        fig_h = px.bar(err_h, x="hour", y="MAE",
                                       title="MAE by Hour of Day",
                                       template="plotly_dark",
                                       color_discrete_sequence=["#FFA500"])
                        fig_h.update_layout(height=280, margin=dict(t=30, b=10))
                        st.plotly_chart(fig_h, use_container_width=True)

                    with c2:
                        err_d = error_by_dow(y_test_series, y_hat)
                        fig_d = px.bar(err_d, x="day", y="MAE",
                                       title="MAE by Day of Week",
                                       template="plotly_dark",
                                       color_discrete_sequence=["#1f77b4"])
                        fig_d.update_layout(height=280, margin=dict(t=30, b=10))
                        st.plotly_chart(fig_d, use_container_width=True)

                    # Feature importance
                    fi = feature_importance(model, feat_cols, top_n=15)
                    fig_fi = px.bar(fi.sort_values("importance"), x="importance", y="feature",
                                    orientation="h",
                                    title="Top 15 Feature Importances (LightGBM)",
                                    template="plotly_dark",
                                    color_discrete_sequence=["#2ca02c"])
                    fig_fi.update_layout(height=400, margin=dict(t=30, b=10))
                    st.plotly_chart(fig_fi, use_container_width=True)

                    # SHAP (if available)
                    with st.expander("SHAP Analysis (sample of 500 test points)"):
                        sample_n = min(500, len(X_test))
                        shap_vals, _ = compute_shap_values(model, X_test[:sample_n], feat_cols)
                        if shap_vals is not None:
                            shap_df = shap_summary_df(shap_vals, feat_cols)
                            fig_shap = px.bar(
                                shap_df.head(15).sort_values("mean_abs_shap"),
                                x="mean_abs_shap", y="feature",
                                orientation="h",
                                title="SHAP Feature Importance (mean |SHAP|)",
                                template="plotly_dark",
                                color_discrete_sequence=["#d62728"],
                            )
                            fig_shap.update_layout(height=400, margin=dict(t=30, b=10))
                            st.plotly_chart(fig_shap, use_container_width=True)
                        else:
                            st.info("Install `shap` to enable SHAP analysis.")

                    st.divider()

            except FileNotFoundError as e:
                st.error(f"Model files not found: {e}")
            except Exception as e:
                st.error(f"Evaluation error: {e}")
                logger.exception("Performance tab error")

# ── Tab 4: Data Explorer ───────────────────────────────────────────────────
with tab_data:
    st.subheader("Training Data Explorer")

    date_range = st.date_input(
        "Date range",
        value=(datetime.now().date() - timedelta(days=30), datetime.now().date()),
        max_value=datetime.now().date(),
    )

    with st.spinner("Loading data…"):
        try:
            raw_df = fetch_region_data(region, period="30d")
            if len(date_range) == 2:
                start, end = pd.Timestamp(date_range[0], tz="Australia/Sydney"), \
                             pd.Timestamp(date_range[1], tz="Australia/Sydney")
                raw_df = raw_df[(raw_df.index >= start) & (raw_df.index <= end)]

            if raw_df.empty:
                st.info("No data for selected range.")
            else:
                col1, col2, col3 = st.columns(3)
                col1.metric("Rows", f"{len(raw_df):,}")
                if "electricity_price_aud_mwh" in raw_df.columns:
                    col2.metric("Avg Price", f"${raw_df['electricity_price_aud_mwh'].mean():.2f}/MWh")
                if "solar_dispatch_mw" in raw_df.columns:
                    col3.metric("Avg Solar", f"{raw_df['solar_dispatch_mw'].mean():.0f} MW")

                if "electricity_price_aud_mwh" in raw_df.columns:
                    fig = px.line(
                        raw_df.reset_index(), x="datetime", y="electricity_price_aud_mwh",
                        title="Electricity Price (AUD/MWh)",
                        template="plotly_dark",
                        color_discrete_sequence=["#FFA500"],
                    )
                    st.plotly_chart(fig, use_container_width=True)

                if "solar_dispatch_mw" in raw_df.columns:
                    fig2 = px.line(
                        raw_df.reset_index(), x="datetime", y="solar_dispatch_mw",
                        title="Solar Dispatch (MW)",
                        template="plotly_dark",
                        color_discrete_sequence=["#2ca02c"],
                    )
                    st.plotly_chart(fig2, use_container_width=True)

                with st.expander("Raw data table"):
                    st.dataframe(raw_df.head(500), use_container_width=True)

        except Exception as e:
            st.error(f"Data loading error: {e}")
