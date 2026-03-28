"""
Steps 8 & 9 — LightGBM Model Selection, Training, HPO & Experiment Tracking.

  - TimeSeriesSplit cross-validation (no data leakage)
  - Optuna TPE hyperparameter optimisation (n_trials from config)
  - MLflow experiment tracking (all trials logged)
  - Seeds fixed for reproducibility
  - Early stopping on validation loss
  - Separate models trained for price and dispatch targets
"""

import hashlib
import json
import logging
from pathlib import Path

import joblib
import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import numpy as np
import optuna
import pandas as pd
import yaml
from sklearn.model_selection import TimeSeriesSplit

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


def _load_config() -> dict:
    cfg_path = Path(__file__).parents[2] / "config.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


CFG = _load_config()
MODEL_CFG = CFG["models"]
LGBM_CFG  = MODEL_CFG["lgbm"]
MLFLOW_CFG = CFG["mlflow"]
MODELS_DIR = Path(CFG["data"]["models_dir"])
SEED = MODEL_CFG["random_seed"]


# ---------------------------------------------------------------------------
# MLflow setup
# ---------------------------------------------------------------------------

def _setup_mlflow():
    mlflow.set_tracking_uri(MLFLOW_CFG["tracking_uri"])
    mlflow.set_experiment(MLFLOW_CFG["experiment_name"])


# ---------------------------------------------------------------------------
# Data fingerprint for reproducibility logging
# ---------------------------------------------------------------------------

def _data_hash(X: np.ndarray) -> str:
    return hashlib.md5(X.tobytes()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------

def _make_objective(X_train: np.ndarray, y_train: np.ndarray, cv: TimeSeriesSplit):
    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective":        "regression",
            "metric":           "rmse",
            "verbosity":        -1,
            "boosting_type":    "gbdt",
            "num_leaves":       trial.suggest_int("num_leaves", 20, 300),
            "max_depth":        trial.suggest_int("max_depth", 3, 12),
            "learning_rate":    trial.suggest_float("learning_rate", 1e-4, 0.3, log=True),
            "n_estimators":     trial.suggest_int("n_estimators", 100, LGBM_CFG["n_estimators_max"]),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha":        trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "random_state":     SEED,
        }

        fold_rmses = []
        for train_idx, val_idx in cv.split(X_train):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]

            model = lgb.LGBMRegressor(**params)
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(
                    stopping_rounds=LGBM_CFG["early_stopping_rounds"],
                    verbose=False,
                )],
            )
            preds = model.predict(X_val)
            rmse = np.sqrt(np.mean((y_val - preds) ** 2))
            fold_rmses.append(rmse)

        return np.mean(fold_rmses)

    return objective


# ---------------------------------------------------------------------------
# Public training API
# ---------------------------------------------------------------------------

def train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: list[str],
    target: str,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
) -> lgb.LGBMRegressor:
    """
    Run Optuna HPO → retrain best model on full train set → log to MLflow.

    Parameters
    ----------
    X_train, y_train : training arrays (already scaled by pipeline)
    feature_names    : list of feature column names (for SHAP / importance)
    target           : "electricity_price_aud_mwh" or "solar_dispatch_mw"
    X_val, y_val     : optional hold-out val arrays for final evaluation
    """
    _setup_mlflow()

    cv = TimeSeriesSplit(n_splits=LGBM_CFG["cv_folds"])
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
    )

    logger.info("Starting Optuna HPO: %d trials for target '%s'", LGBM_CFG["n_trials"], target)

    with mlflow.start_run(run_name=f"hpo_{target}"):
        mlflow.log_param("target", target)
        mlflow.log_param("n_trials", LGBM_CFG["n_trials"])
        mlflow.log_param("cv_folds", LGBM_CFG["cv_folds"])
        mlflow.log_param("data_hash", _data_hash(X_train))
        mlflow.log_param("n_train_samples", len(X_train))
        mlflow.log_param("n_features", X_train.shape[1])

        study.optimize(
            _make_objective(X_train, y_train, cv),
            n_trials=LGBM_CFG["n_trials"],
            show_progress_bar=False,
        )

        best_params = study.best_params
        best_cv_rmse = study.best_value
        logger.info("Best CV RMSE=%.4f  params=%s", best_cv_rmse, best_params)
        mlflow.log_metric("best_cv_rmse", best_cv_rmse)
        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})

        # Retrain on full training set with best hyperparameters
        final_params = {
            **best_params,
            "objective":     "regression",
            "metric":        "rmse",
            "verbosity":     -1,
            "boosting_type": "gbdt",
            "random_state":  SEED,
        }
        final_model = lgb.LGBMRegressor(**final_params)

        eval_set = [(X_val, y_val)] if X_val is not None else None
        callbacks = []
        if eval_set:
            callbacks.append(lgb.early_stopping(
                stopping_rounds=LGBM_CFG["early_stopping_rounds"], verbose=False,
            ))

        final_model.fit(X_train, y_train, eval_set=eval_set, callbacks=callbacks)

        if X_val is not None:
            val_preds = final_model.predict(X_val)
            val_rmse  = np.sqrt(np.mean((y_val - val_preds) ** 2))
            val_mae   = np.mean(np.abs(y_val - val_preds))
            mlflow.log_metric("val_rmse", val_rmse)
            mlflow.log_metric("val_mae",  val_mae)
            logger.info("Final model  val_RMSE=%.4f  val_MAE=%.4f", val_rmse, val_mae)

        mlflow.lightgbm.log_model(final_model, artifact_path=f"model_{target}")

    return final_model


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

def save_model(model: lgb.LGBMRegressor, target: str) -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / f"lgbm_{target}.joblib"
    meta_path = MODELS_DIR / f"lgbm_{target}_meta.json"

    joblib.dump(model, path)

    meta = {
        "target": target,
        "n_estimators": model.n_estimators_,
        "n_features": model.n_features_in_,
        "best_score": float(list(model.best_score_.get("valid_0", {}).values() or [0])[0])
            if hasattr(model, "best_score_") else None,
    }
    meta_path.write_text(json.dumps(meta, indent=2))

    logger.info("Model saved → %s", path)
    return path


def load_model(target: str) -> lgb.LGBMRegressor:
    path = MODELS_DIR / f"lgbm_{target}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"No model at {path}. Run training first.")
    return joblib.load(path)


def model_exists(target: str) -> bool:
    return (MODELS_DIR / f"lgbm_{target}.joblib").exists()


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict(model: lgb.LGBMRegressor, X: np.ndarray) -> np.ndarray:
    """Raw prediction; caller is responsible for inverse-scaling if needed."""
    return model.predict(X)


def feature_importance(
    model: lgb.LGBMRegressor,
    feature_names: list[str],
    top_n: int = 20,
) -> pd.DataFrame:
    importance = model.feature_importances_
    df = pd.DataFrame({
        "feature":    feature_names,
        "importance": importance,
    }).sort_values("importance", ascending=False).head(top_n)
    return df.reset_index(drop=True)
