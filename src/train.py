"""
Model training pipeline with LightGBM, log1p target transformation, and 5-fold Cross-Validation.
"""
from __future__ import annotations
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb

from src.data_loader import load_train_data, BASE_DIR
from src.features import FeaturePipeline


def train_model():
    print("Loading training data...")
    train_df = load_train_data()
    # Filter out missing target row if present
    train_df = train_df.dropna(subset=["posted_rate"]).reset_index(drop=True)

    pipeline = FeaturePipeline()
    pipeline.fit(train_df)

    X = pipeline.transform(train_df)
    y = train_df["posted_rate"].values
    y_log = np.log1p(y)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds_log = np.zeros(len(train_df))
    models = []

    params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "n_estimators": 2000,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "verbose": -1,
        "n_jobs": -1
    }

    print("\nStarting 5-Fold Cross-Validation...")
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y_log), 1):
        X_train, y_train = X.iloc[train_idx], y_log[train_idx]
        X_val, y_val = X.iloc[val_idx], y_log[val_idx]

        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )

        val_pred = model.predict(X_val)
        oof_preds_log[val_idx] = val_pred
        models.append(model)

        fold_rmse = np.sqrt(mean_squared_error(y[val_idx], np.expm1(val_pred)))
        fold_mae = mean_absolute_error(y[val_idx], np.expm1(val_pred))
        print(f"Fold {fold} - RMSE: ${fold_rmse:.2f}, MAE: ${fold_mae:.2f}")

    oof_preds = np.expm1(oof_preds_log)
    rmse = np.sqrt(mean_squared_error(y, oof_preds))
    mae = mean_absolute_error(y, oof_preds)
    r2 = r2_score(y, oof_preds)
    mape = np.mean(np.abs((y - oof_preds) / y)) * 100

    print("\n=== Out-Of-Fold Cross-Validation Performance ===")
    print(f"RMSE : ${rmse:.2f}")
    print(f"MAE  : ${mae:.2f}")
    print(f"R^2  : {r2:.4f}")
    print(f"MAPE : {mape:.2f}%")

    # Save artifacts
    artifacts = {
        "pipeline": pipeline,
        "models": models
    }
    models_dir = BASE_DIR / "models"
    models_dir.mkdir(exist_ok=True)
    with open(models_dir / "lgb_pipeline.pkl", "wb") as f:
        pickle.dump(artifacts, f)
    print(f"\nModel pipeline saved to {models_dir / 'lgb_pipeline.pkl'}")

    return pipeline, models


if __name__ == "__main__":
    train_model()
