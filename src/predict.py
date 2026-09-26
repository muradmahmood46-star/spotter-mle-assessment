"""
Prediction pipeline to generate validation-predictions.csv and populate december-chart-inputs.csv.
"""
from __future__ import annotations
import sys
import pickle
from pathlib import Path

# Add project root to sys.path for direct execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.data_loader import load_validation_data, load_december_inputs, BASE_DIR


def generate_predictions(model_path: str | Path | None = None):
    if model_path is None:
        model_path = BASE_DIR / "models" / "lgb_pipeline.pkl"
    else:
        model_path = Path(model_path)

    with open(model_path, "rb") as f:
        artifacts = pickle.load(f)

    pipeline = artifacts["pipeline"]
    models = artifacts["models"]

    # 1. Validation predictions
    print("Generating validation predictions...")
    val_df = load_validation_data()
    X_val = pipeline.transform(val_df)

    # Average predictions across folds in log space
    val_preds_log = np.mean([model.predict(X_val) for model in models], axis=0)
    val_preds = np.expm1(val_preds_log)

    # Format exactly as load_id,predicted_rate
    sub_df = pd.DataFrame({
        "load_id": val_df["load_id"],
        "predicted_rate": val_preds.round(2)
    })
    output_pred_path = BASE_DIR / "validation_predictions.csv"
    sub_df.to_csv(output_pred_path, index=False)
    print(f"Saved {output_pred_path} successfully ({len(sub_df):,} rows).")

    # 2. December chart input predictions
    print("Generating December chart predictions...")
    dec_df = load_december_inputs()
    X_dec = pipeline.transform(dec_df)

    dec_preds_log = np.mean([model.predict(X_dec) for model in models], axis=0)
    dec_preds = np.expm1(dec_preds_log)

    dec_df["predicted_rate"] = dec_preds.round(2)
    dec_df["date"] = dec_df["date"].dt.strftime("%Y-%m-%d")
    output_dec_path = BASE_DIR / "december-chart-inputs.csv"
    dec_df.to_csv(output_dec_path, index=False)
    print(f"Updated {output_dec_path} with predictions successfully ({len(dec_df)} rows).")


if __name__ == "__main__":
    generate_predictions()
