# Freight Rate Prediction Challenge - Spotter MLE Assessment

This repository contains an end-to-end Machine Learning pipeline to predict spot freight rates (`posted_rate`) across US freight lanes and diverse equipment types.

---

## 🚀 Quickstart & Run Instructions

### 1. Environment Setup

Clone the repository and install the dependencies:

```bash
git clone https://github.com/muradmahmood46-star/spotter-mle-assessment.git
cd spotter-mle-assessment
```

Install requirements:
```bash
python -m pip install -r requirements.txt
```

*(Or using `uv`):*
```bash
uv venv
uv pip install -r requirements.txt
```

---

### 2. Full Pipeline Execution

Run the complete pipeline from scratch:

```bash
# Step 1: Train the LightGBM 5-Fold Cross-Validation Model
python -m src.train

# Step 2: Generate validation predictions & populate December test inputs
python -m src.predict

# Step 3: Generate the December rate forecast chart
python -m src.chart

# Step 4: Validate predictions format and outputs using the official scorer
python score.py --predictions validation_predictions.csv --december-predictions december-chart-inputs.csv
```

---

## 📊 Modeling & Validation Strategy

### 1. Data Split & Validation Approach
- **5-Fold Cross-Validation (`KFold(n_splits=5, shuffle=True, random_state=42)`)**: Evaluates out-of-fold generalization across historical loads without temporal or lane target leakage.
- **Log Transformation**: The target `posted_rate` is right-skewed and strictly positive. Training on $\log(1 + \text{posted\_rate})$ stabilizes variance, penalizes relative errors symmetrically, and ensures all reconstructed predictions are strictly positive ($> 0$).
- **Fold Averaging (Ensemble)**: Predictions on `validation.csv` and `december-chart-inputs.csv` are computed as the arithmetic mean of fold predictions in log space before exponentiating ($\text{expm1}$).

### 2. Feature Engineering
Implemented in `src/features.py`:
- **Spatial / Coordinate Features**: Haversine distance, latitude/longitude absolute differences, distance-to-great-circle ratio.
- **Missing Value Handling**: Missing `weight` values (~10%) are imputed using equipment-specific medians (e.g. Reefer vs Dry Van distributions).
- **Temporal Encodings**: Day of week, day of month, day of year, weekend flag, and cyclical trigonometric encodings ($\sin/\cos$).
- **Market Dynamics Interactions**: Cross interactions between distance, `market_index`, and `quote_signal` (`market_dist`, `quote_dist`, `quote_market`, `weight_dist`).
- **Categorical Encodings**: Equipment, pickup location, delivery location, and origin-destination lane IDs.

---

## 📈 Cross-Validation Performance

| Fold | RMSE ($) | MAE ($) |
| :---: | :---: | :---: |
| Fold 1 | \$73.19 | \$52.61 |
| Fold 2 | \$69.45 | \$50.84 |
| Fold 3 | \$70.36 | \$51.52 |
| Fold 4 | \$68.48 | \$51.53 |
| Fold 5 | \$67.24 | \$49.33 |
| **Overall OOF** | **\$69.78** | **\$51.17** |

- **$R^2$ Score**: **0.9976**
- **MAPE**: **2.50%**

---

## 📁 Repository Structure

```
├── .gitignore
├── pyproject.toml
├── README.md
├── requirements.txt
├── score.py                                  # Official assessment scorer
├── validation_predictions.csv                # 12,000 final predictions
├── validation-predictions.csv                # Alias file for validation predictions
├── december-chart-inputs.csv                 # 31 predicted daily December loads
├── scorer_results/
│   └── candidate_december.png                # Generated scorer chart
├── december-chart.png                        # Pipeline chart output
├── models/
│   └── lgb_pipeline.pkl                      # Saved trained models & feature pipeline
└── src/
    ├── __init__.py
    ├── data_loader.py                        # Dataset loading utilities
    ├── features.py                           # Feature engineering & preprocessing
    ├── train.py                              # 5-fold CV training pipeline
    ├── predict.py                            # Inference & output generation
    └── chart.py                              # Visualization script
```

---

## ✅ Scorer Verification Output

```text
Validated 12,000 final predictions.
Validated 31 fixed December predictions.
Created chart: scorer_results\candidate_december.png
Final validation metrics are calculated by Spotter after submission.
```