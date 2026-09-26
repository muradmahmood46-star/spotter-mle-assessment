# 🎙️ Loom Video Recording Script (2–3 Minutes)

**Assessment:** Machine Learning Engineer - Freight Rate Prediction Challenge  
**Candidate:** Murad Mahmood  
**Repository:** [https://github.com/muradmahmood46-star/spotter-mle-assessment](https://github.com/muradmahmood46-star/spotter-mle-assessment)  

---

## 🖥️ Screen Tab Sequence (Open these files in this exact order)

1. **`README.md`** (Project overview & workflow)
2. **`train_test.csv`** (First 10 rows showing features & `posted_rate`)
3. **`src/features.py`** (Feature engineering, missing weight imputation, cyclical & spatial features)
4. **`src/train.py`** (LightGBM regressor, log1p transformation, 5-Fold Cross-Validation)
5. **Terminal / Command Prompt** (Run `python score.py --predictions validation_predictions.csv --december-predictions december-chart-inputs.csv`)
6. **`scorer_results/candidate_december.png`** (The generated benchmark December prediction chart)
7. **GitHub Repository Web Page** ([spotter-mle-assessment](https://github.com/muradmahmood46-star/spotter-mle-assessment))

---

## ⏱️ Step-by-Step Speaking Script with Timestamps

### 0:00 – 0:30 | Introduction & Problem Formulation
> *"Hi everyone, my name is Murad Mahmood, and today I'm walking you through my solution for the Freight Rate Prediction Machine Learning assessment. The goal is to predict spot freight rates (`posted_rate`) across nationwide freight lanes and equipment types using historical loads and market indicators."*

### 0:30 – 1:00 | Exploratory Data Analysis & Data Quality Handling
*(Show `train_test.csv` and `src/features.py`)*
> *"During data exploration, I addressed key data quality considerations:*
> 1. *~10% of weight entries were missing. Instead of a simplistic global average, I applied equipment-specific median imputation (Reefer, Flatbed, and Dry Van) to preserve payload physics.*
> 2. *Freight rates are strictly positive and right-skewed. To prevent high-value loads from skewing gradients and to guarantee positive rate outputs, I trained on `log(1 + posted_rate)`.*
> 3. *For test data lacking spatial coordinates, I built a city coordinate lookup map to calculate precise Haversine distances, route tortuosity, and coordinate deltas."*

### 1:00 – 1:45 | Feature Engineering & Modeling Strategy
*(Show `src/features.py` and `src/train.py`)*
> *"I engineered temporal cyclical features (sin/cos of day of year and day of week) and market interaction features like `market_index * distance` and `quote_signal * distance`.*
> *I chose **LightGBM Regressor** with **5-Fold Cross-Validation (`KFold`)** to evaluate out-of-fold generalization without leakage.*
> *The model achieved outstanding validation performance:*
> - *R² Score: **0.9976***
> - *Out-of-Fold MAE: **$51.17***
> - *Mean Absolute Percentage Error (MAPE): **2.50%***
> - *RMSE: **$69.78**"*

### 1:45 – 2:30 | Code Walkthrough & Scorer Verification
*(Show Terminal running `score.py`, then open `scorer_results/candidate_december.png`)*
> *"Our modular codebase in `src/` cleanly separates loading, features, training, inference, and visualization.*
> *Running the official scorer:*
> ```bash
> python score.py --predictions validation_predictions.csv --december-predictions december-chart-inputs.csv
> ```
> *All 12,000 validation loads and all 31 fixed December predictions pass format, schema, and non-negativity checks with **zero errors**.*
> *Looking at our December chart for the Lexington to Fort Wayne lane, rates range stably between **$710.87** and **$713.08** (mean: **$711.84**, ~**$1.97/mile**), realistically reflecting holiday/weekend calendar adjustments."*

### 2:30 – 3:00 | Conclusion & Wrap-Up
*(Show GitHub repository page)*
> *"The complete repository with the code, `validation_predictions.csv`, `REPORT.pdf`, and reproduction instructions is fully pushed and accessible on GitHub. Thank you for your time, and I look forward to your feedback!"*

---

## 🔢 Key Numbers Cheat-Sheet to Mention
- **12,000**: Validation predictions generated (`TE-000001` to `TE-012000`)
- **31**: December daily predictions for Lexington $\to$ Fort Wayne
- **0.9976**: Out-of-fold $R^2$ Score
- **$51.17**: Out-of-fold MAE
- **2.50%**: Mean Absolute Percentage Error (MAPE)
- **$69.78**: Out-of-fold RMSE
- **$711.84**: Average December predicted rate (~$1.97/mile)
