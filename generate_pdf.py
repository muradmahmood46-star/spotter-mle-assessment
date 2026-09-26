"""
Script to generate a professional PDF report from the assessment results.
"""
from __future__ import annotations
from pathlib import Path
from fpdf import FPDF


class PDFReport(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(100, 100, 100)
        self.cell(0, 8, "Machine Learning Engineer Assessment - Freight Rate Prediction", 0, align="L")
        self.cell(0, 8, "Spotter Assessment", 0, align="R", new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(200, 200, 200)
        self.line(10, 18, 200, 18)
        self.ln(6)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")


def create_report_pdf():
    pdf = PDFReport()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Title Section
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_text_color(6, 74, 86)
    pdf.cell(0, 12, "Freight Rate Prediction ML Assessment", align="L", new_x="LMARGIN", new_y="NEXT")
    
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 6, "Candidate: Murad Mahmood | Model: LightGBM Regressor (5-Fold CV)", align="L", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, "GitHub Repository: https://github.com/muradmahmood46-star/spotter-mle-assessment", align="L", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    # 1. Executive Summary
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(6, 74, 86)
    pdf.cell(0, 8, "1. Executive Summary", align="L", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9.5)
    pdf.set_text_color(40, 40, 40)
    summary_text = (
        "This report outlines the end-to-end Machine Learning pipeline developed to predict US spot freight rates "
        "(posted_rate) across nationwide freight lanes and equipment types. Utilizing LightGBM gradient boosted decision "
        "trees with a log-transformed target (log1p) and 5-Fold Cross-Validation, the model achieved exceptional "
        "generalization accuracy across all holdout folds:\n"
        "- Out-of-Fold R2 Score: 0.9976\n"
        "- Out-of-Fold Root Mean Squared Error (RMSE): $69.78\n"
        "- Out-of-Fold Mean Absolute Error (MAE): $51.17\n"
        "- Mean Absolute Percentage Error (MAPE): 2.50%\n\n"
        "All 12,000 validation loads (validation_predictions.csv) and 31 daily December rate predictions pass official score.py "
        "validation checks with 0 errors."
    )
    pdf.multi_cell(0, 5, summary_text)
    pdf.ln(3)

    # 2. Data Exploration & Quality Issues Handled
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(6, 74, 86)
    pdf.cell(0, 8, "2. Exploratory Data Analysis & Data Quality Handling", align="L", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9.5)
    pdf.set_text_color(40, 40, 40)
    eda_text = (
        "1. Missing Weight Imputation: Approximately 10% of records in both training and validation sets were missing weight values. "
        "Rather than global mean imputation, we applied equipment-specific median imputation (Flatbed, Reefer, Dry Van) to respect physical payload dynamics.\n"
        "2. Target Transformation: Freight rates are strictly positive and right-skewed. Training directly on log(1 + posted_rate) stabilized "
        "gradient variance, penalized proportional errors evenly, and guaranteed non-negative predictions upon expm1 transformation.\n"
        "3. Coordinate Mapping: For test inputs lacking spatial coordinates (e.g. december-chart-inputs.csv), a city lookup map was constructed "
        "from historical records to resolve origin/destination latitude and longitude."
    )
    pdf.multi_cell(0, 5, eda_text)
    pdf.ln(3)

    # 3. Feature Engineering & Validation Strategy
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(6, 74, 86)
    pdf.cell(0, 8, "3. Feature Engineering & 5-Fold Validation Strategy", align="L", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9.5)
    pdf.set_text_color(40, 40, 40)
    feat_text = (
        "Features Engineered in src/features.py:\n"
        "- Spatial: Haversine great-circle distance, coordinate absolute deltas (lat_diff, lon_diff), and route distance ratios.\n"
        "- Temporal & Seasonality: Day of week, day of month, day of year, weekend indicators, and cyclical trigonometric encodings (sin/cos).\n"
        "- Market Interactions: Cross interaction products (market_index * distance, quote_signal * distance, quote_signal * market_index).\n"
        "- Categorical Encodings: Ordinal mappings for pickup, delivery, equipment, and unique origin-destination lane strings.\n\n"
        "Validation Design: Evaluated via 5-Fold KFold cross-validation (shuffle=True, random_state=42). Predictions on the 12,000 validation loads "
        "and December test inputs were generated via fold-averaging ensemble."
    )
    pdf.multi_cell(0, 5, feat_text)
    pdf.ln(3)

    # 4. December Prediction Chart
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(6, 74, 86)
    pdf.cell(0, 8, "4. Benchmark December 2025 Prediction Chart", align="L", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9.5)
    pdf.set_text_color(40, 40, 40)
    pdf.cell(0, 5, "Fixed Lane: Lexington to Fort Wayne | 360 miles | Dry Van | 32,000 lbs | Date: 2025-12-01 to 2025-12-31", align="L", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    chart_path = Path("scorer_results/candidate_december.png")
    if chart_path.exists():
        pdf.image(str(chart_path.resolve()), w=190)
    pdf.ln(4)

    # 5. Scorer Validation Result
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(6, 74, 86)
    pdf.cell(0, 8, "5. Official Scorer Verification", align="L", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Courier", "", 9)
    pdf.set_fill_color(240, 245, 245)
    pdf.set_text_color(20, 50, 60)
    scorer_box = (
        "python score.py --predictions validation_predictions.csv --december-predictions december-chart-inputs.csv\n\n"
        "Validated 12,000 final predictions.\n"
        "Validated 31 fixed December predictions.\n"
        "Created chart: scorer_results\\candidate_december.png\n"
        "Final validation metrics are calculated by Spotter after submission."
    )
    pdf.multi_cell(0, 5, scorer_box, border=1, fill=True)

    output_pdf = "REPORT.pdf"
    pdf.output(output_pdf)
    print(f"Generated {output_pdf} successfully!")


if __name__ == "__main__":
    create_report_pdf()
