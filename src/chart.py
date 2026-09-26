"""
December 2025 Prediction Visualization.
"""
from __future__ import annotations
import sys
from pathlib import Path

# Add project root to sys.path for direct execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.data_loader import BASE_DIR


def plot_december_chart(input_csv: str | Path | None = None, output_png: str | Path | None = None):
    if input_csv is None:
        input_csv = BASE_DIR / "december-chart-inputs.csv"
    else:
        input_csv = Path(input_csv)

    if output_png is None:
        output_png = BASE_DIR / "december-chart.png"
    else:
        output_png = Path(output_png)

    df = pd.read_csv(input_csv)
    df["date"] = pd.to_datetime(df["date"])

    fig, ax = plt.subplots(figsize=(11, 5), dpi=200)
    color = "#064A56"

    ax.plot(
        df["date"],
        df["predicted_rate"],
        color=color,
        linewidth=2.5,
        marker="o",
        markersize=4,
        label="Predicted Rate ($)"
    )

    floor = float(df["predicted_rate"].min())
    ax.fill_between(
        df["date"],
        df["predicted_rate"],
        floor - max(10.0, floor * 0.02),
        color=color,
        alpha=0.08,
    )

    ax.set_title("December 2025 Predicted Freight Load Rates", loc="left", fontsize=14, fontweight="bold", pad=12)
    ax.set_ylabel("Predicted Rate ($)", fontsize=11)
    ax.grid(axis="y", color="#D9E2E4", linewidth=0.8, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#9DAFB3")
    ax.tick_params(axis="x", rotation=35)
    ax.text(
        0,
        -0.35,
        "Lane: Lexington -> Fort Wayne | 360 miles | Dry Van | 32,000 lbs",
        transform=ax.transAxes,
        fontsize=9.5,
        color="#455A60",
    )

    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(output_png, bbox_inches="tight")
    plt.close(fig)
    print(f"Chart saved to {output_png}")


if __name__ == "__main__":
    plot_december_chart()
