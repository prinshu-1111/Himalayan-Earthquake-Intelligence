"""Reusable exploratory analysis tables and visualizations."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def create_eda_outputs(df: pd.DataFrame, output_dir: Path) -> dict[str, pd.DataFrame]:
    """Create descriptive tables and static plots and return analytical tables."""
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    yearly = df.groupby("year").size().rename("earthquake_count").reset_index()
    monthly = df.groupby(["month", "month_name"]).size().rename("earthquake_count").reset_index()
    magnitude_categories = (
        df["magnitude_category"].astype("string").value_counts(dropna=False)
        .rename("earthquake_count")
        .rename_axis("magnitude_category")
        .reset_index()
    )
    correlation = df[
        ["magnitude", "depth", "latitude", "longitude", "year"]
    ].corr(method="pearson")

    outputs = {
        "yearly": yearly,
        "monthly": monthly,
        "magnitude_categories": magnitude_categories,
        "correlation": correlation,
    }

    yearly.to_csv(output_dir / "yearly_activity.csv", index=False)
    monthly.to_csv(output_dir / "monthly_activity.csv", index=False)
    magnitude_categories.to_csv(output_dir / "magnitude_categories.csv", index=False)
    correlation.to_csv(output_dir / "correlation_matrix.csv")

    figures = (
        (
            output_dir / "magnitude_distribution.png",
            lambda ax: sns.histplot(df["magnitude"], bins=30, kde=True, ax=ax),
            ("Earthquake Magnitude Distribution", "Magnitude", "Number of earthquakes"),
        ),
        (
            output_dir / "depth_analysis.png",
            lambda ax: sns.histplot(df["depth"], bins=30, kde=True, ax=ax),
            ("Earthquake Depth Distribution", "Depth (km)", "Number of earthquakes"),
        ),
        (
            output_dir / "magnitude_depth.png",
            lambda ax: sns.scatterplot(data=df, x="depth", y="magnitude", alpha=0.35, ax=ax),
            ("Magnitude versus Depth", "Depth (km)", "Magnitude"),
        ),
        (
            output_dir / "yearly_earthquakes.png",
            lambda ax: sns.lineplot(data=yearly, x="year", y="earthquake_count", marker="o", ax=ax),
            ("Yearly Earthquake Activity", "Year", "Number of earthquakes"),
        ),
        (
            output_dir / "monthly_activity.png",
            lambda ax: sns.barplot(data=monthly.sort_values("month"), x="month_name", y="earthquake_count", ax=ax),
            ("Monthly Earthquake Activity", "Month", "Number of earthquakes"),
        ),
    )
    for path, plotter, labels in figures:
        fig, ax = plt.subplots(figsize=(10, 6))
        plotter(ax)
        ax.set(title=labels[0], xlabel=labels[1], ylabel=labels[2])
        if "Monthly" in labels[0]:
            ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(correlation, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation of Numeric Earthquake Features")
    fig.tight_layout()
    fig.savefig(output_dir / "correlation_heatmap.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.hexbin(df["longitude"], df["latitude"], gridsize=40, cmap="Reds", mincnt=1)
    ax.set(
        title="Geographical Earthquake Density",
        xlabel="Longitude",
        ylabel="Latitude",
    )
    fig.colorbar(ax.collections[0], ax=ax, label="Earthquake Frequency")
    fig.tight_layout()
    fig.savefig(output_dir / "earthquake_heatmap.png", dpi=150)
    plt.close(fig)
    return outputs
