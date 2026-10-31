"""Gutenberg-Richter and temporal earthquake statistical analysis."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def gutenberg_richter_analysis(
    df: pd.DataFrame,
    output_dir: Path,
    magnitude_bin_width: float = 0.1,
) -> tuple[pd.DataFrame, dict[str, float | None]]:
    """Estimate Gutenberg-Richter frequency and Aki maximum-likelihood b-value."""
    magnitudes = df["magnitude"].dropna().astype(float)
    minimum_magnitude = float(magnitudes.min())
    completeness_adjustment = minimum_magnitude + magnitude_bin_width / 2.0
    complete = magnitudes[magnitudes >= minimum_magnitude]

    thresholds = np.round(np.arange(2.5, np.floor(magnitudes.max()) + 0.5, 0.5), 1)
    frequency = pd.DataFrame(
        {
            "threshold_magnitude": thresholds,
            "cumulative_count": [
                int((complete >= threshold).sum()) for threshold in thresholds
            ],
        }
    )
    frequency["log10_cumulative_count"] = np.log10(
        frequency["cumulative_count"].replace(0, np.nan)
    )

    if complete.size:
        b_value = float(np.log10(np.e) / max(complete.mean() - completeness_adjustment, 1e-12))
    else:
        b_value = None

    output_dir.mkdir(parents=True, exist_ok=True)
    frequency.to_csv(output_dir / "gutenberg_richter.csv", index=False)
    figure, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        frequency["threshold_magnitude"],
        frequency["log10_cumulative_count"],
        marker="o",
    )
    ax.set(
        title="Gutenberg-Richter Frequency-Magnitude Relation",
        xlabel="Magnitude threshold",
        ylabel="log10(cumulative count)",
    )
    ax.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_dir / "gutenberg_richter.png", dpi=150)
    plt.close(figure)

    statistics = {
        "b_value": b_value,
        "minimum_magnitude": minimum_magnitude,
        "event_count_above_minimum": int(complete.size),
        "magnitude_bin_width": magnitude_bin_width,
    }
    return frequency, statistics
