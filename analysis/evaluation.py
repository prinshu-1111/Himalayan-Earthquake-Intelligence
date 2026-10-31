"""Evaluation and parameter comparison for geographical DBSCAN models."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import davies_bouldin_score, silhouette_score

from .clustering import coordinates_in_radians, run_dbscan


def evaluate_labels(
    df: pd.DataFrame,
    labels: np.ndarray,
    *,
    epsilon_km: float,
    min_samples: int,
    execution_time_seconds: float,
) -> dict[str, object]:
    """Evaluate one DBSCAN result while leaving undefined metrics unset."""
    non_noise = labels != -1
    cluster_labels = labels[non_noise]
    unique_clusters = np.unique(cluster_labels)
    cluster_count = len(unique_clusters)
    noise_count = int(np.sum(~non_noise))
    sizes = pd.Series(cluster_labels).value_counts().sort_values(ascending=False)

    silhouette: float | None = None
    davies_bouldin: float | None = None
    if cluster_count >= 2 and len(unique_clusters) == len(np.unique(labels)):
        pass

    # Spatial Euclidean distance on radians has a monotonic relationship to
    # Haversine distance for points considered in this regional analysis.
    if cluster_count >= 2 and cluster_count < non_noise.sum():
        coordinates = coordinates_in_radians(df)[non_noise]
        silhouette = float(silhouette_score(coordinates, cluster_labels))
        davies_bouldin = float(davies_bouldin_score(coordinates, cluster_labels))

    return {
        "epsilon_km": epsilon_km,
        "min_samples": min_samples,
        "execution_time_seconds": execution_time_seconds,
        "number_of_clusters": cluster_count,
        "noise_points": noise_count,
        "noise_percent": round(noise_count / len(labels) * 100, 3),
        "silhouette_score": silhouette,
        "davies_bouldin_index": davies_bouldin,
        "largest_cluster_size": int(sizes.iloc[0]) if len(sizes) else 0,
        "smallest_cluster_size": int(sizes.iloc[-1]) if len(sizes) else 0,
        "median_cluster_size": float(sizes.median()) if len(sizes) else 0.0,
    }


def compare_dbscan_parameters(
    df: pd.DataFrame,
    epsilon_values: tuple[float, ...] = (25.0, 50.0, 75.0),
    min_samples_values: tuple[int, ...] = (5, 10, 20),
) -> pd.DataFrame:
    """Fit and evaluate a bounded set of DBSCAN parameter combinations."""
    evaluations: list[dict[str, object]] = []
    for epsilon_km in epsilon_values:
        for min_samples in min_samples_values:
            result = run_dbscan(df, epsilon_km, min_samples)
            evaluations.append(
                evaluate_labels(
                    df,
                    result.labels,
                    epsilon_km=result.epsilon_km,
                    min_samples=result.min_samples,
                    execution_time_seconds=result.execution_time_seconds,
                )
            )
    return pd.DataFrame(evaluations)


def summarize_clusters(df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Build cluster statistics while excluding DBSCAN noise."""
    clustered = df.copy()
    clustered["cluster"] = labels
    clustered = clustered.loc[clustered["cluster"] != -1]
    if clustered.empty:
        return pd.DataFrame()
    return (
        clustered.groupby("cluster")
        .agg(
            earthquake_count=("id", "count"),
            average_magnitude=("magnitude", "mean"),
            maximum_magnitude=("magnitude", "max"),
            average_depth=("depth", "mean"),
            center_latitude=("latitude", "mean"),
            center_longitude=("longitude", "mean"),
        )
        .reset_index()
        .round(3)
    )
