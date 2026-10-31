"""Haversine DBSCAN clustering for earthquake epicentres."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN


EARTH_RADIUS_KM = 6371.0088


@dataclass(frozen=True)
class ClusteringResult:
    labels: np.ndarray
    model: DBSCAN
    epsilon_km: float
    min_samples: int
    execution_time_seconds: float


def coordinates_in_radians(df: pd.DataFrame) -> np.ndarray:
    """Convert latitude and longitude to radians for the Haversine metric."""
    return np.radians(df[["latitude", "longitude"]].to_numpy(dtype=float))


def run_dbscan(
    df: pd.DataFrame,
    epsilon_km: float = 50.0,
    min_samples: int = 5,
) -> ClusteringResult:
    """Fit a validated geographical DBSCAN model."""
    coordinates = coordinates_in_radians(df)
    epsilon_radians = epsilon_km / EARTH_RADIUS_KM
    model = DBSCAN(
        eps=epsilon_radians,
        min_samples=min_samples,
        metric="haversine",
        algorithm="ball_tree",
    )
    started_at = perf_counter()
    labels = model.fit_predict(coordinates)
    execution_time = perf_counter() - started_at
    return ClusteringResult(
        labels=labels,
        model=model,
        epsilon_km=epsilon_km,
        min_samples=min_samples,
        execution_time_seconds=execution_time,
    )
