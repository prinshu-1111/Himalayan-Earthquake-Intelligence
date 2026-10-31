"""Data quality checks and preprocessing for earthquake records."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd


LOGGER = logging.getLogger(__name__)
REQUIRED_COLUMNS = {
    "id",
    "time",
    "magnitude",
    "longitude",
    "latitude",
    "depth",
}


def summarize_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Return missing-value counts and percentages for each column."""
    counts = df.isna().sum()
    percentages = counts.div(len(df)).mul(100).where(len(df) > 0, 0)
    return pd.DataFrame(
        {"missing_count": counts, "missing_percent": percentages.round(3)}
    ).reset_index(names="column")


def validate_coordinates(df: pd.DataFrame) -> pd.Series:
    """Return valid-row mask for geographic coordinates."""
    latitude = pd.to_numeric(df["latitude"], errors="coerce")
    longitude = pd.to_numeric(df["longitude"], errors="coerce")
    return (
        latitude.between(-90, 90)
        & longitude.between(-180, 180)
        & latitude.notna()
        & longitude.notna()
    )


def preprocess_earthquakes(
    df: pd.DataFrame,
    *,
    min_latitude: float,
    max_latitude: float,
    min_longitude: float,
    max_longitude: float,
    minimum_magnitude: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Clean earthquake records and return data, quality summary, and outliers."""
    missing_columns = REQUIRED_COLUMNS.difference(df.columns)
    if missing_columns:
        raise ValueError(
            "Required earthquake columns are missing: "
            + ", ".join(sorted(missing_columns))
        )

    cleaned = df.copy()
    original_rows = len(cleaned)
    numeric_columns = [
        column
        for column in ("magnitude", "longitude", "latitude", "depth")
        if column in cleaned.columns
    ]
    cleaned[numeric_columns] = cleaned[numeric_columns].apply(
        pd.to_numeric, errors="coerce"
    )

    numeric_time = pd.to_numeric(cleaned["time"], errors="coerce")
    parsed_numeric = pd.to_datetime(numeric_time, unit="ms", utc=True, errors="coerce")
    parsed_text = pd.to_datetime(cleaned["time"], utc=True, errors="coerce")
    cleaned["time"] = parsed_numeric.where(numeric_time.notna(), parsed_text)

    reasons: dict[str, int] = {}
    cleaned["_missing_required"] = cleaned[
        ["time", "magnitude", "longitude", "latitude", "depth"]
    ].isna().any(axis=1)
    reasons["missing_required"] = int(cleaned["_missing_required"].sum())

    duplicate_mask = cleaned["id"].duplicated(keep="first") & cleaned["id"].notna()
    reasons["duplicate_id"] = int(duplicate_mask.sum())

    coordinate_mask = validate_coordinates(cleaned)
    reasons["invalid_coordinates"] = int((~coordinate_mask).sum())

    magnitude_mask = cleaned["magnitude"].between(minimum_magnitude, 10.0)
    reasons["invalid_magnitude"] = int(
        (cleaned["magnitude"].notna() & ~magnitude_mask).sum()
    )
    depth_mask = cleaned["depth"].between(0, 700)
    reasons["invalid_depth"] = int((cleaned["depth"].notna() & ~depth_mask).sum())

    in_bounding_box = (
        cleaned["latitude"].between(min_latitude, max_latitude)
        & cleaned["longitude"].between(min_longitude, max_longitude)
    )
    reasons["outside_study_region"] = int(in_bounding_box.eq(False).sum())

    retained_mask = (
        ~cleaned["_missing_required"]
        & ~duplicate_mask
        & coordinate_mask
        & magnitude_mask.fillna(False)
        & depth_mask.fillna(False)
        & in_bounding_box.fillna(False)
    )
    cleaned = cleaned.loc[retained_mask].copy()
    cleaned.drop(columns="_missing_required", inplace=True)
    cleaned.sort_values("time", inplace=True)
    cleaned.reset_index(drop=True, inplace=True)

    if cleaned.empty:
        raise ValueError("No valid earthquake records remain after preprocessing.")

    cleaned = add_temporal_features(cleaned)
    cleaned["magnitude_category"] = categorize_magnitude(cleaned["magnitude"])
    cleaned["depth_category"] = categorize_depth(cleaned["depth"])
    cleaned["energy_joules"] = 10 ** (1.5 * cleaned["magnitude"] + 4.8)

    q1 = cleaned["depth"].quantile(0.25)
    q3 = cleaned["depth"].quantile(0.75)
    iqr = q3 - q1
    outlier_mask = cleaned["depth"].lt(q1 - 1.5 * iqr) | cleaned["depth"].gt(
        q3 + 1.5 * iqr
    )
    outlier_columns = [
        column
        for column in (
            "id",
            "time",
            "place",
            "magnitude",
            "latitude",
            "longitude",
            "depth",
            "magnitude_category",
        )
        if column in cleaned.columns
    ]
    outliers = cleaned.loc[outlier_mask, outlier_columns].copy()

    quality = pd.DataFrame(
        [
            {"metric": "raw_rows", "value": original_rows},
            {"metric": "retained_rows", "value": len(cleaned)},
            {"metric": "removed_rows", "value": original_rows - len(cleaned)},
            {"metric": "duplicate_rows", "value": reasons["duplicate_id"]},
            {"metric": "missing_required_rows", "value": reasons["missing_required"]},
            {"metric": "invalid_coordinate_rows", "value": reasons["invalid_coordinates"]},
            {"metric": "invalid_magnitude_rows", "value": reasons["invalid_magnitude"]},
            {"metric": "invalid_depth_rows", "value": reasons["invalid_depth"]},
            {"metric": "outside_study_region_rows", "value": reasons["outside_study_region"]},
            {"metric": "statistical_depth_outliers", "value": len(outliers)},
        ]
    )
    quality["value"] = pd.to_numeric(quality["value"])
    LOGGER.info(
        "Preprocessing retained %d of %d earthquake records", len(cleaned), original_rows
    )
    return cleaned, quality, outliers


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar features derived from UTC event time."""
    result = df.copy()
    result["year"] = result["time"].dt.year
    result["month"] = result["time"].dt.month
    result["day"] = result["time"].dt.day
    result["hour"] = result["time"].dt.hour
    result["weekday"] = result["time"].dt.day_name()
    result["month_name"] = result["time"].dt.month_name()
    return result


def categorize_magnitude(magnitude: pd.Series) -> pd.Series:
    """Categorize magnitude without extrapolating outside physically useful bounds."""
    return pd.cut(
        magnitude,
        bins=[-np.inf, 4, 5, 6, 7, np.inf],
        labels=["Minor", "Light", "Moderate", "Strong", "Major"],
        right=True,
        include_lowest=True,
    )


def categorize_depth(depth: pd.Series) -> pd.Series:
    """Apply standard shallow/intermediate/deep depth categories."""
    return np.select(
        [depth <= 70, depth <= 300],
        ["Shallow", "Intermediate"],
        default="Deep",
    )
