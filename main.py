
# ============================================================
# HIMALAYAN EARTHQUAKE INTELLIGENCE SYSTEM
# Complete Data Science Project
# Data Source: USGS Earthquake Catalog
# ============================================================

import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
import joblib
import time
import warnings
import logging

from pathlib import Path
from datetime import datetime, timezone
from folium.plugins import HeatMap, MarkerCluster
from sklearn.cluster import DBSCAN

from analysis.preprocessing import preprocess_earthquakes
from analysis.eda import create_eda_outputs
from analysis.clustering import EARTH_RADIUS_KM, run_dbscan
from analysis.evaluation import compare_dbscan_parameters, evaluate_labels, summarize_clusters
from analysis.statistical_analysis import gutenberg_richter_analysis

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_DIR = BASE_DIR / "models"

DATA_FILE = DATA_DIR / "earthquakes.csv"

# Automatically create folders if they do not exist
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

# Himalayan region bounding box
MIN_LATITUDE = 25
MAX_LATITUDE = 38

MIN_LONGITUDE = 70
MAX_LONGITUDE = 100

# Historical period: 20 years
START_DATE = "2006-11-01"
END_DATE = "2026-11-01"

# Minimum earthquake magnitude
MIN_MAGNITUDE = 2.5

# USGS API
USGS_API_URL = (
    "https://earthquake.usgs.gov/fdsnws/event/1/query"
)


# ============================================================
# 2. DOWNLOAD EARTHQUAKE DATA FROM USGS
# ============================================================

def download_earthquakes():

    print("\nStarting USGS earthquake data collection...")

    all_earthquakes = []

    session = requests.Session()

    start = pd.Timestamp(START_DATE, tz="UTC")
    end = pd.Timestamp(END_DATE, tz="UTC")

    current_date = start

    while current_date < end:

        next_date = min(
            current_date + pd.DateOffset(months=1),
            end
        )

        print(
            f"Downloading: {current_date.date()} "
            f"to {next_date.date()}"
        )

        params = {

            "format": "geojson",

            "starttime": current_date.strftime(
                "%Y-%m-%dT%H:%M:%S"
            ),

            "endtime": next_date.strftime(
                "%Y-%m-%dT%H:%M:%S"
            ),

            "minlatitude": MIN_LATITUDE,
            "maxlatitude": MAX_LATITUDE,

            "minlongitude": MIN_LONGITUDE,
            "maxlongitude": MAX_LONGITUDE,

            "minmagnitude": MIN_MAGNITUDE,

            "orderby": "time-asc",

            "limit": 20000
        }

        # Retry failed API requests
        response = None

        for attempt in range(3):

            try:

                response = session.get(
                    USGS_API_URL,
                    params=params,
                    timeout=60
                )

                response.raise_for_status()

                break

            except requests.RequestException as error:

                print(
                    f"Request failed (attempt {attempt + 1}/3): "
                    f"{error}"
                )

                if attempt == 2:

                    raise RuntimeError(
                        "USGS download failed. "
                        "Check your internet connection and try again."
                    )

                time.sleep(3)

        data = response.json()

        features = data.get("features", [])

        print(f"Records received: {len(features)}")

        # Extract earthquake information
        for feature in features:

            properties = feature.get("properties", {})
            geometry = feature.get("geometry", {})

            coordinates = geometry.get("coordinates", [])

            if len(coordinates) < 3:
                continue

            longitude = coordinates[0]
            latitude = coordinates[1]
            depth = coordinates[2]

            earthquake = {

                "id": feature.get("id"),

                "magnitude": properties.get("mag"),

                "place": properties.get("place"),

                "time": properties.get("time"),

                "longitude": longitude,

                "latitude": latitude,

                "depth": depth,

                "mag_type": properties.get("magType"),

                "significance": properties.get("sig"),

                "felt": properties.get("felt"),

                "tsunami": properties.get("tsunami"),

                "event_type": properties.get("type"),

                "url": properties.get("url")
            }

            all_earthquakes.append(earthquake)

        # The USGS API has a maximum result limit.
        if len(features) >= 20000:

            raise RuntimeError(
                "A monthly request reached the 20,000-event API limit. "
                "Reduce the date interval or increase the magnitude "
                "threshold before continuing."
            )

        current_date = next_date

        time.sleep(0.15)

    session.close()

    # Convert records into a DataFrame
    df = pd.DataFrame(all_earthquakes)

    if df.empty:

        raise RuntimeError(
            "USGS returned no earthquake records. "
            "Check the date range, geographic boundaries and API."
        )

    # Remove duplicate events
    df.drop_duplicates(subset="id", inplace=True)

    # Save only after successful download
    temporary_file = DATA_DIR / "earthquakes_temp.csv"

    df.to_csv(temporary_file, index=False)

    temporary_file.replace(DATA_FILE)

    print("\nDownload completed successfully!")

    print("Total earthquake records:", len(df))

    print("Saved to:", DATA_FILE)

    return df


# ============================================================
# 3. LOAD OR DOWNLOAD DATA
# ============================================================

def load_earthquake_data():

    if DATA_FILE.exists() and DATA_FILE.stat().st_size > 0:

        print("\nChecking existing earthquake dataset...")

        try:

            df = pd.read_csv(DATA_FILE)

            if not df.empty:

                print("Existing dataset found.")

                print("Records:", len(df))

                return df

            print("Existing CSV contains no records.")

        except (pd.errors.EmptyDataError, pd.errors.ParserError):

            print("Existing CSV is empty or invalid.")

    else:

        print("No valid earthquake CSV found.")

    print("\nDownloading fresh earthquake data...")

    return download_earthquakes()


# ============================================================
# 4. DATA CLEANING
# ============================================================

def clean_data(df):
    """Backward-compatible cleaning entry point backed by the preprocessing module."""
    print("\nCleaning earthquake dataset...")
    cleaned, quality, outliers = preprocess_earthquakes(
        df,
        min_latitude=MIN_LATITUDE,
        max_latitude=MAX_LATITUDE,
        min_longitude=MIN_LONGITUDE,
        max_longitude=MAX_LONGITUDE,
        minimum_magnitude=MIN_MAGNITUDE,
    )
    quality.to_csv(OUTPUT_DIR / "preprocessing_summary.csv", index=False)
    outliers.to_csv(OUTPUT_DIR / "depth_outliers.csv", index=False)
    print("Cleaning completed.")
    print("Records remaining:", len(cleaned))
    return cleaned


# ============================================================
# 5. FEATURE ENGINEERING
# ============================================================

def feature_engineering(df):

    print("\nPerforming feature engineering...")

    df = df.copy()

    # Extract date-related information
    df["year"] = df["time"].dt.year

    df["month"] = df["time"].dt.month

    df["day"] = df["time"].dt.day

    df["hour"] = df["time"].dt.hour

    df["weekday"] = df["time"].dt.day_name()

    # Magnitude classification
    df["magnitude_category"] = pd.cut(

        df["magnitude"],

        bins=[2.5, 4, 5, 6, 7, 10],

        labels=[
            "Minor",
            "Light",
            "Moderate",
            "Strong",
            "Major"
        ],

        include_lowest=True
    )

    # Depth classification
    df["depth_category"] = np.select(

        [
            df["depth"] <= 70,

            df["depth"] <= 300
        ],

        [
            "Shallow",

            "Intermediate"
        ],

        default="Deep"
    )

    # Approximate earthquake energy in Joules
    df["energy_joules"] = (
        10 ** (1.5 * df["magnitude"] + 4.8)
    )

    print("Feature engineering completed.")

    return df


# ============================================================
# 6. STATISTICAL ANALYSIS
# ============================================================

def statistical_analysis(df):

    print("\n" + "=" * 55)

    print("HIMALAYAN EARTHQUAKE STATISTICAL REPORT")

    print("=" * 55)

    print("\nTotal earthquakes:", len(df))

    print(
        "Average magnitude:",
        round(df["magnitude"].mean(), 2)
    )

    print(
        "Maximum magnitude:",
        round(df["magnitude"].max(), 2)
    )

    print(
        "Median magnitude:",
        round(df["magnitude"].median(), 2)
    )

    print(
        "Average depth:",
        round(df["depth"].mean(), 2),
        "km"
    )

    print(
        "Maximum depth:",
        round(df["depth"].max(), 2),
        "km"
    )

    print("\nMagnitude category distribution:")

    print(
        df["magnitude_category"].value_counts(
            sort=False
        )
    )

    print("\nDepth category distribution:")

    print(
        df["depth_category"].value_counts()
    )

    print("\nStatistical description:")

    print(
        df[["magnitude", "depth", "latitude", "longitude"]]
        .describe()
        .round(2)
    )


# ============================================================
# 7. DATA VISUALISATION
# ============================================================

def create_visualisations(df):

    print("\nGenerating visualisations...")

    sns.set_theme(style="whitegrid")

    # --------------------------------------------------------
    # 7.1 Magnitude Distribution
    # --------------------------------------------------------

    plt.figure(figsize=(10, 6))

    sns.histplot(
        data=df,
        x="magnitude",
        bins=30,
        kde=True
    )

    plt.title("Earthquake Magnitude Distribution")

    plt.xlabel("Magnitude")

    plt.ylabel("Number of Earthquakes")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "magnitude_distribution.png"
    )

    plt.close()

    # --------------------------------------------------------
    # 7.2 Yearly Earthquake Frequency
    # --------------------------------------------------------

    yearly = df.groupby("year").size()

    plt.figure(figsize=(12, 6))

    yearly.plot(
        kind="line",
        marker="o"
    )

    plt.title("Yearly Earthquake Frequency")

    plt.xlabel("Year")

    plt.ylabel("Number of Earthquakes")

    plt.xticks(rotation=45)

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "yearly_earthquakes.png"
    )

    plt.close()

    # --------------------------------------------------------
    # 7.3 Earthquake Depth Distribution
    # --------------------------------------------------------

    plt.figure(figsize=(10, 6))

    sns.histplot(
        data=df,
        x="depth",
        bins=30,
        kde=True
    )

    plt.title("Earthquake Depth Distribution")

    plt.xlabel("Depth (km)")

    plt.ylabel("Frequency")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "depth_analysis.png"
    )

    plt.close()

    # --------------------------------------------------------
    # 7.4 Magnitude vs Depth
    # --------------------------------------------------------

    plt.figure(figsize=(10, 6))

    sns.scatterplot(

        data=df,

        x="depth",

        y="magnitude",

        alpha=0.5
    )

    plt.title("Earthquake Magnitude vs Depth")

    plt.xlabel("Depth (km)")

    plt.ylabel("Magnitude")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "magnitude_depth.png"
    )

    plt.close()

    # --------------------------------------------------------
    # 7.5 Monthly Earthquake Activity
    # --------------------------------------------------------

    monthly = df.groupby("month").size()

    plt.figure(figsize=(10, 6))

    monthly.plot(
        kind="bar"
    )

    plt.title("Monthly Earthquake Activity")

    plt.xlabel("Month")

    plt.ylabel("Number of Earthquakes")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "monthly_activity.png"
    )

    plt.close()

    # --------------------------------------------------------
    # 7.6 Geographical Earthquake Density
    # --------------------------------------------------------

    plt.figure(figsize=(10, 7))

    plt.hexbin(

        df["longitude"],

        df["latitude"],

        gridsize=40,

        cmap="Reds",

        mincnt=1
    )

    plt.colorbar(
        label="Earthquake Frequency"
    )

    plt.title("Geographical Earthquake Density")

    plt.xlabel("Longitude")

    plt.ylabel("Latitude")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "earthquake_heatmap.png"
    )

    plt.close()

    print("Visualisations saved successfully.")


# ============================================================
# 8. DBSCAN CLUSTERING
# ============================================================

def earthquake_clustering(df):
    """Fit the reference DBSCAN model and compare bounded parameter alternatives."""
    print("\nPerforming DBSCAN clustering...")
    df = df.copy()
    result = run_dbscan(df, epsilon_km=50.0, min_samples=5)
    df["cluster"] = result.labels
    summary = summarize_clusters(df, result.labels)

    evaluation = compare_dbscan_parameters(df)
    evaluation.to_csv(OUTPUT_DIR / "model_evaluation.csv", index=False)
    reference_evaluation = evaluate_labels(
        df,
        result.labels,
        epsilon_km=result.epsilon_km,
        min_samples=result.min_samples,
        execution_time_seconds=result.execution_time_seconds,
    )
    evaluation = evaluation.sort_values(
        ["silhouette_score", "noise_percent"],
        ascending=[False, True],
        na_position="last",
    )

    if not summary.empty:
        summary.to_csv(OUTPUT_DIR / "cluster_summary.csv", index=False)
        (DATA_DIR / "cluster_summary.csv").parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(DATA_DIR / "cluster_summary.csv", index=False)
        public_data_dir = BASE_DIR / "public" / "data"
        public_data_dir.mkdir(parents=True, exist_ok=True)
        summary.to_csv(public_data_dir / "cluster_summary.csv", index=False)
    joblib.dump(
        {
            "model": result.model,
            "epsilon_km": result.epsilon_km,
            "earth_radius_km": EARTH_RADIUS_KM,
            "min_samples": result.min_samples,
            "evaluation": reference_evaluation,
        },
        MODEL_DIR / "dbscan_model.pkl",
    )

    # --------------------------------------------------------
    # Cluster Visualisation
    # --------------------------------------------------------

    plt.figure(figsize=(12, 7))

    sns.scatterplot(

        data=df,

        x="longitude",

        y="latitude",

        hue="cluster",

        palette="tab20",

        s=15,

        alpha=0.7,

        legend=False
    )

    plt.title("DBSCAN Earthquake Geographical Clusters")

    plt.xlabel("Longitude")

    plt.ylabel("Latitude")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "cluster_analysis.png"
    )

    plt.close()

    print("Clustering completed.")

    return df, summary


# ============================================================
# 9. INTERACTIVE EARTHQUAKE MAP
# ============================================================

def create_interactive_map(df, summary):

    print("\nCreating interactive earthquake map...")

    earthquake_map = folium.Map(

        location=[31.5, 85],

        zoom_start=5,

        tiles="OpenStreetMap"
    )

    # --------------------------------------------------------
    # Heatmap Layer
    # --------------------------------------------------------

    heatmap_layer = folium.FeatureGroup(
        name="Earthquake Heatmap"
    )

    # Limit the heatmap points to control map size
    heatmap_data = df

    if len(heatmap_data) > 20000:

        heatmap_data = heatmap_data.sample(
            n=20000,
            random_state=42
        )

    heat_points = [

        [
            row["latitude"],

            row["longitude"],

            max(float(row["magnitude"]), 0.1)

        ]

        for _, row in heatmap_data.iterrows()

    ]

    HeatMap(
        heat_points,
        radius=12,
        blur=15,
        name="Earthquake Density"
    ).add_to(heatmap_layer)

    heatmap_layer.add_to(earthquake_map)

    # --------------------------------------------------------
    # Major Earthquake Markers
    # --------------------------------------------------------

    marker_layer = folium.FeatureGroup(
        name="Major Earthquakes"
    )

    marker_cluster = MarkerCluster()

    major_events = df[
        df["magnitude"] >= 4.5
    ].sort_values(
        "magnitude",
        ascending=False
    ).head(2000)

    for _, row in major_events.iterrows():

        popup_text = f"""
        <b>Location:</b> {row['place']}<br>
        <b>Magnitude:</b> {row['magnitude']}<br>
        <b>Depth:</b> {row['depth']} km<br>
        <b>Date:</b> {row['time']}
        """

        folium.Marker(

            location=[
                row["latitude"],
                row["longitude"]
            ],

            popup=folium.Popup(
                popup_text,
                max_width=300
            ),

            tooltip=f"Magnitude: {row['magnitude']}",

            icon=folium.Icon(
                color="red",
                icon="info-sign"
            )

        ).add_to(marker_cluster)

    marker_cluster.add_to(marker_layer)

    marker_layer.add_to(earthquake_map)

    # --------------------------------------------------------
    # DBSCAN Cluster Centres
    # --------------------------------------------------------

    if not summary.empty:

        cluster_layer = folium.FeatureGroup(
            name="DBSCAN Cluster Centres"
        )

        for _, row in summary.iterrows():

            folium.CircleMarker(

                location=[
                    row["center_latitude"],
                    row["center_longitude"]
                ],

                radius=8,

                color="blue",

                fill=True,

                fill_opacity=0.7,

                popup=(
                    f"Cluster: {row['cluster']}<br>"
                    f"Earthquakes: {row['earthquake_count']}<br>"
                    f"Average Magnitude: "
                    f"{row['average_magnitude']}"
                )

            ).add_to(cluster_layer)

        cluster_layer.add_to(earthquake_map)

    # Layer control
    folium.LayerControl().add_to(
        earthquake_map
    )

    # Save map
    map_path = OUTPUT_DIR / "earthquake_map.html"

    earthquake_map.save(str(map_path))

    print("Interactive map saved:", map_path)


# ============================================================
# 10. SAVE PROCESSED DATA
# ============================================================

def save_data(df):

    output_file = (
        OUTPUT_DIR / "processed_earthquakes.csv"
    )

    df.to_csv(
        output_file,
        index=False
    )

    # Keep deployable dashboard copies synchronized with the primary outputs.
    data_output = DATA_DIR / "processed_earthquakes.csv"
    df.to_csv(data_output, index=False)

    deployment_output_dir = BASE_DIR / "public" / "data"
    deployment_output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(deployment_output_dir / "processed_earthquakes.csv", index=False)

    print("\nProcessed dataset saved.")

    print(output_file)

def generate_advanced_analysis(df):
    """Generate reusable EDA and Gutenberg-Richter outputs."""
    create_eda_outputs(df, OUTPUT_DIR)
    frequency, statistics = gutenberg_richter_analysis(df, OUTPUT_DIR)
    pd.DataFrame([statistics]).to_csv(
        OUTPUT_DIR / "gutenberg_richter_summary.csv", index=False
    )
    frequency.to_csv(OUTPUT_DIR / "public_gutenberg_richter.csv", index=False)


# ============================================================
# 11. FINAL PROJECT REPORT
# ============================================================

def generate_report(df, summary):

    print("\n" + "=" * 55)

    print("FINAL PROJECT REPORT")

    print("=" * 55)

    print(
        "Study Period:",
        START_DATE,
        "to",
        END_DATE
    )

    print(
        "Geographical Latitude:",
        MIN_LATITUDE,
        "to",
        MAX_LATITUDE
    )

    print(
        "Geographical Longitude:",
        MIN_LONGITUDE,
        "to",
        MAX_LONGITUDE
    )

    print(
        "Total Earthquakes:",
        len(df)
    )

    print(
        "Average Magnitude:",
        round(df["magnitude"].mean(), 2)
    )

    print(
        "Largest Recorded Magnitude:",
        round(df["magnitude"].max(), 2)
    )

    print(
        "Average Depth:",
        round(df["depth"].mean(), 2),
        "km"
    )

    if not summary.empty:

        print(
            "Number of DBSCAN Clusters:",
            len(summary)
        )

    else:

        print("Number of DBSCAN Clusters: 0")

    print("\nGenerated files are available in:")

    print(OUTPUT_DIR)

    print("\nIMPORTANT:")

    print(
        "This project analyses historical earthquake patterns. "
        "It does not predict future earthquakes."
    )

    print("=" * 55)


# ============================================================
# 12. MAIN EXECUTION
# ============================================================

def main():

    print("\n")
    print("=" * 60)

    print("HIMALAYAN EARTHQUAKE INTELLIGENCE SYSTEM")

    print("=" * 60)

    # Step 1: Load or download data
    df = load_earthquake_data()

    # Step 2: Clean data
    df = clean_data(df)

    # Step 3: Feature engineering
    df = feature_engineering(df)

    # Step 4: Statistical analysis
    statistical_analysis(df)

    # Step 5: Generate charts
    create_visualisations(df)
    generate_advanced_analysis(df)

    # Step 6: Apply DBSCAN clustering
    df, summary = earthquake_clustering(df)

    # Step 7: Generate interactive map
    create_interactive_map(df, summary)

    # Step 8: Save processed dataset
    save_data(df)

    # Step 9: Generate final report
    generate_report(df, summary)

    print("\nPROJECT EXECUTION COMPLETED SUCCESSFULLY!")


# ============================================================
# RUN PROJECT
# ============================================================

if __name__ == "__main__":

    main()
