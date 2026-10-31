
# ============================================================
# HIMALAYAN EARTHQUAKE INTELLIGENCE DASHBOARD
# STREAMLIT WEB APPLICATION
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import folium
import plotly.express as px

from pathlib import Path
from folium.plugins import HeatMap, MarkerCluster
from streamlit_folium import st_folium


# ------------------------------------------------------------
# 1. PAGE CONFIGURATION
# ------------------------------------------------------------

st.set_page_config(

    page_title="Himalayan Earthquake Intelligence",

    page_icon="🌍",

    layout="wide"

)


# ------------------------------------------------------------
# 2. PROJECT PATHS
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

DATA_FILE = BASE_DIR / "outputs" / "processed_earthquakes.csv"

CLUSTER_FILE = BASE_DIR / "outputs" / "cluster_summary.csv"


# ------------------------------------------------------------
# 3. LOAD DATA
# ------------------------------------------------------------

@st.cache_data

def load_data():

    df = pd.read_csv(
        DATA_FILE,
        parse_dates=["time"]
    )

    return df


@st.cache_data

def load_clusters():

    if CLUSTER_FILE.exists():

        return pd.read_csv(CLUSTER_FILE)

    return pd.DataFrame()


if not DATA_FILE.exists():

    st.error(
        "Earthquake dataset not found. "
        "Please run python main.py first."
    )

    st.stop()


df = load_data()

clusters = load_clusters()


# ------------------------------------------------------------
# 4. HEADER
# ------------------------------------------------------------

st.title("🌍 Himalayan Earthquake Intelligence System")

st.markdown(
    """
    **A Data Science and Geospatial Analytics Platform**

    Explore 20 years of historical earthquake activity
    across the Himalayan region using real USGS data.
    """
)

st.caption(
    "Historical seismic activity analysis. "
    "This application does not predict future earthquakes."
)


# ------------------------------------------------------------
# 5. SIDEBAR FILTERS
# ------------------------------------------------------------

st.sidebar.title("Analysis Filters")

minimum_magnitude = float(df["magnitude"].min())

maximum_magnitude = float(df["magnitude"].max())

selected_magnitude = st.sidebar.slider(

    "Minimum Magnitude",

    min_value=minimum_magnitude,

    max_value=maximum_magnitude,

    value=minimum_magnitude,

    step=0.1

)

minimum_date = pd.to_datetime(df["time"]).min().date()

maximum_date = pd.to_datetime(df["time"]).max().date()

selected_dates = st.sidebar.date_input(

    "Select Date Range",

    value=(minimum_date, maximum_date),

    min_value=minimum_date,

    max_value=maximum_date

)


# ------------------------------------------------------------
# 6. APPLY FILTERS
# ------------------------------------------------------------

filtered_df = df[
    df["magnitude"] >= selected_magnitude
].copy()


if isinstance(selected_dates, (tuple, list)) and len(selected_dates) == 2:

    start_date, end_date = selected_dates

    filtered_df = filtered_df[

        (filtered_df["time"].dt.date >= start_date)

        &

        (filtered_df["time"].dt.date <= end_date)

    ]


if filtered_df.empty:

    st.warning("No earthquakes found for these filters.")

    st.stop()


# ------------------------------------------------------------
# 7. KEY PERFORMANCE INDICATORS
# ------------------------------------------------------------

st.subheader("Earthquake Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric(

    "Total Earthquakes",

    f"{len(filtered_df):,}"

)

col2.metric(

    "Average Magnitude",

    f"{filtered_df['magnitude'].mean():.2f}"

)

col3.metric(

    "Maximum Magnitude",

    f"{filtered_df['magnitude'].max():.2f}"

)

col4.metric(

    "Average Depth",

    f"{filtered_df['depth'].mean():.1f} km"

)


st.divider()


# ------------------------------------------------------------
# 8. INTERACTIVE EARTHQUAKE MAP
# ------------------------------------------------------------

st.subheader("📍 Geographical Earthquake Activity")

st.write(
    "The heatmap shows historical earthquake concentration. "
    "The markers represent individual larger earthquakes."
)

earthquake_map = folium.Map(

    location=[31.5, 85],

    zoom_start=5,

    tiles="CartoDB positron"

)


# Heatmap layer

heat_points = filtered_df[
    ["latitude", "longitude", "magnitude"]
].dropna()


HeatMap(

    heat_points.values.tolist(),

    radius=12,

    blur=18,

    name="Earthquake Density"

).add_to(earthquake_map)


# Major earthquake markers

major_events = filtered_df[
    filtered_df["magnitude"] >= 4.5
]

marker_group = MarkerCluster(
    name="Major Earthquakes"
)

# Limit markers for better browser performance

major_events = major_events.head(3000)


for _, row in major_events.iterrows():

    popup = f"""
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

        popup=popup,

        tooltip=f"Magnitude {row['magnitude']}"

    ).add_to(marker_group)


marker_group.add_to(earthquake_map)

folium.LayerControl().add_to(earthquake_map)


st_folium(

    earthquake_map,

    width=None,

    height=600,

    key="earthquake_map"

)


# ------------------------------------------------------------
# 9. MAGNITUDE DISTRIBUTION
# ------------------------------------------------------------

st.divider()

st.subheader("📊 Magnitude Distribution")

fig_magnitude = px.histogram(

    filtered_df,

    x="magnitude",

    nbins=30,

    title="Distribution of Earthquake Magnitudes",

    color_discrete_sequence=["#2563eb"]

)

fig_magnitude.update_layout(

    xaxis_title="Magnitude",

    yaxis_title="Number of Earthquakes"

)

st.plotly_chart(
    fig_magnitude,
    use_container_width=True
)


# ------------------------------------------------------------
# 10. YEARLY EARTHQUAKE TREND
# ------------------------------------------------------------

st.subheader("📈 Yearly Earthquake Activity")

yearly = (

    filtered_df.groupby("year")

    .size()

    .reset_index(name="earthquake_count")

)

fig_yearly = px.line(

    yearly,

    x="year",

    y="earthquake_count",

    markers=True,

    title="Yearly Earthquake Frequency"

)

st.plotly_chart(
    fig_yearly,
    use_container_width=True
)


# ------------------------------------------------------------
# 11. DEPTH ANALYSIS
# ------------------------------------------------------------

st.subheader("🌋 Earthquake Depth Analysis")

fig_depth = px.histogram(

    filtered_df,

    x="depth",

    nbins=30,

    title="Earthquake Depth Distribution",

    color_discrete_sequence=["#e8793e"]

)

fig_depth.update_layout(

    xaxis_title="Depth (km)",

    yaxis_title="Frequency"

)

st.plotly_chart(
    fig_depth,
    use_container_width=True
)


# ------------------------------------------------------------
# 12. MAGNITUDE VS DEPTH
# ------------------------------------------------------------

st.subheader("Magnitude vs Depth Relationship")

fig_scatter = px.scatter(

    filtered_df,

    x="depth",

    y="magnitude",

    color="magnitude_category",

    hover_data=["place", "time"],

    title="Magnitude and Depth Relationship"

)

st.plotly_chart(
    fig_scatter,
    use_container_width=True
)


# ------------------------------------------------------------
# 13. MONTHLY ACTIVITY
# ------------------------------------------------------------

st.subheader("Monthly Earthquake Activity")

monthly = (

    filtered_df.groupby("month")

    .size()

    .reindex(range(1, 13), fill_value=0)

    .reset_index(name="count")

)

monthly["month_name"] = pd.to_datetime(

    monthly["month"],

    format="%m"

).dt.strftime("%B")


fig_monthly = px.bar(

    monthly,

    x="month_name",

    y="count",

    title="Earthquake Frequency by Month"

)

st.plotly_chart(
    fig_monthly,
    use_container_width=True
)


# ------------------------------------------------------------
# 14. DBSCAN CLUSTER ANALYSIS
# ------------------------------------------------------------

st.divider()

st.subheader("🧠 Geographical Clustering Analysis")

st.write(
    """
    DBSCAN identifies groups of earthquake epicentres
    based on geographical distance.

    Cluster -1 represents events not assigned to
    a sufficiently dense cluster.
    """
)


if not clusters.empty:

    if "cluster" in filtered_df.columns:

        filtered_clusters = filtered_df[
            filtered_df["cluster"] != -1
        ]

        cluster_counts = (

            filtered_clusters.groupby("cluster")

            .size()

            .reset_index(name="earthquake_count")

        )

        if not cluster_counts.empty:

            fig_clusters = px.bar(

                cluster_counts,

                x="cluster",

                y="earthquake_count",

                title="Earthquakes per Geographical Cluster"

            )

            st.plotly_chart(

                fig_clusters,

                use_container_width=True

            )

        else:

            st.info(
                "No DBSCAN clusters found for the selected filters."
            )


    st.subheader("Cluster Summary")

    st.dataframe(

        clusters,

        use_container_width=True

    )

else:

    st.info("Cluster summary has not been generated yet.")


# ------------------------------------------------------------
# 15. EARTHQUAKE RECORDS
# ------------------------------------------------------------

st.divider()

st.subheader("📋 Earthquake Records")

display_columns = [

    "time",

    "place",

    "magnitude",

    "depth",

    "latitude",

    "longitude",

    "magnitude_category",

    "cluster"

]

display_columns = [

    column for column in display_columns

    if column in filtered_df.columns

]

st.dataframe(

    filtered_df[display_columns]

    .sort_values("time", ascending=False)

    .head(500),

    use_container_width=True

)


# ------------------------------------------------------------
# 16. DOWNLOAD FILTERED DATA
# ------------------------------------------------------------

st.subheader("Download Analysis Data")

csv = filtered_df.to_csv(
    index=False
).encode("utf-8")

st.download_button(

    label="Download Filtered Earthquake Data",

    data=csv,

    file_name="himalayan_earthquake_analysis.csv",

    mime="text/csv"

)


# ------------------------------------------------------------
# 17. FOOTER
# ------------------------------------------------------------

st.divider()

st.caption(

    """
    Data Source: United States Geological Survey (USGS)
    Earthquake Catalog / ComCat.

    This project analyses historical earthquake observations.
    It is not an earthquake early-warning or prediction system.
    """

)

st.success("Earthquake analysis dashboard loaded successfully!")
