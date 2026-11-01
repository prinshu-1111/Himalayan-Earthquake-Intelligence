# Himalayan Earthquake Intelligence System

An end-to-end Data Science portfolio project that explores historical earthquake
activity across the Himalayan region using USGS catalogue data.

[Open the live dashboard](https://himalayan-earthquake-intelligence.vercel.app)

## What This Project Does

The project collects earthquake observations, validates and cleans them,
performs exploratory analysis, identifies geographic clusters with Haversine
DBSCAN, evaluates clustering configurations, and estimates Gutenberg-Richter
frequency-magnitude behaviour.

It includes both:

- an interactive Streamlit application for detailed analysis and downloads;
- a framework-free static dashboard deployed on Vercel.

> This project supports historical, descriptive analysis only. It does not
> predict earthquakes, issue alerts, or replace official seismic-hazard products.

## Live Demo

**https://himalayan-earthquake-intelligence.vercel.app**

## Main Features

- USGS API data collection and monthly catalogue retrieval
- Data-quality checks, duplicate detection, type validation, and range checks
- Missing-value handling and statistical outlier investigation
- Time features for year, month, day, hour, and weekday
- Annual, monthly, magnitude, depth, regional, and correlation analysis
- Interactive maps and magnitude/depth visualizations
- Haversine DBSCAN clustering in kilometres
- Comparison of DBSCAN epsilon and minimum-sample parameters
- Silhouette, Davies-Bouldin, noise, and cluster-size evaluation
- Gutenberg-Richter analysis and catalogue limitation reporting
- Downloadable processed datasets and analytical outputs

## Dataset

The source is the [USGS Earthquake Catalog](https://earthquake.usgs.gov/fdsnws/event/1/).

- Geographic region: 25-38 degrees N, 70-100 degrees E
- Study catalogue: 2006-11-01 to 2026-11-01
- Minimum requested magnitude: 2.5
- Raw observations: 8,153 records
- Required fields: event ID, UTC time, latitude, longitude, depth, and magnitude

Catalogue completeness varies with time, geography, reporting practices, and
instrument sensitivity. These limitations are reported rather than hidden.

## Technology Stack

- Python 3.11+
- Pandas and NumPy
- Matplotlib, Seaborn, Plotly, and Folium
- Scikit-learn
- Streamlit
- USGS FDSN Event API
- GitHub, Vercel, and static HTML/CSS/JavaScript

## Project Architecture

```text
Himalayan-Earthquake-Intelligence/
├── analysis/
│   ├── preprocessing.py       # Cleaning, validation, quality report
│   ├── eda.py                 # Exploratory analysis and figures
│   ├── clustering.py          # Haversine DBSCAN implementation
│   ├── evaluation.py          # Parameter comparison and model metrics
│   └── statistical_analysis.py # Gutenberg-Richter calculations
├── data/                      # Deployment-ready data copies
├── public/                    # Static dashboard assets and data
├── assets/                    # Static CSS and JavaScript
├── outputs/                   # Generated datasets and visualizations
├── app.py                     # Streamlit dashboard
├── main.py                    # End-to-end analysis pipeline
├── index.html                 # Static Vercel dashboard
└── vercel.json                # Static deployment configuration
```

## Run Locally

### 1. Create a virtual environment

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 2. Run the complete pipeline

```bash
python main.py
```

This processes the catalogue, creates exploratory outputs, evaluates DBSCAN
parameters, and updates the dashboard-ready datasets.

### 3. Launch Streamlit

```bash
python -m streamlit run app.py
```

### 4. Preview the static dashboard

```bash
python -m http.server 8080
```

Open `http://localhost:8080`.

## Machine Learning Approach

DBSCAN is applied to latitude and longitude after converting them to radians.
The model uses the Haversine metric and converts each epsilon value from
kilometres to radians using Earth's mean radius:

```text
epsilon_radians = epsilon_kilometres / 6371.0088
```

The pipeline compares 25 km, 50 km, and 75 km epsilon values with 5, 10, and
20 minimum samples. Evaluation records cluster count, noise percentage,
cluster sizes, execution time, Silhouette Score, and Davies-Bouldin Index.
Silhouette and Davies-Bouldin are reported only when mathematically valid.

These metrics assess geometric separation; they do not prove seismic
significance, future earthquake risk, or fault activity.

## Generated Outputs

| File | Purpose |
|---|---|
| `outputs/processed_earthquakes.csv` | Clean analysis dataset |
| `outputs/preprocessing_summary.csv` | Missing values and record handling |
| `outputs/depth_outliers.csv` | Investigated depth outliers |
| `outputs/model_evaluation.csv` | DBSCAN parameter comparison |
| `outputs/cluster_summary.csv` | Cluster statistics |
| `outputs/gutenberg_richter.csv` | Frequency-magnitude data |
| `outputs/gutenberg_richter_summary.csv` | Gutenberg-Richter estimate |

## Deployment

The static dashboard is deployed on Vercel through `vercel.json`.

The Streamlit dashboard can be deployed with Streamlit Community Cloud:

1. Push the repository to GitHub.
2. Create a Streamlit Community Cloud application.
3. Select `app.py` as the entry point.
4. Use `requirements.txt` for dependencies.

Generated datasets are included so deployment does not require API credentials
or a manual pipeline run.

## Limitations

- USGS records are observations, not a complete catalogue of every event.
- Detection sensitivity has changed over time.
- The rectangular study boundary is analytical, not a tectonic boundary.
- Heterogeneous magnitude types and measurement uncertainty affect comparisons.
- Gutenberg-Richter estimates depend on catalogue completeness and the selected
  magnitude threshold.
- Correlation does not establish causation.
- Geographical clusters describe historical spatial concentration, not future
  locations or hazard forecasts.

## Recruiter and Interview Guide

For a detailed explanation of the problem, architecture, technical decisions,
project results, limitations, and interview talking points, see:

[RECRUITER_PROJECT_GUIDE.md](RECRUITER_PROJECT_GUIDE.md)
