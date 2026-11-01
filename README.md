
# Himalayan Earthquake Intelligence System

An end-to-end, portfolio-ready Data Science project for exploring historical
earthquake activity across the Himalayan region. The project combines USGS
catalogue data, robust preprocessing, exploratory analysis, Haversine DBSCAN
clustering, model evaluation, Gutenberg-Richter analysis, and an interactive
Streamlit dashboard.

> The system performs historical observational analysis only. It does not
> predict earthquakes, issue alerts, or replace official seismic-hazard products.

## Live Demo

The deployed static dashboard is available at:

**https://himalayan-earthquake-intelligence.vercel.app**

## Problem Statement

Earthquake observations are spatially, temporally, and observationally uneven.
This project examines that catalogue data to answer descriptive questions about
frequency, magnitude, depth, location, temporal patterns, and geographical
clustering without claiming that historical correlations establish causation.

## Dataset

The raw observations are collected from the
[USGS Earthquake Catalog](https://earthquake.usgs.gov/earthquakes/search/).

- Geographic study region: 25°N to 38°N and 70°E to 100°E
- Observation period: 2006-11-01 to 2026-11-01
- Minimum requested magnitude: 2.5
- Required fields: event ID, UTC time, magnitude, latitude, longitude, depth
- Catalogue duplicates and records outside the defined region or requested value
  ranges are checked before analysis.

## What Happens in This Project?

1. **Data Collection:** We collect earthquake records from the USGS API for the selected Himalayan study region, covering November 2006 to November 2026.

2. **Data Cleaning:** We remove duplicate and incomplete records, correct data types and filter earthquakes according to our geographical boundaries and magnitude threshold.

3. **Data Analysis:** We analyse earthquake frequency, magnitude distribution, depth and yearly and monthly activity.

4. **Feature Engineering:** We create additional features such as year, month, magnitude category, depth category and estimated earthquake energy.

5. **Machine Learning:** We use the DBSCAN clustering algorithm to identify geographical groups of earthquake epicentres.

6. **Geographical Mapping:** We use Folium to create interactive earthquake maps showing earthquake locations, heatmaps and identified clusters.

7. **Interactive Dashboard:** We use Streamlit to bring the analysis together in a dashboard where users can explore earthquake data, view charts, apply filters and download records.

## Technologies Used

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn (DBSCAN)
- Folium
- Plotly
- Streamlit
- USGS Earthquake API

## Initial Results

Our first successful data collection and analysis produced:

- **8,153** earthquake records
- **4.35** average earthquake magnitude
- **7.8** maximum recorded magnitude
- **78.96 km** average earthquake depth
- **31** geographical clusters identified using DBSCAN

## Project Structure

```text
Himalayan-Earthquake-Intelligence/
│
├── data/
│   └── earthquakes.csv
│
├── outputs/
│   ├── magnitude_distribution.png
│   ├── yearly_earthquakes.png
│   ├── depth_analysis.png
│   ├── magnitude_depth.png
│   ├── monthly_activity.png
│   ├── earthquake_heatmap.png
│   ├── cluster_analysis.png
│   ├── cluster_summary.csv
│   ├── earthquake_map.html
│   └── processed_earthquakes.csv
│
├── models/
│   └── dbscan_model.pkl
│
├── main.py
├── app.py
├── requirements.txt
└── README.md
```

## Run the dashboard locally

Serve the repository from a local HTTP server:

```bash
python -m http.server 8080
```

Then open `http://localhost:8080`.

## Deploy to Vercel

The website is framework-free and deploys directly from the repository root:

1. Push the project to GitHub.
2. Import the repository in Vercel.
3. Leave the build command and output directory empty.
4. Deploy.

The checked-in `vercel.json` handles routing as a static site. No environment variables are required.
The current production deployment is available at
[himalayan-earthquake-intelligence.vercel.app](https://himalayan-earthquake-intelligence.vercel.app).

## Run the analysis pipeline

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Run the data analysis pipeline:

```bash
python main.py
```

Launch the interactive dashboard:

```bash
python -m streamlit run app.py
```

## Data Source

[USGS Earthquake Catalog](https://earthquake.usgs.gov/fdsnws/event/1/)

## Data Science Methodology

1. Download USGS observations by monthly interval.
2. Preserve the raw catalogue as the source of truth.
3. Coerce and validate numeric/date fields.
4. Remove event-ID duplicates and records outside supported ranges.
5. Filter records to the Himalayan study region.
6. Report missing values, removed-record reasons, and statistical depth outliers.
7. Engineer UTC year, month, day, hour, weekday, depth, magnitude, and energy features.
8. Generate annual, monthly, magnitude, depth, spatial, and correlation analysis.
9. Estimate Gutenberg-Richter frequency-magnitude behavior.

## Machine Learning Methodology

DBSCAN receives latitude/longitude values converted to radians. The configured
distance is converted from kilometres to radians using the mean Earth radius:

`eps_radians = eps_km / 6371.0088`

This allows the Scikit-learn `haversine` metric to operate in kilometres in the
model's parameter space without using incorrect planar coordinate distance.

The pipeline compares several epsilon/minimum-samples combinations and records:

- Number of clusters
- Noise point count and percentage
- Cluster-size distribution
- Silhouette Score
- Davies-Bouldin Index
- Execution time

Silhouette and Davies-Bouldin are calculated only with at least two clusters and
no noise labels in the evaluated sample. Both are geometric measures: they do
not independently establish spatial correctness, seismic significance, or
hazard. Small clusters, mixed noise, catalogue incompleteness, regional boundary
effects, and correlated spatial samples can distort these scores.

## Output Files

The pipeline generates:

| Artifact | Purpose |
|---|---|
| `outputs/preprocessing_summary.csv` | Record retention and removal counts |
| `outputs/depth_outliers.csv` | Statistical depth outliers for investigation |
| `outputs/processed_earthquakes.csv` | Clean modelling and dashboard dataset |
| `outputs/model_evaluation.csv` | DBSCAN parameter comparison |
| `outputs/cluster_summary.csv` | Spatial cluster statistics |
| `outputs/gutenberg_richter.csv` | Magnitude-frequency relation |
| `outputs/gutenberg_richter_summary.csv` | Estimated b-value and assumptions |
| `outputs/correlation_matrix.csv` | Numeric EDA correlation matrix |
| `outputs/*.png` | Static analytical figures |

The static deployment receives synchronized copies under `data/` and
`public/data/`, so the dashboards can run without invoking preprocessing.

## Installation

Use Python 3.11 or newer:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run Locally

Run the complete collection/analysis pipeline:

```bash
python main.py
```

Launch Streamlit:

```bash
streamlit run app.py
```

The analysis pipeline uses the existing downloaded catalogue when available.
Delete the raw dataset deliberately if a fresh USGS download is required.

## Deploy to Streamlit Community Cloud

1. Commit `app.py`, `main.py`, `analysis/`, `requirements.txt`, and deployable
   generated datasets.
2. Push the repository to GitHub.
3. Create a new Streamlit Community Cloud application.
4. Select this repository and branch.
5. Set the main file to `app.py`.
6. Deploy.

The application loads `outputs/processed_earthquakes.csv` first and falls back to
the repository copies under `data/` or `public/data/`. It does not require an
API key or manual preprocessing in Streamlit Cloud.

## Optional Static Deployment

The existing framework-free Vercel site remains available through
`index.html` and the `assets/` directory. Sync generated CSV files before
redeploying that interface.

## Limitations

- USGS catalogues are observational records, not complete observations of all
  earthquakes throughout the whole period.
- Detection sensitivity has changed over time.
- The rectangular region is an analytical convenience, not a tectonic boundary.
- Catalogue magnitude types and measurement uncertainty can vary.
- Gutenberg-Richter estimates depend strongly on completeness magnitude and
  data selection.
- DBSCAN clustering does not predict future seismic activity.
- Geographical density is affected by population, reporting, and cataloguing.

## Disclaimer

This project analyses historical earthquake activity and identifies geographical patterns. It does not predict future earthquakes or provide official seismic hazard assessments.

## Future Scope

- Improve geographical boundary accuracy.
- Analyse earthquake patterns across different time periods.
- Experiment with different clustering parameters.
- Incorporate tectonic plate and fault-line information.
- Improve the interactive dashboard.