
# Himalayan Earthquake Intelligence System

An end-to-end Data Science project that analyses 20 years of historical earthquake data across the Himalayan region using real-world data from the USGS Earthquake Catalog.

## About the Project

The aim of this project is to understand historical earthquake activity in the Himalayan region by analysing earthquake frequency, magnitude, depth and geographical locations.

We collect real earthquake records, process the data, perform statistical analysis and use Machine Learning to identify geographical clusters of earthquake activity.

The results are presented through interactive visualizations and a Streamlit dashboard.

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

## How to Run

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

## Disclaimer

This project analyses historical earthquake activity and identifies geographical patterns. It does not predict future earthquakes or provide official seismic hazard assessments.

## Future Scope

- Improve geographical boundary accuracy.
- Analyse earthquake patterns across different time periods.
- Experiment with different clustering parameters.
- Incorporate tectonic plate and fault-line information.
- Improve the interactive dashboard.