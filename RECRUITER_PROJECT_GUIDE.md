# Himalayan Earthquake Intelligence: Recruiter and Interview Guide

This document is a personal speaking guide for portfolio reviews and Data
Science interviews. It explains the project in depth, including what you built,
why each decision was made, how to discuss results honestly, and which technical
questions you may be asked.

## 1. Project Summary

Himalayan Earthquake Intelligence is an end-to-end Data Science project built to
analyse historical earthquake observations in a defined Himalayan study region.
The project covers data acquisition, validation, preprocessing, exploratory
analysis, geospatial clustering, model evaluation, statistical analysis, and
interactive reporting.

The final system includes:

- a modular Python analysis pipeline;
- a Streamlit dashboard;
- a deployed static dashboard;
- generated analytical datasets and visualizations;
- a GitHub repository and Vercel deployment.

Use this concise summary in an interview:

> “I built an end-to-end historical earthquake analysis project using the USGS
> earthquake catalogue. The pipeline validates the catalogue, creates analytical
> features, performs EDA, evaluates Haversine DBSCAN configurations, estimates
> Gutenberg-Richter frequency-magnitude behavior, and presents results through
> Streamlit and a static Vercel dashboard. The project is descriptive rather than
> predictive, which lets me discuss uncertainty and limitations honestly.”

## 2. Problem Statement

Earthquake records are distributed geographically and over time, while catalogue
coverage can vary because of instrument sensitivity, reporting practices, and
regional monitoring networks. The project investigates where historical activity
is concentrated, how events vary by magnitude and depth, whether activity has
temporal patterns, and which spatial groups emerge from the catalogue.

The system does not attempt to predict the time, location, or magnitude of a
future earthquake. It should be described as an observational intelligence and
pattern-analysis system, not as a forecasting model.

## 3. Business and Portfolio Value

The project demonstrates more than dashboard creation. It shows ability to:

- work with a real external catalogue;
- inspect data-quality problems before modeling;
- preserve raw and processed data separately;
- apply meaningful domain-specific distance calculations;
- compare model configurations instead of choosing an arbitrary model;
- separate valid metrics from unsupported conclusions;
- deliver the same analytical assets through two interfaces;
- document limitations and deployment decisions.

## 4. Dataset and Scope

Data comes from the USGS FDSN Event API. A rectangular boundary is used for the
study:

- latitude: 25 degrees N to 38 degrees N;
- longitude: 70 degrees E to 100 degrees E;
- period: 2006-11-01 to 2026-11-01;
- minimum requested magnitude: 2.5.

The cleaned catalogue contains 8,153 records. The region and date range should
be presented as explicit analytical filters, not as complete tectonic or national
boundaries.

Important required fields are:

- event identifier;
- UTC event time;
- latitude;
- longitude;
- depth;
- magnitude.

Optional fields, such as “felt” responses, are handled differently from required
seismic measurements.

## 5. Data Collection and Pipeline

`main.py` provides the complete pipeline. It requests USGS data in bounded time
intervals, loads the catalogue when available, then delegates work to modules in
`analysis/`.

The pipeline flow is:

1. collect or load the raw dataset;
2. detect and audit data-quality issues;
3. normalize data types and time fields;
4. detect duplicate event identifiers;
5. validate coordinates and required numeric fields;
6. engineer calendar and derived features;
7. perform EDA;
8. compare Haversine DBSCAN configurations;
9. calculate Gutenberg-Richter summaries;
10. generate dashboard-ready datasets and visualizations.

## 6. Data Cleaning and Preprocessing

The preprocessing module under `analysis/preprocessing.py` validates:

- numeric latitude and longitude;
- geographic coordinate bounds;
- valid magnitude and depth ranges;
- required versus optional missing values;
- duplicate event identifiers;
- UTC date parsing and derived calendar features.

Outliers are not automatically deleted. Statistical depth outliers are reported
for investigation because a very large depth may be genuine seismic activity or
an incorrect observation. Magnitude and depth range rules are applied only where
records are clearly invalid or inconsistent.

Explain this distinction:

> “I deliberately separated suspicious observations from genuine extreme values.
> Extreme earthquakes are scientifically important and should not be removed
> simply because they are statistical outliers.”

## 7. Exploratory Data Analysis

`analysis/eda.py` creates temporal, distribution, relationship, and spatial
outputs. Analysis includes:

- earthquake frequency by year and month;
- magnitude distribution;
- depth distribution;
- magnitude-versus-depth relationship;
- correlation analysis;
- geographic distribution;
- regional earthquake frequency;
- event categories based on magnitude;
- temporal trends in catalogued activity.

These outputs help show whether apparent temporal changes are real seismic
changes or effects of catalogue reporting and detection sensitivity.

## 8. Why Haversine Distance Matters

Latitude and longitude cannot be clustered safely as plain Cartesian coordinates.
Degrees of longitude do not represent equal physical distances at every latitude,
and Euclidean calculations would distort the geography.

The project therefore:

- converts coordinates to radians;
- uses Scikit-learn's Haversine metric;
- converts epsilon from kilometres to radians using Earth's radius:

```text
eps_radians = eps_km / 6371.0088
```

In ordinary language:

> “I use geographic distance rather than treating latitude and longitude like
> simple screen pixels. The epsilon becomes physically meaningful in kilometres.”

## 9. DBSCAN Methodology

DBSCAN is used because earthquake epicentres may form irregular groups and the
dataset may contain noise. It does not require a preselected number of clusters.

The pipeline evaluates combinations of:

- epsilon: 25 km, 50 km, and 75 km;
- minimum samples: 5, 10, and 20.

The reference result of 50 km with `min_samples=5` produces:

- 31 clusters;
- 151 noise points;
- approximately 1.85 percent noise;
- one dominant cluster containing 5,574 observations.

The implementation also reports the complete parameter comparison. The dominant
cluster demonstrates that a low number of clusters does not automatically mean a
balanced or useful clustering result.

If asked why one configuration was selected, explain:

> “I selected it as an interpretable reference configuration, not as a claim that
> it is the uniquely best tectonic segmentation. The parameter comparison is
> intentionally retained so spatial density and evaluation trade-offs remain
> visible.”

## 10. Model Evaluation

`analysis/evaluation.py` records:

- execution time;
- number of clusters;
- noise points and noise percentage;
- largest, smallest, and median cluster size;
- Silhouette Score when mathematically valid;
- Davies-Bouldin Index when applicable.

For the 50 km and `min_samples=5` reference configuration, the current output
contains a negative Silhouette Score and a Davies-Bouldin score of about 0.83.
These are geometric calculations. They should not be quoted as evidence of
real-world seismic correctness.

Correct interview explanation:

> “The metrics help compare parameter configurations and identify geometric
> overlap or fragmentation. They are not independent proof of seismological
> validity. I also report noise and cluster imbalance, which are often more
> important than a single aggregate score.”

## 11. Gutenberg-Richter Analysis

`analysis/statistical_analysis.py` analyses cumulative earthquake counts above
magnitude thresholds. Gutenberg-Richter relationships generally describe how
frequency decreases as magnitude increases.

A simple catalogue estimate can be unstable when the selected threshold and
completeness magnitude do not match the data characteristics. The pipeline
therefore stores the threshold, count, bin width, and estimate rather than
presenting the b-value as a guaranteed physical parameter.

Say this explicitly:

> “The resulting estimate depends on catalogue completeness, magnitude binning,
> and the selected minimum magnitude. It is a descriptive estimate, not a
> definitive regional b-value.”

## 12. Dashboard Architecture

The Streamlit application in `app.py` provides detailed portfolio sections:

- project overview;
- dataset quality;
- exploratory analysis;
- geospatial maps;
- DBSCAN clustering;
- model evaluation;
- Gutenberg-Richter output;
- findings and downloadable records.

The static dashboard uses root-level HTML, CSS, and JavaScript plus synchronized
copies under `data/` and `public/data/`. Vercel serves it directly, so production
does not run Python preprocessing.

This separation makes the repository resilient to hosting differences while
preserving the existing Streamlit functionality.

## 13. Deployment Decisions

The project has two deployment surfaces:

- Streamlit: preserves interactive Python analysis and filters;
- Vercel: serves the framework-free static dashboard quickly without Python.

Deployment data is generated ahead of time and committed where appropriate.
The dashboard therefore does not require API keys or manual processing after
deployment.

## 14. Key Projects Files

| File | Responsibility |
|---|---|
| `main.py` | Runs the complete pipeline |
| `app.py` | Streamlit dashboard |
| `analysis/preprocessing.py` | Cleaning and data-quality summary |
| `analysis/eda.py` | Exploratory analysis and figures |
| `analysis/clustering.py` | Haversine DBSCAN |
| `analysis/evaluation.py` | Parameter and metric comparison |
| `analysis/statistical_analysis.py` | Gutenberg-Richter analysis |
| `vercel.json` | Static deployment routing |

## 15. Results to Mention

Current repository figures that are safe to state include:

- 8,153 catalogue records in the processed dataset;
- a magnitude range of 2.6 to 7.8;
- 31 clusters in the 50 km, `min_samples=5` reference configuration;
- 151 noise points, or approximately 1.85 percent;
- 9 DBSCAN parameter combinations evaluated;
- 23 engineered columns in the processed dataset;
- 0 duplicated event IDs and no invalid required-coordinate records in the
  audited raw dataset.

Do not represent these as perfect historical coverage or prove that the
recorded maximum is the maximum event that could ever occur in the region.

## 16. Limitations to State

Be prepared to explain:

- USGS data are not complete for all places and all historical dates;
- detection thresholds change over time;
- spatial catalogue coverage can reflect monitoring networks;
- a rectangular boundary is not a geological boundary;
- clusters are sensitive to epsilon and minimum-samples choices;
- DBSCAN labels can be dominated by dense regions;
- noise is not automatically an incorrect earthquake;
- temporal trends can be affected by reporting;
- Gutenberg-Richter estimates depend on completeness;
- correlation does not establish causation.

## 17. Challenges and Solutions

### Challenge: Different Earthquake Counts Across Years

Solution: report catalogue coverage and detection sensitivity instead of
automatically attributing every change to geological activity.

### Challenge: Geographic Distance

Solution: use radians, Haversine distance, and kilometre-based epsilon values.

### Challenge: Cluster Evaluation

Solution: compare multiple parameter combinations and report noise, cluster
sizes, execution time, and valid geometric metrics.

### Challenge: Extreme Values

Solution: investigate statistical outliers without deleting them automatically.

### Challenge: Deployment Without Reprocessing

Solution: generate deployment-ready CSV files and synchronize them to the static
site.

### Challenge: Framework Detection

Solution: add explicit static `builds` configuration in `vercel.json` so Python
files do not cause Vercel to infer a Python handler application.

## 18. Suggested Technical Answers

### “Why did you use DBSCAN?”

DBSCAN identifies dense, irregular spatial regions and can classify sparse
events as noise without requiring the number of clusters in advance.

### “Why Haversine?”

Because latitude and longitude are spherical geographic coordinates and planar
distance would distort real distance, especially at different latitudes.

### “How do you choose the epsilon?”

I test physically interpretable kilometre values, compare cluster counts, noise
and cluster size, and report evaluation metrics. Epsilon is not chosen solely
to maximize the metric or cluster count.

### “Why include silhouette if noise exists?”

I calculate it only where the labels support a mathematically valid score and
I exclude noise appropriately. It is a geometric quality check, not proof of
seismic meaning.

### “What does the project predict?”

It does not predict earthquakes. It describes recorded historical activity and
identifies spatial concentration.

### “How would you improve the project?”

Possible improvements include:

- fault-line and tectonic-plate features;
- completeness-magnitude modelling;
- uncertainty intervals for Gutenberg-Richter estimates;
- temporal catalogue bias corrections;
- more rigorous spatial cross-validation;
- automated tests and data-quality tests;
- CI/CD validation before deployment.

## 19. Interview Story Structure

Use the STAR method:

1. **Situation:** Real earthquake data are incomplete and spatially irregular.
2. **Task:** Build an end-to-end descriptive analysis system.
3. **Action:** Validate the source, modularize preprocessing and analysis, use
   Haversine DBSCAN, compare configurations, and build dashboards.
4. **Result:** Produce a deployed portfolio application with reproducible
   outputs and transparent evaluation limitations.

## 20. Closing Statement

A strong closing statement is:

> “This project shows my ability to take a public dataset from ingestion through
> cleaning, EDA, spatial modelling, evaluation, deployment, and documentation.
> More importantly, I can explain the technical decisions and the limitations
> rather than presenting descriptive clustering as prediction.”
