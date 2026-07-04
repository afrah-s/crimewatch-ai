# CrimeWatch AI Prototype

AI-Driven Crime Analytics & Visualization Platform for Challenge 02.

## What this prototype covers

- Interactive crime dashboard
- District-level drilldown
- Trend alerts and anomaly detection
- Police station/outpost geospatial map
- Explainable risk scoring
- Natural-language search bar that maps plain English to safe dashboard filters
- Demo layer for network/link analysis and repeat offender tracking

## How to run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Data files used

- `ka-district-wise-2025(1).csv`
- `crime_review_for_the_month_of_december_2025_9(1).csv`
- `police_stations_processed.csv` converted from uploaded KML files
- Demo-only anonymized network data generated inside the app

## Important note

Public datasets uploaded here are aggregated. Real criminal network and repeat offender tracking need case-level records with `case_id` and anonymized `offender_id` / `accused_id` fields.
