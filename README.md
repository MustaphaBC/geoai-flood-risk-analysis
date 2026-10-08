# GeoAI Flood-Risk Analysis

An interactive Streamlit application for exploring building-level flood risk from geospatial layers. The interface loads building, river, road, and elevation data, trains a Random Forest classifier, displays risk summaries, and exports maps or geospatial results.

## Features

- Upload Shapefile components for buildings, rivers, roads, and elevation data.
- Generate sample data or load compatible layers from `data/`.
- Configure model size and train/test split before analysis.
- View interactive Folium maps and export GeoJSON or Shapefile results.

## Requirements

Python 3.9 or later and the geospatial Python stack are required:

```bash
python -m pip install streamlit geopandas pandas numpy matplotlib folium streamlit-folium scikit-learn shapely
```

## Run

From the project root:

```bash
streamlit run geoai6.py
```

Use the sample-data option in the interface or upload compatible geospatial layers. Shapefile datasets and generated reports/exports are excluded from the public repository.
