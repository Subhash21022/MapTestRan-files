# Project Context for QGIS Mapping

## Overview
This project predicts ridership potential for the 128 under-construction stations of Chennai Metro Phase 2. It uses a **transfer-learned direct demand model** trained on Bengaluru's Namma Metro data. The model is built using the **5D framework** (Density, Diversity, Design, Destination accessibility, Distance to transit). 

The pipeline generates spatial features and final ridership predictions, which need to be visualised and styled in QGIS to produce planning maps (e.g., feeder-bus priorities, TOD upzoning candidates).

## Coordinate Reference Systems (CRS)
Always ensure layers are correctly projected before running spatial operations or styling in QGIS:
*   **Storage/Web Display**: EPSG:4326 (WGS 84)
*   **Chennai (Prediction Target)**: EPSG:32644 (UTM Zone 44N)
*   **Bengaluru (Training Data)**: EPSG:32643 (UTM Zone 43N)

## Key Directories
*   `data/raw/`: Raw input data (OSM, Bhuvan LULC, GHS-POP, etc.).
*   `data/processed/`: Analysis-ready spatial layers (GeoPackages) and feature tables (CSV). **Use these for QGIS mapping.**
*   `qgis/`: Where the QGIS project file (`.qgz`) and print layouts should be saved.
*   `outputs/maps/`: Rendered map outputs.

## Core Spatial Layers (in `data/processed/`)
When mapping in QGIS, you will primarily work with these generated files (prefixed with the city name, e.g., `chennai_` or `bengaluru_`):

1.  **`{city}_stations.gpkg`**: Point layer of the metro stations.
    *   *Styling*: Often styled by line colour, operational phase, or predicted ridership (proportional symbols or graduated colours).
2.  **`{city}_catchments_buffer.gpkg`**: Euclidean buffers around stations.
    *   Contains 400m (core), 800m (standard), and 1200m (extended) radii.
    *   *Styling*: Useful for showing basic catchment areas and clipping raster data (like population or land use).
3.  **`{city}_catchments_network.gpkg`**: Network service areas (walksheds) computed along the street network.
    *   *Styling*: Often compared against the Euclidean buffer to illustrate the Pedestrian Network Ratio (PNR) / severance.
4.  **`{city}_features.csv`**: The assembled 5D features for each station.
    *   *Usage*: Join this CSV to the `stations.gpkg` in QGIS (using station name or ID) to map specific variables like `intersection_density`, `landuse_entropy`, or `pnr`.

## Predictions & Scenarios
*   Final predictions and scenarios are saved as CSVs in `outputs/tables/` (e.g., `{city}_predictions.csv`). 
*   To map the final ridership predictions, join this prediction CSV to the `stations.gpkg` layer in QGIS.

## QGIS Mapping Goals
When assisting with QGIS, focus on:
*   Joining the generated CSV feature tables and prediction outputs to the spatial geometries (stations and catchments).
*   Applying effective symbology (e.g., graduated symbols for ridership volume, bivariate choropleths if comparing two features).
*   Setting up clean, professional Print Layouts with legends, scale bars, and standard cartographic elements.
