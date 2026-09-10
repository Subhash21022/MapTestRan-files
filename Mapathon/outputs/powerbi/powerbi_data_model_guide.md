# Power BI Data Model & Dashboard Setup Guide

## Overview
This document provides step-by-step instructions for importing the Chennai Metro Phase 2 Power BI asset package into **Power BI Desktop**, creating relationships, applying DAX measures, and arranging visuals to match the 2024 Mapathon poster template.

---

## Data Files Included (`outputs/powerbi/`)
1. **`chennai_powerbi_stations.csv`**: Primary facts & dimension table containing all 134 metro stations with 5D spatial attributes, predictions, confidence intervals, and TOD action classifications.
2. **`chennai_powerbi_corridors.csv`**: Aggregated metrics for Phase 2 corridors (Purple Line, Red Line, Yellow Line).
3. **`chennai_powerbi_elasticities.csv`**: Direct Demand Model feature importance & elasticity coefficients.
4. **`chennai_powerbi_model_performance.csv`**: ML model cross-validation benchmarks ($R^2$, MAE, RMSE, MAPE, Spearman).
5. **`chennai_powerbi_tod_matrix.csv`**: Summary of TOD Strategic Action Quadrants.
6. **`dax_measures.dax`**: Complete suite of production DAX measures.

---

## Power BI Desktop Integration Steps

### Step 1: Import Datasets
1. Open **Power BI Desktop**.
2. Click **Get Data** > **Text/CSV**.
3. Import all 5 CSV files from `outputs/powerbi/`.
4. Ensure data types are correctly assigned:
   - `Station_ID`, `Station_Name`, `Phase`, `Corridor_Line`, `TOD_Quadrant`, `Strategic_Action`: **Text**
   - `Predicted_Boardings`, `Lower_CI_95`, `Upper_CI_95`, `Population_800m`, `Bus_Stops_500m`: **Whole Number**
   - `Density_Pop_per_km2`, `PNR_Walkability_Ratio`, `Access_Index`, `Potential_Index`: **Decimal Number**
   - `Is_Top15_Phase2`, `Is_Intermodal_Hub`, `Is_Interchange`: **Whole Number / True-False**

### Step 2: Establish Model Relationships
In the **Model View**, set up the following relationships:
- `chennai_powerbi_corridors[Corridor_Line]` **(1)** $ightarrow$ **(*)** `chennai_powerbi_stations[Corridor_Line]`
- `chennai_powerbi_tod_matrix[Strategic_Action]` **(1)** $ightarrow$ **(*)** `chennai_powerbi_stations[Strategic_Action]`

### Step 3: Add DAX Measures
1. In the **Report View**, click **New Measure**.
2. Copy and paste measures from `dax_measures.dax`.
3. Group measures in a dedicated table named `_Measures`.

---

## Recommended Dashboard Layout (Poster Alignment)

| Section | Visual Type | Fields / DAX Measures | Purpose |
|---|---|---|---|
| **Header KPIs** | Card Visuals | `Phase 2 Total Boardings`, `Top 15 Boardings Share Pct`, `Avg PNR Walkability Ratio`, `Feeder Bus Priority Count` | Instant executive stats header |
| **Corridor Analysis** | Clustered Bar Chart | Axis: `Corridor_Line`<br>Values: `Total Predicted Boardings`, `Average Boardings per Station` | Compare Purple, Red, and Yellow lines |
| **5D Model Driver** | Horizontal Bar Chart | Axis: `Feature_Label`<br>Values: `Elasticity_Coeff`<br>Color: `Impact_Direction` | Show key ridership drivers (Bus feeder, Pop, Centrality) |
| **TOD Quadrants** | Scatter Chart | X-Axis: `Access_Index`<br>Y-Axis: `Potential_Index`<br>Legend: `Strategic_Action` | Identify TOD upzoning vs feeder priority candidate stations |
| **Top 15 Stations** | Table / Rank Card | Columns: `Phase2_Ridership_Rank`, `Station_Name`, `Corridor_Line`, `Predicted_Boardings` | Top performing station leaderboard |
