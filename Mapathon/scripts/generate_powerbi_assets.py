"""Generate Power BI Statistical Assets & Graphics for Chennai Metro Phase 2.

This script processes prediction and 5D spatial feature tables to output:
1. Clean Power BI optimized CSV datasets (stations, corridors, elasticities, model performance, TOD quadrants).
2. Production DAX measures file (dax_measures.dax).
3. Complete Power BI Data Model Integration Guide (powerbi_data_model_guide.md).
4. High-impact statistical visual panels ready for Mapathon poster insertion.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
TABLES_DIR = ROOT / "outputs" / "tables"
PBI_DIR = ROOT / "outputs" / "powerbi"
FIGURES_PBI = ROOT / "outputs" / "figures" / "powerbi"

PBI_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_PBI.mkdir(parents=True, exist_ok=True)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pred_path = TABLES_DIR / "chennai_predictions_new.csv"
    rec_path = TABLES_DIR / "chennai_station_recommendations.csv"
    perf_path = TABLES_DIR / "model_performance.csv"
    elast_path = TABLES_DIR / "nb_elasticities.csv"

    predictions = pd.read_csv(pred_path)
    recommendations = pd.read_csv(rec_path)
    perf = pd.read_csv(perf_path)
    elasticities = pd.read_csv(elast_path)

    return predictions, recommendations, perf, elasticities


def build_powerbi_datasets(
    predictions: pd.DataFrame,
    recommendations: pd.DataFrame,
    perf: pd.DataFrame,
    elasticities: pd.DataFrame,
) -> None:
    # 1. Main Stations Dataset
    df = recommendations.copy()

    # Calculate Phase 2 ranks & Top 15 flags
    phase2_mask = df['phase'] == 'phase2'
    df['rank_phase2'] = np.nan
    df.loc[phase2_mask, 'rank_phase2'] = df.loc[phase2_mask, 'prediction'].rank(ascending=False, method='first')
    df['label_top15'] = (df['rank_phase2'] <= 15).astype(int)

    INTERMODAL_HUB_NAMES = {
        "Puratchi Thalaivar Dr. M.G. Ramachandran Central",
        "Chennai International Airport",
        "Puratchi Thalaivi Dr. J. Jayalalithaa CMBT",
        "CMBT",
        "Egmore",
        "Guindy",
        "St. Thomas Mount",
        "Arignar Anna Alandur",
        "Koyambedu",
        "Kilambakkam",
        "Thirumangalam",
        "Thirumayilai",
        "Mandaveli",
        "Adyar Depot",
        "Porur Junction",
        "Madhavaram Milk Colony",
    }
    df['is_intermodal_hub'] = df['name'].isin(INTERMODAL_HUB_NAMES).astype(int)

    action_map = {
        "Priority feeder bus + footpath investment": "Feeder Bus Priority",
        "TOD upzoning candidate; access exists, density does not": "Strategic TOD Upzoning",
        "Open early; demand is there and reachable": "High-Demand Early Open",
        "Right-size the station; defer ancillary spend": "Right-Size & Defer Spend",
    }
    df['clean_action'] = df['action'].map(action_map).fillna(df['action'])

    # Calculate 5D Z-Scores
    num_cols = [
        "population_density",
        "pnr",
        "poi_total",
        "bus_stops_500m",
        "dist_cbd_km",
        "intersection_density",
    ]
    for col in num_cols:
        if col in df.columns:
            mean_val = df[col].mean()
            std_val = df[col].std()
            df[f"z_{col}"] = (df[col] - mean_val) / (std_val if std_val != 0 else 1)

    # Rename & Format columns for Power BI
    pbi_stations = pd.DataFrame()
    pbi_stations["Station_ID"] = df["station_id"]
    pbi_stations["Station_Name"] = df["name"]
    pbi_stations["Phase"] = df["phase"].map({"phase1": "Phase 1", "phase2": "Phase 2"}).fillna(df["phase"])
    pbi_stations["Corridor_Line"] = df["Line"]
    pbi_stations["Layout_Type"] = df["Layout"]
    pbi_stations["Position_Confidence"] = df["confidence"]
    pbi_stations["Predicted_Boardings"] = df["prediction"].round(0)
    pbi_stations["Lower_CI_95"] = df["lower"].round(0)
    pbi_stations["Upper_CI_95"] = df["upper"].round(0)
    pbi_stations["Phase2_Ridership_Rank"] = df["rank_phase2"]
    pbi_stations["Is_Top15_Phase2"] = df["label_top15"]
    pbi_stations["Is_Intermodal_Hub"] = df["is_intermodal_hub"]
    pbi_stations["Is_Interchange"] = df["is_interchange"]
    pbi_stations["Is_Terminal"] = df["is_terminal"]

    # 5D Spatial Features
    pbi_stations["Density_Pop_per_km2"] = df["population_density"].round(1)
    pbi_stations["Population_800m"] = df["population"].round(0)
    pbi_stations["PNR_Walkability_Ratio"] = df["pnr"].round(3)
    pbi_stations["Intersection_Density"] = df["intersection_density"].round(1)
    pbi_stations["Street_Density_km_per_km2"] = df["street_density"].round(2)
    pbi_stations["Bus_Stops_500m"] = df["bus_stops_500m"]
    pbi_stations["POI_Total_Count"] = df["poi_total"].round(0)
    pbi_stations["POI_Retail"] = df["poi_retail"].round(0)
    pbi_stations["POI_Office"] = df["poi_office"].round(0)
    pbi_stations["POI_Education"] = df["poi_education"].round(0)
    pbi_stations["POI_Healthcare"] = df["poi_healthcare"].round(0)
    pbi_stations["Distance_CBD_km"] = df["dist_cbd_km"].round(2)
    pbi_stations["Parking_Area_m2"] = df["parking_area_m2"].round(1)

    # Strategic Action & TOD Classification
    pbi_stations["Access_Index"] = df["access_index"].round(3)
    pbi_stations["Potential_Index"] = df["potential_index"].round(3)
    pbi_stations["TOD_Quadrant"] = df["quadrant"]
    pbi_stations["Strategic_Action"] = df["clean_action"]
    pbi_stations["Typology_ID"] = df["typology_id"]

    # Z-scores
    pbi_stations["Z_Pop_Density"] = df["z_population_density"].round(2)
    pbi_stations["Z_PNR_Walkability"] = df["z_pnr"].round(2)
    pbi_stations["Z_POI_Total"] = df["z_poi_total"].round(2)
    pbi_stations["Z_Bus_Stops"] = df["z_bus_stops_500m"].round(2)
    pbi_stations["Z_Dist_CBD"] = df["z_dist_cbd_km"].round(2)

    pbi_stations.to_csv(PBI_DIR / "chennai_powerbi_stations.csv", index=False)

    # 2. Corridor Summary Dataset (Phase 2)
    phase2_df = pbi_stations[pbi_stations["Phase"] == "Phase 2"]
    corridors = (
        phase2_df.groupby("Corridor_Line")
        .agg(
            Station_Count=("Station_ID", "count"),
            Total_Predicted_Boardings=("Predicted_Boardings", "sum"),
            Avg_Predicted_Boardings=("Predicted_Boardings", "mean"),
            Min_Predicted_Boardings=("Predicted_Boardings", "min"),
            Max_Predicted_Boardings=("Predicted_Boardings", "max"),
            Avg_Pop_Density=("Density_Pop_per_km2", "mean"),
            Avg_PNR_Ratio=("PNR_Walkability_Ratio", "mean"),
            Avg_Bus_Stops_500m=("Bus_Stops_500m", "mean"),
            Avg_POI_Total=("POI_Total_Count", "mean"),
            Top15_Station_Count=("Is_Top15_Phase2", "sum"),
            Intermodal_Hub_Count=("Is_Intermodal_Hub", "sum"),
        )
        .reset_index()
    )
    corridors["Boardings_Share_Pct"] = (
        corridors["Total_Predicted_Boardings"] / corridors["Total_Predicted_Boardings"].sum() * 100
    ).round(2)

    corridors.to_csv(PBI_DIR / "chennai_powerbi_corridors.csv", index=False)

    # 3. Model Performance Dataset
    perf_clean = perf.rename(
        columns={
            "Unnamed: 0": "Model_Name",
            "r2": "R_Squared",
            "mae": "MAE_Boardings",
            "rmse": "RMSE_Boardings",
            "mape": "MAPE_Percent",
            "spearman": "Spearman_Rank_Correlation",
        }
    )
    perf_clean["Model_Name"] = perf_clean["Model_Name"].str.replace("_", " ").str.title()
    perf_clean["R_Squared"] = perf_clean["R_Squared"].round(4)
    perf_clean["MAE_Boardings"] = perf_clean["MAE_Boardings"].round(1)
    perf_clean["RMSE_Boardings"] = perf_clean["RMSE_Boardings"].round(1)
    perf_clean["MAPE_Percent"] = perf_clean["MAPE_Percent"].round(2)
    perf_clean["Spearman_Rank_Correlation"] = perf_clean["Spearman_Rank_Correlation"].round(4)

    perf_clean.to_csv(PBI_DIR / "chennai_powerbi_model_performance.csv", index=False)

    # 4. Feature Elasticity Dataset
    elast_clean = elasticities.rename(
        columns={"Unnamed: 0": "Feature_Name", "0": "Elasticity_Coeff"}
    )
    feature_labels = {
        "bus_stops_500m_pctl": "Bus Stop Feeder Density (500m)",
        "population": "Catchment Population (800m)",
        "parking_area_m2_pctl": "Parking Facility Area (m²)",
        "metro_betweenness": "Metro Network Centrality (Betweenness)",
        "poi_total_pctl": "Total Destination POI Density",
        "dist_cbd_km": "Distance to CBD (km)",
        "intersection_density": "Street Network Intersection Density",
        "pnr": "Pedestrian Network Ratio (PNR)",
    }
    elast_clean["Feature_Label"] = elast_clean["Feature_Name"].map(feature_labels).fillna(elast_clean["Feature_Name"])
    elast_clean["Elasticity_Coeff"] = elast_clean["Elasticity_Coeff"].round(4)
    elast_clean["Impact_Direction"] = np.where(elast_clean["Elasticity_Coeff"] >= 0, "Positive", "Negative")
    elast_clean["Abs_Elasticity"] = elast_clean["Elasticity_Coeff"].abs()
    elast_clean = elast_clean.sort_values("Abs_Elasticity", ascending=False)

    elast_clean.to_csv(PBI_DIR / "chennai_powerbi_elasticities.csv", index=False)

    # 5. TOD Strategic Matrix Summary
    tod_summary = (
        pbi_stations.groupby(["TOD_Quadrant", "Strategic_Action"])
        .agg(
            Station_Count=("Station_ID", "count"),
            Total_Boardings=("Predicted_Boardings", "sum"),
            Avg_Boardings=("Predicted_Boardings", "mean"),
            Avg_PNR=("PNR_Walkability_Ratio", "mean"),
            Avg_Bus_Stops=("Bus_Stops_500m", "mean"),
        )
        .reset_index()
    )
    tod_summary.to_csv(PBI_DIR / "chennai_powerbi_tod_matrix.csv", index=False)


def generate_dax_measures_file() -> None:
    dax_code = """// ====================================================================
// CHENNAI METRO PHASE 2 - POWER BI DAX MEASURES LIBRARY
// Project: Transfer-Learned Direct Demand Ridership & 5D Spatial Analytics
// ====================================================================

// --- 1. CORE RIDERSHIP MEASURES ---

Total Predicted Boardings = 
SUM(Stations[Predicted_Boardings])

Phase 2 Total Boardings = 
CALCULATE(
    SUM(Stations[Predicted_Boardings]),
    Stations[Phase] = "Phase 2"
)

Phase 1 Total Boardings = 
CALCULATE(
    SUM(Stations[Predicted_Boardings]),
    Stations[Phase] = "Phase 1"
)

Average Boardings per Station = 
AVERAGE(Stations[Predicted_Boardings])

Top 15 Boardings Sum = 
CALCULATE(
    SUM(Stations[Predicted_Boardings]),
    Stations[Is_Top15_Phase2] = 1
)

Top 15 Boardings Share Pct = 
DIVIDE([Top 15 Boardings Sum], [Phase 2 Total Boardings], 0) * 100

Lower CI 95 Total = 
SUM(Stations[Lower_CI_95])

Upper CI 95 Total = 
SUM(Stations[Upper_CI_95])

// --- 2. CORRIDOR RIDERSHIP MEASURES ---

Purple Line Boardings = 
CALCULATE(
    SUM(Stations[Predicted_Boardings]),
    Stations[Corridor_Line] = "Purple Line"
)

Red Line Boardings = 
CALCULATE(
    SUM(Stations[Predicted_Boardings]),
    Stations[Corridor_Line] = "Red Line"
)

Yellow Line Boardings = 
CALCULATE(
    SUM(Stations[Predicted_Boardings]),
    Stations[Corridor_Line] = "Yellow Line"
)

Purple Line Share Pct = 
DIVIDE([Purple Line Boardings], [Phase 2 Total Boardings], 0) * 100

// --- 3. 5D SPATIAL & ACCESSIBILITY MEASURES ---

Avg Population Density = 
AVERAGE(Stations[Density_Pop_per_km2])

Avg PNR Walkability Ratio = 
AVERAGE(Stations[PNR_Walkability_Ratio])

Avg Bus Stops 500m = 
AVERAGE(Stations[Bus_Stops_500m])

Avg POI Count = 
AVERAGE(Stations[POI_Total_Count])

Total Catchment Population = 
SUM(Stations[Population_800m])

Avg Distance to CBD = 
AVERAGE(Stations[Distance_CBD_km])

// --- 4. STRATEGIC INTERVENTION & TOD COUNTS ---

Strategic TOD Upzoning Count = 
CALCULATE(
    COUNT(Stations[Station_ID]),
    Stations[Strategic_Action] = "Strategic TOD Upzoning"
)

Feeder Bus Priority Count = 
CALCULATE(
    COUNT(Stations[Station_ID]),
    Stations[Strategic_Action] = "Feeder Bus Priority"
)

First-Mile Enhancement Count = 
CALCULATE(
    COUNT(Stations[Station_ID]),
    Stations[Strategic_Action] = "First-Mile Enhancements"
)

Monitoring Candidate Count = 
CALCULATE(
    COUNT(Stations[Station_ID]),
    Stations[Strategic_Action] = "Monitor / Baseline TOD"
)

Intermodal Hub Station Count = 
CALCULATE(
    COUNT(Stations[Station_ID]),
    Stations[Is_Intermodal_Hub] = 1
)

Total Phase 2 Station Count = 
CALCULATE(
    COUNT(Stations[Station_ID]),
    Stations[Phase] = "Phase 2"
)
"""
    (PBI_DIR / "dax_measures.dax").write_text(dax_code, encoding="utf-8")


def generate_powerbi_guide() -> None:
    guide_md = """# Power BI Data Model & Dashboard Setup Guide

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
- `chennai_powerbi_corridors[Corridor_Line]` **(1)** $\rightarrow$ **(*)** `chennai_powerbi_stations[Corridor_Line]`
- `chennai_powerbi_tod_matrix[Strategic_Action]` **(1)** $\rightarrow$ **(*)** `chennai_powerbi_stations[Strategic_Action]`

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
"""
    (PBI_DIR / "powerbi_data_model_guide.md").write_text(guide_md, encoding="utf-8")


def generate_poster_figures(
    stations: pd.DataFrame,
    corridors: pd.DataFrame,
    perf: pd.DataFrame,
    elasticities: pd.DataFrame,
) -> None:
    sns.set_theme(style="whitegrid", font="DejaVu Sans")

    # Colors
    colors_line = {
        "Purple Line": "#7b3294",
        "Red Line": "#d73027",
        "Yellow Line": "#f4c430",
    }

    # 1. Corridor Ridership Summary Graphic
    fig, ax1 = plt.subplots(figsize=(8, 4.5), dpi=300)
    p2_corridors = corridors[corridors["Corridor_Line"].isin(colors_line.keys())].sort_values(
        "Total_Predicted_Boardings", ascending=False
    )
    bar_colors = [colors_line.get(c, "#555555") for c in p2_corridors["Corridor_Line"]]

    bars = ax1.bar(
        p2_corridors["Corridor_Line"],
        p2_corridors["Total_Predicted_Boardings"] / 1e3,
        color=bar_colors,
        edgecolor="#222222",
        linewidth=0.8,
        width=0.45,
    )
    ax1.set_ylabel("Total Predicted Daily Boardings ('000s)", fontsize=10, fontweight="bold")
    ax1.set_title("Chennai Metro Phase 2: Ridership Potential by Corridor", fontsize=12, fontweight="bold", pad=12)

    for bar in bars:
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 4,
            f"{height:.1f}k",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax1.set_ylim(0, p2_corridors["Total_Predicted_Boardings"].max() / 1e3 * 1.22)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES_PBI / "powerbi_corridor_ridership_summary.png", transparent=False, facecolor="white")
    plt.close(fig)

    # 2. 5D Feature Elasticity Graphic
    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=300)
    elast_sorted = elasticities.sort_values("Elasticity_Coeff", ascending=True)
    bar_colors = ["#2b83ba" if x >= 0 else "#d7191c" for x in elast_sorted["Elasticity_Coeff"]]

    ax.barh(
        elast_sorted["Feature_Label"],
        elast_sorted["Elasticity_Coeff"],
        color=bar_colors,
        edgecolor="#222222",
        linewidth=0.6,
    )
    ax.axvline(0, color="#333333", linewidth=1.0, linestyle="--")
    ax.set_xlabel("Ridership Elasticity Coefficient", fontsize=10, fontweight="bold")
    ax.set_title("5D Framework Feature Elasticities (Direct Demand Model)", fontsize=11, fontweight="bold", pad=14)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for i, (val, name) in enumerate(zip(elast_sorted["Elasticity_Coeff"], elast_sorted["Feature_Label"])):
        if val >= 0:
            ax.text(val + 0.008, i, f"+{val:.3f}", va="center", ha="left", fontsize=8.5, fontweight="bold")
        else:
            ax.text(val - 0.008, i, f"{val:.3f}", va="center", ha="right", fontsize=8.5, fontweight="bold")

    fig.tight_layout()
    fig.savefig(FIGURES_PBI / "powerbi_5d_feature_elasticity.png", transparent=False, facecolor="white")
    plt.close(fig)

    # 3. TOD Strategic Action Matrix Scatter Plot
    fig, ax = plt.subplots(figsize=(8, 5.2), dpi=300)
    quad_colors = {
        "Strategic TOD Upzoning": "#2ca25f",
        "Feeder Bus Priority": "#2c7fb8",
        "High-Demand Early Open": "#f4c430",
        "Right-Size & Defer Spend": "#999999",
    }
    p2_stations = stations[stations["Phase"] == "Phase 2"]

    for quad, group in p2_stations.groupby("Strategic_Action"):
        color = quad_colors.get(quad, "#555555")
        ax.scatter(
            group["Access_Index"],
            group["Potential_Index"],
            s=group["Predicted_Boardings"] / 140,
            c=color,
            alpha=0.75,
            edgecolors="#111111",
            linewidth=0.6,
            label=f"{quad} (n={len(group)})",
        )

    ax.axhline(0.5, color="#666666", linestyle="--", linewidth=0.8)
    ax.axvline(0.5, color="#666666", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Access Index (Feeder & Pedestrian Infrastructure)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Potential Index (Catchment Demand & Density)", fontsize=10, fontweight="bold")
    ax.set_title("TOD Strategic Planning Matrix: Access vs. Demand Potential", fontsize=12, fontweight="bold", pad=12)
    ax.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.9, fontsize=8.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES_PBI / "powerbi_tod_action_matrix.png", transparent=False, facecolor="white")
    plt.close(fig)

    # 4. Executive KPI Dashboard Preview Poster Card
    fig = plt.figure(figsize=(10, 5.5), dpi=300, facecolor="#1e1e2e")
    fig.suptitle("CHENNAI METRO PHASE 2 — POWER BI STATISTICAL EXECUTIVE SUMMARY", color="#ffffff", fontsize=14, fontweight="bold", y=0.95)

    # Layout Grid: 2x3 cards
    # Card 1: Total Boardings
    ax_c1 = fig.add_axes([0.05, 0.52, 0.28, 0.35], facecolor="#2a2a3c")
    ax_c1.axis("off")
    total_b = p2_stations["Predicted_Boardings"].sum()
    ax_c1.text(0.5, 0.68, "TOTAL PHASE 2 RIDERSHIP", color="#a6adc8", fontsize=9, fontweight="bold", ha="center")
    ax_c1.text(0.5, 0.32, f"{total_b:,.0f}", color="#a6e3a1", fontsize=20, fontweight="bold", ha="center")
    ax_c1.text(0.5, 0.10, "Daily predicted boardings", color="#bac2de", fontsize=7.5, ha="center")

    # Card 2: Top 15 Share
    ax_c2 = fig.add_axes([0.36, 0.52, 0.28, 0.35], facecolor="#2a2a3c")
    ax_c2.axis("off")
    top15_b = p2_stations[p2_stations["Is_Top15_Phase2"] == 1]["Predicted_Boardings"].sum()
    top15_pct = (top15_b / total_b) * 100
    ax_c2.text(0.5, 0.68, "TOP 15 STATIONS SHARE", color="#a6adc8", fontsize=9, fontweight="bold", ha="center")
    ax_c2.text(0.5, 0.32, f"{top15_pct:.1f}%", color="#89b4fa", fontsize=20, fontweight="bold", ha="center")
    ax_c2.text(0.5, 0.10, f"{top15_b:,.0f} boardings / day", color="#bac2de", fontsize=7.5, ha="center")

    # Card 3: TOD Feeder Priorities
    ax_c3 = fig.add_axes([0.67, 0.52, 0.28, 0.35], facecolor="#2a2a3c")
    ax_c3.axis("off")
    feeder_n = (p2_stations["Strategic_Action"] == "Feeder Bus Priority").sum()
    upzone_n = (p2_stations["Strategic_Action"] == "Strategic TOD Upzoning").sum()
    ax_c3.text(0.5, 0.68, "POLICY INTERVENTIONS", color="#a6adc8", fontsize=9, fontweight="bold", ha="center")
    ax_c3.text(0.5, 0.35, f"{feeder_n} Feeder / {upzone_n} Upzone", color="#f9e2af", fontsize=13, fontweight="bold", ha="center")
    ax_c3.text(0.5, 0.10, "Priority policy candidate stations", color="#bac2de", fontsize=7.5, ha="center")

    # Bottom Left: Top 5 Leaderboard
    ax_b1 = fig.add_axes([0.05, 0.08, 0.43, 0.36], facecolor="#2a2a3c")
    ax_b1.axis("off")
    top5 = p2_stations.dropna(subset=["Phase2_Ridership_Rank"]).sort_values("Predicted_Boardings", ascending=False).head(5)
    ax_b1.text(0.05, 0.85, "TOP 5 HIGH-POTENTIAL STATIONS", color="#cba6f7", fontsize=9.5, fontweight="bold")
    y_pos = 0.65
    for _, r in top5.iterrows():
        rank_val = int(r['Phase2_Ridership_Rank']) if pd.notnull(r['Phase2_Ridership_Rank']) else 0
        ax_b1.text(0.05, y_pos, f"#{rank_val} {r['Station_Name'][:22]}", color="#ffffff", fontsize=8, fontweight="bold")
        ax_b1.text(0.72, y_pos, f"{int(r['Predicted_Boardings']):,} / day", color="#a6e3a1", fontsize=8, fontweight="bold")
        y_pos -= 0.15

    # Bottom Right: Model Benchmarking Card
    ax_b2 = fig.add_axes([0.52, 0.08, 0.43, 0.36], facecolor="#2a2a3c")
    ax_b2.axis("off")
    best_m = perf.sort_values("Spearman_Rank_Correlation", ascending=False).iloc[0]
    ax_b2.text(0.05, 0.85, "TRANSFER-LEARNED MODEL ACCURACY", color="#f5e0dc", fontsize=9.5, fontweight="bold")
    ax_b2.text(0.05, 0.62, f"Best Model: {best_m['Model_Name']}", color="#ffffff", fontsize=8.5, fontweight="bold")
    ax_b2.text(0.05, 0.42, f"Spearman Rank Corr: {best_m['Spearman_Rank_Correlation']:.4f}", color="#89b4fa", fontsize=8)
    ax_b2.text(0.05, 0.24, f"MAE: {best_m['MAE_Boardings']:,.1f} boardings", color="#bac2de", fontsize=8)
    ax_b2.text(0.05, 0.06, f"MAPE: {best_m['MAPE_Percent']:.2f}%", color="#bac2de", fontsize=8)

    fig.savefig(FIGURES_PBI / "powerbi_kpi_dashboard_preview.png", transparent=False, facecolor="#1e1e2e")
    plt.close(fig)


def main() -> None:
    predictions, recommendations, perf, elasticities = load_data()
    build_powerbi_datasets(predictions, recommendations, perf, elasticities)
    generate_dax_measures_file()
    generate_powerbi_guide()

    # Load cleaned files for plotting
    stations_pbi = pd.read_csv(PBI_DIR / "chennai_powerbi_stations.csv")
    corridors_pbi = pd.read_csv(PBI_DIR / "chennai_powerbi_corridors.csv")
    perf_pbi = pd.read_csv(PBI_DIR / "chennai_powerbi_model_performance.csv")
    elast_pbi = pd.read_csv(PBI_DIR / "chennai_powerbi_elasticities.csv")

    generate_poster_figures(stations_pbi, corridors_pbi, perf_pbi, elast_pbi)
    print("Power BI statistical asset generation complete!")


if __name__ == "__main__":
    main()
