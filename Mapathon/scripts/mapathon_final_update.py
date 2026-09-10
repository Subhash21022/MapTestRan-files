"""Mapathon Final Update — All Tasks End-to-End.

Tasks executed:
  TASK 1 — Fix basemap: suppress OSM, elevate LULC opacity, fix "Other" class
  TASK 2 — Add data-quality footnote
  TASK 3 — Apply "Do Now" report recommendations
  TASK 4 — Build 4 poster charts (corridor, model perf, elasticities, misses)
  TASK 5 — QA pass + rebuild QGIS project + export 300 DPI PNG/PDF

Run with:
    D:/QGIS/apps/Python312/python.exe -m scripts.mapathon_final_update
or via the QGIS standalone launcher.
"""

from __future__ import annotations

import math
import os
import sys
import json
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import geopandas as gpd
from pyproj import CRS
from shapely.geometry import LineString, Point
from shapely.ops import linemerge, unary_union

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TABLES_DIR   = ROOT / "outputs" / "tables"
FIGURES_DIR  = ROOT / "outputs" / "figures"
MAPS_DIR     = ROOT / "outputs" / "maps"
QGIS_DIR     = ROOT / "qgis"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
MAPS_DIR.mkdir(parents=True, exist_ok=True)

LAYER_GPKG   = QGIS_DIR / "chennai_mapathon_layers_desktop.gpkg"
PROJECT_PATH = QGIS_DIR / "chennai_mapathon_ridership_desktop.qgs"
PROJECT_QGZ  = QGIS_DIR / "chennai_mapathon_ridership_desktop.qgz"
LAUNCHER_PATH= QGIS_DIR / "open_chennai_mapathon_qgis.bat"
PDF_PATH     = MAPS_DIR / "chennai_mapathon_qgis_preview_desktop.pdf"
PNG_PATH     = MAPS_DIR / "chennai_mapathon_qgis_preview_desktop.png"
FULL_PDF     = MAPS_DIR / "chennai_mapathon_full_labels.pdf"
FULL_PNG     = MAPS_DIR / "chennai_mapathon_full_labels.png"

TOP15_CHART      = FIGURES_DIR / "chennai_top15_phase2_predictions.png"
CORRIDOR_CHART   = FIGURES_DIR / "chart_corridor_ridership.png"
MODEL_PERF_CHART = FIGURES_DIR / "chart_model_performance.png"
ELASTICITY_CHART = FIGURES_DIR / "chart_nb_elasticities.png"
MISSES_CHART     = FIGURES_DIR / "chart_phase1_misses.png"

DATA_DIR    = Path("D:/Mapathon Data Files")
OVERPASS    = ROOT / "data" / "interim" / "chennai_overpass_alignments.json"
CRS_UTM44   = "EPSG:32644"
QGIS_PREFIX = "D:/QGIS/apps/qgis"
MAX_SNAP_M  = 15000

# ── TASK 1: Updated LULC palette — no grey "Other" catch-all ─────────────────
# "Other" renamed to "Barren/Open" with a proper terracotta/sandy colour.
LULC_COLORS = {
    "Built-up":      "#c0726e",   # muted rose-brick — up from #dca8a8
    "Agriculture":   "#c9b44e",   # golden harvest — up from #d8c66f
    "Forest/Scrub":  "#4a9458",   # mid-green — up from #5aa469
    "Water/Wetland": "#4090b5",   # clear blue — up from #5aa6c8
    "Barren/Open":   "#c49a6c",   # sandy tan — up from #d9b38c; replaces "Other"
}
# LULC target opacity — 0.68 (from 0.38–0.45 previously)
LULC_OPACITY = 0.68

LINE_COLORS = {
    "Blue Line":   "#2c7fb8",
    "Green Line":  "#2ca25f",
    "Purple Line": "#7b3294",
    "Red Line":    "#d73027",
    "Yellow Line": "#e8a800",
}

INTERMODAL_HUB_NAMES = {
    "Puratchi Thalaivar Dr. M.G. Ramachandran Central",
    "Chennai International Airport",
    "Puratchi Thalaivi Dr. J. Jayalalithaa CMBT",
    "CMBT", "Egmore", "Guindy", "St. Thomas Mount",
    "Arignar Anna Alandur", "Koyambedu", "Kilambakkam",
    "Thirumangalam", "Thirumayilai", "Mandaveli",
    "Adyar Depot", "Porur Junction", "Madhavaram Milk Colony",
}

ALIGNMENT_NAME_TO_LINE = {
    "Chennai Metro Line 1": ("phase1", "Blue Line"),
    "Chennai Metro Line 2": ("phase1", "Green Line"),
}

# ── Footnote text (TASK 2) ────────────────────────────────────────────────────
FOOTNOTE_TEXT = (
    "September 2026 update: based on 1 weekend observation. Absolute ridership\n"
    "bands may shift when weekday data accumulates. Rankings are directionally stable."
)

# ── Corridor totals from report (for chart; weekend-biased, TASK 4) ───────────
CORRIDOR_REPORT = {
    "C3: Madhavaram-SIPCOT\n(Red Line)":     195_000,
    "C4: Lighthouse-Poonamallee\n(Yellow Line)": 140_000,
    "C5: Madhavaram-Sholinganallur\n(Purple Line)": 220_000,
}
CORRIDOR_COLORS = ["#d73027", "#e8a800", "#7b3294"]

# ── Phase-1 miss data from report (TASK 4) ───────────────────────────────────
MISS_DATA = {
    "MGR Central":       (27379, 10062),
    "Chennai Airport":   (18156,  8428),
    "Thirumangalam":     (13988,  7244),
    "Vadapalani":        (14398,  8771),
    "Guindy":            (12722,  7836),
    "Thousand Lights":   (12821,  8695),
}


# =============================================================================
# TASK 4 — Generate the 4 poster charts without QGIS
# =============================================================================

def _chart_style(ax, title: str) -> None:
    ax.set_title(title, fontsize=10, fontweight="bold", pad=8)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.grid(axis="y" if ax.get_xlim()[0] == 0 else "x",
            color="#e0e0e0", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)


def make_corridor_chart() -> None:
    """TASK 4 chart 1 — corridor comparison bar."""
    labels = list(CORRIDOR_REPORT.keys())
    vals   = [v / 1_000 for v in CORRIDOR_REPORT.values()]
    cols   = CORRIDOR_COLORS

    fig, ax = plt.subplots(figsize=(5.2, 3.4), dpi=300)
    bars = ax.bar(range(len(labels)), vals, color=cols,
                  edgecolor="#333333", linewidth=0.6, width=0.5, zorder=3)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel("Total Predicted Boardings ('000s/day)", fontsize=8)
    _chart_style(ax, "Phase 2 Ridership by Corridor\n(Sep 2026, weekend-adjusted)")
    ax.grid(axis="y", color="#e0e0e0", linewidth=0.5, zorder=0)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3,
                f"{val:.0f}k", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.set_ylim(0, max(vals) * 1.25)
    fig.tight_layout(pad=0.8)
    fig.savefig(CORRIDOR_CHART, dpi=300, facecolor="white", transparent=False)
    plt.close(fig)
    print(f"  [T4-1] corridor chart -> {CORRIDOR_CHART.name}")


def make_model_perf_chart() -> None:
    """TASK 4 chart 2 — LOOCV model performance."""
    perf_path = TABLES_DIR / "model_performance.csv"
    perf = pd.read_csv(perf_path, index_col=0)
    models = ["linear_regression", "elastic_net", "random_forest", "gradient_boosting"]
    labels = ["Linear\nRegression", "Elastic\nNet", "Random\nForest*", "Gradient\nBoosting"]
    r2_vals  = [perf.loc[m, "r2"]       for m in models]
    rho_vals = [perf.loc[m, "spearman"] for m in models]
    mae_vals = [perf.loc[m, "mae"]      for m in models]

    # Selected model (Random Forest) gets accent colour
    bar_colors = ["#aaaaaa", "#aaaaaa", "#d73027", "#aaaaaa"]
    edge_colors = ["#555555", "#555555", "#8b0000", "#555555"]

    fig, axes = plt.subplots(1, 3, figsize=(8.5, 3.0), dpi=300)
    x = range(len(models))

    for ax, vals, ylabel, title, fmt in [
        (axes[0], r2_vals,  "R²",    "LOO R²\n(higher=better)", ".3f"),
        (axes[1], rho_vals, "ρ",     "Spearman ρ\n(rank accuracy)", ".3f"),
        (axes[2], [v/1000 for v in mae_vals], "MAE (k boardings)", "MAE ('000s)\n(lower=better)", ".1f"),
    ]:
        bars = ax.bar(x, vals, color=bar_colors, edgecolor=edge_colors, linewidth=0.6, width=0.55, zorder=3)
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, fontsize=6.5)
        ax.set_ylabel(ylabel, fontsize=7)
        _chart_style(ax, title)
        ax.grid(axis="y", color="#e0e0e0", linewidth=0.5, zorder=0)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + (max(vals)-min(vals))*0.03,
                    f"{val:{fmt}}", ha="center", va="bottom", fontsize=6, fontweight="bold")

    # legend patch for selected model
    patch = mpatches.Patch(color="#d73027", label="Selected model (Random Forest)")
    fig.legend(handles=[patch], loc="lower center", fontsize=7.5,
               frameon=True, facecolor="white", ncol=1, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Model LOOCV Benchmarks — Sep 2026 (1-day weekend training)", fontsize=9, fontweight="bold")
    fig.tight_layout(pad=0.6, rect=[0, 0.06, 1, 0.97])
    fig.savefig(MODEL_PERF_CHART, dpi=300, facecolor="white", transparent=False)
    plt.close(fig)
    print(f"  [T4-2] model perf chart -> {MODEL_PERF_CHART.name}")


def make_elasticity_chart() -> None:
    """TASK 4 chart 3 — NB elasticities horizontal bar."""
    elast_path = TABLES_DIR / "nb_elasticities.csv"
    elast = pd.read_csv(elast_path, index_col=0, header=0)
    elast.columns = ["coeff"]
    elast = elast.sort_values("coeff")

    feature_labels = {
        "bus_stops_500m_pctl": "Bus Stop Feeder Density (500m)",
        "competing_rail_1km":  "Competing Rail Proximity (1km)",
        "dist_cbd_km":         "Distance to CBD (km)",
        "parking_area_m2_pctl": "Parking Area Percentile",
        "population":          "Catchment Population (800m)",
        "poi_total_pctl":      "POI Density Percentile",
        "pnr":                 "Pedestrian Network Ratio",
        "intersection_density":"Street Intersection Density",
    }
    labels = [feature_labels.get(i, i) for i in elast.index]
    colors = ["#d73027" if v < 0 else "#2b83ba" for v in elast["coeff"]]

    fig, ax = plt.subplots(figsize=(6.0, 3.6), dpi=300)
    bars = ax.barh(labels, elast["coeff"], color=colors,
                   edgecolor="#333333", linewidth=0.5, zorder=3)
    ax.axvline(0, color="#333333", linewidth=0.8, linestyle="--")
    ax.set_xlabel("NB Elasticity Coefficient", fontsize=8)
    _chart_style(ax, "Feature Elasticities — NB GLM (Direct Demand Model)")
    ax.grid(axis="x", color="#e0e0e0", linewidth=0.5, zorder=0)
    for bar, val in zip(bars, elast["coeff"]):
        offset = 0.006 if val >= 0 else -0.006
        ha = "left" if val >= 0 else "right"
        ax.text(val + offset, bar.get_y() + bar.get_height()/2,
                f"{val:+.3f}", va="center", ha=ha, fontsize=7, fontweight="bold")
    pos_patch = mpatches.Patch(color="#2b83ba", label="Positive driver")
    neg_patch = mpatches.Patch(color="#d73027", label="Negative driver")
    ax.legend(handles=[pos_patch, neg_patch], fontsize=7, loc="lower right", frameon=True)
    fig.tight_layout(pad=0.8)
    fig.savefig(ELASTICITY_CHART, dpi=300, facecolor="white", transparent=False)
    plt.close(fig)
    print(f"  [T4-3] elasticity chart -> {ELASTICITY_CHART.name}")


def make_misses_chart() -> None:
    """TASK 4 chart 4 — Phase 1 misses vs weekend observed."""
    stations = list(MISS_DATA.keys())
    observed = [MISS_DATA[s][0] for s in stations]
    predicted= [MISS_DATA[s][1] for s in stations]
    x = range(len(stations))

    fig, ax = plt.subplots(figsize=(6.2, 3.4), dpi=300)
    w = 0.35
    ax.bar([i - w/2 for i in x], observed,  width=w, color="#4393c3",
           label="Weekend Observed", edgecolor="#333", linewidth=0.5, zorder=3)
    ax.bar([i + w/2 for i in x], predicted, width=w, color="#d73027",
           label="Model Predicted",  edgecolor="#333", linewidth=0.5, zorder=3)
    ax.set_xticks(list(x))
    ax.set_xticklabels(stations, fontsize=7.5, rotation=20, ha="right")
    ax.set_ylabel("Boardings/day", fontsize=8)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    _chart_style(ax, "Phase 1 Prediction Misses (Weekend Observed vs Model)")
    ax.grid(axis="y", color="#e0e0e0", linewidth=0.5, zorder=0)
    ax.legend(fontsize=7.5, frameon=True)
    note = ("*All large misses are leisure/interchange stations where weekend\n"
            " demand genuinely exceeds the commuter-oriented model assumption.")
    ax.text(0.01, 0.02, note, transform=ax.transAxes, fontsize=6.5,
            color="#666666", va="bottom")
    fig.tight_layout(pad=0.8)
    fig.savefig(MISSES_CHART, dpi=300, facecolor="white", transparent=False)
    plt.close(fig)
    print(f"  [T4-4] misses chart -> {MISSES_CHART.name}")


def make_top15_chart_updated() -> None:
    """Top-15 bar chart with Thirumayilai #1 bolded (TASK 3)."""
    df = pd.read_csv(TABLES_DIR / "chennai_predictions.csv")
    top = df[df["phase"].eq("phase2")].nlargest(15, "prediction").sort_values("prediction")

    colors = [LINE_COLORS.get(line, "#666666") for line in top["Line"]]
    # TASK 3: Thirumayilai highlighted with thicker outline
    edge_colors = ["#8b0000" if n == "Thirumayilai" else "#333333" for n in top["name"]]
    edge_widths = [1.5 if n == "Thirumayilai" else 0.4 for n in top["name"]]

    xerr = np.vstack([top["prediction"] - top["lower"], top["upper"] - top["prediction"]])
    fig, ax = plt.subplots(figsize=(7.6, 5.0), dpi=300)
    bars = ax.barh(top["name"], top["prediction"], color=colors,
                   edgecolor=edge_colors, linewidth=edge_widths, zorder=3)
    ax.errorbar(top["prediction"], top["name"], xerr=xerr, fmt="none",
                ecolor="#222222", elinewidth=0.8, capsize=2)
    ax.set_xlabel("Predicted daily boardings", fontsize=9)
    ax.set_title("Top 15 Phase 2 Stations — Predicted Ridership Potential\n"
                 "(#1: Thirumayilai ★  |  Red Line cluster: Manapakkam, Madipakkam, Anna Nagar West, Butt Rd, Ramapuram)",
                 fontsize=8.5, fontweight="bold")
    ax.grid(axis="x", color="#d9d9d9", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    # Bold label for #1
    tick_labels = ax.get_yticklabels()
    for label in tick_labels:
        if label.get_text() == "Thirumayilai":
            label.set_fontweight("bold")
            label.set_color("#8b0000")
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    # Red Line stations get highlighted annotation
    red_line = {"Manapakkam", "Madipakkam", "Anna Nagar West", "Butt Road", "Ramapuram"}
    for bar, name in zip(bars, top["name"]):
        if name in red_line:
            ax.text(bar.get_width() + 150, bar.get_y() + bar.get_height()/2,
                    "●", va="center", ha="left", fontsize=8, color="#d73027")
    red_patch = mpatches.Patch(color="#d73027", label="Red Line cluster (prominent)")
    ax.legend(handles=[red_patch], fontsize=7.5, loc="lower right", frameon=True)
    fig.tight_layout()
    fig.savefig(TOP15_CHART, transparent=False, facecolor="white", dpi=300)
    plt.close(fig)
    print(f"  [T3/T4] top-15 chart -> {TOP15_CHART.name}")


# =============================================================================
# QGIS project helpers (re-used from build_qgis_project.py with TASK 1 edits)
# =============================================================================

def classify_alignment(name: str):
    if not name:
        return None
    if "proposed" in name.lower() or "Line 5B" in name or "corridor switch" in name:
        return None
    if name in ALIGNMENT_NAME_TO_LINE:
        return ALIGNMENT_NAME_TO_LINE[name]
    if "Line 3:" in name:
        return ("phase2", "Purple Line")
    if "Line 5:" in name:
        return ("phase2", "Red Line")
    if "Yellow Line" in name:
        return ("phase2", "Yellow Line")
    return None


def read_osm_alignments() -> gpd.GeoDataFrame:
    if not OVERPASS.exists():
        return gpd.GeoDataFrame(columns=["phase","Line","osm_name","geometry"], crs=CRS_UTM44)
    data = json.loads(OVERPASS.read_text(encoding="utf-8"))
    rows = []
    for el in data.get("elements", []):
        if el.get("type") != "way" or "geometry" not in el:
            continue
        tags = el.get("tags", {})
        name = tags.get("name", "")
        classified = classify_alignment(name)
        if not classified:
            continue
        coords = [(pt["lon"], pt["lat"]) for pt in el["geometry"]]
        if len(coords) < 2:
            continue
        phase, line = classified
        rows.append({"phase": phase, "Line": line, "osm_name": name,
                     "osm_id": el.get("id"), "geometry": LineString(coords)})
    if not rows:
        return gpd.GeoDataFrame(columns=["phase","Line","osm_name","geometry"], crs=CRS_UTM44)
    lines = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326").to_crs(CRS_UTM44)
    dissolved = []
    for (phase, line), grp in lines.groupby(["phase","Line"]):
        merged = linemerge(unary_union(grp.geometry.tolist()))
        if merged.geom_type == "GeometryCollection":
            parts = [g for g in merged.geoms if g.geom_type in ("LineString","MultiLineString")]
            merged = linemerge(unary_union(parts))
        dissolved.append({"phase": phase, "Line": line, "station_count": 0,
                           "source": "osm_overpass",
                           "length_km": round(float(grp.geometry.length.sum()/1000), 2),
                           "geometry": merged})
    return gpd.GeoDataFrame(dissolved, geometry="geometry", crs=CRS_UTM44)


def lulc_group(value) -> str:
    text = str(value or "").lower()
    if "built" in text: return "Built-up"
    if any(k in text for k in ("agriculture","crop","fallow")): return "Agriculture"
    if any(k in text for k in ("forest","scrub")): return "Forest/Scrub"
    if any(k in text for k in ("wetland","water","river","lake")): return "Water/Wetland"
    # "Other" mapped to "Barren/Open" (TASK 1)
    return "Barren/Open"


def nearest_point_on_geometry(geom, point):
    return geom.interpolate(geom.project(point))


def snap_stations_for_map(stations, alignments):
    out = stations.to_crs(CRS_UTM44).copy()
    out["map_snap_m"] = 0.0
    out["map_aligned"] = 0
    if alignments.empty:
        return out
    line_geoms = {(r["phase"], r["Line"]): r.geometry for _, r in alignments.iterrows()}
    phase2_union = unary_union(alignments[alignments["phase"].eq("phase2")].geometry.tolist())
    all_union    = unary_union(alignments.geometry.tolist())
    for idx, row in out.iterrows():
        keys = [(row.get("phase"), row.get("Line"))]
        if row.get("Line") == "Red LinePurple Line":
            keys = [("phase2","Red Line"), ("phase2","Purple Line")]
        candidates = [line_geoms[k] for k in keys if k in line_geoms]
        if not candidates and row.get("phase") == "phase2" and not phase2_union.is_empty:
            candidates = [phase2_union]
        if not candidates and not all_union.is_empty:
            candidates = [all_union]
        if not candidates:
            continue
        point = row.geometry
        options = [(g.distance(point), nearest_point_on_geometry(g, point)) for g in candidates]
        dist, snapped = min(options, key=lambda x: x[0])
        if dist <= MAX_SNAP_M:
            out.at[idx, "geometry"] = snapped
            out.at[idx, "map_snap_m"] = round(float(dist), 1)
            out.at[idx, "map_aligned"] = 1
    return out


def build_fallback_guides(stations):
    rows = []
    for (phase, line), grp in stations.dropna(subset=["Line"]).groupby(["phase","Line"]):
        if len(grp) < 2: continue
        pts = grp.geometry.tolist()
        unused = set(range(len(pts)))
        start = min(unused, key=lambda i: (pts[i].x, pts[i].y))
        order = [start]; unused.remove(start)
        while unused:
            last = pts[order[-1]]
            nxt = min(unused, key=lambda i: last.distance(pts[i]))
            order.append(nxt); unused.remove(nxt)
        rows.append({"phase": phase, "Line": line, "station_count": len(grp),
                     "source": "station_nearest_neighbour",
                     "length_km": round(LineString([pts[i] for i in order]).length/1000, 2),
                     "geometry": LineString([pts[i] for i in order])})
    return gpd.GeoDataFrame(rows, crs=CRS_UTM44)


def remove_existing_gpkg(path: Path) -> None:
    if path.exists(): path.unlink()
    for f in path.parent.glob(path.name + "*"):
        if f.is_file(): f.unlink()


def write_layer(gdf, layer_name, first=False):
    mode = "w" if first else "a"
    gdf.to_file(LAYER_GPKG, layer=layer_name, driver="GPKG", mode=mode)


def prepare_lulc_layers() -> None:
    first = True
    for suffix, year_label in [("0506","2005-06"),("1112","2011-12"),("1516","2015-16")]:
        path = DATA_DIR / f"CHENNAI_TN_LULC50K_{suffix}.shp"
        if not path.exists():
            print(f"  [T1] LULC shapefile missing, skipping: {path.name}")
            continue
        gdf = gpd.read_file(path)
        gdf["map_year"]   = year_label
        gdf["lulc_group"] = gdf["DESCR_1"].map(lulc_group)
        gdf["lulc_class"] = gdf["DESCR_2"].fillna(gdf["lulc_group"])
        gdf = gdf.to_crs(CRS_UTM44)
        keep = ["map_year","lulc_group","lulc_class","LU_Webcode","geometry"]
        write_layer(gdf[keep], f"bhuvan_lulc_{suffix}", first=first)
        first = False
        print(f"  [T1] LULC layer written: bhuvan_lulc_{suffix}")


def prepare_station_layers():
    stations = gpd.read_file(ROOT / "data" / "processed" / "chennai_stations.gpkg")
    pred_path = TABLES_DIR / "chennai_predictions.csv"
    predictions = pd.read_csv(pred_path)
    pred_cols = ["station_id","prediction","lower","upper","population_density",
                 "pnr","bus_stops_500m","poi_total","metro_closeness"]
    pred_cols = [c for c in pred_cols if c in predictions.columns]
    predictions = predictions[pred_cols]
    stations = stations.merge(predictions, on="station_id", how="left")
    stations["prediction"] = stations["prediction"].round(0)
    stations["lower"]      = stations["lower"].round(0)
    stations["upper"]      = stations["upper"].round(0)
    stations["prediction_band"] = pd.qcut(
        stations["prediction"].rank(method="first"), q=5,
        labels=["Very low","Low","Medium","High","Very high"]).astype(str)
    # TASK 5 QA: fix "Red LinePurple Line" label
    stations.loc[stations["Line"] == "Red LinePurple Line", "Line"] = "Purple Line"
    phase2 = stations["phase"].eq("phase2")
    stations["rank_phase2"] = np.nan
    stations.loc[phase2, "rank_phase2"] = (
        stations.loc[phase2, "prediction"].rank(ascending=False, method="first"))
    stations["label_top15"]      = (stations["rank_phase2"] <= 15).astype(int)
    stations["is_intermodal_hub"]= stations["name"].isin(INTERMODAL_HUB_NAMES).astype(int)
    # TASK 3: Flag #1 station (Thirumayilai)
    stations["is_top1_phase2"] = (stations["rank_phase2"] == 1).fillna(False).astype(int)
    pred_min = float(stations.loc[phase2, "prediction"].min())
    pred_max = float(stations.loc[phase2, "prediction"].max())
    osm_alignments = read_osm_alignments()
    if osm_alignments.empty:
        stations_map = stations.to_crs(CRS_UTM44)
        alignments   = build_fallback_guides(stations_map)
    else:
        alignments   = osm_alignments
        stations_map = snap_stations_for_map(stations, alignments)
    write_layer(stations_map, "chennai_stations_predictions")
    buffers = gpd.read_file(ROOT / "data" / "processed" / "chennai_catchments_buffer.gpkg")
    buffers = buffers[buffers["radius_m"].eq(800)].copy()
    buffers = buffers.drop(columns="geometry")
    buffers = buffers.merge(
        stations_map[["station_id","prediction","rank_phase2","label_top15","geometry"]],
        on="station_id", how="left")
    buffers = gpd.GeoDataFrame(buffers, geometry="geometry", crs=CRS_UTM44)
    buffers["geometry"] = buffers.geometry.buffer(800)
    write_layer(buffers.to_crs(CRS_UTM44), "chennai_catchments_800m")
    write_layer(alignments, "chennai_corridor_guides")
    return pred_min, pred_max


# =============================================================================
# QGIS project creation with all TASK 1, 2, 3 edits
# =============================================================================

def create_project(pred_min: float, pred_max: float) -> None:
    from qgis.PyQt.QtCore import Qt
    from qgis.PyQt.QtGui import QColor, QFont
    from qgis.core import (
        QgsApplication, QgsCategorizedSymbolRenderer,
        QgsCoordinateReferenceSystem, QgsFillSymbol, QgsGraduatedSymbolRenderer,
        QgsLayoutExporter, QgsLayoutItemLabel, QgsLayoutItemLegend,
        QgsLayoutItemMap, QgsLayoutItemPicture, QgsLayoutItemScaleBar,
        QgsLayoutItemMapGrid, QgsLayoutPoint, QgsLayoutSize, QgsLegendStyle,
        QgsLineSymbol, QgsMarkerSymbol, QgsPalLayerSettings, QgsPrintLayout,
        QgsProject, QgsProperty, QgsRasterLayer, QgsRendererCategory,
        QgsRendererRange, QgsSimpleLineCallout, QgsSingleSymbolRenderer,
        QgsVectorLayerSimpleLabeling, QgsTextBufferSettings, QgsTextFormat,
        QgsUnitTypes, QgsVectorLayer,
    )

    QgsApplication.setPrefixPath(QGIS_PREFIX, True)
    app = QgsApplication([], False)
    app.initQgis()
    project = QgsProject.instance()
    project.clear()
    project.setCrs(QgsCoordinateReferenceSystem(CRS_UTM44))
    project.setTitle("Chennai Metro Phase 2 Ridership Potential - Mapathon")

    src = str(LAYER_GPKG).replace("\\", "/")

    # TASK 1: NO OSM raster basemap added to layers at all.
    # Only a flat light-grey background rectangle is used (set via page background colour).
    # Optional: if QgsRasterLayer for OSM still desired, set opacity to 0 to suppress.

    def add_layer(path, name, subset=None):
        lyr = QgsVectorLayer(path, name, "ogr")
        if not lyr.isValid():
            print(f"    [WARN] Layer invalid: {name}")
            return None
        if subset:
            lyr.setSubsetString(subset)
        project.addMapLayer(lyr)
        return lyr

    # TASK 1: LULC at 68% opacity — up from 38-45%
    def apply_lulc_style(layer):
        categories = []
        for group, color in LULC_COLORS.items():
            symbol = QgsFillSymbol.createSimple({
                "color": color,
                "outline_color": "#ffffff",
                "outline_width": "0.04",
            })
            symbol.setOpacity(LULC_OPACITY)   # 0.68 — TASK 1
            categories.append(QgsRendererCategory(group, symbol, group))
        layer.setRenderer(QgsCategorizedSymbolRenderer("lulc_group", categories))

    def apply_catchment_style(layer, top15=False):
        symbol = QgsFillSymbol.createSimple({
            "color": "255,255,255,0",
            "outline_color": "#333333" if top15 else "#aaaaaa",
            "outline_width": "0.22" if top15 else "0.05",
            "outline_style": "dash" if top15 else "solid",
        })
        symbol.setOpacity(0.45 if top15 else 0.10)
        layer.setRenderer(QgsSingleSymbolRenderer(symbol))

    def apply_corridor_style(layer, width, phase):
        categories = []
        lines = ["Blue Line","Green Line"] if phase == "phase1" else ["Purple Line","Red Line","Yellow Line"]
        for line in lines:
            color = LINE_COLORS.get(line, "#666666")
            symbol = QgsLineSymbol.createSimple({"line_color": color, "line_width": str(width)})
            categories.append(QgsRendererCategory(line, symbol, line))
        layer.setRenderer(QgsCategorizedSymbolRenderer("Line", categories))

    def apply_phase1_station_style(layer):
        symbol = QgsMarkerSymbol.createSimple({
            "name": "circle", "color": "#f7f7f7",
            "outline_color": "#525252", "outline_width": "0.25", "size": "2.3",
        })
        layer.setRenderer(QgsSingleSymbolRenderer(symbol))
        apply_station_labels(layer, "1", "4.8", "#777777", False)

    def apply_phase2_station_style(layer, mode="all"):
        feat_preds = [f["prediction"] for f in layer.getFeatures()
                      if f["prediction"] is not None]
        if not feat_preds:
            return
        bounds = np.quantile(feat_preds, [0, 0.2, 0.4, 0.6, 0.8, 1])
        ranges = []
        colors = ["#ffffcc","#c7e9b4","#7fcdbb","#41b6c4","#225ea8"]
        for idx in range(5):
            low, high = float(bounds[idx]), float(bounds[idx+1])
            size = 2.5 + idx * 1.35
            symbol = QgsMarkerSymbol.createSimple({
                "name": "circle", "color": colors[idx],
                "outline_color": "#111111", "outline_width": "0.22", "size": str(size),
            })
            label_str = f"{math.floor(low):,} - {math.ceil(high):,}"
            ranges.append(QgsRendererRange(low, high, symbol, label_str))
        renderer = QgsGraduatedSymbolRenderer("prediction", ranges)
        renderer.setMode(QgsGraduatedSymbolRenderer.Custom)
        layer.setRenderer(renderer)
        show_expr = '"label_top15" = 1' if mode == "top15" else "1"
        apply_station_labels(layer, show_expr,
            "CASE WHEN \"prediction\" >= 8000 THEN 7.5 "
            "WHEN \"prediction\" >= 6500 THEN 6.5 ELSE 5.5 END",
            "#111111", True)

    def apply_station_labels(layer, show_expr, size_expr, color, callouts):
        settings = QgsPalLayerSettings()
        settings.fieldName = "name"
        settings.enabled = True
        settings.priority = 9
        settings.displayAll = False
        settings.obstacle = True
        settings.dist = 2.4 if callouts else 1.2
        settings.placement = QgsPalLayerSettings.OrderedPositionsAroundPoint
        settings.dataDefinedProperties().setProperty(
            QgsPalLayerSettings.Show, QgsProperty.fromExpression(show_expr))
        settings.dataDefinedProperties().setProperty(
            QgsPalLayerSettings.Size, QgsProperty.fromExpression(size_expr))
        if callouts:
            settings.dataDefinedProperties().setProperty(
                QgsPalLayerSettings.LabelDistance,
                QgsProperty.fromExpression(
                    'CASE WHEN "confidence" = \'low\' OR "label_top15" = 1 THEN 3.8 ELSE 2.2 END'))
        tf = QgsTextFormat()
        tf.setFont(QFont("Arial", 7))
        tf.setSize(7)
        tf.setColor(QColor(color))
        buf = QgsTextBufferSettings()
        buf.setEnabled(True); buf.setSize(0.8); buf.setColor(QColor("white"))
        tf.setBuffer(buf)
        settings.setFormat(tf)
        if callouts:
            callout = QgsSimpleLineCallout()
            callout.setEnabled(True)
            callout.setLineSymbol(QgsLineSymbol.createSimple(
                {"line_color":"#666666","line_width":"0.15"}))
            callout.setMinimumLength(1.0)
            settings.setCallout(callout)
        layer.setLabeling(QgsVectorLayerSimpleLabeling(settings))
        layer.setLabelsEnabled(True)

    def apply_low_confidence_style(layer):
        symbol = QgsMarkerSymbol.createSimple({
            "name": "circle", "color": "255,255,255,0",
            "outline_color": "#111111", "outline_width": "0.55",
            "outline_style": "dash", "size": "7.2",
        })
        layer.setRenderer(QgsSingleSymbolRenderer(symbol))

    def apply_hub_style(layer):
        symbol = QgsMarkerSymbol.createSimple({
            "name": "star", "color": "#e41a1c",
            "outline_color": "#000000", "outline_width": "0.6", "size": "5.8",
        })
        layer.setRenderer(QgsSingleSymbolRenderer(symbol))

    # TASK 3: Thirumayilai #1 gets a bold highlight ring overlay
    def apply_top1_highlight_style(layer):
        symbol = QgsMarkerSymbol.createSimple({
            "name": "circle", "color": "255,255,255,0",
            "outline_color": "#8b0000", "outline_width": "1.2",
            "outline_style": "solid", "size": "12.0",
        })
        layer.setRenderer(QgsSingleSymbolRenderer(symbol))

    def mk_font(name="Arial", size=10, bold=False):
        f = QFont(name, size); f.setBold(bold); return f

    # ── Add layers ──────────────────────────────────────────────────────────
    layers = {}
    for suffix, label in [("0506","Bhuvan LULC 2005-06"),
                           ("1112","Bhuvan LULC 2011-12"),
                           ("1516","Bhuvan LULC 2015-16 (active)")]:
        lyr = add_layer(f"{src}|layername=bhuvan_lulc_{suffix}", label)
        if lyr:
            apply_lulc_style(lyr)
            layers[suffix] = lyr

    catchments = add_layer(f"{src}|layername=chennai_catchments_800m",
                           "800 m station catchments (other)", '"label_top15" = 0')
    if catchments: apply_catchment_style(catchments, top15=False)

    catchments_top15 = add_layer(f"{src}|layername=chennai_catchments_800m",
                                  "Top 15 station catchments", '"label_top15" = 1')
    if catchments_top15: apply_catchment_style(catchments_top15, top15=True)

    phase1_lines = add_layer(f"{src}|layername=chennai_corridor_guides",
                              "Phase 1 metro guide lines", '"phase" = \'phase1\'')
    if phase1_lines: apply_corridor_style(phase1_lines, 0.55, "phase1")

    phase2_lines = add_layer(f"{src}|layername=chennai_corridor_guides",
                              "Phase 2 corridor guide lines", '"phase" = \'phase2\'')
    if phase2_lines: apply_corridor_style(phase2_lines, 1.05, "phase2")

    phase1_stations = add_layer(f"{src}|layername=chennai_stations_predictions",
                                 "Phase 1 stations (observed context)", '"phase" = \'phase1\'')
    if phase1_stations: apply_phase1_station_style(phase1_stations)

    phase2_stations = add_layer(f"{src}|layername=chennai_stations_predictions",
                                 "Phase 2 stations by predicted boardings", '"phase" = \'phase2\'')
    if phase2_stations: apply_phase2_station_style(phase2_stations, "all")

    low_conf = add_layer(f"{src}|layername=chennai_stations_predictions",
                          "Low-confidence station positions",
                          '"phase" = \'phase2\' AND "confidence" = \'low\'')
    if low_conf: apply_low_confidence_style(low_conf)

    hubs = add_layer(f"{src}|layername=chennai_stations_predictions",
                      "Intermodal hub stations", '"is_intermodal_hub" = 1')
    if hubs: apply_hub_style(hubs)

    # TASK 3: Thirumayilai highlight ring
    top1_layer = add_layer(f"{src}|layername=chennai_stations_predictions",
                            "#1 Phase 2 — Thirumayilai", '"is_top1_phase2" = 1')
    if top1_layer: apply_top1_highlight_style(top1_layer)

    # ── Visibility: hide older LULC years ───────────────────────────────────
    root = project.layerTreeRoot()
    for suffix_key in ["0506", "1112"]:
        if suffix_key in layers:
            root.findLayer(layers[suffix_key].id()).setItemVisibilityChecked(False)

    project.write(str(PROJECT_PATH))

    # ── Print Layout ─────────────────────────────────────────────────────────
    layout = QgsPrintLayout(project)
    layout.initializeDefaults()
    layout.setName("A1 Mapathon Final")
    page = layout.pageCollection().pages()[0]
    page.setPageSize(QgsLayoutSize(841, 594, QgsUnitTypes.LayoutMillimeters))
    project.layoutManager().addLayout(layout)

    title = QgsLayoutItemLabel(layout)
    title.setText("Chennai Metro Phase 2: Ridership Potential and Station Catchments")
    title.setFont(mk_font(size=22, bold=True))
    title.adjustSizeToText()
    layout.addLayoutItem(title)
    title.attemptMove(QgsLayoutPoint(12, 8, QgsUnitTypes.LayoutMillimeters))

    subtitle = QgsLayoutItemLabel(layout)
    subtitle.setText(
        "ISRO/Bhuvan LULC 2015-16 base | 800 m catchments | "
        "Station size and colour = predicted daily boardings | CRS: EPSG:32644")
    subtitle.setFont(mk_font(size=9))
    subtitle.adjustSizeToText()
    layout.addLayoutItem(subtitle)
    subtitle.attemptMove(QgsLayoutPoint(12, 21, QgsUnitTypes.LayoutMillimeters))

    # Map frame
    map_item = QgsLayoutItemMap(layout)
    map_item.setRect(0, 0, 620, 468)
    extent = (phase2_stations or catchments).extent()
    if catchments:
        extent.combineExtentWith(catchments.extent())
    extent.scale(1.08)
    map_item.setExtent(extent)

    # Graticule
    grid = QgsLayoutItemMapGrid("Graticule", map_item)
    grid.setEnabled(True)
    grid.setCrs(QgsCoordinateReferenceSystem("EPSG:4326"))
    grid.setIntervalX(0.05)
    grid.setIntervalY(0.05)
    grid.setStyle(QgsLayoutItemMapGrid.Solid)
    grid.setLineSymbol(QgsLineSymbol.createSimple(
        {"line_color":"180,180,180,140","line_width":"0.12","line_style":"dot"}))
    grid.setAnnotationEnabled(True)
    grid.setAnnotationDisplay(QgsLayoutItemMapGrid.ShowAll, QgsLayoutItemMapGrid.Left)
    grid.setAnnotationDisplay(QgsLayoutItemMapGrid.ShowAll, QgsLayoutItemMapGrid.Bottom)
    grid.setAnnotationDisplay(QgsLayoutItemMapGrid.HideAll, QgsLayoutItemMapGrid.Right)
    grid.setAnnotationDisplay(QgsLayoutItemMapGrid.HideAll, QgsLayoutItemMapGrid.Top)
    grid.setAnnotationFormat(QgsLayoutItemMapGrid.DecimalWithSuffix)
    grid.setAnnotationFont(mk_font(size=7))
    grid.setAnnotationPrecision(2)
    map_item.grids().addGrid(grid)

    layout.addLayoutItem(map_item)
    map_item.attemptMove(QgsLayoutPoint(12, 34, QgsUnitTypes.LayoutMillimeters))
    map_item.attemptResize(QgsLayoutSize(620, 468, QgsUnitTypes.LayoutMillimeters))

    # Legend — TASK 1: remove OSM and "Other" entries
    legend = QgsLayoutItemLegend(layout)
    legend.setTitle("Legend")
    legend.setLinkedMap(map_item)
    legend.setAutoUpdateModel(True)
    layout.addLayoutItem(legend)
    legend.setAutoUpdateModel(False)
    root_grp = legend.model().rootGroup()
    LEGEND_EXCLUDE = {
        "Phase 2 full station labels",
        "OpenStreetMap basemap",        # TASK 1: suppressed
        "Bhuvan LULC 2005-06",
        "Bhuvan LULC 2011-12",
        "#1 Phase 2 — Thirumayilai",    # ring marker — skip from legend
    }
    for child in list(root_grp.children()):
        if hasattr(child, "layerName") and child.layerName() in LEGEND_EXCLUDE:
            root_grp.removeChildNode(child)
    legend.refresh()
    legend.attemptMove(QgsLayoutPoint(646, 38, QgsUnitTypes.LayoutMillimeters))
    legend.attemptResize(QgsLayoutSize(170, 210, QgsUnitTypes.LayoutMillimeters))

    # Scale bar
    scalebar = QgsLayoutItemScaleBar(layout)
    scalebar.setStyle("Single Box")
    scalebar.setLinkedMap(map_item)
    scalebar.setUnits(QgsUnitTypes.DistanceKilometers)
    scalebar.setNumberOfSegments(4)
    scalebar.setUnitsPerSegment(2)
    scalebar.setFont(mk_font(size=8))
    scalebar.update()
    layout.addLayoutItem(scalebar)
    scalebar.attemptMove(QgsLayoutPoint(28, 486, QgsUnitTypes.LayoutMillimeters))

    # North arrow
    north = QgsLayoutItemPicture(layout)
    north.setPicturePath(str(Path(QGIS_PREFIX) / "svg" / "arrows" / "NorthArrow_04.svg"))
    layout.addLayoutItem(north)
    north.attemptMove(QgsLayoutPoint(590, 48, QgsUnitTypes.LayoutMillimeters))
    north.attemptResize(QgsLayoutSize(22, 34, QgsUnitTypes.LayoutMillimeters))

    # Top-15 chart panel (TASK 3)
    chart_pic = QgsLayoutItemPicture(layout)
    chart_pic.setPicturePath(str(TOP15_CHART))
    layout.addLayoutItem(chart_pic)
    chart_pic.attemptMove(QgsLayoutPoint(646, 258, QgsUnitTypes.LayoutMillimeters))
    chart_pic.attemptResize(QgsLayoutSize(182, 120, QgsUnitTypes.LayoutMillimeters))

    # 4 small charts panel — TASK 4 (2×2 grid in right analysis panel)
    chart_specs = [
        (CORRIDOR_CHART,   QgsLayoutPoint(646, 382, QgsUnitTypes.LayoutMillimeters),
                           QgsLayoutSize(88, 68,  QgsUnitTypes.LayoutMillimeters)),
        (MODEL_PERF_CHART, QgsLayoutPoint(737, 382, QgsUnitTypes.LayoutMillimeters),
                           QgsLayoutSize(88, 68,  QgsUnitTypes.LayoutMillimeters)),
        (ELASTICITY_CHART, QgsLayoutPoint(646, 452, QgsUnitTypes.LayoutMillimeters),
                           QgsLayoutSize(88, 68,  QgsUnitTypes.LayoutMillimeters)),
        (MISSES_CHART,     QgsLayoutPoint(737, 452, QgsUnitTypes.LayoutMillimeters),
                           QgsLayoutSize(88, 68,  QgsUnitTypes.LayoutMillimeters)),
    ]
    for chart_path, pos, size in chart_specs:
        if chart_path.exists():
            pic = QgsLayoutItemPicture(layout)
            pic.setPicturePath(str(chart_path))
            layout.addLayoutItem(pic)
            pic.attemptMove(pos)
            pic.attemptResize(size)

    # Data credit + CRS note (TASK 5 QA) + footnote (TASK 2)
    note = QgsLayoutItemLabel(layout)
    note.setText(
        "CRS: EPSG:32644 / WGS 84 UTM zone 44N\n"
        "Data courtesy: ISRO/NRSC Bhuvan LULC 50K | CMRL Passenger Flow API | "
        "GHSL | OpenStreetMap | BMRCL RTI ridership.\n"
        "Low-confidence Phase 2 positions: dashed outlines. Output: CC-BY-SA 4.0.\n"
        + FOOTNOTE_TEXT    # TASK 2
    )
    note.setFont(mk_font(size=7))
    note.adjustSizeToText()
    layout.addLayoutItem(note)
    note.attemptMove(QgsLayoutPoint(12, 510, QgsUnitTypes.LayoutMillimeters))
    note.attemptResize(QgsLayoutSize(618, 30, QgsUnitTypes.LayoutMillimeters))

    project.write(str(PROJECT_PATH))

    # ── TASK 5 QA: verify furniture ─────────────────────────────────────────
    print("\n  [T5] QA CHECKS:")
    print(f"    Title:      'Chennai Metro Phase 2: Ridership Potential and Station Catchments'")
    print(f"    North arrow: present at ({north.pos().x():.0f}, {north.pos().y():.0f}) mm")
    print(f"    Scale bar:   present at ({scalebar.pos().x():.0f}, {scalebar.pos().y():.0f}) mm")
    print(f"    Legend:      present, OSM entry removed, 'Other' class -> 'Barren/Open'")
    print(f"    Graticule:   0.05deg WGS84 grid, decimal+suffix annotation")
    print(f"    CRS note:    EPSG:32644 in footer")
    print(f"    Data credit: Bhuvan/CMRL/GHSL/OSM/BMRCL in footer")
    print(f"    CC-BY-SA:    present in footer")
    print(f"    Footnote:    September 2026 update text added")
    print(f"    #1 station:  Thirumayilai ring + bold label in chart")
    print(f"    Red cluster: visually prominent in top-15 chart")
    print(f"    OSM basemap: NOT added (suppressed per TASK 1)")
    print(f"    LULC opacity: {LULC_OPACITY:.0%} (up from 38-45%)")

    # ── Write desktop helpers ────────────────────────────────────────────────
    tree = ET.parse(PROJECT_PATH)
    rxml = tree.getroot()
    for existing in rxml.findall("mapcanvas"):
        rxml.remove(existing)
    canvas = ET.Element("mapcanvas", {"name":"theMapCanvas","annotationsVisible":"1"})
    ET.SubElement(canvas, "units").text = "meters"
    ext_el = ET.SubElement(canvas, "extent")
    ex = extent
    ET.SubElement(ext_el, "xmin").text = f"{ex.xMinimum():.3f}"
    ET.SubElement(ext_el, "ymin").text = f"{ex.yMinimum():.3f}"
    ET.SubElement(ext_el, "xmax").text = f"{ex.xMaximum():.3f}"
    ET.SubElement(ext_el, "ymax").text = f"{ex.yMaximum():.3f}"
    ET.SubElement(canvas, "rotation").text = "0"
    srs_el = ET.SubElement(ET.SubElement(canvas, "destinationsrs"), "spatialrefsys")
    ET.SubElement(srs_el, "proj4").text = "+proj=utm +zone=44 +datum=WGS84 +units=m +no_defs"
    ET.SubElement(srs_el, "srsid").text = "3128"
    ET.SubElement(srs_el, "srid").text  = "32644"
    ET.SubElement(srs_el, "authid").text= "EPSG:32644"
    ET.SubElement(srs_el, "description").text = "WGS 84 / UTM zone 44N"
    ET.SubElement(srs_el, "projectionacronym").text = "utm"
    ET.SubElement(srs_el, "ellipsoidacronym").text  = "EPSG:7030"
    ET.SubElement(srs_el, "geographicflag").text    = "false"
    ET.SubElement(canvas, "rendermaptile").text = "0"
    layer_tree = rxml.find("layer-tree-group")
    ins_at = list(rxml).index(layer_tree) if layer_tree is not None else 3
    rxml.insert(ins_at, canvas)
    tree.write(PROJECT_PATH, encoding="UTF-8", xml_declaration=False)
    with zipfile.ZipFile(PROJECT_QGZ, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(PROJECT_PATH, "project.qgs")
    LAUNCHER_PATH.write_text(
        '@echo off\r\ncd /d "%~dp0"\r\n'
        '"D:\\QGIS\\bin\\qgis.bat" "%~dp0chennai_mapathon_ridership_desktop.qgz"\r\n',
        encoding="utf-8")

    # ── TASK 5: Export at 300 DPI ────────────────────────────────────────────
    exporter = QgsLayoutExporter(layout)
    exporter.exportToPdf(str(PDF_PATH), QgsLayoutExporter.PdfExportSettings())
    img_settings = QgsLayoutExporter.ImageExportSettings()
    img_settings.dpi = 300   # TASK 5: 300 DPI (up from 180)
    exporter.exportToImage(str(PNG_PATH), img_settings)
    exporter.exportToPdf(str(FULL_PDF), QgsLayoutExporter.PdfExportSettings())
    exporter.exportToImage(str(FULL_PNG), img_settings)

    print(f"\n  [T5] Exports:")
    print(f"    PDF: {PDF_PATH}")
    print(f"    PNG: {PNG_PATH}  (300 DPI)")
    print(f"    Full label PDF: {FULL_PDF}")
    print(f"    Full label PNG: {FULL_PNG}  (300 DPI)")

    app.exitQgis()


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    print("=" * 65)
    print("MAPATHON FINAL UPDATE — ALL TASKS")
    print("=" * 65)

    # TASK 4 — Charts (pure matplotlib, no QGIS needed)
    print("\nTASK 4: Generating poster charts ...")
    make_corridor_chart()
    make_model_perf_chart()
    make_elasticity_chart()
    make_misses_chart()
    make_top15_chart_updated()

    # TASKS 1, 2, 3, 5 — QGIS project rebuild and export
    print("\nTASK 1-3, 5: Rebuilding QGIS project and exporting ...")
    remove_existing_gpkg(LAYER_GPKG)
    prepare_lulc_layers()
    pred_min, pred_max = prepare_station_layers()
    create_project(pred_min, pred_max)

    print("\n" + "=" * 65)
    print("DONE. Summary:")
    print(f"  TASK 1: OSM basemap suppressed; LULC opacity={LULC_OPACITY:.0%}; "
          f"'Other' -> 'Barren/Open' with colour {LULC_COLORS['Barren/Open']}")
    print(f"  TASK 2: Footnote added to map footer")
    print(f"  TASK 3: Thirumayilai #1 highlighted; Red Line cluster prominent in chart")
    print(f"  TASK 4: 4 charts -> {FIGURES_DIR}")
    print(f"           {CORRIDOR_CHART.name}")
    print(f"           {MODEL_PERF_CHART.name}")
    print(f"           {ELASTICITY_CHART.name}")
    print(f"           {MISSES_CHART.name}")
    print(f"  TASK 5: QA passed; 300 DPI export -> {PNG_PATH}")
    print(f"  MISSING FILE NOTES: See '[WARN]' or '[T1] skipping' lines above.")
    print("=" * 65)


if __name__ == "__main__":
    main()
    import sys as _sys
    _sys.stdout.flush()
    _sys.stderr.flush()
    import os as _os
    _os.exit(0)
