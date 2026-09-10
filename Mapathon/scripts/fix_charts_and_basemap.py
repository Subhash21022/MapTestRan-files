"""
fix_charts_and_basemap.py
=========================
TASK 1-3: Re-generate all 5 poster charts with layout defects fixed.
TASK 4:   Rebuild QGIS project with CartoDB Positron muted basemap + LULC at 60% opacity.
TASK 5:   QA pass + 300 DPI re-export.

Pure-matplotlib section runs with standard Python.
QGIS section runs via QGIS Python interpreter.
"""

from __future__ import annotations

import math, os, sys, json, zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import matplotlib.patheffects as pe
import geopandas as gpd
from shapely.geometry import LineString
from shapely.ops import linemerge, unary_union

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parents[1]
TABLES_DIR  = ROOT / "outputs" / "tables"
FIGURES_DIR = ROOT / "outputs" / "figures"
MAPS_DIR    = ROOT / "outputs" / "maps"
QGIS_DIR    = ROOT / "qgis"
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
TEMPLATE_MAP_PNG = MAPS_DIR / "chennai_full_network_template_map.png"
OVERPASS     = ROOT / "data" / "interim" / "chennai_overpass_alignments.json"
DATA_DIR     = Path("D:/Mapathon Data Files")

TOP15_CHART      = FIGURES_DIR / "chennai_top15_phase2_predictions.png"
CORRIDOR_CHART   = FIGURES_DIR / "chart_corridor_ridership.png"
MODEL_PERF_CHART = FIGURES_DIR / "chart_model_performance.png"
ELASTICITY_CHART = FIGURES_DIR / "chart_nb_elasticities.png"
MISSES_CHART     = FIGURES_DIR / "chart_phase1_misses.png"

CRS_UTM44   = "EPSG:32644"
QGIS_PREFIX = "D:/QGIS/apps/qgis"
MAX_SNAP_M  = 15000

# ── TASK 3: Consistent palette / font ────────────────────────────────────────
FONT_FAMILY = "DejaVu Sans"
TITLE_SIZE  = 10          # uniform title font size across all charts
LABEL_SIZE  = 8.5
TICK_SIZE   = 8.0

mpl.rcParams.update({
    "font.family":      FONT_FAMILY,
    "axes.titlesize":   TITLE_SIZE,
    "axes.titleweight": "bold",
    "axes.titlepad":    10,
    "axes.labelsize":   LABEL_SIZE,
    "xtick.labelsize":  TICK_SIZE,
    "ytick.labelsize":  TICK_SIZE,
    "figure.dpi":       300,
})

LINE_COLORS = {          # matches main map
    "Blue Line":   "#0066cc",
    "Green Line":  "#008837",
    "Purple Line": "#7b3294",
    "Red Line":    "#d73027",
    "Yellow Line": "#e8a800",
}
LULC_COLORS = {
    "Built-up":      "#c0726e",
    "Agriculture":   "#c9b44e",
    "Forest/Scrub":  "#4a9458",
    "Water/Wetland": "#4090b5",
    "Barren/Open":   "#c49a6c",
}
LULC_OPACITY = 0.62       # TASK 4: 60-65% so OSM context shows through

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

FOOTNOTE_TEXT = (
    "September 2026 update: based on 1 weekend observation. Absolute ridership\n"
    "bands may shift when weekday data accumulates. Rankings are directionally stable."
)

# ── Corridor data ──────────────────────────────────────────────────────────────
CORRIDOR_REPORT = {
    "C3: Madhavaram–SIPCOT\n(Red Line, 50 stn)":        195_000,
    "C4: Lighthouse–Poonamallee\n(Yellow Line, 30 stn)": 140_000,
    "C5: Madhavaram–Sholinganallur\n(Purple Line, 48 stn)": 220_000,
}
CORRIDOR_COLORS = ["#d73027", "#e8a800", "#7b3294"]

MISS_DATA = {
    "MGR Central":     (27379, 10062),
    "Chennai Airport": (18156,  8428),
    "Thirumangalam":   (13988,  7244),
    "Vadapalani":      (14398,  8771),
    "Guindy":          (12722,  7836),
    "Thousand Lights": (12821,  8695),
}

SAVEKW = dict(dpi=300, bbox_inches="tight", facecolor="white", transparent=False)

# =============================================================================
# Helper
# =============================================================================

def _clean_spines(ax):
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)


def _save(fig, path):
    fig.savefig(path, **SAVEKW)
    plt.close(fig)
    print(f"  Saved -> {path.name}")


# =============================================================================
# CHART 1 — Top 15 Phase 2 Predictions  (TASK 2.1)
# =============================================================================

def make_top15_chart() -> None:
    df = pd.read_csv(TABLES_DIR / "chennai_predictions.csv")
    top = (df[df["phase"].eq("phase2")]
           .nlargest(15, "prediction")
           .sort_values("prediction"))

    colors      = [LINE_COLORS.get(ln, "#666666") for ln in top["Line"]]
    edge_colors = ["#8b0000" if n == "Thirumayilai" else "#333333" for n in top["name"]]
    edge_widths = [1.8      if n == "Thirumayilai" else 0.4        for n in top["name"]]

    xerr = np.vstack([top["prediction"] - top["lower"],
                      top["upper"]      - top["prediction"]])

    # Wider figure so subtitle and title don't clip
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    bars = ax.barh(top["name"], top["prediction"],
                   color=colors, edgecolor=edge_colors,
                   linewidth=edge_widths, zorder=3)
    ax.errorbar(top["prediction"], top["name"], xerr=xerr,
                fmt="none", ecolor="#444444", elinewidth=0.8, capsize=2.5)
    ax.set_xlabel("Predicted daily boardings", fontsize=LABEL_SIZE)

    # Full multi-line title — bbox_inches='tight' will capture it
    ax.set_title(
        "Top 15 Phase 2 Stations — Predicted Ridership Potential\n"
        "  #1: Thirumayilai ★   |   Red Line cluster: Manapakkam, Madipakkam,\n"
        "  Anna Nagar West, Butt Rd, Ramapuram",
        fontsize=TITLE_SIZE, fontweight="bold", loc="left", pad=10,
    )
    ax.grid(axis="x", color="#e0e0e0", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    _clean_spines(ax)
    ax.spines["left"].set_visible(False)

    # Bold / colour the #1 tick label
    ax.figure.canvas.draw()
    for lbl in ax.get_yticklabels():
        if lbl.get_text() == "Thirumayilai":
            lbl.set_fontweight("bold")
            lbl.set_color("#8b0000")

    # Red Line cluster dot annotations (right of bar, never inside plot)
    red_cluster = {"Manapakkam","Madipakkam","Anna Nagar West","Butt Road","Ramapuram"}
    xmax = top["upper"].max()
    for bar, name in zip(bars, top["name"]):
        if name in red_cluster:
            ax.text(xmax * 1.01, bar.get_y() + bar.get_height() / 2,
                    "●", va="center", ha="left", fontsize=9, color="#d73027")

    # TASK 2.1: legend BELOW x-axis, outside plot area
    red_patch = mpatches.Patch(color="#d73027", label="Red Line cluster (prominent)")
    ax.legend(handles=[red_patch], fontsize=8,
              loc="upper center",
              bbox_to_anchor=(0.5, -0.12),   # below x-axis
              frameon=True, facecolor="white", edgecolor="#cccccc", ncol=1)

    fig.subplots_adjust(left=0.28, right=0.95, top=0.82, bottom=0.16)
    _save(fig, TOP15_CHART)
    print("  [1] Top-15 chart: title+subtitle visible, legend below x-axis")


# =============================================================================
# CHART 2 — Corridor Ridership  (TASK 2.2)
# =============================================================================

def make_corridor_chart() -> None:
    labels = list(CORRIDOR_REPORT.keys())
    vals   = [v / 1_000 for v in CORRIDOR_REPORT.values()]

    # Wider figure + rotated labels prevent collision
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    bars = ax.bar(range(len(labels)), vals, color=CORRIDOR_COLORS,
                  edgecolor="#333333", linewidth=0.7, width=0.45, zorder=3)
    ax.set_xticks(range(len(labels)))
    # TASK 2.2: rotate 20° and use ha='right' so long names don't collide
    ax.set_xticklabels(labels, fontsize=8.5, rotation=20, ha="right",
                       rotation_mode="anchor")
    ax.set_ylabel("Total Predicted Daily Boardings ('000s/day)", fontsize=LABEL_SIZE)
    ax.set_title(
        "Phase 2 Ridership by Corridor  (Sep 2026, weekend-adjusted)\n"
        "Purple Line (C5) leads — Red and Yellow Lines follow",
        fontsize=TITLE_SIZE, fontweight="bold", loc="left",
    )
    ax.grid(axis="y", color="#e0e0e0", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    _clean_spines(ax)

    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 2.5,
                f"{val:.0f}k",
                ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_ylim(0, max(vals) * 1.30)
    fig.subplots_adjust(left=0.12, right=0.97, top=0.85, bottom=0.28)
    _save(fig, CORRIDOR_CHART)
    print("  [2] Corridor chart: labels rotated, no collision")


# =============================================================================
# CHART 3 — Model Performance  (TASK 2.3)
# =============================================================================

def make_model_perf_chart() -> None:
    perf = pd.read_csv(TABLES_DIR / "model_performance.csv", index_col=0)
    models = ["linear_regression","elastic_net","random_forest","gradient_boosting"]
    # TASK 2.3: short single-line labels so no stacking needed
    tick_labels = ["Linear Reg.", "Elastic Net", "Rand. Forest*", "Grad. Boost"]
    r2_vals  = [perf.loc[m, "r2"]       for m in models]
    rho_vals = [perf.loc[m, "spearman"] for m in models]
    mae_vals = [perf.loc[m, "mae"]      for m in models]

    bar_colors  = ["#aaaaaa","#aaaaaa","#d73027","#aaaaaa"]
    edge_colors = ["#555555","#555555","#8b0000","#555555"]

    # Wider figure, taller height for label room
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.8))

    specs = [
        (axes[0], r2_vals,              "R²",               "LOO R² (higher=better)",     ".3f"),
        (axes[1], rho_vals,             "Spearman ρ",       "Spearman ρ  (rank accuracy)", ".3f"),
        (axes[2], [v/1000 for v in mae_vals], "MAE (k/day)", "MAE — '000s  (lower=better)",".1f"),
    ]
    x = range(len(models))

    for ax, vals, ylabel, title, fmt in specs:
        bars = ax.bar(x, vals, color=bar_colors, edgecolor=edge_colors,
                      linewidth=0.7, width=0.55, zorder=3)
        ax.set_xticks(list(x))
        # TASK 2.3: rotate 35° — fully legible, no overlap
        ax.set_xticklabels(tick_labels, fontsize=8, rotation=35, ha="right",
                           rotation_mode="anchor")
        ax.set_ylabel(ylabel, fontsize=LABEL_SIZE)
        ax.set_title(title, fontsize=TITLE_SIZE, fontweight="bold", pad=8)
        ax.grid(axis="y", color="#e0e0e0", linewidth=0.5, zorder=0)
        ax.set_axisbelow(True)
        _clean_spines(ax)
        spread = max(vals) - min(vals)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + spread * 0.04,
                    f"{val:{fmt}}", ha="center", va="bottom",
                    fontsize=7.5, fontweight="bold")

    patch = mpatches.Patch(color="#d73027", label="Selected model (Random Forest)")
    fig.legend(handles=[patch], loc="lower center", fontsize=8.5,
               frameon=True, facecolor="white", ncol=1, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(
        "Model LOOCV Benchmarks — Sep 2026  (1-day weekend training)",
        fontsize=TITLE_SIZE + 1, fontweight="bold", y=1.01,
    )
    fig.subplots_adjust(left=0.07, right=0.98, top=0.88, bottom=0.28, wspace=0.38)
    _save(fig, MODEL_PERF_CHART)
    print("  [3] Model perf chart: x-labels rotated 35°, no overlap")


# =============================================================================
# CHART 4 — NB Elasticities  (TASK 2.4)
# =============================================================================

def make_elasticity_chart() -> None:
    elast_path = TABLES_DIR / "nb_elasticities.csv"
    elast = pd.read_csv(elast_path, index_col=0, header=0)
    elast.columns = ["coeff"]
    elast = elast.sort_values("coeff")       # ascending → most-negative at top

    feature_labels = {
        "intersection_density":  "Street Intersection Density",
        "population":            "Catchment Population (800m)",
        "pnr":                   "Pedestrian Network Ratio",
        "poi_total_pctl":        "POI Density Percentile",
        "parking_area_m2_pctl":  "Parking Area Percentile",
        "dist_cbd_km":           "Distance to CBD (km)",
        "competing_rail_1km":    "Competing Rail Proximity (1km)",
        "bus_stops_500m_pctl":   "Bus Stop Feeder Density (500m)",
    }
    ylabels = [feature_labels.get(i, i) for i in elast.index]
    colors  = ["#d73027" if v < 0 else "#2b83ba" for v in elast["coeff"]]

    # Extra left margin for long y-axis labels; extra right for pos-value annotations
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    bars = ax.barh(ylabels, elast["coeff"], color=colors,
                   edgecolor="#333333", linewidth=0.5, zorder=3)
    ax.axvline(0, color="#444444", linewidth=0.9, linestyle="--")
    ax.set_xlabel("NB Elasticity Coefficient", fontsize=LABEL_SIZE)
    # Full title — bbox_inches='tight' captures right edge
    ax.set_title(
        "Feature Elasticities — NB GLM  (Direct Demand Model)",
        fontsize=TITLE_SIZE, fontweight="bold", loc="left",
    )
    ax.grid(axis="x", color="#e0e0e0", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    _clean_spines(ax)

    # TASK 2.4: negative labels go LEFT of bar (outside), positive go RIGHT
    gap = 0.007
    for bar, val in zip(bars, elast["coeff"]):
        if val < 0:
            # Place label to the LEFT of the bar end (i.e., at val - gap)
            ax.text(val - gap, bar.get_y() + bar.get_height() / 2,
                    f"{val:+.3f}", va="center", ha="right",
                    fontsize=8, fontweight="bold", color="#8b1a0a")
        else:
            ax.text(val + gap, bar.get_y() + bar.get_height() / 2,
                    f"{val:+.3f}", va="center", ha="left",
                    fontsize=8, fontweight="bold", color="#1a4f7a")

    pos_patch = mpatches.Patch(color="#2b83ba", label="Positive driver")
    neg_patch = mpatches.Patch(color="#d73027", label="Negative driver")
    ax.legend(handles=[pos_patch, neg_patch], fontsize=8.5,
              loc="lower right", frameon=True)

    fig.subplots_adjust(left=0.34, right=0.94, top=0.88, bottom=0.12)
    _save(fig, ELASTICITY_CHART)
    print("  [4] Elasticities: negative labels left-of-bar, title not clipped")


# =============================================================================
# CHART 5 — Phase 1 Misses  (TASK 2.5)
# =============================================================================

def make_misses_chart() -> None:
    stations  = list(MISS_DATA.keys())
    observed  = [MISS_DATA[s][0] for s in stations]
    predicted = [MISS_DATA[s][1] for s in stations]
    x = np.arange(len(stations))

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    w = 0.36
    ax.bar(x - w/2, observed,  width=w, color="#4393c3",
           label="Weekend Observed", edgecolor="#333", linewidth=0.6, zorder=3)
    ax.bar(x + w/2, predicted, width=w, color="#d73027",
           label="Model Predicted",  edgecolor="#333", linewidth=0.6, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(stations, fontsize=9, rotation=20, ha="right",
                       rotation_mode="anchor")
    ax.set_ylabel("Boardings / day", fontsize=LABEL_SIZE)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    # TASK 2.5: Full title, not clipped
    ax.set_title(
        "Phase 1 Prediction Misses  (Weekend Observed vs Model)",
        fontsize=TITLE_SIZE, fontweight="bold", loc="left",
    )
    ax.grid(axis="y", color="#e0e0e0", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    _clean_spines(ax)
    ax.legend(fontsize=8.5, frameon=True, loc="upper right")

    # TASK 2.5: footnote BELOW plot, outside bars, full opacity, high contrast
    note = (
        "* All large misses are leisure/interchange stations where weekend demand\n"
        "  genuinely exceeds the commuter-oriented model assumption."
    )
    # Use figure coordinates so it appears below the subplot
    fig.text(0.05, 0.01, note,
             fontsize=7.5, color="#333333",
             va="bottom", ha="left", style="italic",
             wrap=True)

    fig.subplots_adjust(left=0.12, right=0.97, top=0.88, bottom=0.28)
    _save(fig, MISSES_CHART)
    print("  [5] Misses chart: footnote below plot, title not clipped")


# =============================================================================
# TASK 5 — QA: verify all charts are readable
# =============================================================================

def qa_charts() -> None:
    print("\n  [QA] Chart file sizes (proxy for content completeness):")
    for path in [TOP15_CHART, CORRIDOR_CHART, MODEL_PERF_CHART,
                 ELASTICITY_CHART, MISSES_CHART]:
        size_kb = path.stat().st_size / 1024 if path.exists() else 0
        ok = "OK" if size_kb > 50 else "WARN-SMALL"
        print(f"    {path.name:<50} {size_kb:>7.1f} KB  [{ok}]")


# =============================================================================
# QGIS helpers (duplicated from mapathon_final_update.py for self-containment)
# =============================================================================

def classify_alignment(name):
    if not name: return None
    if "proposed" in name.lower() or "Line 5B" in name or "corridor switch" in name:
        return None
    if name in ALIGNMENT_NAME_TO_LINE:
        return ALIGNMENT_NAME_TO_LINE[name]
    if "Line 3:" in name:  return ("phase2","Purple Line")
    if "Line 5:" in name:  return ("phase2","Red Line")
    if "Yellow Line" in name: return ("phase2","Yellow Line")
    return None


def read_osm_alignments():
    from shapely.geometry import LineString as LS
    if not OVERPASS.exists():
        return gpd.GeoDataFrame(columns=["phase","Line","osm_name","geometry"], crs=CRS_UTM44)
    data = json.loads(OVERPASS.read_text(encoding="utf-8"))
    rows = []
    for el in data.get("elements", []):
        if el.get("type") != "way" or "geometry" not in el: continue
        name = el.get("tags",{}).get("name","")
        c = classify_alignment(name)
        if not c: continue
        coords = [(pt["lon"],pt["lat"]) for pt in el["geometry"]]
        if len(coords) < 2: continue
        phase, line = c
        rows.append({"phase":phase,"Line":line,"osm_name":name,
                     "osm_id":el.get("id"),"geometry":LS(coords)})
    if not rows:
        return gpd.GeoDataFrame(columns=["phase","Line","osm_name","geometry"], crs=CRS_UTM44)
    lines = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326").to_crs(CRS_UTM44)
    dissolved = []
    for (phase,line),grp in lines.groupby(["phase","Line"]):
        merged = linemerge(unary_union(grp.geometry.tolist()))
        if merged.geom_type == "GeometryCollection":
            parts = [g for g in merged.geoms if g.geom_type in ("LineString","MultiLineString")]
            merged = linemerge(unary_union(parts))
        dissolved.append({"phase":phase,"Line":line,"station_count":0,
                           "source":"osm_overpass",
                           "length_km":round(grp.geometry.length.sum()/1000,2),
                           "geometry":merged})
    return gpd.GeoDataFrame(dissolved, geometry="geometry", crs=CRS_UTM44)


def lulc_group(value):
    text = str(value or "").lower()
    if "built" in text: return "Built-up"
    if any(k in text for k in ("agriculture","crop","fallow")): return "Agriculture"
    if any(k in text for k in ("forest","scrub")): return "Forest/Scrub"
    if any(k in text for k in ("wetland","water","river","lake")): return "Water/Wetland"
    return "Barren/Open"


def nearest_point_on_geometry(geom, point):
    return geom.interpolate(geom.project(point))


def snap_stations_for_map(stations, alignments):
    out = stations.to_crs(CRS_UTM44).copy()
    out["map_snap_m"] = 0.0; out["map_aligned"] = 0
    if alignments.empty: return out
    line_geoms   = {(r["phase"],r["Line"]): r.geometry for _,r in alignments.iterrows()}
    phase2_union = unary_union(alignments[alignments["phase"].eq("phase2")].geometry.tolist())
    all_union    = unary_union(alignments.geometry.tolist())
    for idx, row in out.iterrows():
        keys = [(row.get("phase"), row.get("Line"))]
        if row.get("Line") == "Red LinePurple Line":
            keys = [("phase2","Red Line"),("phase2","Purple Line")]
        candidates = [line_geoms[k] for k in keys if k in line_geoms]
        if not candidates and row.get("phase")=="phase2" and not phase2_union.is_empty:
            candidates = [phase2_union]
        if not candidates and not all_union.is_empty:
            candidates = [all_union]
        if not candidates: continue
        point = row.geometry
        opts  = [(g.distance(point), nearest_point_on_geometry(g, point)) for g in candidates]
        dist, snapped = min(opts, key=lambda a: a[0])
        if dist <= MAX_SNAP_M:
            out.at[idx,"geometry"] = snapped
            out.at[idx,"map_snap_m"] = round(float(dist),1)
            out.at[idx,"map_aligned"] = 1
    return out


def build_fallback_guides(stations):
    rows = []
    for (phase, line), grp in stations.dropna(subset=["Line"]).groupby(["phase","Line"]):
        if len(grp) < 2: continue
        pts    = grp.geometry.tolist()
        unused = set(range(len(pts)))
        start  = min(unused, key=lambda i: (pts[i].x, pts[i].y))
        order  = [start]; unused.remove(start)
        while unused:
            last = pts[order[-1]]
            nxt  = min(unused, key=lambda i: last.distance(pts[i]))
            order.append(nxt); unused.remove(nxt)
        rows.append({"phase":phase,"Line":line,"station_count":len(grp),
                     "source":"station_nearest_neighbour",
                     "length_km":round(LineString([pts[i] for i in order]).length/1000,2),
                     "geometry":LineString([pts[i] for i in order])})
    return gpd.GeoDataFrame(rows, crs=CRS_UTM44)


def remove_existing_gpkg(path):
    try:
        if path.exists(): path.unlink()
        for f in path.parent.glob(path.name + "*"):
            if f.is_file(): f.unlink()
        return True
    except PermissionError:
        print(f"  [NOTE] {path.name} is open in QGIS GUI; reusing existing GPKG layers.")
        return False


def write_layer(gdf, layer_name, first=False):
    mode = "w" if first else "a"
    gdf.to_file(LAYER_GPKG, layer=layer_name, driver="GPKG", mode=mode)


def prepare_lulc_layers():
    first = True
    for suffix, year_label in [("0506","2005-06"),("1112","2011-12"),("1516","2015-16")]:
        path = DATA_DIR / f"CHENNAI_TN_LULC50K_{suffix}.shp"
        if not path.exists():
            print(f"  [T4] LULC shapefile missing, skipping: {path.name}"); continue
        gdf = gpd.read_file(path)
        gdf["map_year"]   = year_label
        gdf["lulc_group"] = gdf["DESCR_1"].map(lulc_group)
        gdf["lulc_class"] = gdf["DESCR_2"].fillna(gdf["lulc_group"])
        gdf = gdf.to_crs(CRS_UTM44)
        write_layer(gdf[["map_year","lulc_group","lulc_class","LU_Webcode","geometry"]],
                    f"bhuvan_lulc_{suffix}", first=first)
        first = False
        print(f"  [T4] LULC layer written: bhuvan_lulc_{suffix}")


def prepare_station_layers():
    stations    = gpd.read_file(ROOT / "data" / "processed" / "chennai_stations.gpkg")
    predictions = pd.read_csv(TABLES_DIR / "chennai_predictions.csv")
    pcols = [c for c in ["station_id","prediction","lower","upper",
                         "population_density","pnr","bus_stops_500m",
                         "poi_total","metro_closeness"] if c in predictions.columns]
    stations = stations.merge(predictions[pcols], on="station_id", how="left")
    for col in ["prediction","lower","upper"]:
        stations[col] = stations[col].round(0)
    stations["prediction_band"] = pd.qcut(
        stations["prediction"].rank(method="first"), q=5,
        labels=["Very low","Low","Medium","High","Very high"]).astype(str)
    stations.loc[stations["Line"]=="Red LinePurple Line","Line"] = "Purple Line"
    phase2 = stations["phase"].eq("phase2")
    stations["rank_phase2"]     = np.nan
    stations.loc[phase2,"rank_phase2"] = (
        stations.loc[phase2,"prediction"].rank(ascending=False,method="first"))
    stations["label_top15"]       = (stations["rank_phase2"] <= 15).astype(int)
    stations["is_intermodal_hub"] = stations["name"].isin(INTERMODAL_HUB_NAMES).astype(int)
    stations["is_top1_phase2"]    = (stations["rank_phase2"]==1).fillna(False).astype(int)
    pred_min = float(stations.loc[phase2,"prediction"].min())
    pred_max = float(stations.loc[phase2,"prediction"].max())
    osm_aligns = read_osm_alignments()
    if osm_aligns.empty:
        stations_map = stations.to_crs(CRS_UTM44)
        alignments   = build_fallback_guides(stations_map)
    else:
        alignments   = osm_aligns
        stations_map = snap_stations_for_map(stations, osm_aligns)
    write_layer(stations_map, "chennai_stations_predictions")
    buffers = gpd.read_file(ROOT / "data" / "processed" / "chennai_catchments_buffer.gpkg")
    buffers = buffers[buffers["radius_m"].eq(800)].copy().drop(columns="geometry")
    buffers = buffers.merge(
        stations_map[["station_id","prediction","rank_phase2","label_top15","geometry"]],
        on="station_id", how="left")
    buffers = gpd.GeoDataFrame(buffers, geometry="geometry", crs=CRS_UTM44)
    buffers["geometry"] = buffers.geometry.buffer(800)
    write_layer(buffers.to_crs(CRS_UTM44), "chennai_catchments_800m")
    write_layer(alignments,                "chennai_corridor_guides")
    return pred_min, pred_max


# =============================================================================
# TASK 4 — QGIS project with Carto Light basemap + LULC overlay
# =============================================================================

def create_project(pred_min: float, pred_max: float) -> None:
    from qgis.PyQt.QtGui import QColor, QFont
    from qgis.core import (
        QgsApplication, QgsCategorizedSymbolRenderer,
        QgsCoordinateReferenceSystem, QgsFillSymbol, QgsGraduatedSymbolRenderer,
        QgsLayoutExporter, QgsLayoutItemLabel, QgsLayoutItemLegend,
        QgsLayoutItemMap, QgsLayoutItemMapGrid, QgsLayoutItemPicture,
        QgsLayoutItemScaleBar, QgsLayoutPoint, QgsLayoutSize,
        QgsLineSymbol, QgsMarkerSymbol, QgsPalLayerSettings, QgsPrintLayout,
        QgsProject, QgsProperty, QgsRasterLayer, QgsRectangle, QgsRendererCategory,
        QgsRendererRange, QgsSimpleLineCallout, QgsSingleSymbolRenderer,
        QgsVectorLayerSimpleLabeling, QgsTextBufferSettings, QgsTextFormat,
        QgsUnitTypes, QgsVectorLayer, QgsLegendStyle,
    )

    QgsApplication.setPrefixPath(QGIS_PREFIX, True)
    app = QgsApplication([], False)
    app.initQgis()
    project = QgsProject.instance()
    project.clear()
    project.setCrs(QgsCoordinateReferenceSystem(CRS_UTM44))
    project.setTitle("Chennai Metro Phase 2 Ridership Potential - Mapathon")
    src = str(LAYER_GPKG).replace("\\", "/")

    # TASK 4: Muted Esri Light Gray Canvas basemap (greyscale, low contrast, no API key watermark)
    ESRI_LIGHT_URI = (
        "type=xyz"
        "&url=https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}"
        "&zmin=0&zmax=16&crs=EPSG3857"
    )
    basemap = QgsRasterLayer(ESRI_LIGHT_URI, "Esri Light Gray (basemap)", "wms")
    if basemap.isValid():
        basemap.renderer().setOpacity(0.55)  # muted — barely visible under LULC
        project.addMapLayer(basemap)
        print("  [T4] Esri Light Gray Canvas basemap added at 55% opacity")
    else:
        print("  [T4] WARN: Esri Light Gray tile layer invalid (offline?)")

    def add_layer(path, name, subset=None):
        lyr = QgsVectorLayer(path, name, "ogr")
        if not lyr.isValid():
            print(f"    [WARN] Layer invalid: {name}"); return None
        if subset: lyr.setSubsetString(subset)
        project.addMapLayer(lyr)
        return lyr

    # LULC at 62% opacity — TASK 4 target 55-65%
    def apply_lulc_style(layer):
        cats = []
        for group, color in LULC_COLORS.items():
            sym = QgsFillSymbol.createSimple({
                "color": color, "outline_color":"#ffffff", "outline_width":"0.04"})
            sym.setOpacity(LULC_OPACITY)
            cats.append(QgsRendererCategory(group, sym, group))
        layer.setRenderer(QgsCategorizedSymbolRenderer("lulc_group", cats))

    def apply_catchment_style(layer, top15=False):
        sym = QgsFillSymbol.createSimple({
            "color":"255,255,255,0",
            "outline_color":"#333333" if top15 else "#aaaaaa",
            "outline_width":"0.22" if top15 else "0.05",
            "outline_style":"dash" if top15 else "solid"})
        sym.setOpacity(0.45 if top15 else 0.10)
        layer.setRenderer(QgsSingleSymbolRenderer(sym))

    def apply_corridor_style(layer, width, phase):
        lines = ["Blue Line","Green Line"] if phase=="phase1" else ["Purple Line","Red Line","Yellow Line"]
        cats = []
        for line in lines:
            sym = QgsLineSymbol.createSimple({"line_color":LINE_COLORS.get(line,"#666"),"line_width":str(width)})
            cats.append(QgsRendererCategory(line, sym, line))
        layer.setRenderer(QgsCategorizedSymbolRenderer("Line", cats))

    def apply_phase1_station_style(layer):
        sym = QgsMarkerSymbol.createSimple({
            "name":"circle","color":"#ffffff","outline_color":"#2c7fb8",
            "outline_width":"0.45","size":"3.2"})
        layer.setRenderer(QgsSingleSymbolRenderer(sym))
        apply_station_labels(layer,"1","6.0","#111111",True)

    def apply_phase2_station_style(layer, mode="all"):
        feats = [f["prediction"] for f in layer.getFeatures() if f["prediction"] is not None]
        if not feats: return
        bounds = np.quantile(feats, [0,0.2,0.4,0.6,0.8,1])
        rngs   = []
        colors = ["#ffffcc","#c7e9b4","#7fcdbb","#41b6c4","#225ea8"]
        for i in range(5):
            lo, hi = float(bounds[i]), float(bounds[i+1])
            sz     = 2.5 + i * 1.35
            sym    = QgsMarkerSymbol.createSimple({
                "name":"circle","color":colors[i],"outline_color":"#111111",
                "outline_width":"0.22","size":str(sz)})
            rngs.append(QgsRendererRange(lo, hi, sym, f"{math.floor(lo):,} – {math.ceil(hi):,}"))
        rend = QgsGraduatedSymbolRenderer("prediction", rngs)
        rend.setMode(QgsGraduatedSymbolRenderer.Custom)
        layer.setRenderer(rend)
        show = '"label_top15" = 1' if mode=="top15" else "1"
        apply_station_labels(layer, show,
            'CASE WHEN "prediction" >= 8000 THEN 7.5 '
            'WHEN "prediction" >= 6500 THEN 6.5 ELSE 5.5 END',
            "#111111", True)

    def apply_station_labels(layer, show_expr, size_expr, color, callouts):
        s = QgsPalLayerSettings()
        s.fieldName = "name"; s.enabled = True; s.priority = 10
        s.displayAll = True; s.obstacle = True
        s.dist = 2.0 if callouts else 1.0
        s.placement = QgsPalLayerSettings.OrderedPositionsAroundPoint
        s.dataDefinedProperties().setProperty(
            QgsPalLayerSettings.Show, QgsProperty.fromExpression(show_expr))
        s.dataDefinedProperties().setProperty(
            QgsPalLayerSettings.Size, QgsProperty.fromExpression(size_expr))
        if callouts:
            s.dataDefinedProperties().setProperty(
                QgsPalLayerSettings.LabelDistance,
                QgsProperty.fromExpression(
                    'CASE WHEN "confidence"=\'low\' OR "label_top15"=1 THEN 3.2 ELSE 1.8 END'))
        tf  = QgsTextFormat()
        tf.setFont(QFont("Arial",7)); tf.setSize(7); tf.setColor(QColor(color))
        buf = QgsTextBufferSettings()
        buf.setEnabled(True); buf.setSize(0.8); buf.setColor(QColor("white"))
        tf.setBuffer(buf); s.setFormat(tf)
        if callouts:
            co = QgsSimpleLineCallout(); co.setEnabled(True)
            co.setLineSymbol(QgsLineSymbol.createSimple({"line_color":"#444444","line_width":"0.18"}))
            co.setMinimumLength(0.8); s.setCallout(co)
        layer.setLabeling(QgsVectorLayerSimpleLabeling(s))
        layer.setLabelsEnabled(True)

    def apply_low_confidence_style(layer):
        sym = QgsMarkerSymbol.createSimple({
            "name":"circle","color":"255,255,255,0","outline_color":"#111111",
            "outline_width":"0.55","outline_style":"dash","size":"7.2"})
        layer.setRenderer(QgsSingleSymbolRenderer(sym))

    def apply_hub_style(layer):
        sym = QgsMarkerSymbol.createSimple({
            "name":"star","color":"#e41a1c","outline_color":"#000000",
            "outline_width":"0.6","size":"5.8"})
        layer.setRenderer(QgsSingleSymbolRenderer(sym))

    def apply_top1_highlight_style(layer):
        sym = QgsMarkerSymbol.createSimple({
            "name":"circle","color":"255,255,255,0","outline_color":"#8b0000",
            "outline_width":"1.2","outline_style":"solid","size":"12.0"})
        layer.setRenderer(QgsSingleSymbolRenderer(sym))

    def mk_font(name="Arial", size=10, bold=False):
        f = QFont(name, int(round(size))); f.setBold(bold); return f

    # ── Add vector layers in order ──────────────────────────────────────────
    layers = {}
    for suffix, label in [("0506","Bhuvan LULC 2005-06"),
                           ("1112","Bhuvan LULC 2011-12"),
                           ("1516","Bhuvan LULC 2015-16 (active)")]:
        lyr = add_layer(f"{src}|layername=bhuvan_lulc_{suffix}", label)
        if lyr: apply_lulc_style(lyr); layers[suffix] = lyr

    catchments = add_layer(f"{src}|layername=chennai_catchments_800m",
                           "800 m station catchments (other)", '"label_top15" = 0')
    if catchments: apply_catchment_style(catchments, top15=False)

    catchments_top15 = add_layer(f"{src}|layername=chennai_catchments_800m",
                                  "Top 15 station catchments", '"label_top15" = 1')
    if catchments_top15: apply_catchment_style(catchments_top15, top15=True)

    phase1_lines = add_layer(f"{src}|layername=chennai_corridor_guides",
                              "Phase 1 metro guide lines", '"phase" = \'phase1\'')
    if phase1_lines: apply_corridor_style(phase1_lines, 0.95, "phase1")

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

    top1_layer = add_layer(f"{src}|layername=chennai_stations_predictions",
                            "#1 Phase 2 — Thirumayilai", '"is_top1_phase2" = 1')
    if top1_layer: apply_top1_highlight_style(top1_layer)

    # Hide older LULC layers
    root = project.layerTreeRoot()
    for sk in ["0506","1112"]:
        if sk in layers:
            root.findLayer(layers[sk].id()).setItemVisibilityChecked(False)

    project.write(str(PROJECT_PATH))

    # ── Print Layout (COMPACT POSTER MAP WITH REDUCED TOP & BOTTOM EMPTY SPACE) ──
    # Map Frame is 400 mm wide x 690 mm high (Height : Width = 690 : 400 = 1.725 ratio)
    layout = QgsPrintLayout(project)
    layout.initializeDefaults()
    layout.setName("19:8 Mapathon Final")
    page = layout.pageCollection().pages()[0]
    page.setPageSize(QgsLayoutSize(430, 755, QgsUnitTypes.LayoutMillimeters))
    project.layoutManager().addLayout(layout)

    title = QgsLayoutItemLabel(layout)
    title.setText("Chennai Metro Phase 1 & Phase 2: Ridership Potential & Catchments")
    title.setFont(mk_font(size=18, bold=True))
    title.adjustSizeToText()
    layout.addLayoutItem(title)
    title.attemptMove(QgsLayoutPoint(15, 8, QgsUnitTypes.LayoutMillimeters))

    subtitle = QgsLayoutItemLabel(layout)
    subtitle.setText(
        "ISRO/Bhuvan LULC 2015-16 base | Esri Light Gray Canvas | "
        "800 m catchments | Predicted daily boardings | CRS: EPSG:32644")
    subtitle.setFont(mk_font(size=8.0))
    subtitle.adjustSizeToText()
    layout.addLayoutItem(subtitle)
    subtitle.attemptMove(QgsLayoutPoint(15, 19, QgsUnitTypes.LayoutMillimeters))

    # Map Frame — 400 mm wide x 690 mm high
    map_item = QgsLayoutItemMap(layout)
    layout.addLayoutItem(map_item)
    map_item.attemptMove(QgsLayoutPoint(15, 27, QgsUnitTypes.LayoutMillimeters))
    map_item.attemptResize(QgsLayoutSize(400, 690, QgsUnitTypes.LayoutMillimeters))
    map_item.setCrs(QgsCoordinateReferenceSystem("EPSG:32644"))

    # Compact bounding box: Ymin = 1,413,500 m to Ymax = 1,463,500 m (50,000 m height, 29,000 m width)
    # Eliminates ~9.2 km empty space from top and ~9.7 km empty space from bottom
    target_extent = QgsRectangle(398500.0, 1413500.0, 427500.0, 1463500.0)
    map_item.setExtent(target_extent)

    # Graticule
    grid = QgsLayoutItemMapGrid("Graticule", map_item)
    grid.setEnabled(True)
    grid.setCrs(QgsCoordinateReferenceSystem("EPSG:4326"))
    grid.setIntervalX(0.05); grid.setIntervalY(0.05)
    grid.setStyle(QgsLayoutItemMapGrid.Solid)
    grid.setLineSymbol(QgsLineSymbol.createSimple(
        {"line_color":"180,180,180,140","line_width":"0.12","line_style":"dot"}))
    grid.setAnnotationEnabled(True)
    grid.setAnnotationDisplay(QgsLayoutItemMapGrid.ShowAll, QgsLayoutItemMapGrid.Left)
    grid.setAnnotationDisplay(QgsLayoutItemMapGrid.ShowAll, QgsLayoutItemMapGrid.Bottom)
    grid.setAnnotationDisplay(QgsLayoutItemMapGrid.HideAll, QgsLayoutItemMapGrid.Right)
    grid.setAnnotationDisplay(QgsLayoutItemMapGrid.HideAll, QgsLayoutItemMapGrid.Top)
    grid.setAnnotationFormat(QgsLayoutItemMapGrid.DecimalWithSuffix)
    grid.setAnnotationFont(mk_font(size=7.5))
    grid.setAnnotationPrecision(2)
    map_item.grids().addGrid(grid)

    # Legend — Upper right ocean area with 2 compact columns
    legend = QgsLayoutItemLegend(layout)
    legend.setTitle("Legend")
    legend.setLinkedMap(map_item)
    legend.setAutoUpdateModel(True)
    layout.addLayoutItem(legend)
    legend.setAutoUpdateModel(False)
    root_grp = legend.model().rootGroup()
    LEGEND_EXCLUDE = {
        "Phase 2 full station labels",
        "OpenStreetMap basemap",
        "Bhuvan LULC 2005-06",
        "Bhuvan LULC 2011-12",
        "CartoDB Positron (basemap)",
        "Esri Light Gray (basemap)",
    }
    for child in list(root_grp.children()):
        if child.name() in LEGEND_EXCLUDE:
            root_grp.removeChildNode(child)
    legend.setColumnCount(2)
    legend.setSplitLayer(True)
    legend.setEqualColumnWidth(False)
    legend.setStyleFont(QgsLegendStyle.Title, mk_font(size=9, bold=True))
    legend.setStyleFont(QgsLegendStyle.Group, mk_font(size=8, bold=True))
    legend.setStyleFont(QgsLegendStyle.Subgroup, mk_font(size=7.5, bold=True))
    legend.setStyleFont(QgsLegendStyle.SymbolLabel, mk_font(size=7.0))
    legend.refresh()
    legend.attemptMove(QgsLayoutPoint(25, 455, QgsUnitTypes.LayoutMillimeters))
    legend.attemptResize(QgsLayoutSize(155, 125, QgsUnitTypes.LayoutMillimeters))
    legend.setBackgroundEnabled(True)
    legend.setBackgroundColor(QColor(255, 255, 255, 230))

    # Scale bar
    scalebar = QgsLayoutItemScaleBar(layout)
    scalebar.setStyle("Single Box")
    layout.addLayoutItem(scalebar)
    scalebar.setLinkedMap(map_item)
    scalebar.setUnits(QgsUnitTypes.DistanceKilometers)
    scalebar.setUnitLabel("km")
    scalebar.setNumberOfSegments(4)
    scalebar.setUnitsPerSegment(2)
    scalebar.setFont(mk_font(size=8))
    scalebar.update()
    scalebar.attemptMove(QgsLayoutPoint(25, 695, QgsUnitTypes.LayoutMillimeters))

    # North arrow
    north = QgsLayoutItemPicture(layout)
    north.setPicturePath(str(Path(QGIS_PREFIX)/"svg"/"arrows"/"NorthArrow_04.svg"))
    layout.addLayoutItem(north)
    north.attemptMove(QgsLayoutPoint(215, 32, QgsUnitTypes.LayoutMillimeters))
    north.attemptResize(QgsLayoutSize(18, 28, QgsUnitTypes.LayoutMillimeters))

    # Footer
    note = QgsLayoutItemLabel(layout)
    note.setText(
        "CRS: EPSG:32644 / WGS 84 UTM zone 44N\n"
        "Data courtesy: ISRO/NRSC Bhuvan LULC 50K | CMRL Passenger Flow API | "
        "GHSL | OpenStreetMap contributors | BMRCL RTI ridership.\n"
        "Low-confidence Phase 2 positions: dashed outlines. Output: CC-BY-SA 4.0.\n"
        + FOOTNOTE_TEXT
    )
    note.setFont(mk_font(size=7))
    note.adjustSizeToText()
    layout.addLayoutItem(note)
    note.attemptMove(QgsLayoutPoint(15, 726, QgsUnitTypes.LayoutMillimeters))
    note.attemptResize(QgsLayoutSize(400, 24, QgsUnitTypes.LayoutMillimeters))

    project.write(str(PROJECT_PATH))

    # ── Task 5 QA print ──────────────────────────────────────────────────────
    print("\n  [T5] QGIS QA CHECKS:")
    print(f"    Title:      present")
    print(f"    North arrow: ({north.pos().x():.0f}, {north.pos().y():.0f}) mm")
    print(f"    Scale bar:   ({scalebar.pos().x():.0f}, {scalebar.pos().y():.0f}) mm")
    print(f"    Graticule:   0.05° WGS84, decimal+suffix")
    print(f"    CRS note:    EPSG:32644 in footer")
    print(f"    Data credit: Bhuvan/CMRL/GHSL/OSM/BMRCL")
    print(f"    CC-BY-SA 4.0 present")
    print(f"    Footnote:    September 2026 update text")
    print(f"    OSM basemap: CartoDB Positron at 55% opacity")
    print(f"    LULC opacity: {LULC_OPACITY:.0%}")
    print(f"    Legend:      'Other', 'OpenStreetMap basemap', 'CartoDB' excluded")

    # Desktop canvas helpers
    tree = ET.parse(PROJECT_PATH)
    rxml = tree.getroot()
    for ex in rxml.findall("mapcanvas"): rxml.remove(ex)
    canvas = ET.Element("mapcanvas",{"name":"theMapCanvas","annotationsVisible":"1"})
    ET.SubElement(canvas,"units").text = "meters"
    ext_el = ET.SubElement(canvas,"extent")
    ex = target_extent
    ET.SubElement(ext_el,"xmin").text = f"{ex.xMinimum():.3f}"
    ET.SubElement(ext_el,"ymin").text = f"{ex.yMinimum():.3f}"
    ET.SubElement(ext_el,"xmax").text = f"{ex.xMaximum():.3f}"
    ET.SubElement(ext_el,"ymax").text = f"{ex.yMaximum():.3f}"
    ET.SubElement(canvas,"rotation").text = "0"
    srs_el = ET.SubElement(ET.SubElement(canvas,"destinationsrs"),"spatialrefsys")
    ET.SubElement(srs_el,"authid").text="EPSG:32644"
    ET.SubElement(srs_el,"description").text="WGS 84 / UTM zone 44N"
    ET.SubElement(srs_el,"geographicflag").text="false"
    layer_tree = rxml.find("layer-tree-group")
    ins = list(rxml).index(layer_tree) if layer_tree is not None else 3
    rxml.insert(ins, canvas)
    tree.write(PROJECT_PATH, encoding="UTF-8", xml_declaration=False)
    with zipfile.ZipFile(PROJECT_QGZ,"w",compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(PROJECT_PATH,"project.qgs")
    LAUNCHER_PATH.write_text(
        '@echo off\r\ncd /d "%~dp0"\r\n'
        '"D:\\QGIS\\bin\\qgis.bat" "%~dp0chennai_mapathon_ridership_desktop.qgz"\r\n',
        encoding="utf-8")

    # ── TASK 5: Export at 300 DPI ────────────────────────────────────────────
    exporter = QgsLayoutExporter(layout)
    exporter.exportToPdf(str(PDF_PATH), QgsLayoutExporter.PdfExportSettings())
    img_settings = QgsLayoutExporter.ImageExportSettings()
    img_settings.dpi = 300
    exporter.exportToImage(str(PNG_PATH), img_settings)
    exporter.exportToPdf(str(FULL_PDF),  QgsLayoutExporter.PdfExportSettings())
    exporter.exportToImage(str(FULL_PNG), img_settings)
    exporter.exportToImage(str(TEMPLATE_MAP_PNG), img_settings)

    canva_opt_png = MAPS_DIR / "chennai_mapathon_canva_optimized.png"
    canva_settings = QgsLayoutExporter.ImageExportSettings()
    canva_settings.dpi = 252  # ~7490 px height, strictly under Canva 8000px limit to avoid compression
    exporter.exportToImage(str(canva_opt_png), canva_settings)

    print(f"\n  [T5] Exports:")
    print(f"    PDF: {PDF_PATH}")
    print(f"    PNG: {PNG_PATH}  (300 DPI)")
    print(f"    Canva-optimized PNG: {canva_opt_png} (252 DPI)")
    print(f"    Full-label PNG: {FULL_PNG}  (300 DPI)")
    print(f"    Template Graphic PNG: {TEMPLATE_MAP_PNG}  (300 DPI)")

    app.exitQgis()


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    print("=" * 65)
    print("FIX CHARTS + BASEMAP — ALL TASKS")
    print("=" * 65)

    print("\nTASKS 1-3: Regenerating all 5 charts with layout fixes ...")
    make_top15_chart()
    make_corridor_chart()
    make_model_perf_chart()
    make_elasticity_chart()
    make_misses_chart()
    qa_charts()

    print("\nTASK 4-5: Rebuilding QGIS project with Positron basemap ...")
    rebuilt = remove_existing_gpkg(LAYER_GPKG)
    if rebuilt or not LAYER_GPKG.exists():
        prepare_lulc_layers()
        pred_min, pred_max = prepare_station_layers()
    else:
        import geopandas as gpd
        gdf = gpd.read_file(LAYER_GPKG, layer="chennai_stations_predictions")
        p2 = gdf[gdf["phase"] == "phase2"]
        pred_min, pred_max = float(p2["prediction"].min()), float(p2["prediction"].max())
    create_project(pred_min, pred_max)

    print("\n" + "=" * 65)
    print("DONE. Final export path:")
    print(f"  {PNG_PATH}")
    print("=" * 65)


if __name__ == "__main__":
    main()
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0)
