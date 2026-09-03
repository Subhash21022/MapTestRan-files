"""Build the Chennai Mapathon QGIS project.

This script treats source documents and data files as inputs only. It prepares
QGIS-ready layers from the processed model outputs and the Bhuvan LULC files,
then writes a styled QGIS project plus first PDF/PNG layout exports.
"""

from __future__ import annotations

import math
import os
import sys
import json
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyproj import CRS
from shapely.geometry import LineString, Point
from shapely.ops import linemerge, unary_union
from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtGui import QColor, QFont
from qgis.core import (
    QgsApplication,
    QgsCategorizedSymbolRenderer,
    QgsCoordinateReferenceSystem,
    QgsFeatureRequest,
    QgsFillSymbol,
    QgsGraduatedSymbolRenderer,
    QgsLayoutExporter,
    QgsLayoutItemLabel,
    QgsLayoutItemLegend,
    QgsLayoutItemMap,
    QgsLayoutItemPicture,
    QgsLayoutItemScaleBar,
    QgsLayoutPoint,
    QgsLayoutSize,
    QgsLegendStyle,
    QgsLineSymbol,
    QgsMarkerSymbol,
    QgsPalLayerSettings,
    QgsPrintLayout,
    QgsProject,
    QgsProperty,
    QgsRasterLayer,
    QgsRendererCategory,
    QgsRendererRange,
    QgsRuleBasedLabeling,
    QgsSingleSymbolRenderer,
    QgsVectorLayerSimpleLabeling,
    QgsTextBufferSettings,
    QgsTextFormat,
    QgsUnitTypes,
    QgsVectorFileWriter,
    QgsVectorLayer,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path("D:/Mapathon Data Files")
QGIS_DIR = ROOT / "qgis"
QGIS_DIR.mkdir(exist_ok=True)
FIGURES_DIR = ROOT / "outputs" / "figures"
MAPS_DIR = ROOT / "outputs" / "maps"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
MAPS_DIR.mkdir(parents=True, exist_ok=True)

LAYER_GPKG = QGIS_DIR / "chennai_mapathon_layers_desktop.gpkg"
PROJECT_PATH = QGIS_DIR / "chennai_mapathon_ridership_desktop.qgs"
PROJECT_QGZ_PATH = QGIS_DIR / "chennai_mapathon_ridership_desktop.qgz"
LAUNCHER_PATH = QGIS_DIR / "open_chennai_mapathon_qgis.bat"
PDF_PATH = MAPS_DIR / "chennai_mapathon_qgis_preview_desktop.pdf"
PNG_PATH = MAPS_DIR / "chennai_mapathon_qgis_preview_desktop.png"
TOP15_CHART = FIGURES_DIR / "chennai_top15_phase2_predictions.png"
OVERPASS_ALIGNMENTS = ROOT / "data" / "interim" / "chennai_overpass_alignments.json"

CRS_UTM44 = "EPSG:32644"
QGIS_PREFIX = "D:/QGIS/apps/qgis"
# This is a cartographic alignment threshold, not analytical editing. Low
# confidence stations keep their uncertainty styling and source coordinates.
MAX_MAP_SNAP_M = 15000

LINE_COLORS = {
    "Blue Line": "#2c7fb8",
    "Green Line": "#2ca25f",
    "Purple Line": "#7b3294",
    "Red Line": "#d73027",
    "Yellow Line": "#f4c430",
    "Red LinePurple Line": "#525252",
}

LULC_COLORS = {
    "Built-up": "#c94c4c",
    "Agriculture": "#d8c66f",
    "Forest/Scrub": "#5aa469",
    "Water/Wetland": "#5aa6c8",
    "Barren/Open": "#d9b38c",
    "Other": "#bdbdbd",
}

ALIGNMENT_NAME_TO_LINE = {
    "Chennai Metro Line 1": ("phase1", "Blue Line"),
    "Chennai Metro Line 2": ("phase1", "Green Line"),
}


def classify_alignment(name: str) -> tuple[str, str] | None:
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
    """Read locally cached Overpass linework into the map CRS."""
    if not OVERPASS_ALIGNMENTS.exists():
        return gpd.GeoDataFrame(columns=["phase", "Line", "osm_name", "geometry"], crs=CRS_UTM44)

    data = json.loads(OVERPASS_ALIGNMENTS.read_text(encoding="utf-8"))
    rows = []
    for element in data.get("elements", []):
        if element.get("type") != "way" or "geometry" not in element:
            continue
        tags = element.get("tags", {})
        name = tags.get("name", "")
        classified = classify_alignment(name)
        if not classified:
            continue
        coords = [(pt["lon"], pt["lat"]) for pt in element["geometry"]]
        if len(coords) < 2:
            continue
        phase, line = classified
        rows.append(
            {
                "phase": phase,
                "Line": line,
                "osm_name": name,
                "osm_id": element.get("id"),
                "geometry": LineString(coords),
            }
        )

    if not rows:
        return gpd.GeoDataFrame(columns=["phase", "Line", "osm_name", "geometry"], crs=CRS_UTM44)

    lines = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326").to_crs(CRS_UTM44)
    dissolved_rows = []
    for (phase, line), group in lines.groupby(["phase", "Line"]):
        merged = linemerge(unary_union(group.geometry.tolist()))
        if merged.geom_type == "GeometryCollection":
            parts = [geom for geom in merged.geoms if geom.geom_type in ("LineString", "MultiLineString")]
            merged = linemerge(unary_union(parts))
        dissolved_rows.append(
            {
                "phase": phase,
                "Line": line,
                "station_count": 0,
                "source": "osm_overpass",
                "length_km": round(float(group.geometry.length.sum() / 1000), 2),
                "geometry": merged,
            }
        )
    return gpd.GeoDataFrame(dissolved_rows, geometry="geometry", crs=CRS_UTM44)


def lulc_group(value: object) -> str:
    text = str(value or "").lower()
    if "built" in text:
        return "Built-up"
    if "agriculture" in text or "crop" in text or "fallow" in text:
        return "Agriculture"
    if "forest" in text or "scrub" in text:
        return "Forest/Scrub"
    if "wetland" in text or "water" in text or "river" in text or "lake" in text:
        return "Water/Wetland"
    if "barren" in text or "waste" in text or "sandy" in text:
        return "Barren/Open"
    return "Other"


def remove_existing_gpkg(path: Path) -> None:
    if path.exists():
        path.unlink()
    for sidecar in path.parent.glob(path.name + "*"):
        if sidecar.is_file():
            sidecar.unlink()


def write_layer(gdf: gpd.GeoDataFrame, layer_name: str, first: bool = False) -> None:
    mode = "w" if first else "a"
    gdf.to_file(LAYER_GPKG, layer=layer_name, driver="GPKG", mode=mode)


def prepare_lulc_layers() -> None:
    first = True
    for suffix, year_label in [
        ("0506", "2005-06"),
        ("1112", "2011-12"),
        ("1516", "2015-16"),
    ]:
        path = DATA_DIR / f"CHENNAI_TN_LULC50K_{suffix}.shp"
        gdf = gpd.read_file(path)
        gdf["map_year"] = year_label
        gdf["lulc_group"] = gdf["DESCR_1"].map(lulc_group)
        gdf["lulc_class"] = gdf["DESCR_2"].fillna(gdf["lulc_group"])
        gdf = gdf.to_crs(CRS_UTM44)
        keep = ["map_year", "lulc_group", "lulc_class", "LU_Webcode", "geometry"]
        write_layer(gdf[keep], f"bhuvan_lulc_{suffix}", first=first)
        first = False


def nearest_point_on_geometry(geometry, point: Point) -> Point:
    """Return the closest point on a LineString or MultiLineString."""
    return geometry.interpolate(geometry.project(point))


def snap_stations_for_map(
    stations: gpd.GeoDataFrame,
    alignments: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Create a cartographic station layer aligned to metro linework."""
    out = stations.to_crs(CRS_UTM44).copy()
    out["source_x"] = out.geometry.x.round(3)
    out["source_y"] = out.geometry.y.round(3)
    out["map_snap_m"] = 0.0
    out["map_aligned"] = 0

    if alignments.empty:
        return out

    line_geoms = {(row["phase"], row["Line"]): row.geometry for _, row in alignments.iterrows()}
    phase2_union = unary_union(alignments[alignments["phase"].eq("phase2")].geometry.tolist())
    all_union = unary_union(alignments.geometry.tolist())

    for idx, row in out.iterrows():
        keys = [(row.get("phase"), row.get("Line"))]
        if row.get("Line") == "Red LinePurple Line":
            keys = [("phase2", "Red Line"), ("phase2", "Purple Line")]

        candidates = [line_geoms[key] for key in keys if key in line_geoms]
        if not candidates and row.get("phase") == "phase2" and not phase2_union.is_empty:
            candidates = [phase2_union]
        if not candidates and not all_union.is_empty:
            candidates = [all_union]
        if not candidates:
            continue

        point = row.geometry
        snapped_options = [(geom.distance(point), nearest_point_on_geometry(geom, point)) for geom in candidates]
        distance, snapped = min(snapped_options, key=lambda item: item[0])
        if distance <= MAX_MAP_SNAP_M:
            out.at[idx, "geometry"] = snapped
            out.at[idx, "map_snap_m"] = round(float(distance), 1)
            out.at[idx, "map_aligned"] = 1

    return out


def build_fallback_guides(stations: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Fallback guide lines if real OSM alignment linework is unavailable."""
    guide_rows = []
    for (phase, line), group in stations.dropna(subset=["Line"]).groupby(["phase", "Line"]):
        if len(group) < 2:
            continue
        points = group.geometry.tolist()
        unused = set(range(len(points)))
        start = min(unused, key=lambda i: (points[i].x, points[i].y))
        order = [start]
        unused.remove(start)
        while unused:
            last = points[order[-1]]
            nxt = min(unused, key=lambda i: last.distance(points[i]))
            order.append(nxt)
            unused.remove(nxt)
        guide_rows.append(
            {
                "phase": phase,
                "Line": line,
                "station_count": len(group),
                "source": "station_nearest_neighbour",
                "length_km": round(LineString([points[i] for i in order]).length / 1000, 2),
                "geometry": LineString([points[i] for i in order]),
            }
        )
    return gpd.GeoDataFrame(guide_rows, crs=CRS_UTM44)


def prepare_station_layers() -> tuple[float, float]:
    stations = gpd.read_file(ROOT / "data" / "processed" / "chennai_stations.gpkg")
    predictions = pd.read_csv(ROOT / "outputs" / "tables" / "chennai_predictions_new.csv")
    predictions = predictions[
        [
            "station_id",
            "prediction",
            "lower",
            "upper",
            "population_density",
            "pnr",
            "bus_stops_500m",
            "poi_total",
            "metro_closeness",
        ]
    ]
    stations = stations.merge(predictions, on="station_id", how="left")
    stations["prediction"] = stations["prediction"].round(0)
    stations["lower"] = stations["lower"].round(0)
    stations["upper"] = stations["upper"].round(0)
    stations["prediction_band"] = pd.qcut(
        stations["prediction"].rank(method="first"),
        q=5,
        labels=["Very low", "Low", "Medium", "High", "Very high"],
    ).astype(str)

    phase2 = stations["phase"].eq("phase2")
    stations["rank_phase2"] = np.nan
    stations.loc[phase2, "rank_phase2"] = (
        stations.loc[phase2, "prediction"].rank(ascending=False, method="first")
    )
    stations["label_top15"] = (stations["rank_phase2"] <= 15).astype(int)
    pred_min = float(stations.loc[phase2, "prediction"].min())
    pred_max = float(stations.loc[phase2, "prediction"].max())
    stations["marker_size"] = np.interp(
        stations["prediction"].fillna(pred_min),
        [pred_min, pred_max],
        [2.2, 8.5],
    ).round(2)

    osm_alignments = read_osm_alignments()
    if osm_alignments.empty:
        stations_map = stations.to_crs(CRS_UTM44)
        alignments = build_fallback_guides(stations_map)
    else:
        alignments = osm_alignments
        stations_map = snap_stations_for_map(stations, alignments)

    write_layer(stations_map, "chennai_stations_predictions")

    buffers = gpd.read_file(ROOT / "data" / "processed" / "chennai_catchments_buffer.gpkg")
    buffers = buffers[buffers["radius_m"].eq(800)].copy()
    buffers = buffers.drop(columns="geometry")
    buffers = buffers.merge(
        stations_map[["station_id", "prediction", "rank_phase2", "label_top15", "geometry"]],
        on="station_id",
        how="left",
    )
    buffers = gpd.GeoDataFrame(buffers, geometry="geometry", crs=CRS_UTM44)
    buffers["geometry"] = buffers.geometry.buffer(800)
    write_layer(buffers.to_crs(CRS_UTM44), "chennai_catchments_800m")
    write_layer(alignments, "chennai_corridor_guides")
    return pred_min, pred_max


def make_top15_chart() -> None:
    df = pd.read_csv(ROOT / "outputs" / "tables" / "chennai_predictions_new.csv")
    top = df[df["phase"].eq("phase2")].nlargest(15, "prediction").sort_values("prediction")
    fig, ax = plt.subplots(figsize=(7.6, 4.4), dpi=180)
    xerr = np.vstack([top["prediction"] - top["lower"], top["upper"] - top["prediction"]])
    colors = [LINE_COLORS.get(line, "#666666") for line in top["Line"]]
    ax.barh(top["name"], top["prediction"], color=colors, edgecolor="#333333", linewidth=0.4)
    ax.errorbar(top["prediction"], top["name"], xerr=xerr, fmt="none", ecolor="#222222", elinewidth=0.8, capsize=2)
    ax.set_xlabel("Predicted daily boardings")
    ax.set_title("Top 15 Phase 2 stations by predicted potential")
    ax.grid(axis="x", color="#d9d9d9", linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(TOP15_CHART, transparent=False, facecolor="white")
    plt.close(fig)


def add_layer(path: str, name: str, subset: str | None = None) -> QgsVectorLayer:
    layer = QgsVectorLayer(path, name, "ogr")
    if not layer.isValid():
        raise RuntimeError(f"Layer failed to load: {name} from {path}")
    if subset:
        layer.setSubsetString(subset)
    QgsProject.instance().addMapLayer(layer)
    return layer


def add_osm_basemap() -> QgsRasterLayer | None:
    uri = (
        "type=xyz&url=https://tile.openstreetmap.org/{z}/{x}/{y}.png"
        "&zmin=0&zmax=19&crs=EPSG3857"
    )
    layer = QgsRasterLayer(uri, "OpenStreetMap basemap", "wms")
    if not layer.isValid():
        return None
    layer.renderer().setOpacity(0.42)
    QgsProject.instance().addMapLayer(layer)
    return layer


def apply_lulc_style(layer: QgsVectorLayer) -> None:
    categories = []
    for group, color in LULC_COLORS.items():
        symbol = QgsFillSymbol.createSimple(
            {
                "color": color,
                "outline_color": "#ffffff",
                "outline_width": "0.05",
            }
        )
        symbol.setOpacity(0.58)
        categories.append(QgsRendererCategory(group, symbol, group))
    layer.setRenderer(QgsCategorizedSymbolRenderer("lulc_group", categories))


def apply_catchment_style(layer: QgsVectorLayer) -> None:
    symbol = QgsFillSymbol.createSimple(
        {
            "color": "255,255,255,0",
            "outline_color": "#404040",
            "outline_width": "0.18",
            "outline_style": "dash",
        }
    )
    symbol.setOpacity(0.65)
    layer.setRenderer(QgsSingleSymbolRenderer(symbol))


def apply_corridor_style(layer: QgsVectorLayer, width: float) -> None:
    categories = []
    for line, color in LINE_COLORS.items():
        symbol = QgsLineSymbol.createSimple({"line_color": color, "line_width": str(width)})
        categories.append(QgsRendererCategory(line, symbol, line))
    layer.setRenderer(QgsCategorizedSymbolRenderer("Line", categories))


def apply_phase1_station_style(layer: QgsVectorLayer) -> None:
    symbol = QgsMarkerSymbol.createSimple(
        {
            "name": "circle",
            "color": "#f7f7f7",
            "outline_color": "#525252",
            "outline_width": "0.25",
            "size": "2.3",
        }
    )
    layer.setRenderer(QgsSingleSymbolRenderer(symbol))


def apply_phase2_station_style(layer: QgsVectorLayer, pred_min: float, pred_max: float) -> None:
    ranges = []
    colors = ["#ffffcc", "#c7e9b4", "#7fcdbb", "#41b6c4", "#225ea8"]
    bounds = np.quantile(
        [feature["prediction"] for feature in layer.getFeatures() if feature["prediction"] is not None],
        [0, 0.2, 0.4, 0.6, 0.8, 1],
    )
    for idx in range(5):
        low = float(bounds[idx])
        high = float(bounds[idx + 1])
        size = 2.5 + idx * 1.35
        symbol = QgsMarkerSymbol.createSimple(
            {
                "name": "circle",
                "color": colors[idx],
                "outline_color": "#111111",
                "outline_width": "0.22",
                "size": str(size),
            }
        )
        label = f"{math.floor(low):,} - {math.ceil(high):,}"
        ranges.append(QgsRendererRange(low, high, symbol, label))
    renderer = QgsGraduatedSymbolRenderer("prediction", ranges)
    renderer.setMode(QgsGraduatedSymbolRenderer.Custom)
    layer.setRenderer(renderer)

    settings = QgsPalLayerSettings()
    settings.fieldName = "name"
    settings.enabled = True
    settings.priority = 9
    settings.dataDefinedProperties().setProperty(
        QgsPalLayerSettings.Show,
        QgsProperty.fromExpression('"label_top15" = 1'),
    )
    text_format = QgsTextFormat()
    text_format.setFont(QFont("Arial", 8))
    text_format.setSize(8)
    buffer = QgsTextBufferSettings()
    buffer.setEnabled(True)
    buffer.setSize(1)
    buffer.setColor(QColor("white"))
    text_format.setBuffer(buffer)
    settings.setFormat(text_format)
    layer.setLabeling(QgsVectorLayerSimpleLabeling(settings))
    layer.setLabelsEnabled(True)


def apply_low_confidence_style(layer: QgsVectorLayer) -> None:
    symbol = QgsMarkerSymbol.createSimple(
        {
            "name": "circle",
            "color": "255,255,255,0",
            "outline_color": "#111111",
            "outline_width": "0.55",
            "outline_style": "dash",
            "size": "7.2",
        }
    )
    layer.setRenderer(QgsSingleSymbolRenderer(symbol))


def font(name: str = "Arial", size: int = 10, bold: bool = False) -> QFont:
    out = QFont(name, size)
    out.setBold(bold)
    return out


def write_desktop_helpers(extent) -> None:
    """Save a desktop canvas extent and a double-click launcher."""
    tree = ET.parse(PROJECT_PATH)
    root = tree.getroot()
    for existing in root.findall("mapcanvas"):
        root.remove(existing)

    canvas = ET.Element("mapcanvas", {"name": "theMapCanvas", "annotationsVisible": "1"})
    ET.SubElement(canvas, "units").text = "meters"
    extent_el = ET.SubElement(canvas, "extent")
    ET.SubElement(extent_el, "xmin").text = f"{extent.xMinimum():.3f}"
    ET.SubElement(extent_el, "ymin").text = f"{extent.yMinimum():.3f}"
    ET.SubElement(extent_el, "xmax").text = f"{extent.xMaximum():.3f}"
    ET.SubElement(extent_el, "ymax").text = f"{extent.yMaximum():.3f}"
    ET.SubElement(canvas, "rotation").text = "0"
    srs = ET.SubElement(canvas, "destinationsrs")
    spatialrefsys = ET.SubElement(srs, "spatialrefsys")
    ET.SubElement(spatialrefsys, "proj4").text = "+proj=utm +zone=44 +datum=WGS84 +units=m +no_defs"
    ET.SubElement(spatialrefsys, "srsid").text = "3128"
    ET.SubElement(spatialrefsys, "srid").text = "32644"
    ET.SubElement(spatialrefsys, "authid").text = "EPSG:32644"
    ET.SubElement(spatialrefsys, "description").text = "WGS 84 / UTM zone 44N"
    ET.SubElement(spatialrefsys, "projectionacronym").text = "utm"
    ET.SubElement(spatialrefsys, "ellipsoidacronym").text = "EPSG:7030"
    ET.SubElement(spatialrefsys, "geographicflag").text = "false"
    ET.SubElement(canvas, "rendermaptile").text = "0"

    layer_tree = root.find("layer-tree-group")
    insert_at = list(root).index(layer_tree) if layer_tree is not None else 3
    root.insert(insert_at, canvas)
    tree.write(PROJECT_PATH, encoding="UTF-8", xml_declaration=False)

    with zipfile.ZipFile(PROJECT_QGZ_PATH, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(PROJECT_PATH, "project.qgs")

    LAUNCHER_PATH.write_text(
        '@echo off\r\n'
        'cd /d "%~dp0"\r\n'
        '"D:\\QGIS\\bin\\qgis.bat" "%~dp0chennai_mapathon_ridership_desktop.qgz"\r\n',
        encoding="utf-8",
    )


def create_project(pred_min: float, pred_max: float) -> None:
    QgsApplication.setPrefixPath(QGIS_PREFIX, True)
    app = QgsApplication([], False)
    app.initQgis()
    project = QgsProject.instance()
    project.clear()
    project.setCrs(QgsCoordinateReferenceSystem(CRS_UTM44))
    project.setTitle("Chennai Metro Phase 2 Ridership Potential - Mapathon")

    src = str(LAYER_GPKG).replace("\\", "/")
    add_osm_basemap()
    layers = {}
    for suffix, label in [
        ("0506", "Bhuvan LULC 2005-06"),
        ("1112", "Bhuvan LULC 2011-12"),
        ("1516", "Bhuvan LULC 2015-16"),
    ]:
        layer = add_layer(f"{src}|layername=bhuvan_lulc_{suffix}", label)
        apply_lulc_style(layer)
        layers[suffix] = layer
    catchments = add_layer(f"{src}|layername=chennai_catchments_800m", "800 m station catchments")
    apply_catchment_style(catchments)
    phase1_lines = add_layer(
        f"{src}|layername=chennai_corridor_guides",
        "Phase 1 metro guide lines",
        "\"phase\" = 'phase1'",
    )
    apply_corridor_style(phase1_lines, 0.55)
    phase2_lines = add_layer(
        f"{src}|layername=chennai_corridor_guides",
        "Phase 2 corridor guide lines",
        "\"phase\" = 'phase2'",
    )
    apply_corridor_style(phase2_lines, 1.05)
    phase1_stations = add_layer(
        f"{src}|layername=chennai_stations_predictions",
        "Phase 1 stations (observed context)",
        "\"phase\" = 'phase1'",
    )
    apply_phase1_station_style(phase1_stations)
    phase2_stations = add_layer(
        f"{src}|layername=chennai_stations_predictions",
        "Phase 2 stations by predicted boardings",
        "\"phase\" = 'phase2'",
    )
    apply_phase2_station_style(phase2_stations, pred_min, pred_max)
    low_conf = add_layer(
        f"{src}|layername=chennai_stations_predictions",
        "Low-confidence station positions",
        "\"phase\" = 'phase2' AND \"confidence\" = 'low'",
    )
    apply_low_confidence_style(low_conf)

    root = project.layerTreeRoot()
    for lyr in [layers["0506"], layers["1112"]]:
        root.findLayer(lyr.id()).setItemVisibilityChecked(False)

    project.write(str(PROJECT_PATH))

    layout = QgsPrintLayout(project)
    layout.initializeDefaults()
    layout.setName("A1 Mapathon Preview")
    page = layout.pageCollection().pages()[0]
    page.setPageSize(QgsLayoutSize(841, 594, QgsUnitTypes.LayoutMillimeters))
    project.layoutManager().addLayout(layout)

    title = QgsLayoutItemLabel(layout)
    title.setText("Chennai Metro Phase 2: Ridership Potential and Station Catchments")
    title.setFont(font(size=22, bold=True))
    title.adjustSizeToText()
    layout.addLayoutItem(title)
    title.attemptMove(QgsLayoutPoint(12, 8, QgsUnitTypes.LayoutMillimeters))

    subtitle = QgsLayoutItemLabel(layout)
    subtitle.setText("Bhuvan LULC 2015-16 base with 800 m catchments; station size and colour show predicted daily boardings")
    subtitle.setFont(font(size=10))
    subtitle.adjustSizeToText()
    layout.addLayoutItem(subtitle)
    subtitle.attemptMove(QgsLayoutPoint(12, 21, QgsUnitTypes.LayoutMillimeters))

    map_item = QgsLayoutItemMap(layout)
    map_item.setRect(0, 0, 620, 468)
    extent = phase2_stations.extent()
    extent.combineExtentWith(catchments.extent())
    extent.scale(1.08)
    map_item.setExtent(extent)
    layout.addLayoutItem(map_item)
    map_item.attemptMove(QgsLayoutPoint(12, 34, QgsUnitTypes.LayoutMillimeters))
    map_item.attemptResize(QgsLayoutSize(620, 468, QgsUnitTypes.LayoutMillimeters))

    legend = QgsLayoutItemLegend(layout)
    legend.setTitle("Legend")
    legend.setLinkedMap(map_item)
    legend.setStyleFont(QgsLegendStyle.Title, font(size=12, bold=True))
    legend.setStyleFont(QgsLegendStyle.Group, font(size=9, bold=True))
    legend.setStyleFont(QgsLegendStyle.SymbolLabel, font(size=8))
    layout.addLayoutItem(legend)
    legend.attemptMove(QgsLayoutPoint(646, 38, QgsUnitTypes.LayoutMillimeters))
    legend.attemptResize(QgsLayoutSize(170, 210, QgsUnitTypes.LayoutMillimeters))

    scalebar = QgsLayoutItemScaleBar(layout)
    scalebar.setStyle("Single Box")
    scalebar.setLinkedMap(map_item)
    scalebar.setUnits(QgsUnitTypes.DistanceKilometers)
    scalebar.setNumberOfSegments(4)
    scalebar.setUnitsPerSegment(2)
    scalebar.setFont(font(size=8))
    scalebar.update()
    layout.addLayoutItem(scalebar)
    scalebar.attemptMove(QgsLayoutPoint(28, 486, QgsUnitTypes.LayoutMillimeters))

    north = QgsLayoutItemPicture(layout)
    north.setPicturePath(str(Path(QGIS_PREFIX) / "svg" / "arrows" / "NorthArrow_04.svg"))
    layout.addLayoutItem(north)
    north.attemptMove(QgsLayoutPoint(590, 48, QgsUnitTypes.LayoutMillimeters))
    north.attemptResize(QgsLayoutSize(22, 34, QgsUnitTypes.LayoutMillimeters))

    chart = QgsLayoutItemPicture(layout)
    chart.setPicturePath(str(TOP15_CHART))
    layout.addLayoutItem(chart)
    chart.attemptMove(QgsLayoutPoint(646, 258, QgsUnitTypes.LayoutMillimeters))
    chart.attemptResize(QgsLayoutSize(180, 104, QgsUnitTypes.LayoutMillimeters))

    note = QgsLayoutItemLabel(layout)
    note.setText(
        "CRS: EPSG:32644 / WGS 84 UTM zone 44N\n"
        "Data courtesy: ISRO/NRSC Bhuvan LULC 50K, CMRL, GHSL, OSM, BMRCL RTI ridership.\n"
        "Low-confidence Phase 2 positions are ringed with dashed outlines. Output: CC-BY-SA 4.0."
    )
    note.setFont(font(size=8))
    note.adjustSizeToText()
    layout.addLayoutItem(note)
    note.attemptMove(QgsLayoutPoint(646, 516, QgsUnitTypes.LayoutMillimeters))

    project.write(str(PROJECT_PATH))
    write_desktop_helpers(extent)
    exporter = QgsLayoutExporter(layout)
    exporter.exportToPdf(str(PDF_PATH), QgsLayoutExporter.PdfExportSettings())
    image_settings = QgsLayoutExporter.ImageExportSettings()
    image_settings.dpi = 180
    exporter.exportToImage(str(PNG_PATH), image_settings)
    app.exitQgis()


def main() -> None:
    remove_existing_gpkg(LAYER_GPKG)
    prepare_lulc_layers()
    pred_min, pred_max = prepare_station_layers()
    make_top15_chart()
    create_project(pred_min, pred_max)
    print(f"Wrote {PROJECT_PATH}")
    print(f"Wrote {PROJECT_QGZ_PATH}")
    print(f"Wrote {LAUNCHER_PATH}")
    print(f"Wrote {LAYER_GPKG}")
    print(f"Wrote {PDF_PATH}")
    print(f"Wrote {PNG_PATH}")


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
