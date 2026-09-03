"""Extract POIs, bus stops and parking from a Geofabrik .osm.pbf.

Why this exists
---------------
Overpass proved unusable for these layers: the whole-bbox queries never
returned, and tiled requests managed 2 of 24 tiles in forty minutes. A single
regional extract removes the dependency entirely - one sustained download, then
local extraction in seconds, repeatable offline and reproducible for anyone
checking the methodology.

Both cities are extracted here with the *same* code path. That matters more than
it looks: POI counts feed the model as a predictor, so if Chennai's came from
GDAL and Bengaluru's from OSMnx, any difference in tag handling would make the
feature mean different things in the training and prediction sets and quietly
invalidate the transfer.

Requires GDAL with the OSM driver - already present via QGIS at D:\\QGIS\\bin.

    python -m src.pbf_extract --city chennai --city bengaluru
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import geopandas as gpd
import pandas as pd

from src import config as C

PBF = C.RAW_OSM / "pbf" / "southern-zone-latest.osm.pbf"

# GDAL's OSM driver promotes some tags to real columns and dumps the rest into
# an `other_tags` HSTORE. Crucially the promoted set is **per layer**, and they
# differ a lot: `points` does not promote amenity/shop/office/leisure, while
# `multipolygons` does. Filtering on a column the layer lacks made GDAL drop the
# WHERE clause silently and return every tagged feature in the bbox - 440,812
# "bus stops" for Chennai, and the same 40,428 points for two different queries.
#
# Read from osmconf.ini rather than hardcoded, so this cannot drift from the
# GDAL build actually in use.
FALLBACK_PROMOTED = {
    "points": {"name", "barrier", "highway", "ref", "address", "is_in", "place",
               "man_made"},
    "lines": {"name", "highway", "waterway", "aerialway", "barrier", "man_made",
              "railway"},
    "multipolygons": {"name", "type", "aeroway", "amenity", "admin_level",
                      "barrier", "boundary", "building", "craft", "geological",
                      "historic", "land_area", "landuse", "leisure", "man_made",
                      "military", "natural", "office", "place", "shop", "sport",
                      "tourism"},
}

_promoted_cache: dict[str, set[str]] = {}


def _promoted_attrs(layer: str) -> set[str]:
    """Columns GDAL promotes for a given OSM layer, read from osmconf.ini."""
    if _promoted_cache:
        return _promoted_cache.get(layer, FALLBACK_PROMOTED.get(layer, set()))

    env = _gdal_env()
    conf = Path(env.get("GDAL_DATA", "")) / "osmconf.ini"
    if conf.exists():
        current = None
        for line in conf.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if line.startswith("[") and line.endswith("]"):
                current = line[1:-1]
            elif line.startswith("attributes=") and current:
                _promoted_cache[current] = {
                    a.strip() for a in line.split("=", 1)[1].split(",") if a.strip()
                }
    if not _promoted_cache:
        _promoted_cache.update(FALLBACK_PROMOTED)
    return _promoted_cache.get(layer, FALLBACK_PROMOTED.get(layer, set()))


def _gdal_env() -> dict:
    """Environment for the GDAL subprocess.

    The QGIS-bundled ogr2ogr does not set GDAL_DATA itself, so it cannot find
    osmconf.ini - the file that defines which OSM tags become columns - and
    refuses to open a .pbf at all. PROJ_LIB is set for the same reason.
    """
    import os

    env = os.environ.copy()
    for var, candidates in {
        "GDAL_DATA": [r"D:\QGIS\apps\gdal\share\gdal",
                      r"C:\Program Files\QGIS\apps\gdal\share\gdal"],
        "PROJ_LIB": [r"D:\QGIS\share\proj", r"C:\Program Files\QGIS\share\proj"],
    }.items():
        if env.get(var) and Path(env[var]).exists():
            continue
        for candidate in candidates:
            if Path(candidate).exists():
                env[var] = candidate
                break
    return env


def _gdal_bin(name: str) -> str:
    """Locate a GDAL executable, preferring one already on PATH."""
    found = shutil.which(name)
    if found:
        return found
    for candidate in (Path(r"D:\QGIS\bin") / f"{name}.exe",
                      Path(r"C:\Program Files\QGIS\bin") / f"{name}.exe"):
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError(
        f"{name} not found. It ships with QGIS; check D:\\QGIS\\bin."
    )


def _hstore_filter(key: str, values=None) -> str:
    """SQL predicate for a tag that GDAL leaves inside `other_tags`."""
    if values is None:
        return f"other_tags LIKE '%\"{key}\"=>%'"
    ors = " OR ".join(f"other_tags LIKE '%\"{key}\"=>\"{v}\"%'" for v in values)
    return f"({ors})"


def _tag_filter(key: str, values, layer: str) -> str:
    """SQL predicate for one tag on one layer, column or hstore as appropriate."""
    if key in _promoted_attrs(layer):
        if values is True:
            return f"{key} IS NOT NULL"
        quoted = ", ".join(f"'{v}'" for v in values)
        return f"{key} IN ({quoted})"
    return _hstore_filter(key, None if values is True else values)


def extract_layer(city: str, tags: dict, layers=("points", "multipolygons"),
                  pbf: Path | None = None) -> gpd.GeoDataFrame:
    """Pull features matching `tags` from the PBF, clipped to the city bbox.

    `tags` follows the same shape as the OSMnx queries in config, and keys are
    OR-ed together exactly as OSMnx does.
    """
    pbf = pbf or PBF
    if not pbf.exists():
        raise FileNotFoundError(
            f"{pbf} missing. Download it with:\n"
            "  https://download.geofabrik.de/asia/india/southern-zone-latest.osm.pbf"
        )

    west, south, east, north = C.CITIES[city]["bbox"]
    ogr2ogr = _gdal_bin("ogr2ogr")

    frames = []
    for layer in layers:
        # The WHERE clause is built per layer: the same tag is a column on one
        # layer and an hstore entry on another.
        where = " OR ".join(_tag_filter(k, v, layer) for k, v in tags.items())

        out = C.INTERIM / f"_pbf_{city}_{layer}.gpkg"
        if out.exists():
            out.unlink()
        cmd = [
            ogr2ogr, "-f", "GPKG", str(out), str(pbf), layer,
            "-spat", str(west), str(south), str(east), str(north),
            "-where", where,
            "-nlt", "PROMOTE_TO_MULTI",
            "--config", "OGR_INTERLEAVED_READING", "YES",
            # No -skipfailures: it turned a malformed WHERE clause into a silent
            # pass-everything, which is how 440,812 "bus stops" happened. A bad
            # filter must fail loudly.
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600,
                                env=_gdal_env())
        if result.returncode != 0:
            print(f"    {layer}: ogr2ogr failed\n{result.stderr[:500]}")
            continue
        if not out.exists():
            print(f"    {layer}: no output")
            continue
        try:
            gdf = gpd.read_file(out)
        except Exception as exc:  # noqa: BLE001 - empty layer reads as an error
            print(f"    {layer}: unreadable ({type(exc).__name__})")
            continue
        if len(gdf):
            gdf["source_layer"] = layer
            frames.append(gdf)
        print(f"    {layer}: {len(gdf):,}")

        # A filter that quietly stops filtering returns tens of thousands of
        # features. Nothing we query for is that common, so flag it.
        if len(gdf) > 30_000:
            print(f"    WARNING: {len(gdf):,} features from {layer} for {tags} "
                  f"looks like the WHERE clause was not applied - verify before "
                  f"trusting this layer")

    if not frames:
        return gpd.GeoDataFrame(geometry=[], crs=C.CRS_GEO)

    combined = pd.concat(frames, ignore_index=True)
    return gpd.GeoDataFrame(combined, geometry="geometry", crs=C.CRS_GEO)


def _to_points(gdf: gpd.GeoDataFrame, crs_proj: str) -> gpd.GeoDataFrame:
    if gdf.empty:
        return gdf
    projected = gdf.to_crs(crs_proj)
    projected["geometry"] = projected.geometry.centroid
    return projected


def build_pois(city: str, save: bool = True) -> gpd.GeoDataFrame:
    crs_proj = C.city_crs(city)
    frames = []
    for category, tags in C.OSM_POI_CATEGORIES.items():
        print(f"[{city}] POIs: {category}", flush=True)
        gdf = extract_layer(city, tags)
        if gdf.empty:
            continue
        pts = _to_points(gdf, crs_proj)
        pts["category"] = category
        keep = [c for c in ("name", "amenity", "shop", "office", "leisure",
                            "category", "geometry") if c in pts.columns]
        frames.append(pts[keep])

    if not frames:
        return gpd.GeoDataFrame(geometry=[], crs=crs_proj)

    pois = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True),
                            geometry="geometry", crs=crs_proj)
    pois["city"] = city
    print(f"[{city}] {len(pois):,} POIs")
    print(pois["category"].value_counts().to_string())

    if save:
        out = C.RAW_OSM / f"{city}_pois.gpkg"
        pois.to_file(out, driver="GPKG")
        print(f"[{city}] -> {out}")
    return pois


def build_bus_stops(city: str, save: bool = True) -> gpd.GeoDataFrame:
    crs_proj = C.city_crs(city)
    print(f"[{city}] bus stops", flush=True)
    gdf = extract_layer(city, C.OSM_BUS_STOP_TAGS)
    if gdf.empty:
        return gpd.GeoDataFrame(geometry=[], crs=crs_proj)

    stops = _to_points(gdf, crs_proj)
    keep = [c for c in ("name", "highway", "amenity", "geometry") if c in stops.columns]
    stops = stops[keep]
    stops["city"] = city
    print(f"[{city}] {len(stops):,} bus stops")

    if save:
        out = C.RAW_OSM / f"{city}_bus_stops.gpkg"
        stops.to_file(out, driver="GPKG")
        print(f"[{city}] -> {out}")
    return stops


def build_parking(city: str, save: bool = True) -> gpd.GeoDataFrame:
    crs_proj = C.city_crs(city)
    print(f"[{city}] parking", flush=True)
    gdf = extract_layer(city, C.OSM_PARKING_TAGS)
    if gdf.empty:
        return gpd.GeoDataFrame(geometry=[], crs=crs_proj)

    # Area matters more than count for park-and-ride, so polygons stay whole.
    parking = gdf.to_crs(crs_proj)
    parking["area_m2"] = parking.geometry.area
    keep = [c for c in ("name", "amenity", "area_m2", "geometry") if c in parking.columns]
    parking = parking[keep]
    parking["city"] = city
    print(f"[{city}] {len(parking):,} parking features, "
          f"{parking['area_m2'].sum() / 1e6:.2f} km2")

    if save:
        out = C.RAW_OSM / f"{city}_parking.gpkg"
        parking.to_file(out, driver="GPKG")
        print(f"[{city}] -> {out}")
    return parking


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--city", action="append", choices=list(C.CITIES), default=None)
    parser.add_argument("--what", action="append",
                        choices=("pois", "bus", "parking", "all"), default=None)
    parser.add_argument("--pbf", type=Path, default=None)
    args = parser.parse_args()

    cities = args.city or list(C.CITIES)
    what = set(args.what or ["all"])
    everything = "all" in what

    for city in cities:
        print(f"\n{'=' * 60}\n{C.CITIES[city]['name']}\n{'=' * 60}", flush=True)
        if everything or "pois" in what:
            build_pois(city)
        if everything or "bus" in what:
            build_bus_stops(city)
        if everything or "parking" in what:
            build_parking(city)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
