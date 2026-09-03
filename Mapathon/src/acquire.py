"""Data acquisition: OSM layers, street networks, and training ridership.

Everything is cached to data/raw so re-runs are cheap and the pipeline stays
reproducible offline once the first fetch has completed.

    python -m src.acquire --city chennai --what stations
    python -m src.acquire --city chennai --city bengaluru --what all
    python -m src.acquire --what bmrcl

Under-construction metro stations are tagged inconsistently in OSM, so
`fetch_metro_stations` deliberately casts a wide net and reports what tag
combinations it actually found rather than assuming a scheme. Inspect the
diagnostic output before trusting the classification.
"""

from __future__ import annotations

import argparse
import io
import json
import time
import zipfile
from pathlib import Path

import geopandas as gpd
import networkx as nx
import osmnx as ox
import pandas as pd
import requests

from src import config as C

BMRCL_REPO = "Vonter/bmrcl-ridership-hourly"
GITHUB_API = "https://api.github.com/repos/{repo}/git/trees/{ref}?recursive=1"
GITHUB_RAW = "https://raw.githubusercontent.com/{repo}/{ref}/{path}"


# --------------------------------------------------------------------------
# OSM setup
# --------------------------------------------------------------------------

# Overpass mirrors rate-limit anonymous traffic. Asked for their status, both
# kumi.systems and private.coffee answer: "Please include a meaningful
# User-Agent string with your requests to avoid rate-limiting", and return
# HTTP 429 without one. OSMnx sends its own default UA, which is exactly what
# gets throttled - this is what stalled the Chennai POI and bus-stop fetches
# for tens of minutes with no error surfaced.
USER_AGENT = (
    "IITB-FOSSEE-Mapathon-2026 chennai-metro-ridership "
    "(student research; https://github.com/; contact via repo)"
)


def setup_osmnx() -> None:
    """Point OSMnx at our own cache directory and identify ourselves politely."""
    cache = C.RAW_OSM / "_osmnx_cache"
    cache.mkdir(parents=True, exist_ok=True)
    ox.settings.use_cache = True
    ox.settings.cache_folder = str(cache)
    ox.settings.log_console = False
    # 90 s, not 300. A mirror that has not answered a bbox query in 90 seconds
    # is overloaded, and waiting the full five minutes on each of four mirrors
    # before falling back to tiles wasted the better part of an hour per layer.
    ox.settings.requests_timeout = 90
    ox.settings.http_user_agent = USER_AGENT
    # Some builds read the UA from the headers dict instead.
    headers = dict(getattr(ox.settings, "http_headers", {}) or {})
    headers["User-Agent"] = USER_AGENT
    ox.settings.http_headers = headers


def _bbox(city: str) -> tuple[float, float, float, float]:
    """OSMnx 2.x bbox order: (west, south, east, north)."""
    return C.CITIES[city]["bbox"]


# The public Overpass instance throttles aggressively once a session has run a
# few large queries, so mirrors are tried in turn before giving up.
# Order matters a great deal. Every mirror that fails costs up to
# `requests_timeout` before the next is tried, so the most reliable one must be
# first. overpass-api.de answers a plain `requests.post` in under two seconds
# yet ConnectTimeouts consistently under OSMnx here, burning 90 s on every
# single tile - so it is demoted rather than removed.
OVERPASS_MIRRORS = [
    "https://overpass.kumi.systems/api",
    "https://overpass.private.coffee/api",
    "https://overpass-api.de/api",
]

# Deliberately NOT included:
#   maps.mail.ru       - returns HTTP 504 under load
#   overpass.osm.ch    - answers quickly with zero elements; Swiss data only.
#     A mirror that returns an empty result rather than an error is worse than
#     one that is down, because the pipeline cannot tell "nothing here" from
#     "wrong server".



def _features(city: str, tags: dict, attempts: int = 2,
              tiled: bool = False) -> gpd.GeoDataFrame:
    """Query OSM features, optionally going straight to tiled requests.

    `tiled=True` skips the whole-bbox attempt entirely. Worth using for broad
    queries such as `shop=True`, which over a metropolitan bbox can occupy a
    mirror for minutes and then fail: with four mirrors and two attempts the
    fallback to tiling was costing up to 40 minutes of dead time before the
    first tile was even requested.
    """
    if tiled:
        got = _features_by_tiles(city, tags)
        ox.settings.overpass_url = OVERPASS_MIRRORS[0]
        return got if got is not None else gpd.GeoDataFrame(geometry=[], crs=C.CRS_GEO)
    return _features_whole_then_tiled(city, tags, attempts)


def _features_whole_then_tiled(city: str, tags: dict, attempts: int = 2) -> gpd.GeoDataFrame:
    """Query OSM features in the city bbox, returning an empty frame on no match.

    Distinguishes "nothing matched" (a real, final answer) from a transport
    failure (worth retrying elsewhere) - conflating the two would silently
    produce empty layers that look like genuine absence of data.
    """
    last_error = None

    for attempt in range(attempts):
        for mirror in OVERPASS_MIRRORS:
            ox.settings.overpass_url = mirror
            try:
                gdf = ox.features_from_bbox(_bbox(city), tags)
                if mirror != OVERPASS_MIRRORS[0]:
                    print(f"    (via {mirror})")
                return gdf.reset_index()
            except ox._errors.InsufficientResponseError:
                # Server answered, nothing matched. Not a transport problem.
                print(f"    no features match {tags}")
                return gpd.GeoDataFrame(geometry=[], crs=C.CRS_GEO)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                print(f"    {mirror.split('//')[1].split('/')[0]}: "
                      f"{type(exc).__name__}")
                continue
        if attempt < attempts - 1:
            print(f"    all mirrors failed, waiting 30 s before retry "
                  f"{attempt + 2}/{attempts}...")
            time.sleep(30)

    print(f"    whole-bbox refused ({type(last_error).__name__}); trying tiles")
    tiled = _features_by_tiles(city, tags)
    ox.settings.overpass_url = OVERPASS_MIRRORS[0]
    if tiled is not None:
        return tiled

    print(f"    giving up on {tags}")
    return gpd.GeoDataFrame(geometry=[], crs=C.CRS_GEO)


def _features_by_tiles(city: str, tags: dict, nx_tiles: int = 3, ny_tiles: int = 4):
    """Fetch a feature query as a grid of smaller bbox requests.

    Unfiltered queries like `shop=True` over a whole metropolitan bbox are
    among the heaviest things Overpass can be asked for, and Chennai's simply
    never returned. The same query in twelve pieces goes through.
    """
    west, south, east, north = _bbox(city)
    frames = []
    total = nx_tiles * ny_tiles

    for i in range(nx_tiles):
        for j in range(ny_tiles):
            w = west + (east - west) * i / nx_tiles
            e = west + (east - west) * (i + 1) / nx_tiles
            s = south + (north - south) * j / ny_tiles
            n = south + (north - south) * (j + 1) / ny_tiles

            index = i * ny_tiles + j + 1
            got = None
            for mirror in OVERPASS_MIRRORS:
                ox.settings.overpass_url = mirror
                try:
                    got = ox.features_from_bbox((w, s, e, n), tags)
                    break
                except ox._errors.InsufficientResponseError:
                    got = None
                    break                      # genuinely nothing here
                except Exception:              # noqa: BLE001 - try next mirror
                    continue
            if got is not None and len(got):
                frames.append(got.reset_index())
            # flush: stdout is a pipe when run in the background, so without
            # this there is no way to tell a slow fetch from a hung one.
            print(f"      tile {index}/{total}: {0 if got is None else len(got)}",
                  flush=True)

    if not frames:
        return None

    combined = pd.concat(frames, ignore_index=True)
    # Tiles share boundary features; de-duplicate on the OSM element id.
    if "id" in combined.columns:
        before = len(combined)
        combined = combined.drop_duplicates(subset="id")
        print(f"      combined {before:,} -> {len(combined):,} after de-duplication")
    return gpd.GeoDataFrame(combined, geometry="geometry", crs=C.CRS_GEO)


def _to_points(gdf: gpd.GeoDataFrame, crs_proj: str) -> gpd.GeoDataFrame:
    """Collapse mixed geometry (nodes, platform ways, station areas) to points.

    Station objects appear in OSM as nodes, closed ways and relations. Taking
    the projected centroid gives one comparable point per station.
    """
    if gdf.empty:
        return gdf
    projected = gdf.to_crs(crs_proj)
    projected["geometry"] = projected.geometry.centroid
    return projected


# --------------------------------------------------------------------------
# Metro stations
# --------------------------------------------------------------------------

# Broad probe: catches operational subway stations, suburban/MRTS stations,
# and the various ways mappers tag things that are not built yet.
STATION_PROBE = {
    "railway": ["station", "halt", "construction", "proposed"],
    "station": ["subway"],
    "construction:railway": ["station", "halt"],
    "proposed:railway": ["station", "halt"],
    "public_transport": ["station"],
}

TAG_COLUMNS = [
    "railway", "station", "subway", "construction", "construction:railway",
    "proposed", "proposed:railway", "public_transport", "network", "operator",
    "name", "name:en",
]


def classify_station(row: pd.Series) -> str:
    """Bucket an OSM station record into metro-operational / metro-construction / other-rail.

    Kept deliberately simple and readable so it can be corrected by hand once
    the diagnostic report shows what Chennai's mappers actually used.
    """
    def val(key):
        v = row.get(key)
        return str(v).lower() if pd.notna(v) else ""

    railway = val("railway")
    station = val("station")
    subway = val("subway")
    constr_rail = val("construction:railway")
    constr = val("construction")
    proposed_rail = val("proposed:railway")
    name = f"{val('name')} {val('name:en')}"
    network = f"{val('network')} {val('operator')}"

    looks_metro = (
        station == "subway"
        or subway == "yes"
        or "metro" in name
        or "metro" in network
        or "cmrl" in network
        or "bmrcl" in network
    )

    # Not yet built: any of the construction/proposed lifecycle prefixes.
    not_built = (
        railway in {"construction", "proposed"}
        or constr_rail in {"station", "halt"}
        or proposed_rail in {"station", "halt"}
        or constr in {"station", "subway", "light_rail"}
    )

    if not_built:
        return "metro_construction" if looks_metro or constr_rail or proposed_rail else "other_construction"
    if looks_metro:
        return "metro_operational"
    if railway in {"station", "halt"}:
        return "other_rail"          # suburban / MRTS - competition + interchange
    return "other"


def fetch_metro_stations(city: str, save: bool = True) -> gpd.GeoDataFrame:
    """Fetch all rail-station-like features and classify them.

    Returns a projected point layer with a `category` column. Always read the
    printed diagnostic: OSM coverage of under-construction alignments is
    uneven, and Phase 2 will almost certainly need reconciling against the
    Wikipedia station table and the CMRL DPR.
    """
    setup_osmnx()
    crs_proj = C.city_crs(city)
    print(f"[{city}] querying OSM for rail stations...")

    gdf = _features(city, STATION_PROBE)
    if gdf.empty:
        print(f"[{city}] nothing returned - check the bbox in config.CITIES")
        return gdf

    keep = [c for c in TAG_COLUMNS if c in gdf.columns]
    gdf = gdf[keep + ["geometry", "element", "id"] if "element" in gdf.columns else keep + ["geometry"]]

    points = _to_points(gdf, crs_proj)
    points["category"] = points.apply(classify_station, axis=1)
    points["city"] = city

    # Diagnostic: what did we actually match?
    print(f"[{city}] {len(points)} station-like features")
    print(points["category"].value_counts().to_string())

    combos = (
        points[[c for c in ("railway", "station", "construction:railway", "construction") if c in points.columns]]
        .fillna("-")
        .astype(str)
        .value_counts()
        .head(15)
    )
    print(f"\n[{city}] top tag combinations:\n{combos.to_string()}")

    if save:
        out = C.RAW_OSM / f"{city}_stations_raw.gpkg"
        points.to_file(out, driver="GPKG")
        print(f"[{city}] -> {out}")
    return points


# --------------------------------------------------------------------------
# Street network
# --------------------------------------------------------------------------

def fetch_street_network(
    city: str,
    network_type: str = "walk",
    attempts: int = 3,
) -> nx.MultiDiGraph:
    """Download (or load cached) the street graph for a city.

    The walk network is what catchments and intersection density are built on;
    it includes footways and pedestrian links that the drive network omits.

    This is by far the heaviest Overpass request in the pipeline - a few
    hundred thousand nodes - and the one most likely to be refused, so it
    rotates mirrors the same way `_features` does rather than failing outright.
    """
    setup_osmnx()
    path = C.RAW_OSM / f"{city}_{network_type}.graphml"
    if path.exists():
        print(f"[{city}] loading cached {network_type} graph")
        return ox.load_graphml(path)

    print(f"[{city}] downloading {network_type} network (this takes a few minutes)...")

    graph = _graph_with_mirrors(_bbox(city), network_type)
    if graph is None:
        print(f"[{city}] whole-bbox request refused; falling back to tiles")
        graph = _graph_by_tiles(city, network_type)

    if graph is None:
        ox.settings.overpass_url = OVERPASS_MIRRORS[0]
        raise RuntimeError(
            f"could not download the {city} {network_type} network, whole or tiled"
        )

    ox.save_graphml(graph, path)
    print(f"[{city}] {graph.number_of_nodes():,} nodes, "
          f"{graph.number_of_edges():,} edges -> {path}")
    return graph


def _graph_with_mirrors(bbox, network_type: str, quiet: bool = False):
    """One graph request, tried across each Overpass mirror. None if all refuse."""
    for mirror in OVERPASS_MIRRORS:
        ox.settings.overpass_url = mirror
        try:
            return ox.graph_from_bbox(bbox, network_type=network_type, simplify=True)
        except ox._errors.InsufficientResponseError:
            return None          # genuinely empty area - a tile over water
        except Exception as exc:  # noqa: BLE001
            if not quiet:
                host = mirror.split("//")[1].split("/")[0]
                print(f"      {host}: {type(exc).__name__}")
            continue
    return None


def _graph_by_tiles(city: str, network_type: str, nx_tiles: int = 3, ny_tiles: int = 4):
    """Compose the street graph from a grid of smaller bbox requests.

    Overpass will refuse a walk-network query over a whole metropolitan bbox
    when it is under load - Chennai's simply never returned - but serves the
    same area happily in pieces. Tiles overlap slightly so that ways crossing a
    boundary are present in both, and composing on OSM node ids stitches them
    back into one connected graph.
    """
    west, south, east, north = _bbox(city)
    overlap = 0.006  # ~650 m, comfortably longer than a typical block

    graphs = []
    total = nx_tiles * ny_tiles
    for i in range(nx_tiles):
        for j in range(ny_tiles):
            w = west + (east - west) * i / nx_tiles - (overlap if i else 0)
            e = west + (east - west) * (i + 1) / nx_tiles + (overlap if i < nx_tiles - 1 else 0)
            s = south + (north - south) * j / ny_tiles - (overlap if j else 0)
            n = south + (north - south) * (j + 1) / ny_tiles + (overlap if j < ny_tiles - 1 else 0)

            index = i * ny_tiles + j + 1
            print(f"    tile {index}/{total} ({w:.3f},{s:.3f},{e:.3f},{n:.3f})...", end=" ")
            tile = _graph_with_mirrors((w, s, e, n), network_type, quiet=True)
            if tile is None:
                print("empty/failed")
                continue
            print(f"{tile.number_of_nodes():,} nodes")
            graphs.append(tile)

    if not graphs:
        return None

    composed = nx.compose_all(graphs)
    # compose_all drops the graph-level attrs OSMnx needs downstream.
    composed.graph.update(graphs[0].graph)
    print(f"    composed {len(graphs)} tiles -> {composed.number_of_nodes():,} nodes")
    return composed


# --------------------------------------------------------------------------
# Corridor alignments
# --------------------------------------------------------------------------

# Chennai's under-construction corridors are mapped as `railway=subway` route
# relations rather than `railway=construction`, with "(u/c)" in the name.
#
# Issued as separate single-value queries rather than one combined query: a
# combined `{"railway": [...]}` becomes a regex filter and is a different
# Overpass request, which would miss the on-disk cache from the individual
# queries. Given how hard Overpass throttles, cache reuse matters more here
# than saving two round trips.
ALIGNMENT_TAG_SETS = [
    {"railway": "subway"},
    {"railway": "construction"},
    {"railway": "light_rail"},
]


def fetch_alignments(city: str, save: bool = True) -> gpd.GeoDataFrame:
    """Metro corridor linework, used to snap approximate station positions.

    A geocoded station name lands on a locality centroid, which can sit
    hundreds of metres off the actual alignment. Projecting it onto its own
    corridor removes most of that error, because a station is necessarily on
    the line.
    """
    setup_osmnx()
    crs_proj = C.city_crs(city)

    frames = []
    for tags in ALIGNMENT_TAG_SETS:
        part = _features(city, tags, attempts=1)
        if not part.empty:
            frames.append(part)

    if not frames:
        print(f"[{city}] no alignment linework found")
        return gpd.GeoDataFrame(geometry=[], crs=crs_proj)

    gdf = pd.concat(frames, ignore_index=True)
    gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs=C.CRS_GEO)

    lines = gdf[gdf.geometry.geom_type.isin(["LineString", "MultiLineString"])].copy()
    keep = [c for c in ("name", "railway", "construction", "colour", "ref", "operator") if c in lines.columns]
    lines = lines[keep + ["geometry"]].to_crs(crs_proj)
    lines = lines[lines["name"].notna()]
    lines["length_km"] = lines.geometry.length / 1000
    lines["city"] = city

    summary = lines.groupby("name")["length_km"].agg(["sum", "count"]).sort_values("sum", ascending=False)
    print(f"[{city}] {len(lines)} alignment features, {lines['length_km'].sum():.0f} km")
    print(summary.head(20).round(1).to_string())

    if save:
        out = C.RAW_OSM / f"{city}_alignments.gpkg"
        lines.to_file(out, driver="GPKG")
        print(f"[{city}] -> {out}")
    return lines


# --------------------------------------------------------------------------
# POIs, bus stops, parking
# --------------------------------------------------------------------------

def fetch_pois(city: str, save: bool = True, tiled: bool = False) -> gpd.GeoDataFrame:
    """Fetch POIs by the categories defined in config.OSM_POI_CATEGORIES.

    The category definitions must stay identical across cities: POI counts feed
    the model as a predictor, so narrowing `shop=True` for one city and not the
    other would make the feature mean different things in the training and
    prediction sets and quietly invalidate the transfer.
    """
    setup_osmnx()
    crs_proj = C.city_crs(city)
    frames = []

    for category, tags in C.OSM_POI_CATEGORIES.items():
        print(f"[{city}] POIs: {category}", flush=True)
        gdf = _features(city, tags, tiled=tiled)
        if gdf.empty:
            continue
        keep = [c for c in ("name", "amenity", "shop", "office", "leisure") if c in gdf.columns]
        gdf = gdf[keep + ["geometry"]]
        pts = _to_points(gdf, crs_proj)
        pts["category"] = category
        frames.append(pts)

    if not frames:
        return gpd.GeoDataFrame(geometry=[], crs=crs_proj)

    pois = pd.concat(frames, ignore_index=True)
    pois = gpd.GeoDataFrame(pois, geometry="geometry", crs=crs_proj)
    pois["city"] = city
    print(f"[{city}] {len(pois):,} POIs\n{pois['category'].value_counts().to_string()}")

    if save:
        out = C.RAW_OSM / f"{city}_pois.gpkg"
        pois.to_file(out, driver="GPKG")
        print(f"[{city}] -> {out}")
    return pois


def fetch_bus_stops(city: str, save: bool = True, tiled: bool = False) -> gpd.GeoDataFrame:
    """Bus stops and stations - the feeder-access variable.

    Preferred over the Transitland MTC GTFS feed, whose Chennai vintage is
    around 2010 and no longer reflects the network.
    """
    setup_osmnx()
    crs_proj = C.city_crs(city)
    gdf = _features(city, C.OSM_BUS_STOP_TAGS, tiled=tiled)
    if gdf.empty:
        return gpd.GeoDataFrame(geometry=[], crs=crs_proj)

    keep = [c for c in ("name", "highway", "amenity") if c in gdf.columns]
    stops = _to_points(gdf[keep + ["geometry"]], crs_proj)
    stops["city"] = city
    print(f"[{city}] {len(stops):,} bus stops")

    if save:
        out = C.RAW_OSM / f"{city}_bus_stops.gpkg"
        stops.to_file(out, driver="GPKG")
        print(f"[{city}] -> {out}")
    return stops


def fetch_parking(city: str, save: bool = True, tiled: bool = False) -> gpd.GeoDataFrame:
    """Parking areas - park-and-ride potential for terminal stations."""
    setup_osmnx()
    crs_proj = C.city_crs(city)
    gdf = _features(city, C.OSM_PARKING_TAGS, tiled=tiled)
    if gdf.empty:
        return gpd.GeoDataFrame(geometry=[], crs=crs_proj)

    keep = [c for c in ("name", "amenity", "parking", "capacity") if c in gdf.columns]
    # Area matters more than count here, so keep polygons intact.
    parking = gdf[keep + ["geometry"]].to_crs(crs_proj)
    parking["area_m2"] = parking.geometry.area
    parking["city"] = city
    print(f"[{city}] {len(parking):,} parking features")

    if save:
        out = C.RAW_OSM / f"{city}_parking.gpkg"
        parking.to_file(out, driver="GPKG")
        print(f"[{city}] -> {out}")
    return parking


# --------------------------------------------------------------------------
# BMRCL training ridership
# --------------------------------------------------------------------------

def _github_tree(repo: str, ref: str = "main") -> list[dict]:
    resp = requests.get(GITHUB_API.format(repo=repo, ref=ref), timeout=60)
    if resp.status_code == 404 and ref == "main":
        resp = requests.get(GITHUB_API.format(repo=repo, ref="master"), timeout=60)
    resp.raise_for_status()
    return resp.json().get("tree", [])


def download_bmrcl_ridership(save_dir: Path | None = None) -> list[Path]:
    """Download BMRCL station-wise ridership (the regression training target).

    Lists the repo tree via the GitHub API rather than hardcoding filenames,
    which have changed between releases. Prefers parquet, falls back to zipped
    CSV, and skips the station-pair OD files (large, not needed for a
    station-level demand model).
    """
    save_dir = save_dir or C.RAW_RIDERSHIP
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"listing {BMRCL_REPO}...")
    tree = _github_tree(BMRCL_REPO)
    data_files = [
        item for item in tree
        if item["type"] == "blob"
        and item["path"].lower().endswith((".parquet", ".zip", ".csv"))
    ]

    if not data_files:
        print("no data files found - inspect the repo manually")
        return []

    print(f"found {len(data_files)} candidate files:")
    for item in data_files:
        size_mb = item.get("size", 0) / 1e6
        print(f"  {item['path']}  ({size_mb:.1f} MB)")

    # Station-level entries/exits are what we need; OD pair files are far
    # larger and only useful for a future transfer/interchange model.
    wanted = [f for f in data_files if "pair" not in f["path"].lower()]

    # Prefer parquet for the bulk series, but always keep the small lookup
    # CSVs under raw/ - they carry canonical station names and codes, which
    # the OSM geometry join needs.
    lookups = [f for f in wanted if f["path"].startswith("raw/")]
    bulk = [f for f in wanted if not f["path"].startswith("raw/")]
    preferred = ([f for f in bulk if f["path"].endswith(".parquet")] or bulk) + lookups

    written = []
    for item in preferred:
        url = GITHUB_RAW.format(repo=BMRCL_REPO, ref="main", path=item["path"])
        dest = save_dir / Path(item["path"]).name
        if dest.exists():
            print(f"  cached {dest.name}")
            written.append(dest)
            continue
        print(f"  downloading {item['path']}...")
        resp = requests.get(url, timeout=300)
        if resp.status_code != 200:
            print(f"    failed ({resp.status_code}), skipping")
            continue
        dest.write_bytes(resp.content)
        written.append(dest)

        if dest.suffix == ".zip":
            with zipfile.ZipFile(dest) as zf:
                zf.extractall(save_dir)
            print(f"    extracted {len(zf.namelist())} file(s)")

    # Record provenance for the methodology page and licence compliance.
    manifest = save_dir / "bmrcl_manifest.json"
    manifest.write_text(json.dumps({
        "repo": BMRCL_REPO,
        "url": f"https://github.com/{BMRCL_REPO}",
        "licence": "ODbL-1.0",
        "sourced_via": "RTI to BMRCL",
        "files": [p.name for p in written],
    }, indent=2))
    print(f"-> {len(written)} file(s) in {save_dir}")
    return written


# --------------------------------------------------------------------------
# GHSL population and built-up rasters
# --------------------------------------------------------------------------

GHSL_BASE = "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL"

# GHSL publishes on a global Mollweide (ESRI:54009) grid of 36 columns x 18
# rows, each tile 1,000,000 m square.
GHSL_X_MIN, GHSL_Y_MAX, GHSL_TILE = -18041000.0, 9020048.0, 1_000_000.0

# Products worth having: population now, population in 2030 (the future-year
# scenario), and built-up surface/volume as activity and employment proxies.
GHSL_PRODUCTS = [
    ("GHS_POP", "E2025"),
    ("GHS_POP", "E2030"),
    ("GHS_BUILT_S", "E2025"),
    ("GHS_BUILT_V", "E2025"),
]


def ghsl_tiles_for_bbox(bbox: tuple[float, float, float, float]) -> set[tuple[int, int]]:
    """Mollweide tile (row, col) IDs covering a lon/lat bounding box.

    All four corners are converted rather than just the centroid, so a study
    area straddling a tile edge still gets every tile it needs.
    """
    from pyproj import Transformer

    transformer = Transformer.from_crs(C.CRS_GEO, "ESRI:54009", always_xy=True)
    west, south, east, north = bbox

    tiles = set()
    for lon in (west, east):
        for lat in (south, north):
            x, y = transformer.transform(lon, lat)
            col = int((x - GHSL_X_MIN) // GHSL_TILE) + 1
            row = int((GHSL_Y_MAX - y) // GHSL_TILE) + 1
            tiles.add((row, col))
    return tiles


def download_ghsl(cities: list[str] | None = None, save_dir: Path | None = None) -> list[Path]:
    """Download and extract the GHSL tiles covering the study cities.

    Chennai and Bengaluru both fall in tile R8_C26, so the set is small.
    """
    save_dir = save_dir or C.RAW_GHSL
    save_dir.mkdir(parents=True, exist_ok=True)
    cities = cities or list(C.CITIES)

    tiles: set[tuple[int, int]] = set()
    for city in cities:
        city_tiles = ghsl_tiles_for_bbox(C.CITIES[city]["bbox"])
        print(f"[{city}] tiles: {sorted(f'R{r}_C{c}' for r, c in city_tiles)}")
        tiles |= city_tiles

    written = []
    for product, epoch in GHSL_PRODUCTS:
        for row, col in sorted(tiles):
            stem = f"{product}_{epoch}_GLOBE_R2023A_54009_100"
            fname = f"{stem}_V1_0_R{row}_C{col}.zip"
            url = f"{GHSL_BASE}/{product}_GLOBE_R2023A/{stem}/V1-0/tiles/{fname}"

            tif = save_dir / fname.replace(".zip", ".tif")
            if tif.exists():
                print(f"  cached {tif.name}")
                written.append(tif)
                continue

            print(f"  downloading {fname}...")
            try:
                resp = requests.get(url, timeout=900, stream=True)
                resp.raise_for_status()
            except Exception as exc:  # noqa: BLE001
                print(f"    failed: {type(exc).__name__} - skipping")
                continue

            zip_path = save_dir / fname
            with open(zip_path, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    fh.write(chunk)

            with zipfile.ZipFile(zip_path) as zf:
                for member in zf.namelist():
                    if member.endswith(".tif"):
                        zf.extract(member, save_dir)
                        written.append(save_dir / member)
                        print(f"    extracted {member} "
                              f"({(save_dir / member).stat().st_size / 1e6:.0f} MB)")
            zip_path.unlink()

    manifest = save_dir / "ghsl_manifest.json"
    manifest.write_text(json.dumps({
        "source": "JRC Global Human Settlement Layer, release R2023A",
        "url": "https://ghsl.jrc.ec.europa.eu/download.php",
        "licence": "CC BY 4.0",
        "crs": "ESRI:54009 (World Mollweide)",
        "resolution_m": 100,
        "tiles": sorted(f"R{r}_C{c}" for r, c in tiles),
        "files": [p.name for p in written],
    }, indent=2))

    print(f"-> {len(written)} raster(s) in {save_dir}")
    return written


# --------------------------------------------------------------------------
# Canonical station lists (Wikipedia)
# --------------------------------------------------------------------------

WIKI_CHENNAI_STATIONS = "https://en.wikipedia.org/wiki/List_of_Chennai_Metro_stations"


def fetch_wikipedia_station_lists(url: str = WIKI_CHENNAI_STATIONS, save: bool = True):
    """Scrape the canonical Chennai station name lists from Wikipedia.

    OSM alone is not enough: it holds all 41 Phase 1 stations but only a
    fraction of the 128 Phase 2 stations, and mixes Tamil-script duplicates in.
    Wikipedia gives an authoritative *name* list with line and layout for both
    phases -- names that then get positions from OSM, geocoding or the DPR.

    Returns (operational, under_construction) DataFrames. Table structure on
    Wikipedia does change, so the shapes are printed for inspection.
    """
    resp = requests.get(url, timeout=60, headers={"User-Agent": "mapathon-chennai-metro/0.1"})
    resp.raise_for_status()

    # pandas 3.0 no longer accepts a literal HTML string here - it would treat
    # it as a path - so wrap it in a buffer.
    tables = pd.read_html(io.StringIO(resp.text))
    print(f"parsed {len(tables)} tables from {url}")
    for i, tbl in enumerate(tables):
        cols = [str(c)[:28] for c in tbl.columns][:6]
        print(f"  [{i}] {tbl.shape[0]:>4} rows x {tbl.shape[1]:>2} cols  {cols}")

    def looks_like_stations(tbl: pd.DataFrame) -> bool:
        joined = " ".join(str(c).lower() for c in tbl.columns)
        return tbl.shape[0] >= 20 and ("station" in joined or "name" in joined)

    candidates = [t for t in tables if looks_like_stations(t)]
    if not candidates:
        raise RuntimeError("no station-like tables found; inspect the printout above")

    # The operational table is the one mentioning an opening date; the
    # under-construction table is typically the largest remaining one.
    def has_opened(tbl):
        return any("open" in str(c).lower() for c in tbl.columns)

    operational = next((t for t in candidates if has_opened(t)), candidates[0])
    remaining = [t for t in candidates if t is not operational]
    under_construction = max(remaining, key=lambda t: t.shape[0]) if remaining else None

    print(f"\noperational table: {operational.shape}")
    if under_construction is not None:
        print(f"under-construction table: {under_construction.shape}")
        print(f"  (expect ~{C.CHENNAI_PHASE1_STATIONS} operational, "
              f"~{C.CHENNAI_PHASE2_STATIONS} under construction)")

    if save:
        operational.to_csv(C.RAW_BOUNDARIES / "wiki_chennai_operational.csv", index=False)
        if under_construction is not None:
            under_construction.to_csv(
                C.RAW_BOUNDARIES / "wiki_chennai_under_construction.csv", index=False
            )
        print(f"-> {C.RAW_BOUNDARIES}")

    return operational, under_construction


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

WHAT = ("stations", "alignments", "network", "pois", "bus", "parking",
        "bmrcl", "wiki", "ghsl", "all")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--city", action="append", choices=list(C.CITIES), default=None)
    parser.add_argument("--what", action="append", choices=WHAT, default=None)
    parser.add_argument("--tiled", action="store_true",
                        help="skip whole-bbox POI requests and fetch by tiles "
                             "(much faster when a broad query keeps timing out)")
    args = parser.parse_args()

    cities = args.city or list(C.CITIES)
    what = set(args.what or ["all"])
    everything = "all" in what

    for city in cities:
        print(f"\n{'=' * 60}\n{C.CITIES[city]['name']}\n{'=' * 60}")
        if everything or "stations" in what:
            fetch_metro_stations(city)
        if everything or "alignments" in what:
            fetch_alignments(city)
        if everything or "network" in what:
            fetch_street_network(city, "walk")
        if everything or "pois" in what:
            fetch_pois(city, tiled=args.tiled)
        if everything or "bus" in what:
            fetch_bus_stops(city, tiled=args.tiled)
        if everything or "parking" in what:
            fetch_parking(city, tiled=args.tiled)

    if everything or "bmrcl" in what:
        print(f"\n{'=' * 60}\nBMRCL ridership\n{'=' * 60}")
        download_bmrcl_ridership()

    if everything or "wiki" in what:
        print(f"\n{'=' * 60}\nCanonical station lists\n{'=' * 60}")
        fetch_wikipedia_station_lists()

    if everything or "ghsl" in what:
        print(f"\n{'=' * 60}\nGHSL population / built-up rasters\n{'=' * 60}")
        download_ghsl(cities)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
