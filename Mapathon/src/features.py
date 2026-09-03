"""5D feature extraction for station catchments.

Follows Ewing & Cervero's Density / Diversity / Design / Destination
accessibility / Distance-to-transit framework, which is the standard structure
for direct demand models and keeps the variable set interpretable.

Raster-derived variables (population, built-up volume, land-use entropy) are
computed only if the corresponding raster is present in data/raw; the module
reports what it skipped rather than failing, so the OSM-derived features can be
built and inspected before the ISRO and population downloads finish.

    python -m src.features --city bengaluru
"""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
from shapely.geometry import Point

from src import acquire
from src import config as C


# --------------------------------------------------------------------------
# Density - raster zonal statistics
# --------------------------------------------------------------------------

def zonal_sum(catchments: gpd.GeoDataFrame, raster: Path, label: str) -> pd.Series:
    """Sum a population-style raster within each catchment.

    Population rasters are counts per cell, so the correct aggregate is a sum,
    not a mean - taking the mean of a count raster is a common and silent error.
    """
    from rasterstats import zonal_stats

    stats = zonal_stats(
        catchments.to_crs(_raster_crs(raster)),
        str(raster),
        stats=["sum"],
        all_touched=False,
        nodata=None,
    )
    values = pd.Series([s["sum"] if s["sum"] is not None else np.nan for s in stats],
                       index=catchments.index, name=label)
    print(f"  {label}: {values.notna().sum()}/{len(values)} catchments, "
          f"total {values.sum():,.0f}")
    return values


def _raster_crs(raster: Path) -> str:
    import rasterio
    with rasterio.open(raster) as src:
        return src.crs.to_string()


def _find_raster(pattern: str, epoch: str | None = None) -> Path | None:
    """Locate a GHSL raster deterministically.

    Globbing and taking [0] is not safe here: GHS_POP is downloaded for both
    2025 and 2030, and filesystem order would decide which one became the
    baseline population. The epoch is matched explicitly.
    """
    matches = sorted(C.RAW_GHSL.glob(pattern))
    if epoch:
        matches = [p for p in matches if epoch in p.name]
    if not matches:
        matches = sorted(C.RAW_WORLDPOP.glob(pattern))
    return matches[0] if matches else None


def density_features(catchments: gpd.GeoDataFrame) -> pd.DataFrame:
    """Population and built-up density, where the rasters are available.

    `population` is the 2025 epoch and is what the model trains and predicts
    on. `population_2030` is carried alongside purely for the future-year
    scenario and must never enter the design matrix - it is a projection of the
    same quantity and would be near-collinear with the baseline.
    """
    out = pd.DataFrame(index=catchments.index)
    area_km2 = catchments.geometry.area / 1e6
    out["catchment_area_km2"] = area_km2

    candidates = {
        "population": _find_raster("*POP*.tif", epoch="E2025"),
        "built_volume": _find_raster("*BUILT_V*.tif", epoch="E2025"),
        "built_surface": _find_raster("*BUILT_S*.tif", epoch="E2025"),
    }

    for label, path in candidates.items():
        if path is None:
            print(f"  {label}: no raster found - skipped "
                  f"(run: python -m src.acquire --what ghsl)")
            continue
        print(f"  {label}: {path.name}")
        out[label] = zonal_sum(catchments, path, label)
        out[f"{label}_density"] = out[label] / area_km2.replace(0, np.nan)

    # Scenario input only; excluded from the model by src.model.candidate_features.
    future = _find_raster("*POP*.tif", epoch="E2030")
    if future is not None:
        out["population_2030"] = zonal_sum(catchments, future, "population_2030")

    return out


# --------------------------------------------------------------------------
# Diversity - land-use mix
# --------------------------------------------------------------------------

def shannon_entropy(proportions: np.ndarray) -> float:
    """Normalised Shannon entropy of land-use proportions.

    0 = single-use catchment, 1 = perfectly even mix across the classes
    present. Normalising by log(k) keeps catchments comparable when some
    classes are absent.
    """
    p = proportions[proportions > 0]
    if p.size <= 1:
        return 0.0
    return float(-(p * np.log(p)).sum() / np.log(p.size))


def landuse_features(catchments: gpd.GeoDataFrame, lulc_path: Path | None = None) -> pd.DataFrame:
    """Land-use mix entropy and a jobs-housing balance proxy from Bhuvan LULC.

    Expects a categorical raster whose classes have been remapped to the groups
    in config.LULC_GROUPS. Skipped cleanly when the Bhuvan download is not yet
    in place.
    """
    out = pd.DataFrame(index=catchments.index)

    if lulc_path is None:
        found = list(C.RAW_BHUVAN.glob("*.tif"))
        lulc_path = found[0] if found else None

    if lulc_path is None:
        print("  land use: no Bhuvan LULC raster found - skipped")
        return out

    from rasterstats import zonal_stats

    stats = zonal_stats(
        catchments.to_crs(_raster_crs(lulc_path)), str(lulc_path),
        categorical=True, nodata=None,
    )

    entropies, employment_ratio = [], []
    for record in stats:
        if not record:
            entropies.append(np.nan)
            employment_ratio.append(np.nan)
            continue
        total = sum(record.values())
        props = np.array([v / total for v in record.values()])
        entropies.append(shannon_entropy(props))
        # Placeholder until the raster legend is mapped onto LULC_GROUPS;
        # see config.LULC_GROUPS / LULC_EMPLOYMENT_GROUPS.
        employment_ratio.append(np.nan)

    out["landuse_entropy"] = entropies
    out["employment_ratio"] = employment_ratio
    print(f"  land use entropy: computed for {out['landuse_entropy'].notna().sum()} catchments")
    return out


# --------------------------------------------------------------------------
# Design - street network form
# --------------------------------------------------------------------------

def design_features(
    catchments: gpd.GeoDataFrame,
    graph: nx.MultiDiGraph,
) -> pd.DataFrame:
    """Intersection density, street density and circuity per catchment.

    Computed by spatially joining the whole network once rather than
    re-querying per station, which is orders of magnitude faster across
    a few hundred catchments.
    """
    crs = catchments.crs
    graph_proj = ox.project_graph(graph, to_crs=crs)
    nodes, edges = ox.graph_to_gdfs(graph_proj)

    # Real intersections only: degree-1 nodes are cul-de-sacs and degree-2
    # nodes are geometry vertices, neither of which is a route choice.
    degrees = dict(graph_proj.degree())
    nodes["degree"] = nodes.index.map(degrees)
    intersections = nodes[nodes["degree"] >= 3]

    area_km2 = catchments.geometry.area / 1e6
    out = pd.DataFrame(index=catchments.index)

    joined = gpd.sjoin(intersections, catchments[["geometry"]], how="inner", predicate="within")
    counts = joined.groupby("index_right").size()
    out["intersection_count"] = counts.reindex(catchments.index).fillna(0)
    out["intersection_density"] = out["intersection_count"] / area_km2.replace(0, np.nan)

    edge_join = gpd.sjoin(edges, catchments[["geometry"]], how="inner", predicate="intersects")
    street_km = edge_join.groupby("index_right")["length"].sum() / 1000
    out["street_length_km"] = street_km.reindex(catchments.index).fillna(0)
    out["street_density"] = out["street_length_km"] / area_km2.replace(0, np.nan)

    # Circuity: network distance divided by straight-line distance along each
    # edge. Above 1 means indirect routing; a good walkable grid sits near 1.05.
    edges_geo = edges.copy()
    straight = edges_geo.geometry.apply(
        lambda g: Point(g.coords[0]).distance(Point(g.coords[-1])) if g.geom_type == "LineString" else np.nan
    )
    edges_geo["circuity"] = edges_geo["length"] / straight.replace(0, np.nan)
    circ_join = gpd.sjoin(edges_geo, catchments[["geometry"]], how="inner", predicate="intersects")
    out["circuity"] = (
        circ_join.groupby("index_right")["circuity"].median().reindex(catchments.index)
    )

    print(f"  design: median intersection density "
          f"{out['intersection_density'].median():.0f}/km2, "
          f"median street density {out['street_density'].median():.1f} km/km2")
    return out


# --------------------------------------------------------------------------
# Destination accessibility
# --------------------------------------------------------------------------

def poi_features(catchments: gpd.GeoDataFrame, city: str) -> pd.DataFrame:
    """POI counts per catchment, by category."""
    out = pd.DataFrame(index=catchments.index)
    path = C.RAW_OSM / f"{city}_pois.gpkg"
    if not path.exists():
        print(f"  POIs: {path.name} missing - run src.acquire --what pois")
        return out

    pois = gpd.read_file(path).to_crs(catchments.crs)
    joined = gpd.sjoin(pois, catchments[["geometry"]], how="inner", predicate="within")

    for category in C.OSM_POI_CATEGORIES:
        counts = joined[joined["category"] == category].groupby("index_right").size()
        out[f"poi_{category}"] = counts.reindex(catchments.index).fillna(0)

    out["poi_total"] = out.filter(like="poi_").sum(axis=1)
    print(f"  POIs: {int(out['poi_total'].sum()):,} within catchments")
    return out


def accessibility_features(stations: gpd.GeoDataFrame, city: str) -> pd.DataFrame:
    """Distance to the CBD and to the nearest secondary activity centre."""
    crs = stations.crs
    out = pd.DataFrame(index=stations.index)

    cbd = gpd.GeoSeries([Point(*C.CITIES[city]["cbd"])], crs=C.CRS_GEO).to_crs(crs).iloc[0]
    out["dist_cbd_km"] = stations.geometry.distance(cbd) / 1000

    centres = C.CITIES[city].get("secondary_centres", {})
    if centres:
        pts = gpd.GeoSeries([Point(*v) for v in centres.values()], crs=C.CRS_GEO).to_crs(crs)
        out["dist_nearest_centre_km"] = [
            min(geom.distance(p) for p in pts) / 1000 for geom in stations.geometry
        ]

    print(f"  accessibility: CBD distance {out['dist_cbd_km'].min():.1f}"
          f"-{out['dist_cbd_km'].max():.1f} km")
    return out


# --------------------------------------------------------------------------
# Distance to transit - feeder access and network position
# --------------------------------------------------------------------------

def feeder_features(stations: gpd.GeoDataFrame, city: str) -> pd.DataFrame:
    """Bus-stop access, parking and competing rail within reach of each station."""
    crs = stations.crs
    out = pd.DataFrame(index=stations.index)

    bus_path = C.RAW_OSM / f"{city}_bus_stops.gpkg"
    if bus_path.exists():
        stops = gpd.read_file(bus_path).to_crs(crs)
        rings = stations.geometry.buffer(C.FEEDER_RADIUS_M)
        counts = [int(stops.geometry.within(ring).sum()) for ring in rings]
        out["bus_stops_500m"] = counts
        print(f"  feeder: median {np.median(counts):.0f} bus stops within "
              f"{C.FEEDER_RADIUS_M} m")
    else:
        print(f"  feeder: {bus_path.name} missing - run src.acquire --what bus")

    park_path = C.RAW_OSM / f"{city}_parking.gpkg"
    if park_path.exists():
        parking = gpd.read_file(park_path).to_crs(crs)
        rings = stations.geometry.buffer(C.FEEDER_RADIUS_M)
        out["parking_area_m2"] = [
            float(parking.loc[parking.geometry.intersects(ring), "area_m2"].sum())
            for ring in rings
        ]

    rail_path = C.RAW_OSM / f"{city}_stations_raw.gpkg"
    if rail_path.exists():
        raw = gpd.read_file(rail_path).to_crs(crs)
        other_rail = raw[raw["category"] == "other_rail"]
        if len(other_rail):
            out["competing_rail_1km"] = [
                int(other_rail.geometry.distance(geom).lt(C.COMPETING_RAIL_M).sum())
                for geom in stations.geometry
            ]

    return out


def order_stations_along_line(points: gpd.GeoSeries) -> list[int]:
    """Approximate the running order of stations on a corridor.

    Wikipedia lists Phase 2 stations alphabetically, and the OSM alignment is
    incomplete, so the sequence needed for network centrality is reconstructed
    by greedy nearest-neighbour chaining from the most extreme station. Metro
    corridors are close to linear, which is what makes this work; it is
    validated against the known Phase 1 ordering before being trusted.
    """
    if len(points) < 2:
        return list(range(len(points)))

    coords = np.column_stack([points.x.to_numpy(), points.y.to_numpy()])
    # Start from the station furthest along the corridor's principal axis.
    centred = coords - coords.mean(axis=0)
    axis = np.linalg.svd(centred, full_matrices=False)[2][0]
    start = int(np.argmin(centred @ axis))

    order, remaining = [start], set(range(len(coords))) - {start}
    while remaining:
        last = coords[order[-1]]
        nxt = min(remaining, key=lambda i: np.hypot(*(coords[i] - last)))
        order.append(nxt)
        remaining.discard(nxt)
    return order


def _join_lines_at_interchanges(
    graph: nx.Graph,
    stations: gpd.GeoDataFrame,
    line_col: str,
    max_transfer_m: float = 1500,
) -> int:
    """Add transfer edges between corridors so the network is one graph.

    A physical interchange is often a single station node belonging to one
    line - Majestic serves both Purple and Green but appears once - so chaining
    within lines alone leaves the corridors disconnected and betweenness
    measures position along an isolated path rather than in the network.

    For each pair of lines, the mutually closest pair of stations is joined if
    they are within walking/transfer distance. `max_transfer_m` is deliberately
    generous: it is bridging the gap between a shared interchange and its
    neighbour on the other corridor, not asserting a passenger walks that far.
    """
    groups = {line: grp for line, grp in stations.groupby(line_col) if len(grp) >= 2}
    names = list(groups)
    added = 0

    for i, first in enumerate(names):
        for second in names[i + 1:]:
            left, right = groups[first], groups[second]
            best, best_dist = None, np.inf
            for a in left.index:
                distances = right.geometry.distance(stations.geometry[a])
                b = distances.idxmin()
                if distances[b] < best_dist:
                    best, best_dist = (a, b), distances[b]
            if best is not None and best_dist <= max_transfer_m:
                graph.add_edge(*best, weight=max(best_dist, 1.0))
                added += 1

    return added


def network_position_features(
    stations: gpd.GeoDataFrame,
    line_col: str = "Line",
) -> pd.DataFrame:
    """Position of each station within the metro graph.

    Centrality is deliberately computed on the *completed* network - Phase 1
    plus Phase 2 together - because a model trained on today's small system
    would otherwise systematically under-predict a 173 km one. This is the
    single most important correction in the pipeline.
    """
    out = pd.DataFrame(index=stations.index)
    has_lines = line_col in stations.columns and stations[line_col].notna().any()

    graph = nx.Graph()
    if has_lines:
        for line, group in stations.groupby(line_col):
            if pd.isna(line) or len(group) < 2:
                continue
            order = order_stations_along_line(group.geometry)
            idx = group.index.to_numpy()[order]
            for a, b in zip(idx[:-1], idx[1:]):
                dist = stations.geometry[a].distance(stations.geometry[b])
                graph.add_edge(a, b, weight=dist)
    else:
        # Fallback when corridor membership is unknown: a minimum spanning tree
        # over the stations. Metro networks are near-linear, so the MST
        # recovers most of the real topology - but it can bridge two corridors
        # that merely pass close to each other, so prefer real line labels.
        print(f"  network position: no {line_col!r} column - "
              f"approximating topology with a minimum spanning tree")
        complete = nx.Graph()
        coords = np.column_stack([stations.geometry.x, stations.geometry.y])
        for i, a in enumerate(stations.index):
            for j, b in enumerate(stations.index):
                if j <= i:
                    continue
                complete.add_edge(a, b, weight=float(np.hypot(*(coords[i] - coords[j]))))
        graph = nx.minimum_spanning_tree(complete, weight="weight")

    # Connect the lines to each other, or betweenness is computed inside three
    # isolated paths and means nothing.
    #
    # Two cases. Where a corridor is listed once per line (Chennai's Wikipedia
    # table does this), the interchange appears as duplicate names. Where it is
    # a single node serving both lines (Bengaluru's Majestic), no duplicate
    # exists and the lines would stay disconnected - so the nearest pair of
    # stations between each pair of lines is joined instead.
    if has_lines and "name" in stations.columns:
        for _, idxs in stations.groupby("name").groups.items():
            idxs = list(idxs)
            for a, b in zip(idxs[:-1], idxs[1:]):
                graph.add_edge(a, b, weight=1.0)

    if has_lines:
        transfers = _join_lines_at_interchanges(graph, stations, line_col)
        if transfers:
            print(f"  linked {transfers} line pair(s) at their closest stations")

    if graph.number_of_nodes() == 0:
        return out

    components = nx.number_connected_components(graph)
    if components > 1:
        print(f"  WARNING: metro graph has {components} disconnected components; "
              f"betweenness is only comparable within a component")

    betweenness = nx.betweenness_centrality(graph, weight="weight", normalized=True)
    degree = dict(graph.degree())
    closeness = nx.closeness_centrality(graph, distance="weight")

    out["metro_betweenness"] = pd.Series(betweenness).reindex(stations.index)
    out["metro_degree"] = pd.Series(degree).reindex(stations.index).fillna(0)
    out["metro_closeness"] = pd.Series(closeness).reindex(stations.index)
    out["is_interchange"] = (out["metro_degree"] > 2).astype(int)
    out["is_terminal"] = (out["metro_degree"] == 1).astype(int)

    print(f"  network position: {graph.number_of_nodes()} nodes, "
          f"{graph.number_of_edges()} edges, "
          f"{int(out['is_interchange'].sum())} interchanges, "
          f"{int(out['is_terminal'].sum())} terminals")
    return out


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------

# Counts taken straight from OpenStreetMap measure two things at once: how much
# is really there, and how thoroughly local mappers have recorded it. Chennai
# has 2.8x fewer POIs than Bengaluru, 3.0x fewer bus stops and 5.2x less
# parking - far too uniform across categories to be a real difference in urban
# form, and consistent with Bengaluru's much larger OSM contributor community.
#
# Left absolute, a model trained on Bengaluru would read Chennai's sparser
# mapping as sparser land use and under-predict it everywhere. POI count is the
# second-strongest predictor available (r = 0.46), so the error would be large.
#
# Converting to a within-city percentile asks a question mapping effort cannot
# distort: *relative to the other stations in its own city*, how POI-rich is
# this catchment? Rank is preserved, the scale difference disappears.
#
# Only OSM-derived counts are treated this way. Population and built-up volume
# come from modelled global rasters (GHSL), distances are geometric, and PNR,
# circuity and centrality are ratios - none of them depend on how many people
# edited OSM in that city.
NORMALISE_WITHIN_CITY = [
    "poi_total", "poi_education", "poi_healthcare", "poi_retail", "poi_office",
    "poi_government", "poi_transport", "poi_leisure",
    "bus_stops_500m", "parking_area_m2",
]


def normalise_within_city(features: pd.DataFrame,
                          columns: list[str] | None = None) -> pd.DataFrame:
    """Add `<col>_pctl`: each column's percentile rank within this city.

    The raw counts are kept alongside, both for reporting and because the
    scenario layer modifies them directly.
    """
    columns = [c for c in (columns or NORMALISE_WITHIN_CITY) if c in features.columns]
    if not columns:
        return features

    out = features.copy()
    for col in columns:
        out[f"{col}_pctl"] = out[col].rank(pct=True, na_option="keep")

    print(f"\nWithin-city normalisation")
    print(f"  percentile-ranked {len(columns)} OSM count feature(s)")
    sample = [c for c in ("poi_total", "bus_stops_500m") if c in columns]
    for col in sample:
        print(f"    {col}: raw median {out[col].median():,.0f} "
              f"-> pctl median {out[f'{col}_pctl'].median():.2f}")
    return out


def build(city: str, save: bool = True) -> pd.DataFrame:
    """Assemble the full feature table for one city.

    Areal statistics (population, built-up, POIs, street form) are measured on
    the **800 m Euclidean buffer**, not the network service area. The service
    area's shape depends on how densely the street network is noded, so
    measuring density inside it conflates "how many people live here" with "how
    finely the streets are mapped" - which produced a spurious bimodal
    population distribution on the first run.

    Severance is not thrown away: it enters the model as PNR, its own design
    variable, computed from the network catchment. Density and permeability are
    then two separate predictors rather than one confounded measurement.
    """
    stations_path = C.PROCESSED / f"{city}_stations.gpkg"
    buffer_path = C.PROCESSED / f"{city}_catchments_buffer.gpkg"
    network_path = C.PROCESSED / f"{city}_catchments_network.gpkg"

    if not stations_path.exists():
        raise FileNotFoundError(f"{stations_path} missing. Run: python -m src.stations --city {city}")
    for path in (buffer_path, network_path):
        if not path.exists():
            raise FileNotFoundError(f"{path} missing. Run: python -m src.catchment --city {city}")

    stations = gpd.read_file(stations_path).to_crs(C.city_crs(city))
    buffers = gpd.read_file(buffer_path).to_crs(C.city_crs(city))
    network = gpd.read_file(network_path).to_crs(C.city_crs(city))

    catchments = buffers[buffers["radius_m"] == C.CATCHMENT_STANDARD_M].reset_index(drop=True)
    core = buffers[buffers["radius_m"] == C.CATCHMENT_CORE_M].reset_index(drop=True)

    if len(catchments) != len(stations):
        raise ValueError(
            f"{len(catchments)} buffers vs {len(stations)} stations - "
            "the catchment layer is stale; rebuild with src.catchment"
        )

    print(f"[{city}] {len(stations)} stations, "
          f"{C.CATCHMENT_STANDARD_M} m buffers for areal stats, "
          f"network catchments for PNR")

    print("\nDensity")
    density = density_features(catchments)

    # Population within the 400 m core, as a crude distance decay: people who
    # live a 5-minute walk away use the station far more reliably than those at
    # the 10-minute edge, and the core share separates compact catchments from
    # ones whose population sits mostly on the rim.
    core_raster = _find_raster("*POP*.tif", epoch="E2025")
    if core_raster is not None:
        density["population_core400"] = zonal_sum(core, core_raster, "population_core400")
        density["core_share"] = (
            density["population_core400"] / density["population"].replace(0, np.nan)
        )

    print("\nDiversity")
    diversity = landuse_features(catchments)
    print("\nDesign")
    graph = acquire.fetch_street_network(city, "walk")
    design = design_features(catchments, graph)
    # PNR comes from the network service area - the one thing the buffer
    # cannot express - and enters as its own predictor.
    design["pnr"] = network["pnr"].to_numpy() if "pnr" in network.columns else np.nan
    design["network_catchment_km2"] = network["network_area_m2"].to_numpy() / 1e6
    print("\nDestination accessibility")
    pois = poi_features(catchments, city)
    access = accessibility_features(stations, city)
    print("\nDistance to transit")
    feeder = feeder_features(stations, city)
    position = network_position_features(stations)
    
    hubs = C.CITIES[city].get("intermodal_hubs", [])
    if "name" in stations.columns:
        position["is_intermodal_hub"] = stations["name"].isin(hubs).astype(int)
        print(f"  intermodal hubs: {int(position['is_intermodal_hub'].sum())} flagged")
    else:
        position["is_intermodal_hub"] = 0

    keys = [c for c in ("station_id", "name", "phase", "Line", "Layout",
                        "confidence", "position_source",
                        # carries the mid-window-opening exclusion through to
                        # the model; without it every station trains, including
                        # the newly opened line
                        "is_mature") if c in stations.columns]
    target_cols = [c for c in stations.columns if c.startswith(("boardings", "alightings", "throughput"))]

    features = pd.concat(
        [stations[keys + target_cols].reset_index(drop=True),
         density.reset_index(drop=True), diversity.reset_index(drop=True),
         design.reset_index(drop=True), pois.reset_index(drop=True),
         access.reset_index(drop=True), feeder.reset_index(drop=True),
         position.reset_index(drop=True)],
        axis=1,
    )
    features["city"] = city
    features = normalise_within_city(features)

    print(f"\n[{city}] feature table: {features.shape[0]} rows x {features.shape[1]} cols")
    numeric = features.select_dtypes(include=[np.number])
    incomplete = numeric.isna().sum()
    incomplete = incomplete[incomplete > 0]
    if len(incomplete):
        print(f"\ncolumns with missing values:\n{incomplete.to_string()}")

    if save:
        out = C.PROCESSED / f"{city}_features.csv"
        features.to_csv(out, index=False)
        print(f"\n-> {out}")
    return features


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--city", action="append", choices=list(C.CITIES), default=None)
    args = parser.parse_args()

    for city in (args.city or list(C.CITIES)):
        print(f"\n{'=' * 60}\n{C.CITIES[city]['name']}\n{'=' * 60}")
        build(city)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
