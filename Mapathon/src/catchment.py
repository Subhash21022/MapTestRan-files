"""Station catchments: Euclidean buffers, network service areas, and PNR.

Two catchments are built for every station:

  * a plain 800 m buffer - the conventional "10-minute walk" circle, which
    assumes people can walk in straight lines through buildings and rivers;
  * an 800 m *network* service area - everywhere actually reachable along the
    walking network within 800 m of street distance.

The ratio between them is the Pedestrian Network Ratio (PNR). A PNR near 1
means a permeable grid; a low PNR means the catchment is severed by rail lines,
canals, arterial roads or gated layouts, and the nominal catchment population
overstates who can really walk to the station. Chennai has many such stations,
and PNR turns out to be one of the more useful design variables in the model.

    python -m src.catchment --city chennai
"""

from __future__ import annotations

import argparse

import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
import shapely
from shapely.geometry import MultiPoint, Point

from src import acquire
from src import config as C


# --------------------------------------------------------------------------
# Euclidean catchments
# --------------------------------------------------------------------------

def euclidean_catchments(
    stations: gpd.GeoDataFrame,
    radii: tuple[int, ...] = (C.CATCHMENT_CORE_M, C.CATCHMENT_STANDARD_M, C.CATCHMENT_EXTENDED_M),
) -> gpd.GeoDataFrame:
    """Concentric buffers per station, one row per (station, radius).

    Stations must already be in a projected CRS - buffering degrees is
    meaningless and silently wrong.
    """
    if stations.crs is None or stations.crs.is_geographic:
        raise ValueError("stations must be in a projected CRS before buffering")

    frames = []
    for radius in radii:
        ring = stations.copy()
        ring["radius_m"] = radius
        ring["geometry"] = stations.geometry.buffer(radius)
        ring["buffer_area_m2"] = ring.geometry.area
        frames.append(ring)

    out = pd.concat(frames, ignore_index=True)
    return gpd.GeoDataFrame(out, geometry="geometry", crs=stations.crs)


# --------------------------------------------------------------------------
# Network catchments
# --------------------------------------------------------------------------

def _reachable_nodes(graph: nx.Graph, source, radius_m: float) -> list:
    """Node ids within `radius_m` of street distance from `source`."""
    lengths = nx.single_source_dijkstra_path_length(
        graph, source, cutoff=radius_m, weight="length"
    )
    return list(lengths.keys())


_HULL_FALLBACKS = {"count": 0, "degenerate": 0}


def _hull(points: list[tuple[float, float]], ratio: float):
    """Concave hull around reachable nodes, degrading loudly.

    Uses `shapely.concave_hull`, the module-level function. There is no
    `.concave_hull()` *method* on shapely 2.x geometries: calling one raises
    AttributeError, and an earlier bare `except: pass` here swallowed that for
    every station, silently substituting convex hulls throughout. A convex hull
    bridges straight across water, rail corridors and gated land that cannot
    actually be walked, so it inflates the service area and flattens exactly
    the severance signal PNR exists to measure.

    Fallbacks are counted and reported rather than hidden.
    """
    if len(points) < 4:
        _HULL_FALLBACKS["degenerate"] += 1
        return MultiPoint(points).buffer(50) if points else None

    mp = MultiPoint(points)
    hull = shapely.concave_hull(mp, ratio=ratio)
    if hull is not None and not hull.is_empty and hull.geom_type in ("Polygon", "MultiPolygon"):
        return hull

    # Genuinely degenerate input (collinear nodes along a single street).
    _HULL_FALLBACKS["count"] += 1
    convex = mp.convex_hull
    return convex if convex.geom_type in ("Polygon", "MultiPolygon") else convex.buffer(50)


def network_catchments(
    stations: gpd.GeoDataFrame,
    graph: nx.MultiDiGraph,
    radius_m: int = C.CATCHMENT_NETWORK_M,
    ratio: float = C.CONCAVE_HULL_RATIO,
) -> gpd.GeoDataFrame:
    """Walk-network service area per station.

    The graph is treated as undirected: one-way restrictions apply to vehicles,
    not pedestrians, and leaving them in would clip catchments on divided roads.
    """
    if stations.crs is None or stations.crs.is_geographic:
        raise ValueError("stations must be in a projected CRS")

    crs = stations.crs
    graph_proj = ox.project_graph(graph, to_crs=crs)
    undirected = ox.convert.to_undirected(graph_proj)

    xs = stations.geometry.x.to_numpy()
    ys = stations.geometry.y.to_numpy()
    nearest = ox.distance.nearest_nodes(graph_proj, X=xs, Y=ys)
    if np.isscalar(nearest):
        nearest = [nearest]

    coords = {n: (d["x"], d["y"]) for n, d in undirected.nodes(data=True)}

    geoms, n_nodes, snap_dist = [], [], []
    for (_, station), node in zip(stations.iterrows(), nearest):
        reachable = _reachable_nodes(undirected, node, radius_m)
        pts = [coords[n] for n in reachable if n in coords]
        geoms.append(_hull(pts, ratio))
        n_nodes.append(len(pts))
        snap_dist.append(station.geometry.distance(Point(coords[node])))

    out = stations.copy()
    out["geometry"] = geoms
    out["radius_m"] = radius_m
    out["network_area_m2"] = gpd.GeoSeries(geoms, crs=crs).area
    out["n_reachable_nodes"] = n_nodes
    # A large snap distance means the station point sits far from any mapped
    # street - usually a geocoding error or a gap in the walk network.
    out["snap_distance_m"] = snap_dist

    return gpd.GeoDataFrame(out, geometry="geometry", crs=crs)


def pedestrian_network_ratio(
    network: gpd.GeoDataFrame,
    radius_m: int = C.CATCHMENT_NETWORK_M,
) -> pd.Series:
    """PNR = network service area / area of the equivalent circle.

    Values run 0-1 in practice. Schlossberg and Brown report typical urban
    values of 0.3-0.6; below ~0.25 the catchment is badly severed.
    """
    circle_area = np.pi * radius_m ** 2
    return (network["network_area_m2"] / circle_area).clip(upper=1.0)


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------

def build(city: str, stations: gpd.GeoDataFrame | None = None, save: bool = True):
    """Build both catchment families for a city and report PNR diagnostics."""
    crs = C.city_crs(city)

    if stations is None:
        path = C.PROCESSED / f"{city}_stations.gpkg"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} missing - build the station layer first (src.stations)"
            )
        stations = gpd.read_file(path)

    stations = stations.to_crs(crs)
    print(f"[{city}] {len(stations)} stations")

    print(f"[{city}] Euclidean catchments...")
    buffers = euclidean_catchments(stations)

    print(f"[{city}] walk network...")
    graph = acquire.fetch_street_network(city, "walk")

    print(f"[{city}] network service areas (this is the slow part)...")
    network = network_catchments(stations, graph)
    network["pnr"] = pedestrian_network_ratio(network)

    if _HULL_FALLBACKS["count"] or _HULL_FALLBACKS["degenerate"]:
        print(f"\n[{city}] hull fallbacks: "
              f"{_HULL_FALLBACKS['count']} degenerate concave hulls, "
              f"{_HULL_FALLBACKS['degenerate']} catchments with <4 reachable nodes")

    print(f"\n[{city}] PNR summary:")
    print(network["pnr"].describe().to_string())

    severed = network.nsmallest(8, "pnr")
    label = "name" if "name" in severed.columns else network.columns[0]
    print(f"\n[{city}] most severed catchments (lowest PNR):")
    print(severed[[label, "pnr", "n_reachable_nodes"]].to_string(index=False))

    far = network[network["snap_distance_m"] > 200]
    if len(far):
        print(f"\n[{city}] WARNING: {len(far)} station(s) snapped >200 m to the "
              f"walk network - check their coordinates:")
        print(far[[label, "snap_distance_m"]].to_string(index=False))

    if save:
        buf_path = C.PROCESSED / f"{city}_catchments_buffer.gpkg"
        net_path = C.PROCESSED / f"{city}_catchments_network.gpkg"
        buffers.to_file(buf_path, driver="GPKG")
        network.to_file(net_path, driver="GPKG")
        print(f"\n-> {buf_path}\n-> {net_path}")

    return buffers, network


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
