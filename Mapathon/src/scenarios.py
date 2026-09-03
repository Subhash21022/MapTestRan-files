"""The prescriptive layer - turning predictions into planning decisions.

A ranked list of predicted ridership is descriptive. What a planner can act on
is: which stations are being held back by something fixable, which need land-use
change rather than more buses, which should open first, and where the network
misses people entirely. Six outputs, each tied to a decision:

  1. quadrant_analysis   - potential vs current access -> four action types
  2. simulate            - re-run the model under a changed input
  3. station_typology    - k-means archetypes with standard prescriptions
  4. coverage_gaps       - dense areas outside every catchment
  5. priority_phasing    - predicted ridership per crore of capex
  6. severed_catchments  - low-PNR stations needing pedestrian repair

    python -m src.scenarios
"""

from __future__ import annotations

import argparse

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src import config as C

ACCESS_COLUMNS = ("bus_stops_500m", "pnr", "intersection_density")


# --------------------------------------------------------------------------
# 1. Quadrant gap analysis
# --------------------------------------------------------------------------

def access_index(df: pd.DataFrame, columns: tuple[str, ...] = ACCESS_COLUMNS) -> pd.Series:
    """Composite last-mile access score, 0-1.

    Percentile-ranked before averaging so that variables on wildly different
    scales (bus stops in the tens, PNR in the 0-1 range) contribute equally.
    """
    present = [c for c in columns if c in df.columns and df[c].notna().any()]
    if not present:
        raise ValueError(f"none of {columns} present for the access index")
    return df[present].rank(pct=True).mean(axis=1)


def quadrant_analysis(df: pd.DataFrame, prediction_col: str = "prediction") -> pd.DataFrame:
    """Classify stations into four actionable quadrants.

    Split at the median of each axis: the question is not "is this station good"
    but "relative to the rest of this network, what is holding it back".
    """
    out = df.copy()
    out["access_index"] = access_index(out)
    out["potential_index"] = out[prediction_col].rank(pct=True)

    high_potential = out["potential_index"] >= out["potential_index"].median()
    high_access = out["access_index"] >= out["access_index"].median()

    out["quadrant"] = np.select(
        [high_potential & high_access,
         high_potential & ~high_access,
         ~high_potential & high_access],
        ["ready_winner", "latent_potential", "underbuilt"],
        default="low_potential",
    )

    out["action"] = out["quadrant"].map({
        "ready_winner": "Open early; demand is there and reachable",
        "latent_potential": "Priority feeder bus + footpath investment",
        "underbuilt": "TOD upzoning candidate; access exists, density does not",
        "low_potential": "Right-size the station; defer ancillary spend",
    })

    print("quadrants:")
    print(out["quadrant"].value_counts().to_string())
    return out


# --------------------------------------------------------------------------
# 2. Scenario simulation
# --------------------------------------------------------------------------

def simulate(
    model,
    X: pd.DataFrame,
    changes: dict[str, float | tuple[str, float]],
    label: str = "scenario",
) -> pd.Series:
    """Re-predict under modified inputs.

    `changes` maps a feature to either a multiplier, or an ("add", value) /
    ("set", value) tuple. Everything else is held constant, so the difference
    is attributable to the intervention - within the limits of a cross-sectional
    model, which captures association rather than proven causation. State that
    caveat wherever the scenario numbers are presented.
    """
    modified = X.copy()
    for column, change in changes.items():
        if column not in modified.columns:
            print(f"  {label}: {column!r} not in the feature set - skipped")
            continue
        if isinstance(change, tuple):
            op, value = change
            if op == "add":
                modified[column] = modified[column] + value
            elif op == "set":
                modified[column] = value
            else:
                raise ValueError(f"unknown op {op!r}")
        else:
            modified[column] = modified[column] * change

    return pd.Series(np.expm1(model.predict(modified)), index=X.index, name=label).clip(lower=0)


def scenario_table(model, X: pd.DataFrame, baseline: pd.Series) -> pd.DataFrame:
    """Standard scenario set: feeder buses, upzoning, population growth."""
    scenarios = {
        "feeder_bus_plus_20pct": {"bus_stops_500m": 1.2},
        "feeder_bus_plus_50pct": {"bus_stops_500m": 1.5},
        "upzoning_mixed_use": {"landuse_entropy": 1.25, "poi_office": 1.3},
        "population_2030": {"population": 1.15, "population_density": 1.15},
        "pedestrian_repair": {"pnr": ("set", 0.65), "intersection_density": 1.15},
    }

    out = pd.DataFrame({"baseline": baseline})
    for label, changes in scenarios.items():
        applicable = {k: v for k, v in changes.items() if k in X.columns}
        if not applicable:
            print(f"  {label}: no applicable features - skipped")
            continue
        predicted = simulate(model, X, applicable, label)
        out[label] = predicted
        out[f"{label}_delta_pct"] = 100 * (predicted - baseline) / baseline.replace(0, np.nan)

    delta_cols = [c for c in out.columns if c.endswith("_delta_pct")]
    if delta_cols:
        print("\nmean ridership change by scenario:")
        print(out[delta_cols].mean().round(1).to_string())
    return out


# --------------------------------------------------------------------------
# 3. Station typology
# --------------------------------------------------------------------------

TYPOLOGY_HINTS = {
    "high_density_mixed": "CBD/urban core - prioritise pedestrian capacity and crowd management",
    "employment_hub": "Job centre - peak-direction capacity, office shuttle integration",
    "residential_feeder": "Origin station - feeder buses and two-wheeler parking",
    "interchange": "Transfer node - concourse sizing and wayfinding",
    "peripheral_terminal": "Park-and-ride - car/two-wheeler parking, intercity bus links",
}


def station_typology(
    df: pd.DataFrame,
    features: list[str] | None = None,
    n_clusters: int = C.N_TYPOLOGIES,
) -> pd.DataFrame:
    """Cluster stations into planning archetypes.

    Reduces 105 individual stations to a handful of classes that share a design
    prescription, which is how station-area planning is actually resourced.
    Cluster labels are assigned by inspecting the profile table printed below -
    k-means has no idea what a CBD is.
    """
    if features is None:
        preferred = ["population_density", "landuse_entropy", "poi_office", "poi_retail",
                     "intersection_density", "dist_cbd_km", "pnr", "metro_degree"]
        features = [c for c in preferred if c in df.columns]

    if len(features) < 3:
        raise ValueError(f"need at least 3 features for typology, have {features}")

    work = df[features].replace([np.inf, -np.inf], np.nan)
    work = work.fillna(work.median())
    scaled = StandardScaler().fit_transform(work)

    kmeans = KMeans(n_clusters=n_clusters, random_state=C.RANDOM_SEED, n_init=10)
    out = df.copy()
    out["typology_id"] = kmeans.fit_predict(scaled)

    profile = out.groupby("typology_id")[features].median()
    profile["n_stations"] = out.groupby("typology_id").size()
    print("\ntypology profiles (median values - use these to name the clusters):")
    print(profile.round(2).to_string())

    return out


# --------------------------------------------------------------------------
# 4. Coverage gaps
# --------------------------------------------------------------------------

def coverage_gaps(
    catchments: gpd.GeoDataFrame,
    population_raster,
    min_density: float = 10000,
) -> gpd.GeoDataFrame:
    """Dense populated areas falling outside every station catchment.

    These are the infill-station and feeder-route candidates - the part of the
    map that shows what the network misses, not just what it serves.
    """
    import rasterio
    from rasterio.features import shapes

    served = catchments.geometry.union_all()

    with rasterio.open(population_raster) as src:
        data = src.read(1)
        transform = src.transform
        crs = src.crs
        cell_km2 = abs(transform.a * transform.e) / 1e6
        dense = (data / cell_km2) >= min_density

        polygons = [
            {"geometry": geom, "properties": {"value": val}}
            for geom, val in shapes(dense.astype(np.uint8), mask=dense, transform=transform)
        ]

    if not polygons:
        print("no areas above the density threshold")
        return gpd.GeoDataFrame(geometry=[], crs=catchments.crs)

    gdf = gpd.GeoDataFrame.from_features(polygons, crs=crs).to_crs(catchments.crs)
    gdf["geometry"] = gdf.geometry.difference(served)
    gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notna()]
    gdf["area_km2"] = gdf.geometry.area / 1e6
    gdf = gdf[gdf["area_km2"] > 0.05].sort_values("area_km2", ascending=False)

    print(f"{len(gdf)} unserved dense pockets, {gdf['area_km2'].sum():.1f} km2 total")
    return gdf


# --------------------------------------------------------------------------
# 5. Priority phasing
# --------------------------------------------------------------------------

def priority_phasing(
    df: pd.DataFrame,
    prediction_col: str = "prediction",
    corridor_col: str = "Line",
) -> pd.DataFrame:
    """Rank corridors by predicted ridership per crore of capital cost.

    Cost is apportioned across corridors by route length from the sanctioned
    Phase 2 total, and underground stations are weighted more heavily since
    they dominate civil cost. This is an order-of-magnitude prioritisation, not
    a substitute for the DPR's financial appraisal.
    """
    out = df.copy()
    if corridor_col not in out.columns:
        print(f"no {corridor_col!r} column - skipping phasing")
        return out

    total_km = sum(v["km"] for v in C.PHASE2_CORRIDORS.values())
    cost_per_km = C.CHENNAI_PHASE2_COST_CR / total_km

    # Underground construction runs roughly 2.5x elevated per km.
    weight = out.get("Layout", pd.Series(index=out.index, dtype=object))
    out["cost_weight"] = np.where(weight.astype(str).str.lower().eq("underground"), 2.5, 1.0)

    by_corridor = out.groupby(corridor_col).agg(
        stations=("cost_weight", "size"),
        predicted_daily=(prediction_col, "sum"),
        cost_weight=("cost_weight", "sum"),
    )
    by_corridor["cost_share_cr"] = (
        C.CHENNAI_PHASE2_COST_CR * by_corridor["cost_weight"] / by_corridor["cost_weight"].sum()
    )
    by_corridor["riders_per_crore"] = (
        by_corridor["predicted_daily"] / by_corridor["cost_share_cr"]
    )
    by_corridor = by_corridor.sort_values("riders_per_crore", ascending=False)
    by_corridor["open_priority"] = range(1, len(by_corridor) + 1)

    print(f"\ncorridor priority (cost basis: Rs {C.CHENNAI_PHASE2_COST_CR:,} cr / "
          f"{total_km:.1f} km = Rs {cost_per_km:.0f} cr/km):")
    print(by_corridor.round(1).to_string())
    return by_corridor


# --------------------------------------------------------------------------
# 6. Severed catchments
# --------------------------------------------------------------------------

def severed_catchments(df: pd.DataFrame, pnr_threshold: float = 0.30) -> pd.DataFrame:
    """Stations whose walkable catchment is badly cut off.

    Low PNR means the 800 m circle drawn on the plan is not the 800 m people
    can actually walk. These are the cheapest interventions on the whole list -
    a footbridge or a missing link costs a rounding error against a metro
    station, and the model says what it would return.
    """
    if "pnr" not in df.columns:
        print("no PNR column - run src.catchment first")
        return pd.DataFrame()

    severed = df[df["pnr"] < pnr_threshold].copy()
    severed["walkable_area_km2"] = severed["pnr"] * np.pi * C.CATCHMENT_NETWORK_M ** 2 / 1e6
    severed["lost_area_km2"] = (
        (1 - severed["pnr"]) * np.pi * C.CATCHMENT_NETWORK_M ** 2 / 1e6
    )
    severed = severed.sort_values("pnr")

    cols = [c for c in ("name", "Line", "pnr", "lost_area_km2", "prediction") if c in severed.columns]
    print(f"\n{len(severed)} stations with PNR < {pnr_threshold} "
          f"(catchment severed by rail, water or arterials):")
    print(severed[cols].head(15).round(3).to_string(index=False))
    return severed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", default=str(C.OUT_TABLES / "chennai_predictions.csv"))
    args = parser.parse_args()

    path = C.OUT_TABLES / "chennai_predictions.csv" if args.predictions is None else args.predictions
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        print(f"{path} missing. Run the model first: python -m src.model")
        return 1

    df = quadrant_analysis(df)
    df = station_typology(df)
    priority_phasing(df)
    severed_catchments(df)

    out = C.OUT_TABLES / "chennai_station_recommendations.csv"
    df.to_csv(out, index=False)
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
