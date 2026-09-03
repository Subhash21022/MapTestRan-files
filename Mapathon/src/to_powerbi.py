"""Export the pipeline outputs as a Power BI-ready star schema.

    python -m src.to_powerbi

Power BI is used the way the 2024 winning entry used it: for the *graphs*.
The map itself is built in QGIS, which is what the geospatial marks are for.
That entry's own tools table listed QGIS, qgis2web, GitHub, Power BI and MS
Excel side by side, so a proprietary charting tool alongside FOSS GIS is
evidently accepted practice rather than a compliance risk.

Layout
------
Power BI wants a star schema: one row per thing in the dimension tables, long
(not wide) fact tables, and a shared key relating them. Everything joins on
`station_id` or `date`.

    dim_station.csv     one row per station, with latitude/longitude
    dim_date.csv        calendar spanning the observed series
    fact_prediction.csv predicted ridership per station
    fact_features.csv   the 5D predictors, long format
    fact_observed.csv   observed daily boardings, long format
    fact_hourly.csv     hourly profile, long format
    fact_ticket_mix.csv fare medium by day, long format
"""

from __future__ import annotations

import geopandas as gpd
import pandas as pd

from src import config as C

OUT = C.OUTPUTS / "powerbi"

PREDICTIONS = C.OUT_TABLES / "chennai_predictions.csv"
RECOMMENDATIONS = C.OUT_TABLES / "chennai_station_recommendations.csv"
STATIONS = C.PROCESSED / "chennai_stations.gpkg"
OBSERVED = C.RAW_RIDERSHIP / "chennai" / "cmrl_stationflow_daily.csv"
HOURLY = C.RAW_RIDERSHIP / "chennai" / "cmrl_hourly.csv"
TICKETS = C.RAW_RIDERSHIP / "chennai" / "cmrl_ticket_mix.csv"

DIM_COLS = ["station_id", "name", "Line", "phase", "Layout", "confidence",
            "position_source"]


def dim_station() -> pd.DataFrame:
    """One row per station, with WGS84 lat/lon.

    Power BI map visuals need latitude and longitude as separate numeric
    columns - they cannot read a GeoPackage or WKT - so the projected geometry
    is converted back to EPSG:4326 here.
    """
    gdf = gpd.read_file(STATIONS).to_crs(C.CRS_GEO)
    keep = [c for c in DIM_COLS if c in gdf.columns]
    out = gdf[keep].copy()
    out["latitude"] = gdf.geometry.y.round(6)
    out["longitude"] = gdf.geometry.x.round(6)
    out = out.rename(columns={"Line": "line", "Layout": "layout"})

    # Readable labels beat codes in slicers and legends.
    out["phase_label"] = out["phase"].map({
        "phase1": "Phase 1 (operational)",
        "phase2": "Phase 2 (under construction)",
    }).fillna(out["phase"])
    out["position_reliable"] = out["confidence"].map({
        "high": "Surveyed",
        "medium": "Snapped to corridor",
        "low": "Approximate",
    }).fillna("Unknown")
    return out


def fact_prediction() -> pd.DataFrame:
    if not PREDICTIONS.exists():
        return pd.DataFrame()
    df = pd.read_csv(PREDICTIONS)
    cols = ["station_id", "name", "phase", "Line", "confidence",
            "prediction", "lower", "upper"]
    out = df[[c for c in cols if c in df.columns]].copy()
    out = out.rename(columns={
        "Line": "line",
        "prediction": "predicted_boardings",
        "lower": "predicted_lower",
        "upper": "predicted_upper",
    })
    if {"predicted_upper", "predicted_lower"} <= set(out.columns):
        out["interval_width"] = out["predicted_upper"] - out["predicted_lower"]
    if "predicted_boardings" in out.columns:
        p = out["predicted_boardings"]
        span = p.max() - p.min()
        out["potential_score"] = ((p - p.min()) / span * 100).round(1) if span else 0
    return out


def fact_features() -> pd.DataFrame:
    """The 5D predictors in long format, tagged by dimension for slicing."""
    source = RECOMMENDATIONS if RECOMMENDATIONS.exists() else PREDICTIONS
    if not source.exists():
        return pd.DataFrame()
    df = pd.read_csv(source)

    dimension = {
        "population": "Density",
        "population_density": "Density",
        "built_volume": "Density",
        "core_share": "Density",
        "landuse_entropy": "Diversity",
        "intersection_density": "Design",
        "street_density": "Design",
        "circuity": "Design",
        "pnr": "Design",
        "poi_total_pctl": "Destination",
        "dist_cbd_km": "Destination",
        "dist_nearest_centre_km": "Destination",
        "bus_stops_500m_pctl": "Distance to transit",
        "metro_betweenness": "Network position",
        "is_interchange": "Network position",
        "is_terminal": "Network position",
    }
    present = [c for c in dimension if c in df.columns]
    if not present:
        return pd.DataFrame()

    ids = [c for c in ("station_id", "name") if c in df.columns]
    long = df.melt(id_vars=ids, value_vars=present,
                   var_name="feature", value_name="value")
    long["dimension"] = long["feature"].map(dimension)
    return long


def fact_observed() -> pd.DataFrame:
    if not OBSERVED.exists():
        return pd.DataFrame()
    df = pd.read_csv(OBSERVED)
    df["date"] = pd.to_datetime(df["date"])
    df = df[~df["is_partial"].astype(bool)].copy()
    df["day_name"] = df["date"].dt.day_name()
    df["day_type"] = df["date"].dt.dayofweek.map(
        lambda d: "Weekend" if d >= 5 else "Weekday")
    df = df.rename(columns={"cmrl_name": "station_cmrl"})
    return df[["date", "day_name", "day_type", "line", "station_cmrl", "boardings"]]


def fact_hourly() -> pd.DataFrame:
    if not HOURLY.exists():
        return pd.DataFrame()
    df = pd.read_csv(HOURLY)
    df["date"] = pd.to_datetime(df["date"])
    df = df[~df["is_partial"].astype(bool)].copy()
    df["time_label"] = df["hour"].map(lambda h: f"{int(h):02d}:00")
    total = df.groupby("date")["boardings"].transform("sum")
    df["share_of_day"] = (df["boardings"] / total).round(4)
    return df[["date", "hour", "time_label", "boardings", "share_of_day"]]


def fact_ticket_mix() -> pd.DataFrame:
    """One row per date per fare medium."""
    if not TICKETS.exists():
        return pd.DataFrame()
    df = pd.read_csv(TICKETS)
    df["date"] = pd.to_datetime(df["date"])
    df = df[~df["is_partial"].astype(bool)].copy()
    value_cols = [c for c in df.columns
                  if df[c].dtype.kind in "if" and c != "is_partial"
                  and df[c].sum() > 0]
    if not value_cols:
        return pd.DataFrame()
    long = df.melt(id_vars="date", value_vars=value_cols,
                   var_name="medium", value_name="count")
    long["medium"] = (long["medium"]
                      .str.replace("^noOf", "", regex=True)
                      .str.replace("([a-z])([A-Z])", r"\1 \2", regex=True))
    return long[long["count"] > 0]


def dim_date(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Continuous calendar over the observed range, so time slicers behave."""
    parts = [f["date"] for f in frames if not f.empty and "date" in f.columns]
    if not parts:
        return pd.DataFrame()
    dates = pd.concat(parts, ignore_index=True)
    out = pd.DataFrame({"date": pd.date_range(dates.min(), dates.max(), freq="D")})
    out["year"] = out["date"].dt.year
    out["month"] = out["date"].dt.month_name()
    out["day_name"] = out["date"].dt.day_name()
    out["day_type"] = out["date"].dt.dayofweek.map(
        lambda d: "Weekend" if d >= 5 else "Weekday")
    out["iso_week"] = out["date"].dt.isocalendar().week.astype(int)
    return out


def build(save: bool = True) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)

    observed, hourly, tickets = fact_observed(), fact_hourly(), fact_ticket_mix()
    tables = {
        "dim_station": dim_station(),
        "dim_date": dim_date([observed, hourly, tickets]),
        "fact_prediction": fact_prediction(),
        "fact_features": fact_features(),
        "fact_observed": observed,
        "fact_hourly": hourly,
        "fact_ticket_mix": tickets,
    }

    if save:
        for name, frame in tables.items():
            if frame.empty:
                print(f"  {name:<20} (empty - skipped)")
                continue
            # UTF-8 BOM: Power BI and Excel both mis-read plain UTF-8 on
            # Windows, which mangles the Tamil station names.
            frame.to_csv(OUT / f"{name}.csv", index=False, encoding="utf-8-sig")
            print(f"  {name:<20} {frame.shape[0]:>5} rows x {frame.shape[1]:>2} cols")
        _write_readme(tables)
        print(f"\n-> {OUT}")
    return tables


def _write_readme(tables: dict) -> None:
    lines = [
        "# Power BI import guide",
        "",
        "Generated by `python -m src.to_powerbi`. Re-run after any model run.",
        "",
        "Power BI's role is the **graphs**; the map is built in QGIS. That split",
        "follows the 2024 winning entry, whose tools table listed QGIS, qgis2web,",
        "GitHub, Power BI and MS Excel together.",
        "",
        "## Import",
        "",
        "1. Get Data > Text/CSV, select every `*.csv` in this folder.",
        "2. Encoding **65001: Unicode (UTF-8)** if prompted.",
        "3. Model view, create these relationships:",
        "",
        "| From | To | Cardinality |",
        "|---|---|---|",
        "| fact_prediction[station_id] | dim_station[station_id] | many-to-one |",
        "| fact_features[station_id] | dim_station[station_id] | many-to-one |",
        "| fact_observed[date] | dim_date[date] | many-to-one |",
        "| fact_hourly[date] | dim_date[date] | many-to-one |",
        "| fact_ticket_mix[date] | dim_date[date] | many-to-one |",
        "",
        "4. Table tools > Mark as date table, on `dim_date`.",
        "5. On `dim_station`, set Data category: `latitude` -> Latitude and",
        "   `longitude` -> Longitude. Without this the map visual will not plot.",
        "",
        "## Charts worth exporting to the poster",
        "",
        "- **Bar, top 15 by `potential_score`**, with `predicted_lower` /",
        "  `predicted_upper` as error bars. Showing uncertainty is unusual and",
        "  reads as rigour.",
        "- **Line, `fact_hourly` by `time_label`** - the 08:00 peak carrying",
        "  11.9% of the day, with the evening secondary peak.",
        "- **Clustered bar by `day_type`** - weekday ~377k against weekend ~193k.",
        "- **Stacked area, `fact_ticket_mix` over `date`** - the fare-media shift.",
        "- **Matrix of `fact_features`** with `dimension` on rows, comparing the",
        "  5D profile between a strong and a weak station.",
        "",
        "Export each as PNG and place it in the QGIS print layout, the way the",
        "winning entry did.",
        "",
        "## Health warning on fact_prediction",
        "",
        "Station-level validation against observed Phase 1 ridership currently",
        "gives R2 = -0.09 and Spearman = 0.11: the model reproduces the system",
        "total (ratio 1.06) but cannot yet rank individual stations. Present",
        "these as work in progress, not forecasts.",
        "",
        "## Tables",
        "",
    ]
    for name, frame in tables.items():
        if frame.empty:
            continue
        lines.append(f"**{name}** - {frame.shape[0]} rows")
        lines.append(f"`{'`, `'.join(map(str, frame.columns))}`")
        lines.append("")
    (OUT / "README.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    build()
