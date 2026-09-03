"""Station-wise daily boardings from CMRL's public passenger-flow API.

This is the real thing: per-station, per-day ridership for Chennai Phase 1,
published by CMRL themselves. It replaces the Bengaluru transfer as the
training target, which removes every cross-city assumption from the model.

Source
------
The dashboard at https://commuters-data.chennaimetrorail.org/passengerflow is
an Angular app; the numbers come from a public JSON API behind it:

    GET /api/PassengerFlow/stationData/{daysBack}

`daysBack` is an offset from today - 0 is today (partial, and therefore
excluded), 1 is yesterday, and so on. The response carries one object per line,
each with a `series` list whose "Total" entry holds one value per station, in
platform order along the line. Station *names* are not in the response, so the
orderings below were read from the chart configuration in the page and are
applied positionally.

The values are boardings, not footfall: summing every station for a given day
reproduces the dashboard's headline "Total Passengers" exactly (63,818 + 38,139
= 101,957 on 17 Aug 2026), which is only true if each journey is counted once,
at entry.

    python -m src.cmrl_stationflow --days 28
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import date, timedelta

import pandas as pd
import requests

from src import config as C
from src.stations import MANUAL_ALIASES, match_names

API = "https://commuters-dataapi.chennaimetrorail.org/api/PassengerFlow"
HEADERS = {
    "User-Agent": "IITB-FOSSEE-Mapathon-2026 chennai-metro-ridership (student research)",
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://commuters-data.chennaimetrorail.org/passengerflow",
}

# Platform order along each line, taken from the dashboard's chart categories.
# The API returns values positionally with no labels, so these lists and the
# returned arrays must stay the same length - `fetch_day` asserts that.
LINE1_STATIONS = [
    "WIMCO NAGAR DEPOT", "WIMCO NAGAR METRO", "THIRUVOTRIYUR METRO",
    "THIRUVOTRIYUR THERADI METRO", "KALADIPET METRO", "TOLLGATE METRO",
    "NEW WASHERMENPET METRO", "TONDIARPET METRO", "THIYAGARAYA COLLEGE METRO",
    "WASHERMANPET", "MANNADI", "HIGH COURT", "CENTRAL METRO",
    "GOVERNMENT ESTATE", "LIC", "THOUSAND LIGHT", "AG-DMS", "TEYNAMPET",
    "NANDANAM", "SAIDAPET", "LITTLE MOUNT", "GUINDY", "ALANDUR",
    "OTA - NANGANALLUR ROAD", "MEENAMBAKKAM", "CHENNAI AIRPORT",
]

LINE2_STATIONS = [
    "CENTRAL METRO", "EGMORE", "NEHRU PARK", "KILPAUK", "PACHAIAPPA S COLLEGE",
    "SHENOY NAGAR", "ANNA NAGAR EAST", "ANNA NAGAR TOWER", "THIRUMANGALAM",
    "KOYAMBEDU", "CMBT", "ARUMBAKKAM", "VADAPALANI", "ASHOK NAGAR",
    "EKKATTUTHANGAL", "ALANDUR", "St. THOMAS MOUNT",
]

# CMRL's operational shorthand against the canonical Wikipedia names used
# throughout the pipeline. Only the pairs fuzzy matching cannot bridge.
CMRL_ALIASES = {
    "THIYAGARAYA COLLEGE METRO": "Sir Theagaraya College",
    "OTA - NANGANALLUR ROAD": "Nanganallur Road",
    "CHENNAI AIRPORT": "Chennai International Airport",
    "CMBT": "Puratchi Thalaivi Dr. J. Jayalalithaa CMBT",
    "CENTRAL METRO": "Puratchi Thalaivar Dr. M.G. Ramachandran Central",
    "ALANDUR": "Arignar Anna Alandur",
    "THOUSAND LIGHT": "Thousand Lights",
    "PACHAIAPPA S COLLEGE": "Pachaiyappa's College",
    "THIRUVOTRIYUR METRO": "Tiruvottriyur",
    "THIRUVOTRIYUR THERADI METRO": "Tiruvottriyur Theradi",
    "NEW WASHERMENPET METRO": "New Washermanpet",
    "AG-DMS": "AG – DMS",
}

OUT = C.PROCESSED / "chennai_station_target.csv"
RAW_DIR = C.RAW_RIDERSHIP / "chennai"


def fetch_day(days_back: int, session: requests.Session | None = None) -> pd.DataFrame:
    """Station totals for one day. Empty frame if the API has no data."""
    get = (session or requests).get
    resp = get(f"{API}/stationData/{days_back}", headers=HEADERS, timeout=60)
    resp.raise_for_status()
    payload = resp.json()
    if not payload:
        return pd.DataFrame()

    day = date.today() - timedelta(days=days_back)
    rows = []
    for block in payload:
        line = str(block.get("line", "")).strip()
        names = LINE1_STATIONS if line == "01" else LINE2_STATIONS
        total = next((s for s in block.get("series", []) if s.get("name") == "Total"), None)
        if total is None:
            continue
        values = total.get("data", [])
        if len(values) != len(names):
            # Fail loudly: a changed station list silently shifts every value
            # onto the wrong station, which is far worse than no data.
            raise ValueError(
                f"line {line} returned {len(values)} values but {len(names)} station "
                f"names are configured. The network has changed - update "
                f"LINE1_STATIONS / LINE2_STATIONS from the dashboard."
            )
        for name, value in zip(names, values):
            rows.append({"date": day, "line": line.zfill(2), "cmrl_name": name,
                         "boardings": value})
    return pd.DataFrame(rows)


STORE = RAW_DIR / "cmrl_stationflow_daily.csv"


def fetch(pause: float = 0.4) -> pd.DataFrame:
    """Fetch what the API actually offers and append it to the local store.

    **This endpoint is not a history API.** Despite looking like a day offset,
    `daysBack` has only two states: 0 returns the current (partial) day, and
    every value from 1 to 365 returns the *same* single previous day - verified
    by hashing the payloads. An earlier version of this function looped
    1..28 and silently collected 28 identical copies of one day.

    So each run can only ever add one complete day. Results are appended to a
    dated store and de-duplicated, which means running this daily accumulates a
    real series over time. Run it as a scheduled task to build one up.
    """
    session = requests.Session()
    frames = []
    print("fetching from the CMRL passenger-flow API (offsets 0 and 1 only)")

    for offset, label in ((0, "today, partial"), (1, "previous complete day")):
        try:
            frame = fetch_day(offset, session)
        except Exception as exc:  # noqa: BLE001
            print(f"  offset {offset}: {type(exc).__name__} - skipped")
            continue
        if frame.empty:
            print(f"  offset {offset}: empty")
            continue
        frame["is_partial"] = offset == 0
        day = frame["date"].iloc[0]
        total = frame["boardings"].sum()
        print(f"  offset {offset} ({label}): {day} = {total:,} boardings")
        frames.append(frame)
        time.sleep(pause)

    if not frames:
        raise RuntimeError("no data returned by the API")

    fresh = pd.concat(frames, ignore_index=True)

    # Merge with anything collected on previous runs.
    if STORE.exists():
        prior = pd.read_csv(STORE)
        # Normalise the key columns on both sides before de-duplicating.
        # `line` is written as "01"/"02" but read back as integers 1/2, so the
        # composite key never matched and every re-run appended a second copy
        # of the same day - doubling every station's boardings.
        for frame in (prior, fresh):
            frame["date"] = pd.to_datetime(frame["date"]).dt.date
            frame["line"] = frame["line"].astype(str).str.zfill(2)
        combined = pd.concat([prior, fresh], ignore_index=True)
        # A day previously stored as partial must be replaced by its complete
        # version once that becomes available, so keep the last occurrence.
        combined = combined.sort_values("is_partial", ascending=False)
        combined = combined.drop_duplicates(subset=["date", "line", "cmrl_name"],
                                            keep="last")
    else:
        combined = fresh

    combined = combined.sort_values(["date", "line", "cmrl_name"]).reset_index(drop=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_csv(STORE, index=False)

    complete = combined[~combined["is_partial"].astype(bool)]
    days = sorted(complete["date"].unique())
    print(f"\nstore now holds {len(days)} complete day(s): "
          f"{days[0] if days else '-'} to {days[-1] if days else '-'}")

    if len(days) < 5:
        print("  NOTE: too few days for a stable weekday average. Re-run daily "
              "to accumulate; the API exposes only one past day at a time.")
    return combined


def to_station_target(raw: pd.DataFrame, weekdays_only: bool = True) -> pd.DataFrame:
    """Collapse daily records into one mean-weekday-boardings row per station."""
    work = raw.copy()
    if "is_partial" in work.columns:
        work = work[~work["is_partial"].astype(bool)]
    work["date"] = pd.to_datetime(work["date"])
    work["dow"] = work["date"].dt.dayofweek
    work["is_weekday"] = work["dow"] < 5

    n_weekday = work[work["is_weekday"]]["date"].nunique()
    n_weekend = work[~work["is_weekday"]]["date"].nunique()
    print(f"\n{n_weekday} weekday(s) and {n_weekend} weekend day(s) in the store")
    if weekdays_only and n_weekday == 0:
        print("  WARNING: no weekday data yet - falling back to whatever days exist.")
        print("  Weekend ridership is structurally different (leisure rather than")
        print("  commuting), so a target built from it will misrepresent the")
        print("  land-use relationship the model is trying to learn.")

    # Central and Alandur are interchanges and appear on both lines. Their
    # gate entries are reported separately per line; the physical station's
    # boardings are the sum. (Summing every row for a day reproduces the
    # dashboard's headline total, so no double counting is introduced.)
    per_day = (
        work.groupby(["date", "cmrl_name", "is_weekday"], as_index=False)["boardings"]
        .sum()
    )

    subset = per_day[per_day["is_weekday"]] if weekdays_only else per_day
    if subset.empty:
        subset = per_day

    agg = (
        subset.groupby("cmrl_name", as_index=False)
        .agg(boardings_weekday=("boardings", "mean"),
             days_observed=("date", "nunique"),
             boardings_sd=("boardings", "std"))
    )

    weekend = per_day[~per_day["is_weekday"]]
    if not weekend.empty:
        wk = weekend.groupby("cmrl_name", as_index=False)["boardings"].mean()
        wk = wk.rename(columns={"boardings": "boardings_weekend"})
        agg = agg.merge(wk, on="cmrl_name", how="left")
        agg["weekend_ratio"] = agg["boardings_weekend"] / agg["boardings_weekday"]

    return agg.sort_values("boardings_weekday", ascending=False).reset_index(drop=True)


def match_to_canonical(agg: pd.DataFrame) -> pd.DataFrame:
    """Attach the canonical station names the rest of the pipeline uses."""
    import geopandas as gpd

    path = C.PROCESSED / "chennai_stations.gpkg"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing. Run: python -m src.stations --city chennai"
        )
    stations = gpd.read_file(path)
    canonical = stations[stations["phase"] == "phase1"]["name"].dropna().tolist()

    # Explicit aliases first; fuzzy matching only for the remainder.
    resolved, remaining = {}, []
    for name in agg["cmrl_name"]:
        target = CMRL_ALIASES.get(name)
        if target and target in canonical:
            resolved[name] = target
        else:
            remaining.append(name)

    still_free = [c for c in canonical if c not in resolved.values()]
    fuzzy, unmatched = match_names(remaining, still_free, threshold=0.72,
                                   aliases=MANUAL_ALIASES)
    resolved.update(fuzzy)

    agg = agg.copy()
    agg["station"] = agg["cmrl_name"].map(resolved)

    print(f"\nmatched {agg['station'].notna().sum()}/{len(agg)} CMRL stations "
          f"to the canonical list ({len(canonical)} Phase 1 stations)")
    if unmatched:
        print(f"  unmatched CMRL names: {unmatched}")
    missing = sorted(set(canonical) - set(agg["station"].dropna()))
    if missing:
        print(f"  canonical stations with no CMRL data: {missing}")
    return agg


def build(save: bool = True) -> pd.DataFrame:
    raw = fetch()
    agg = to_station_target(raw)
    agg = match_to_canonical(agg)

    out = agg[agg["station"].notna()].copy()
    out["city"] = "chennai"
    out["is_mature"] = True
    out["target_kind"] = "daily_boardings"

    total = out["boardings_weekday"].sum()
    print(f"\n{len(out)} stations with a target")
    print(f"system-wide mean weekday boardings: {total:,.0f}")
    print(f"CMRL published system average:      {C.CHENNAI_OBSERVED_DAILY_MEAN:,}")
    print(f"ratio: {total / C.CHENNAI_OBSERVED_DAILY_MEAN:.2f}")

    print("\ntop 10:")
    cols = ["station", "boardings_weekday", "days_observed"]
    print(out.head(10)[cols].round(0).to_string(index=False))
    print("\nbottom 5:")
    print(out.tail(5)[cols].round(0).to_string(index=False))

    if save:
        out.to_csv(OUT, index=False)
        (RAW_DIR / "provenance.json").write_text(json.dumps({
            "source": "CMRL public passenger-flow API",
            "dashboard": "https://commuters-data.chennaimetrorail.org/passengerflow",
            "endpoint": f"{API}/stationData/{{daysBack}}",
            "measure": "boardings (entries); station sums reproduce the published daily total",
            "days_stored": int(raw["date"].nunique()),
            "api_limitation": (
                "stationData/{n} exposes only two states: n=0 is the current "
                "partial day, and every n>=1 returns the same single previous "
                "day. Run daily to accumulate a series."
            ),
            "date_range": [str(raw["date"].min()), str(raw["date"].max())],
        }, indent=2), encoding="utf-8")
        print(f"\n-> {OUT}")
        print("Now retrain on Chennai:")
        print("    python -m src.model --train-city chennai")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    build()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
