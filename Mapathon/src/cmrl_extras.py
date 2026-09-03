"""The other two CMRL passenger-flow datasets: hourly profile and ticket mix.

The dashboard at commuters-data.chennaimetrorail.org is backed by three
endpoints. `stationData` is the modelling target and lives in
src/cmrl_stationflow.py; the other two are captured here:

    hourlybaseddata/{n}   system-wide boardings by hour of day
    allTicketCount/{n}    system-wide boardings by ticket medium

Neither is station-level, so neither can train the model. They earn their place
elsewhere: the hourly profile gives peak-hour share, which is what actually
sizes a metro (PHPDT drives train frequency and platform capacity), and the
ticket mix documents a striking three-year transition worth a panel on the map.

Same limitation as stationData: offset 0 is today (partial) and every offset
>= 1 returns the same single previous day, so this accumulates by running daily.

    python -m src.cmrl_extras
"""

from __future__ import annotations

import json
from datetime import date, timedelta

import pandas as pd
import requests

from src import config as C
from src.cmrl_stationflow import API, HEADERS, RAW_DIR

HOURLY_STORE = RAW_DIR / "cmrl_hourly.csv"
TICKETS_STORE = RAW_DIR / "cmrl_ticket_mix.csv"


def _day(offset: int) -> date:
    return date.today() - timedelta(days=offset)


def fetch_hourly(offset: int, session=None) -> pd.DataFrame:
    """System-wide boardings by hour for one day."""
    get = (session or requests).get
    resp = get(f"{API}/hourlybaseddata/{offset}", headers=HEADERS, timeout=60)
    resp.raise_for_status()
    payload = resp.json()
    if not payload:
        return pd.DataFrame()

    series = payload.get("series", []) if isinstance(payload, dict) else []
    total = next((s for s in series if s.get("name") == "Total"), None)
    if not total:
        return pd.DataFrame()

    values = total.get("data", [])
    # The chart starts at 00:00 and steps hourly; only elapsed hours are present
    # on a partial day, so the hour is the index rather than a fixed 0-23.
    return pd.DataFrame({
        "date": _day(offset),
        "hour": range(len(values)),
        "boardings": values,
        "is_partial": offset == 0,
    })


def fetch_ticket_mix(offset: int, session=None) -> pd.DataFrame:
    """System-wide boardings by ticket medium for one day."""
    get = (session or requests).get
    resp = get(f"{API}/allTicketCount/{offset}", headers=HEADERS, timeout=60)
    resp.raise_for_status()
    payload = resp.json()
    if not payload:
        return pd.DataFrame()

    row = {k: v for k, v in payload.items() if isinstance(v, (int, float))}
    row["date"] = _day(offset)
    row["is_partial"] = offset == 0
    return pd.DataFrame([row])


def _append(store, fresh: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Merge into the on-disk store, replacing partial days with complete ones."""
    if fresh.empty:
        return fresh
    fresh = fresh.copy()
    fresh["date"] = pd.to_datetime(fresh["date"]).dt.date

    if store.exists():
        prior = pd.read_csv(store)
        prior["date"] = pd.to_datetime(prior["date"]).dt.date
        combined = pd.concat([prior, fresh], ignore_index=True)
        combined = combined.sort_values("is_partial", ascending=False)
        combined = combined.drop_duplicates(subset=keys, keep="last")
    else:
        combined = fresh

    combined = combined.sort_values(keys).reset_index(drop=True)
    combined.to_csv(store, index=False)
    return combined


def build(save: bool = True) -> dict:
    session = requests.Session()
    out = {}

    hourly_frames, ticket_frames = [], []
    for offset in (0, 1):
        try:
            hourly_frames.append(fetch_hourly(offset, session))
            ticket_frames.append(fetch_ticket_mix(offset, session))
        except Exception as exc:  # noqa: BLE001
            print(f"  offset {offset}: {type(exc).__name__}")

    hourly = pd.concat([f for f in hourly_frames if not f.empty], ignore_index=True) \
        if any(not f.empty for f in hourly_frames) else pd.DataFrame()
    tickets = pd.concat([f for f in ticket_frames if not f.empty], ignore_index=True) \
        if any(not f.empty for f in ticket_frames) else pd.DataFrame()

    if not hourly.empty and save:
        stored = _append(HOURLY_STORE, hourly, ["date", "hour"])
        complete = stored[~stored["is_partial"].astype(bool)]
        print(f"hourly profile: {stored['date'].nunique()} day(s) stored "
              f"({complete['date'].nunique()} complete) -> {HOURLY_STORE.name}")
        if not complete.empty:
            latest = complete[complete["date"] == complete["date"].max()]
            peak = latest.loc[latest["boardings"].idxmax()]
            share = peak["boardings"] / latest["boardings"].sum()
            print(f"  {peak['date']}: peak hour {int(peak['hour']):02d}:00 with "
                  f"{peak['boardings']:,} boardings ({share:.1%} of the day)")
        out["hourly"] = stored

    if not tickets.empty and save:
        stored = _append(TICKETS_STORE, tickets, ["date"])
        print(f"ticket mix: {stored['date'].nunique()} day(s) stored "
              f"-> {TICKETS_STORE.name}")
        out["tickets"] = stored

    (RAW_DIR / "provenance_extras.json").write_text(json.dumps({
        "source": "CMRL public passenger-flow API",
        "endpoints": [f"{API}/hourlybaseddata/{{n}}", f"{API}/allTicketCount/{{n}}"],
        "granularity": "system-wide, not station-level",
        "use": "peak-hour share (PHPDT context) and fare-media transition; "
               "neither can train the station-level model",
    }, indent=2), encoding="utf-8")
    return out


if __name__ == "__main__":
    build()
