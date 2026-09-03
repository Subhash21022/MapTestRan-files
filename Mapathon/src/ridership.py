"""Prepare the regression target: average daily boardings per station.

Trains on BMRCL (Bengaluru) station-wise hourly data, which is RTI-sourced and
openly licensed. Chennai publishes no station-wise figures; if an RTI to CMRL
returns them, `load_chennai_target()` becomes the drop-in replacement and
nothing downstream changes.

    python -m src.ridership
"""

from __future__ import annotations

import pandas as pd

from src import config as C

ENTRIES_FILE = "station-hourly.parquet"
EXITS_FILE = "station-hourly-exits.parquet"

HOURS_PER_DAY = 24


def _load(filename: str, value_name: str) -> pd.DataFrame:
    path = C.RAW_RIDERSHIP / filename
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing. Run: python -m src.acquire --what bmrcl"
        )
    df = pd.read_parquet(path)
    df = df.rename(columns={"Ridership": value_name})
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def load_hourly() -> pd.DataFrame:
    """Entries and exits joined on (Date, Hour, Station)."""
    entries = _load(ENTRIES_FILE, "entries")
    exits = _load(EXITS_FILE, "exits")
    merged = entries.merge(exits, on=["Date", "Hour", "Station"], how="outer")
    merged[["entries", "exits"]] = merged[["entries", "exits"]].fillna(0).astype(int)
    return merged


def _complete_station_days(hourly: pd.DataFrame) -> pd.DataFrame:
    """Keep only station-days with all 24 hours present.

    The published series has gaps (48 of the 61 calendar days in Aug-Sep 2025,
    and coverage differs slightly between the entries and exits files).
    Averaging over partial days would silently bias those stations downward,
    so incomplete days are dropped rather than filled.
    """
    counts = hourly.groupby(["Station", "Date"]).size().rename("hours").reset_index()
    complete = counts[counts["hours"] == HOURS_PER_DAY][["Station", "Date"]]
    dropped = len(counts) - len(complete)
    if dropped:
        print(f"  dropped {dropped} incomplete station-days of {len(counts)}")
    return hourly.merge(complete, on=["Station", "Date"], how="inner")


def daily_by_station(hourly: pd.DataFrame) -> pd.DataFrame:
    """Collapse hourly records to one row per station-day."""
    clean = _complete_station_days(hourly)
    daily = (
        clean.groupby(["Station", "Date"])[["entries", "exits"]]
        .sum()
        .reset_index()
    )
    daily["throughput"] = daily["entries"] + daily["exits"]
    daily["dow"] = daily["Date"].dt.dayofweek       # Monday = 0
    daily["is_weekday"] = daily["dow"] < 5
    return daily


def flag_maturity(daily: pd.DataFrame, min_coverage: float = 0.9) -> pd.DataFrame:
    """Identify stations that opened during the observation window.

    Bengaluru's Yellow Line (RV Road - Bommasandra) opened on 10 August 2025,
    inside the 1 Aug - 30 Sep window this data covers. Fifteen of the 83
    stations therefore report ridership for a line that was weeks old: they
    show 38-42 active days against 48 for the rest, and are still ramping
    steeply (Huskur Road +50%, Ragigudda +40% over the window).

    Training on them is actively harmful. Central Silk Board has the catchment
    of a major interchange but recorded 4,316 boardings, so the model learns
    that dense, well-connected catchments produce *low* ridership - it
    predicted 21,905 there and was wrong by 5x in the opposite direction at
    Majestic. Detected from the data rather than a hardcoded station list, so
    the same logic holds if the series is later extended.
    """
    activity = daily[daily["entries"] > 0]
    first_active = activity.groupby("Station")["Date"].min()
    active_days = activity.groupby("Station").size()

    window_start = daily["Date"].min()
    max_days = active_days.max()

    # Ramp-up: last fortnight against the first.
    window_end = daily["Date"].max()
    early = daily[daily["Date"] < window_start + pd.Timedelta(days=14)]
    late = daily[daily["Date"] > window_end - pd.Timedelta(days=14)]
    growth = (
        late.groupby("Station")["entries"].mean()
        / early.groupby("Station")["entries"].mean().replace(0, pd.NA)
    )

    out = pd.DataFrame({
        "first_active": first_active,
        "active_days": active_days,
        "growth_ratio": growth,
    })
    out["opened_mid_window"] = out["first_active"] > window_start
    out["low_coverage"] = out["active_days"] < min_coverage * max_days
    out["is_mature"] = ~(out["opened_mid_window"] | out["low_coverage"])

    immature = out[~out["is_mature"]]
    if len(immature):
        print(f"  {len(immature)} station(s) opened mid-window and are excluded "
              f"from training:")
        report = immature[["first_active", "active_days", "growth_ratio"]].sort_values("first_active")
        report = report.assign(
            first_active=report["first_active"].dt.strftime("%Y-%m-%d"),
            growth_ratio=report["growth_ratio"].astype(float).round(2),
        )
        print(report.to_string())
    return out.reset_index().rename(columns={"Station": "station"})


def station_target(daily: pd.DataFrame) -> pd.DataFrame:
    """Per-station summary used as the model's dependent variable.

    `boardings_weekday` is the primary target - weekday boardings are the
    convention in direct demand modelling, since weekend patterns reflect a
    different trip purpose mix and would blur the land-use signal.
    """
    def agg(frame: pd.DataFrame, suffix: str) -> pd.DataFrame:
        return (
            frame.groupby("Station")
            .agg(**{
                f"boardings_{suffix}": ("entries", "mean"),
                f"alightings_{suffix}": ("exits", "mean"),
                f"throughput_{suffix}": ("throughput", "mean"),
                f"days_{suffix}": ("Date", "nunique"),
            })
        )

    overall = agg(daily, "all")
    weekday = agg(daily[daily["is_weekday"]], "weekday")
    weekend = agg(daily[~daily["is_weekday"]], "weekend")

    out = overall.join(weekday).join(weekend).reset_index()
    out = out.rename(columns={"Station": "station"})

    # Directional imbalance separates origin-dominant (residential) stations
    # from destination-dominant (employment) ones - useful for typology, and a
    # sanity check that entries and exits were not swapped.
    out["boarding_share"] = (
        out["boardings_weekday"] / (out["boardings_weekday"] + out["alightings_weekday"])
    )

    out["weekend_ratio"] = out["throughput_weekend"] / out["throughput_weekday"]
    return out.sort_values("boardings_weekday", ascending=False).reset_index(drop=True)


def build(save: bool = True) -> pd.DataFrame:
    print("loading BMRCL hourly ridership...")
    hourly = load_hourly()
    print(f"  {len(hourly):,} station-hour records")
    print(f"  {hourly['Station'].nunique()} stations, "
          f"{hourly['Date'].min():%Y-%m-%d} to {hourly['Date'].max():%Y-%m-%d}")

    daily = daily_by_station(hourly)
    print(f"  {len(daily):,} complete station-days")

    print("\nmaturity check")
    maturity = flag_maturity(daily)

    target = station_target(daily)
    target = target.merge(maturity, on="station", how="left")
    target["is_mature"] = target["is_mature"].fillna(True)

    system_weekday = target["boardings_weekday"].sum()
    print(f"\n{len(target)} stations with a usable target "
          f"({int(target['is_mature'].sum())} mature, "
          f"{int((~target['is_mature']).sum())} excluded from training)")
    print(f"system-wide mean weekday boardings: {system_weekday:,.0f}")
    print(f"\ntop 10 by weekday boardings:")
    print(target.head(10)[["station", "boardings_weekday", "boarding_share"]].to_string(index=False))
    print(f"\nbottom 5:")
    print(target.tail(5)[["station", "boardings_weekday", "boarding_share"]].to_string(index=False))

    # A 100x spread across stations is normal and is exactly the variation the
    # model has to explain; a near-flat distribution would mean something broke.
    lo, hi = target["boardings_weekday"].min(), target["boardings_weekday"].max()
    print(f"\nspread: {lo:,.0f} to {hi:,.0f}  ({hi / max(lo, 1):.0f}x)")

    if save:
        out = C.PROCESSED / "bmrcl_station_target.csv"
        target.to_csv(out, index=False)
        print(f"\n-> {out}")
    return target


if __name__ == "__main__":
    build()
