"""Export the collected CMRL data to a single Excel workbook.

Open the CSV stores directly in Excel only to look at them - never save from
there. Excel rewrites date columns into its own format and strips leading zeros
from the line codes ("01" becomes 1), which is exactly the corruption that once
broke de-duplication and doubled every station's boardings.

This writes a separate .xlsx instead, so the stores stay untouched:

    python -m src.to_excel

Re-run it after any fetch; it always rebuilds from the current CSVs.
"""

from __future__ import annotations

import pandas as pd

from src import config as C

RAW = C.RAW_RIDERSHIP / "chennai"
OUT = C.OUT_TABLES / "cmrl_data.xlsx"

STATIONS = RAW / "cmrl_stationflow_daily.csv"
HOURLY = RAW / "cmrl_hourly.csv"
TICKETS = RAW / "cmrl_ticket_mix.csv"
TARGET = C.PROCESSED / "chennai_station_target.csv"


def _load(path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df


def build(save: bool = True) -> dict:
    stations = _load(STATIONS)
    if stations.empty:
        raise FileNotFoundError(
            f"{STATIONS} missing. Run: python -m src.cmrl_stationflow"
        )

    sheets: dict[str, pd.DataFrame] = {}

    # --- Daily summary: the sheet to glance at to confirm a fetch landed ---
    daily = (
        stations.groupby(["date", "is_partial"], as_index=False)
        .agg(total_boardings=("boardings", "sum"),
             stations_reported=("cmrl_name", "nunique"))
    )
    daily["day"] = daily["date"].dt.day_name()
    daily["kind"] = daily.apply(
        lambda r: "partial (today)" if r["is_partial"]
        else ("weekend" if r["date"].dayofweek >= 5 else "weekday"), axis=1)
    daily = daily[["date", "day", "kind", "total_boardings", "stations_reported"]]
    daily = daily.sort_values("date", ascending=False)
    sheets["Daily summary"] = daily

    # --- Station x date matrix: a new column means a new day arrived ---
    complete = stations[~stations["is_partial"].astype(bool)]
    if not complete.empty:
        pivot = (
            complete.groupby(["cmrl_name", "date"])["boardings"].sum()
            .unstack("date")
        )
        pivot.columns = [c.strftime("%Y-%m-%d %a") for c in pivot.columns]
        pivot = pivot.sort_values(pivot.columns[-1], ascending=False)
        pivot.insert(0, "mean_weekday", _weekday_mean(complete))
        sheets["Station x date"] = pivot.reset_index().rename(
            columns={"cmrl_name": "station"})

    # --- The model's actual target ---
    target = _load(TARGET)
    if not target.empty:
        sheets["Model target"] = target

    # --- Hourly profile ---
    hourly = _load(HOURLY)
    if not hourly.empty:
        hc = hourly[~hourly["is_partial"].astype(bool)]
        if not hc.empty:
            hp = hc.groupby(["hour", "date"])["boardings"].sum().unstack("date")
            hp.columns = [c.strftime("%Y-%m-%d %a") for c in hp.columns]
            sheets["Hourly profile"] = hp.reset_index()

    # --- Ticket mix ---
    tickets = _load(TICKETS)
    if not tickets.empty:
        keep = ["date"] + [c for c in tickets.columns
                           if tickets[c].dtype.kind in "if" and tickets[c].sum() > 0]
        sheets["Ticket mix"] = tickets[keep].sort_values("date", ascending=False)

    if save:
        C.OUT_TABLES.mkdir(parents=True, exist_ok=True)
        with pd.ExcelWriter(OUT, engine="openpyxl", datetime_format="yyyy-mm-dd") as writer:
            for name, frame in sheets.items():
                frame.to_excel(writer, sheet_name=name[:31], index=False)
                _autosize(writer.sheets[name[:31]], frame)

        print(f"-> {OUT}")
        for name, frame in sheets.items():
            print(f"   {name:<18} {frame.shape[0]:>4} rows x {frame.shape[1]:>3} cols")

    _report(daily, complete)
    return sheets


def _weekday_mean(complete: pd.DataFrame) -> pd.Series:
    wd = complete[complete["date"].dt.dayofweek < 5]
    if wd.empty:
        return pd.Series(dtype=float)
    per_day = wd.groupby(["cmrl_name", "date"])["boardings"].sum()
    return per_day.groupby("cmrl_name").mean().round(0)


def _autosize(sheet, frame: pd.DataFrame) -> None:
    for i, col in enumerate(frame.columns, start=1):
        width = max(len(str(col)), 12)
        if frame[col].dtype == object:
            longest = frame[col].astype(str).str.len().max()
            width = max(width, min(int(longest) + 2, 48))
        sheet.column_dimensions[sheet.cell(row=1, column=i).column_letter].width = width
    sheet.freeze_panes = "B2"


def _report(daily: pd.DataFrame, complete: pd.DataFrame) -> None:
    """Print the same checks the workbook is meant to let you eyeball."""
    print("\ncollection status")
    n_weekday = complete[complete["date"].dt.dayofweek < 5]["date"].nunique()
    n_weekend = complete[complete["date"].dt.dayofweek >= 5]["date"].nunique()
    print(f"  complete days: {complete['date'].nunique()} "
          f"({n_weekday} weekday, {n_weekend} weekend)")

    latest = daily.iloc[0]
    print(f"  most recent:   {latest['date']:%Y-%m-%d} ({latest['day']}), "
          f"{latest['kind']}, {latest['total_boardings']:,.0f} boardings")

    stale = (pd.Timestamp.today().normalize() - daily["date"].max()).days
    if stale > 1:
        print(f"  WARNING: newest record is {stale} days old - the daily fetch "
              f"may not be running. Check the scheduled task.")

    remaining = max(10 - n_weekday, 0)
    if remaining:
        print(f"  {remaining} more weekday(s) until Chennai can replace "
              f"Bengaluru as the training target")
    else:
        print("  enough weekdays collected - retrain with "
              "--train-city chennai")


if __name__ == "__main__":
    build()
