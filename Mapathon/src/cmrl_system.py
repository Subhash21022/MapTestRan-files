"""CMRL system-wide monthly ridership - the calibration benchmark.

This is *not* the training target. CMRL publishes ridership for the network as
a whole, split by ticket media (closed loop card / QR / NCMC), with no station
breakdown - which is the reason the model trains on Bengaluru and transfers.

What it does give us is the one hard number Chennai can be checked against:
predict every Phase 1 station, sum the predictions, and compare the total to
what the system actually carries. That is the whole credibility argument for
the transfer, and it deserves a real figure rather than the hardcoded estimate
it used before.

    python -m src.cmrl_system
"""

from __future__ import annotations

import pandas as pd

from src import config as C

SOURCE = C.RAW_RIDERSHIP / "cmrl_system_monthly.csv"

# Indian digit grouping ("1,04,68,732") defeats pandas' thousands= handling,
# so the separators are stripped outright.
TOTAL_COL = "Total Passenger Flow"
MONTH_COL = "Month"


def load(path=None) -> pd.DataFrame:
    path = path or SOURCE
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing.\n"
            "Download the monthly series from "
            "https://data.opencity.in/dataset/chennai-metro-monthly-usage-data"
        )

    df = pd.read_csv(path)
    df = df[df[MONTH_COL].notna() & (df[MONTH_COL].astype(str).str.strip() != "")]

    df["month"] = pd.to_datetime(df[MONTH_COL], format="%b-%y", errors="coerce")
    df = df[df["month"].notna()].copy()

    df["total"] = pd.to_numeric(
        df[TOTAL_COL].astype(str).str.replace(",", "", regex=False).str.strip(),
        errors="coerce",
    )
    df = df[df["total"].notna()]

    df["days"] = df["month"].dt.days_in_month
    df["daily_mean"] = df["total"] / df["days"]
    return df[["month", "total", "days", "daily_mean"]].sort_values("month").reset_index(drop=True)


def observed_daily(months: int = 12, path=None) -> float:
    """Mean daily ridership over the most recent `months` of published data.

    A trailing window rather than the single latest month: Chennai's monthly
    totals swing by roughly 10% with holidays and monsoon, and calibrating the
    model against one unusually high or low month would move the verdict for no
    real reason.
    """
    series = load(path)
    recent = series.tail(months)
    return float(recent["daily_mean"].mean())


def report(path=None) -> pd.DataFrame:
    series = load(path)
    print(f"CMRL system-wide ridership, {series['month'].min():%b %Y} "
          f"to {series['month'].max():%b %Y}  ({len(series)} months)")

    latest = series.iloc[-1]
    print(f"\nlatest month ({latest['month']:%b %Y}): "
          f"{latest['total']:,.0f} riders, {latest['daily_mean']:,.0f}/day")

    for window in (3, 6, 12, 24):
        if len(series) >= window:
            mean = series.tail(window)["daily_mean"].mean()
            print(f"  trailing {window:>2} months: {mean:,.0f}/day")

    peak = series.loc[series["daily_mean"].idxmax()]
    print(f"\nbusiest month: {peak['month']:%b %Y} at {peak['daily_mean']:,.0f}/day")

    print(f"\nannual growth (same month, year on year):")
    series = series.set_index("month")
    yoy = series["daily_mean"].pct_change(12).dropna() * 100
    for month, change in yoy.tail(6).items():
        print(f"  {month:%b %Y}: {change:+.1f}%")

    calibration = observed_daily(12, path)
    print(f"\n-> calibration benchmark (trailing 12 months): {calibration:,.0f}/day")
    print(f"   config.CHENNAI_OBSERVED_DAILY_MEAN is {C.CHENNAI_OBSERVED_DAILY_MEAN:,}")
    drift = calibration / C.CHENNAI_OBSERVED_DAILY_MEAN - 1
    print(f"   the hardcoded estimate is off by {drift:+.0%}")
    return series


if __name__ == "__main__":
    report()
