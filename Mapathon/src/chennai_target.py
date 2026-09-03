"""Ingest CMRL per-station ridership and turn it into the model's target.

Chennai does not publish station-wise ridership, which is why the pipeline
currently trains on Bengaluru and transfers. If you obtain Chennai Phase 1
numbers - an RTI reply, a CMRL annual report table, a scraped news table - drop
the file in `data/raw/ridership/chennai/` and run:

    python -m src.chennai_target

Training on Chennai directly is strictly better than transferring from
Bengaluru: it removes every cross-city assumption about income, bus
competition and network maturity. Nothing downstream changes - the model picks
up `chennai_station_target.csv` automatically.

The reader is deliberately forgiving about format because the shape of what
arrives is unknown: CSV or Excel, any column names, ridership as daily or
monthly totals, or as high/medium/low categories.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src import config as C
from src.stations import MANUAL_ALIASES, match_names, normalise

INBOX = C.RAW_RIDERSHIP / "chennai"
INBOX.mkdir(parents=True, exist_ok=True)

OUT = C.PROCESSED / "chennai_station_target.csv"

# Column-name hints, matched case-insensitively against normalised headers.
NAME_HINTS = ("station", "name", "stn")
VALUE_HINTS = ("ridership", "footfall", "passenger", "entry", "entries",
               "boarding", "traffic", "count", "daily", "monthly", "total", "avg", "average")
CATEGORY_VALUES = {"high", "medium", "med", "low", "very high", "very low"}

# Used when the source reports a monthly total but the model wants a daily mean.
DAYS_PER_MONTH = 30.4


def _read_any(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(path)
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"could not decode {path}")


def _pick_column(df: pd.DataFrame, hints: tuple[str, ...], numeric: bool) -> str | None:
    """Choose the column whose header best matches the hints.

    Falls back to the first suitable column by dtype, so a headerless-ish table
    still works.
    """
    scored = []
    for col in df.columns:
        header = str(col).lower()
        hits = sum(1 for h in hints if h in header)
        if numeric and not pd.api.types.is_numeric_dtype(df[col]):
            coerced = pd.to_numeric(
                df[col].astype(str).str.replace(r"[,\s]", "", regex=True),
                errors="coerce",
            )
            if coerced.notna().sum() < 0.5 * len(df):
                continue
        if hits:
            scored.append((hits, col))

    if scored:
        return max(scored)[1]

    if numeric:
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                return col
    else:
        for col in df.columns:
            if df[col].dtype == object:
                return col
    return None


def _to_numeric(series: pd.Series) -> pd.Series:
    """Strip thousands separators, footnote marks and stray units."""
    cleaned = (
        series.astype(str)
        .str.replace(r"[,\s]", "", regex=True)
        .str.replace(r"[^\d.\-]", "", regex=True)
    )
    return pd.to_numeric(cleaned, errors="coerce")


def load_raw(path: Path | None = None) -> pd.DataFrame:
    """Read whichever ridership file is in the inbox."""
    if path is None:
        candidates = sorted(
            p for p in INBOX.iterdir()
            if p.suffix.lower() in (".csv", ".xlsx", ".xls") and not p.name.startswith("~$")
        )
        if not candidates:
            raise FileNotFoundError(
                f"no CSV/Excel file in {INBOX}\n"
                "Drop the CMRL station-wise ridership file there and re-run."
            )
        if len(candidates) > 1:
            print(f"{len(candidates)} files found; using the most recent")
            candidates.sort(key=lambda p: p.stat().st_mtime)
        path = candidates[-1]

    print(f"reading {path.name}")
    df = _read_any(path)
    print(f"  {df.shape[0]} rows x {df.shape[1]} cols")
    print(f"  columns: {list(df.columns)}")
    return df


def build(path: Path | None = None, monthly: bool | None = None, save: bool = True) -> pd.DataFrame:
    """Match a CMRL ridership table to the canonical Chennai station list."""
    raw = load_raw(path)

    name_col = _pick_column(raw, NAME_HINTS, numeric=False)
    value_col = _pick_column(raw, VALUE_HINTS, numeric=True)
    if name_col is None:
        raise ValueError(f"could not identify a station-name column in {list(raw.columns)}")
    print(f"  station names <- {name_col!r}")

    # Categorical fallback: some sources only publish high/medium/low bands.
    is_categorical = False
    if value_col is None:
        for col in raw.columns:
            values = set(raw[col].astype(str).str.strip().str.lower().dropna())
            if values and values <= CATEGORY_VALUES | {"nan"}:
                value_col, is_categorical = col, True
                break
    if value_col is None:
        raise ValueError(f"could not identify a ridership column in {list(raw.columns)}")
    print(f"  ridership   <- {value_col!r}{' (categorical)' if is_categorical else ''}")

    work = pd.DataFrame({
        "source_name": raw[name_col].astype(str).str.strip(),
        "raw_value": raw[value_col],
    })
    work = work[work["source_name"].str.len() > 0]

    if is_categorical:
        # Ordered bands, so the model can still regress on a monotone scale.
        band = {"very low": 0, "low": 1, "medium": 2, "med": 2, "high": 3, "very high": 4}
        work["value"] = work["source_name"].map(lambda _: np.nan)
        work["value"] = work["raw_value"].astype(str).str.strip().str.lower().map(band)
        print("  NOTE: categorical target. The model will predict a band index, "
              "not a passenger count; report it as a ranking, not a forecast.")
    else:
        work["value"] = _to_numeric(work["raw_value"])

    work = work[work["value"].notna()]
    if work.empty:
        raise ValueError("no usable rows after cleaning - inspect the source file")

    # Monthly totals are common in annual reports; convert to a daily mean.
    if monthly is None and not is_categorical:
        monthly = bool(re.search(r"month", str(value_col), re.IGNORECASE)) or \
            work["value"].median() > 100_000
    if monthly and not is_categorical:
        print(f"  treating values as monthly totals -> dividing by {DAYS_PER_MONTH}")
        work["value"] = work["value"] / DAYS_PER_MONTH

    # Match against the canonical station names already in the pipeline.
    import geopandas as gpd

    stations_path = C.PROCESSED / "chennai_stations.gpkg"
    if not stations_path.exists():
        raise FileNotFoundError(
            f"{stations_path} missing. Run: python -m src.stations --city chennai"
        )
    stations = gpd.read_file(stations_path)
    phase1 = stations[stations["phase"] == "phase1"]
    canonical = phase1["name"].dropna().tolist()

    mapping, unmatched = match_names(
        work["source_name"].tolist(), canonical,
        threshold=0.80, aliases=MANUAL_ALIASES,
    )
    print(f"\nmatched {len(mapping)}/{len(work)} source rows to "
          f"{len(canonical)} Phase 1 stations")
    if unmatched:
        print(f"  unmatched source rows ({len(unmatched)}): {unmatched[:15]}")

    work["station"] = work["source_name"].map(mapping)
    matched = work[work["station"].notna()].copy()

    out = (
        matched.groupby("station", as_index=False)["value"].mean()
        .rename(columns={"value": "boardings_weekday"})
    )
    out["city"] = "chennai"
    out["is_mature"] = True
    out["target_kind"] = "band" if is_categorical else "daily_boardings"

    missing = sorted(set(canonical) - set(out["station"]))
    print(f"\n{len(out)}/{len(canonical)} Phase 1 stations have a target")
    if missing:
        print(f"  no data for: {missing}")

    print(f"\ntop 10:")
    print(out.nlargest(10, "boardings_weekday").round(0).to_string(index=False))

    if save:
        out.to_csv(OUT, index=False)
        print(f"\n-> {OUT}")
        print("Re-run the model; it will now train on Chennai:")
        print("    python -m src.model --train-city chennai")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", type=Path, default=None,
                        help="specific file to read (default: newest in the inbox)")
    parser.add_argument("--monthly", action="store_true", default=None,
                        help="force treating values as monthly totals")
    args = parser.parse_args()
    build(args.file, monthly=args.monthly)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
