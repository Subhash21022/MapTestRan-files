"""Compare old vs. updated CMRL predictions and produce a summary report.

Usage:
    python scripts/compare_predictions.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OLD_PRED  = ROOT / "outputs/tables/chennai_predictions.csv"
NEW_PRED  = ROOT / "outputs/tables/chennai_predictions_new.csv"
OLD_TARGET = ROOT / "data/processed/chennai_station_target.csv"

# ── load ──────────────────────────────────────────────────────────────────────
old = pd.read_csv(OLD_PRED)
new = pd.read_csv(NEW_PRED)
target = pd.read_csv(OLD_TARGET)   # updated target (today's fetch)

# ── merge on station name ──────────────────────────────────────────────────────
merged = old[["name","phase","Line","prediction","lower","upper"]].merge(
    new[["name","prediction","lower","upper"]],
    on="name", suffixes=("_old","_new")
)
merged["delta"]        = merged["prediction_new"] - merged["prediction_old"]
merged["delta_pct"]    = merged["delta"] / merged["prediction_old"] * 100
merged["rank_old"]     = merged["prediction_old"].rank(ascending=False).astype(int)
merged["rank_new"]     = merged["prediction_new"].rank(ascending=False).astype(int)
merged["rank_change"]  = merged["rank_old"] - merged["rank_new"]   # + = moved up

# ── separate Phase 1 / Phase 2 ────────────────────────────────────────────────
p1 = merged[merged["phase"] == "phase1"].copy()
p2 = merged[merged["phase"] == "phase2"].copy()

SEP = "=" * 72

def section(title):
    print(f"\n{SEP}\n  {title}\n{SEP}")

# ── 1. data provenance ────────────────────────────────────────────────────────
section("DATA PROVENANCE")
print("  Previous target  : 15 weekday observations (hand-collected)")
print("  Updated target   : 1 observation, 2026-09-05 (Saturday — weekend)")
print("  WARNING: today's data is a WEEKEND day. Weekend ridership is")
print("  structurally different from weekday demand. Until more weekday")
print("  observations accumulate (run daily), treat updated predictions")
print("  as directional, not authoritative.")

# ── 2. System-level totals ────────────────────────────────────────────────────
section("SYSTEM-LEVEL TOTALS  (Phase 1  weekday boardings)")
old_total = p1["prediction_old"].sum()
new_total = p1["prediction_new"].sum()
obs_total = target["boardings_weekday"].sum()
print(f"  Old model predicted total  : {old_total:>10,.0f}")
print(f"  New model predicted total  : {new_total:>10,.0f}  ({(new_total-old_total)/old_total*100:+.1f}%)")
print(f"  Observed (Sep 5 weekend)   : {obs_total:>10,.0f}")
print(f"  Published weekday mean     : {'316,121':>10}")

# ── 3. Model performance summary ──────────────────────────────────────────────
try:
    perf = pd.read_csv(ROOT / "outputs/tables/model_performance.csv", index_col=0)
    section("MODEL PERFORMANCE  (new run, leave-one-out CV)")
    for model, row in perf.iterrows():
        print(f"  {model:<20}  R²={float(row.get('r2',float('nan'))):+.3f}  "
              f"rho={float(row.get('spearman',float('nan'))):+.3f}  "
              f"MAE={float(row.get('mae',float('nan'))):,.0f}  "
              f"MAPE={float(row.get('mape',float('nan'))):.0f}%")
except FileNotFoundError:
    pass

# ── 4. Biggest gainers / losers  Phase 2 ─────────────────────────────────────
section("TOP 10 PHASE 2 STATIONS — BIGGEST UPWARD REVISIONS")
cols_show = ["name","Line","prediction_old","prediction_new","delta","delta_pct","rank_old","rank_new","rank_change"]
p2_sorted = p2.sort_values("delta", ascending=False)
print(p2_sorted.head(10)[cols_show].round(1).to_string(index=False))

section("TOP 10 PHASE 2 STATIONS — BIGGEST DOWNWARD REVISIONS")
print(p2_sorted.tail(10)[cols_show].round(1).to_string(index=False))

# ── 5. Phase 2 top-15 by new prediction ───────────────────────────────────────
section("PHASE 2 TOP-15 BY NEW MODEL  (boardings/day)")
p2_new_top = p2.nlargest(15,"prediction_new")
print(p2_new_top[["name","Line","prediction_old","prediction_new","delta_pct",
                   "lower_new","upper_new"]].round(0).to_string(index=False))

# ── 6. Rank changes  Phase 2 ─────────────────────────────────────────────────
section("PHASE 2 RANK MOVERS  (> 5 positions)")
big_movers = p2[p2["rank_change"].abs() > 5].sort_values("rank_change", ascending=False)
if big_movers.empty:
    print("  No station moved more than 5 positions.")
else:
    print(big_movers[["name","Line","rank_old","rank_new","rank_change",
                       "prediction_old","prediction_new"]].round(0).to_string(index=False))

# ── 7. Phase 1 accuracy vs updated observed ───────────────────────────────────
section("PHASE 1 ACCURACY VS UPDATED OBSERVED (Sep 5, weekend)")
p1_obs = p1.merge(
    target[["station","boardings_weekday"]],
    left_on="name", right_on="station", how="inner"
)
if not p1_obs.empty:
    p1_obs["error_new"] = p1_obs["prediction_new"] - p1_obs["boardings_weekday"]
    p1_obs["error_old"] = p1_obs["prediction_old"] - p1_obs["boardings_weekday"]
    p1_obs["abs_err_new"] = p1_obs["error_new"].abs()
    p1_obs["abs_err_old"] = p1_obs["error_old"].abs()
    mae_new = p1_obs["abs_err_new"].mean()
    mae_old = p1_obs["abs_err_old"].mean()
    print(f"  MAE old model vs updated obs : {mae_old:,.0f} boardings/day")
    print(f"  MAE new model vs updated obs : {mae_new:,.0f} boardings/day")
    print(f"  Improvement                  : {mae_old - mae_new:+,.0f} boardings/day")
    print("\n  Largest new-model misses vs updated observations:")
    worst = p1_obs.nlargest(8,"abs_err_new")[
        ["name","boardings_weekday","prediction_new","error_new"]]
    print(worst.round(0).to_string(index=False))

# ── 8. Map change recommendations ─────────────────────────────────────────────
section("MAP CHANGE RECOMMENDATIONS")

# class thresholds  (same as build_qgis_project presumably)
def classify(val, breaks=[5000,8000,11000,15000]):
    if   val < breaks[0]: return "Very Low"
    elif val < breaks[1]: return "Low"
    elif val < breaks[2]: return "Medium"
    elif val < breaks[3]: return "High"
    else:                  return "Very High"

p2["class_old"] = p2["prediction_old"].apply(classify)
p2["class_new"] = p2["prediction_new"].apply(classify)
class_changed = p2[p2["class_old"] != p2["class_new"]][
    ["name","Line","class_old","class_new","prediction_old","prediction_new","delta_pct"]]

print(f"\n  Stations whose RIDERSHIP TIER changed ({len(class_changed)}):")
if class_changed.empty:
    print("    None — no station crossed a tier boundary.")
else:
    print(class_changed.round(0).to_string(index=False))

print(f"""
  Summary of recommended map updates:
  1. COLOUR BAND UPDATES : {len(class_changed)} station(s) need a tier colour change.
  2. LABEL UPDATES       : Re-sort Phase 2 station labels by new predicted volume.
  3. CONFIDENCE BANDS    : Widen prediction intervals for stations based on only
                           1 weekend observation (all Phase 1 stations).
  4. INTERMODAL HUBS     : No change recommended — Hub classification is
                           network-topology driven, not ridership-data driven.
  5. CATCHMENT CIRCLES   : No geometric change needed from model alone.
  6. DATA QUALITY NOTE   : Add a map note that the September update is based on
                           1 weekend day only; wider uncertainty bands apply until
                           weekday observations accumulate (run daily fetch).
""")

# ── 9. Save comparison table ──────────────────────────────────────────────────
out = ROOT / "outputs/tables/prediction_comparison.csv"
merged.to_csv(out, index=False)
print(f"  Full comparison table saved -> {out}")
