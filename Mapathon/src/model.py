"""Fit the direct demand model and transfer it to Chennai.

Three estimators are fitted and compared rather than one being asserted:

  * Negative Binomial GLM  - ridership is an overdispersed count, and the
    coefficients read directly as elasticities, which is what makes the map
    explanatory rather than a black box.
  * Elastic net on log(boardings) - handles the collinearity among the density
    variables and does the final variable selection.
  * Gradient boosting - captures the well-documented non-linearity in the
    density/ridership relationship, interpreted through SHAP.

With ~83 training stations the binding constraint is overfitting, so the
feature set is screened by correlation and VIF, capped at config.MAX_FEATURES,
validated by leave-one-out CV, and reported with bootstrap prediction
intervals rather than bare point estimates.

    python -m src.model
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNetCV, LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src import config as C

TARGET = "boardings_weekday"

# Identifier / metadata columns that must never enter the design matrix.
NON_FEATURES = {
    "station_id", "name", "phase", "Line", "Layout", "city",
    "confidence", "position_source", "station", "osm_name",
    # Scenario inputs, not predictors. population_2030 is a projection of
    # population and would be near-perfectly collinear with it.
    "population_2030", "snap_offset_m", "n_osm_nodes", "line_dist_m",

    # Redundant by construction. Areal statistics are measured on a fixed
    # 800 m buffer, so every "_density" column is its count divided by the
    # same constant - a perfect linear rescaling carrying no extra
    # information. Keeping both wastes correlation-pruning budget and, worse,
    # lets the pruner discard the informative member of the pair. The counts
    # are kept (they are what the scenarios modify); the densities stay in the
    # output table for reporting but are barred from the model.
    "population_density", "built_volume_density", "built_surface_density",
    "intersection_count", "street_length_km",
    "catchment_area_km2",        # constant across stations
    "network_catchment_km2",     # PNR x a constant; PNR is the interpretable form
    "population_core400",        # r=0.97 with population; core_share is the signal

    # poi_total is the exact sum of the category columns, so including it
    # alongside them is perfect multicollinearity - it showed up as poi_total
    # at -0.29 fighting poi_office at +0.26, which is not interpretable.
    # poi_total carries the destination-accessibility dimension in the model;
    # the categories stay in the feature table for the typology clustering.
    "poi_education", "poi_healthcare", "poi_retail", "poi_office",
    "poi_government", "poi_transport", "poi_leisure",
    "poi_education_pctl", "poi_healthcare_pctl", "poi_retail_pctl",
    "poi_office_pctl", "poi_government_pctl", "poi_transport_pctl",
    "poi_leisure_pctl",

    # Raw OSM counts are barred in favour of their `_pctl` versions: absolute
    # counts conflate real land use with how thoroughly each city has been
    # mapped (Chennai has 2.8x fewer POIs and 3.0x fewer bus stops than
    # Bengaluru), which would make the transfer under-predict Chennai.
    # See features.normalise_within_city.
    "poi_total", "bus_stops_500m", "parking_area_m2",

    # metro_degree is collinear with the two dummies derived from it
    # (degree==1 is a terminal, degree>2 an interchange) and drove VIF to
    # infinity. The dummies are the interpretable form.
    "metro_degree",
}


def candidate_features(df: pd.DataFrame) -> list[str]:
    """Numeric columns eligible as predictors.

    Anything derived from ridership is excluded explicitly - leaking the target
    back in as a feature is the failure mode that produces a suspiciously
    perfect R2.
    """
    leaky = ("boardings", "alightings", "throughput", "days_", "boarding_share",
             "weekend_ratio", "active_days", "growth_ratio", "is_mature",
             "opened_mid_window", "low_coverage")
    cols = []
    for col in df.select_dtypes(include=[np.number]).columns:
        if col in NON_FEATURES or col.startswith(leaky) or col == TARGET:
            continue
        cols.append(col)
    return cols


# --------------------------------------------------------------------------
# Feature screening
# --------------------------------------------------------------------------

# One predictor per 5D dimension, kept regardless of what the data-driven
# screening would otherwise do. Ewing & Cervero's framework is the accepted
# specification for direct demand models, and a model missing an entire
# dimension is not interpretable as one - purely correlation-driven pruning
# discarded distance-to-CBD (the strongest single predictor, r = -0.50) simply
# because it correlated with a noisy MST-derived centrality proxy.
# Screening still decides everything outside this core.
PROTECTED_5D = [
    "population",             # Density
    "landuse_entropy",        # Diversity   (present once Bhuvan LULC lands)
    "intersection_density",   # Design
    "pnr",                    # Design - severance
    "dist_cbd_km",            # Destination accessibility
    "poi_total_pctl",         # Destination accessibility - see below
    "bus_stops_500m_pctl",    # Distance to transit - feeder access
    "is_intermodal_hub",      # Distance to transit - interregional access
]


def screen_features(
    X: pd.DataFrame,
    target: pd.Series,
    corr_threshold: float = 0.85,
    vif_threshold: float = 10.0,
    max_features: int = C.MAX_FEATURES,
    protected: list[str] | None = None,
) -> list[str]:
    """Drop near-duplicate and unstable predictors, then cap the count.

    `X` must already be in the modelling representation (log-transformed where
    skewed). Screening the raw values instead inflates VIF enormously for
    heavy-tailed variables and prunes exactly the predictors the model needs.

    Order: correlation pruning first (cheap, removes obvious twins), then VIF
    for multi-way collinearity, then a hard cap by univariate association.
    """
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    work = X.replace([np.inf, -np.inf], np.nan)
    work = work.loc[:, work.notna().sum() > 0.5 * len(work)]
    work = work.fillna(work.median())
    # A predictor with no variance carries no information and breaks VIF.
    work = work.loc[:, work.std() > 0]
    kept = list(work.columns)
    print(f"  {X.shape[1]} candidates -> {len(kept)} after dropping empty/constant")

    protected = [c for c in (protected if protected is not None else PROTECTED_5D) if c in kept]
    print(f"  protected 5D core ({len(protected)}): {protected}")
    strength = work.corrwith(target).abs()

    # Prune one pair at a time, recomputing after each drop.
    #
    # The earlier single-pass version walked every pair and could discard both
    # members of a chain: dist_cbd_km was dropped against metro_closeness, then
    # metro_closeness was dropped against something else, losing the single
    # strongest predictor in the set (r = -0.50) along with its replacement.
    # Removing only the weaker member of the *most* correlated pair each round
    # guarantees one survivor per correlated cluster.
    while len(kept) > 1:
        # Mask the diagonal through the DataFrame: pandas 3.0 exposes a
        # read-only backing array, so np.fill_diagonal on .values raises.
        corr = work[kept].corr().abs().mask(np.eye(len(kept), dtype=bool), 0.0)
        worst = corr.stack().idxmax()
        value = corr.loc[worst]
        if value <= corr_threshold:
            break
        a, b = worst
        if a in protected and b in protected:
            # Both are core: leave them and stop pruning on correlation.
            print(f"    corr {value:.2f}: {a} / {b} both protected - kept")
            break
        if a in protected:
            weaker = b
        elif b in protected:
            weaker = a
        else:
            weaker = a if strength.get(a, 0) < strength.get(b, 0) else b
        keeper = b if weaker == a else a
        print(f"    corr {value:.2f}: dropped {weaker} (kept {keeper})")
        kept.remove(weaker)

    # variance_inflation_factor regresses each column on the others *without*
    # adding an intercept, so a design matrix with no constant column yields
    # VIFs that mostly measure how far the means sit from zero. Every
    # informative predictor here was being pruned as a result. Prepend a
    # constant and offset the indices to read the real values.
    while len(kept) > 1:
        matrix = np.column_stack([np.ones(len(work)), work[kept].to_numpy()])
        vifs = [variance_inflation_factor(matrix, i + 1) for i in range(len(kept))]
        # Only non-protected columns are eligible for VIF removal.
        eligible = [(v, i) for i, v in enumerate(vifs) if kept[i] not in protected]
        if not eligible:
            break
        worst_vif, worst = max(eligible)
        if worst_vif <= vif_threshold:
            break
        print(f"    VIF {worst_vif:.1f}: dropped {kept[worst]}")
        kept.pop(worst)

    if len(kept) > max_features:
        extras = [c for c in kept if c not in protected]
        room = max(max_features - len(protected), 0)
        ranked = work[extras].corrwith(target).abs().sort_values(ascending=False)
        kept = protected + ranked.head(room).index.tolist()
        print(f"    capped at {max_features}: {len(protected)} protected "
              f"+ {room} by univariate association")

    print(f"  final feature set ({len(kept)}): {kept}")
    return kept


def log_transform_skewed(
    X: pd.DataFrame,
    skew_threshold: float = 1.5,
    columns: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """log1p heavily right-skewed non-negative predictors.

    Catchment population spans 624 to 159,060 across Bengaluru's stations - a
    250x range - because the network serves both a very dense old core and
    near-empty tech corridors. Left raw, a handful of dense catchments dominate
    the fit. Logging also makes the coefficients read as elasticities, which is
    the standard specification in the direct demand literature.

    Returns the transformed frame and the list of columns that were logged, so
    the same transform can be reapplied to the prediction city.
    """
    if columns is None:
        columns = []
        for col in X.columns:
            values = X[col].dropna()
            if values.empty or (values < 0).any():
                continue
            # Binary indicators are often "skewed" by the statistic but
            # logging them just rescales 0/1 to 0/0.69 - no benefit, and it
            # obscures the coefficient's reading as a group difference.
            if values.nunique() <= 2:
                continue
            if values.skew() > skew_threshold:
                columns.append(col)

    out = X.copy()
    for col in columns:
        out[col] = np.log1p(out[col].clip(lower=0))
    return out, columns


# --------------------------------------------------------------------------
# Estimators
# --------------------------------------------------------------------------

def fit_negative_binomial(X: pd.DataFrame, y: pd.Series):
    """NB GLM with a log link; coefficients read as semi-elasticities.

    `sm.NegativeBinomial` estimates the dispersion jointly with the
    coefficients by maximum likelihood and, on ~83 rows with a dozen
    standardised predictors, frequently walks off and returns all-NaN
    parameters without raising. That silently produced an elasticity table of
    NaNs on the first run.

    Instead: fit a Poisson GLM first (always well behaved), estimate dispersion
    from its Pearson chi-square, then refit as a GLM with the NegativeBinomial
    family at that fixed alpha. Convergence and the NaN check are both
    reported rather than assumed.
    """
    import statsmodels.api as sm

    design = sm.add_constant(StandardScaler().fit_transform(X))
    counts = np.round(y).astype(float)

    poisson = sm.GLM(counts, design, family=sm.families.Poisson()).fit()

    # Overdispersion: Pearson chi2 / residual df. >1 means Poisson is too tight.
    dispersion = poisson.pearson_chi2 / poisson.df_resid
    alpha = max((dispersion - 1) / max(poisson.mu.mean(), 1e-9), 1e-6)
    print(f"    dispersion {dispersion:.1f} (Poisson assumes 1.0), alpha={alpha:.2e}")

    model = sm.GLM(counts, design, family=sm.families.NegativeBinomial(alpha=alpha)).fit()

    if not np.isfinite(model.params).all():
        print("    NB produced non-finite parameters; reporting the Poisson fit")
        return poisson
    return model


def linear_regression() -> Pipeline:
    """Plain OLS on log(ridership).

    The most explainable option: each coefficient reads as an elasticity ("a
    10% larger catchment population implies x% more boardings"), which is what
    makes the model defensible in a write-up rather than a black box.
    Standardised so coefficients are comparable with one another.
    """
    return Pipeline([
        ("scale", StandardScaler()),
        ("model", LinearRegression()),
    ])


def random_forest() -> RandomForestRegressor:
    """Random forest on log(ridership).

    Handles the non-linearities that matter here - ridership does not rise
    proportionally with density, it steps up sharply near employment centres -
    and each tree sees a bootstrap of the sample, which helps on small data.

    Depth is capped and leaves hold >=3 stations deliberately. With 40-70
    training rows an unconstrained ensemble memorises the sample and
    leave-one-out error goes *up*: that is precisely what gradient boosting did
    here, scoring R2 = -0.009 while a 6-feature linear model scored 0.21.
    """
    return RandomForestRegressor(
        n_estimators=500,
        max_depth=6,
        min_samples_leaf=3,
        max_features="sqrt",
        random_state=C.RANDOM_SEED,
        n_jobs=-1,
    )


def elastic_net() -> Pipeline:
    return Pipeline([
        ("scale", StandardScaler()),
        ("model", ElasticNetCV(l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 1.0],
                               cv=5, random_state=C.RANDOM_SEED, max_iter=10000)),
    ])


def gradient_boosting() -> GradientBoostingRegressor:
    # Shallow trees and heavy subsampling: with <100 rows, depth is where
    # gradient boosting overfits fastest.
    return GradientBoostingRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=2,
        subsample=0.8, random_state=C.RANDOM_SEED,
    )


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

def loocv(make_model, X: pd.DataFrame, y: pd.Series, log_target: bool = True) -> dict:
    """Leave-one-out CV in the original ridership units.

    Errors are reported after back-transforming from log space, because an R2
    on logged values flatters the model and is not what a planner cares about.
    """
    loo = LeaveOneOut()
    preds = np.zeros(len(y))
    values = np.log1p(y) if log_target else y

    for train_idx, test_idx in loo.split(X):
        model = make_model()
        model.fit(X.iloc[train_idx], values.iloc[train_idx])
        pred = model.predict(X.iloc[test_idx])[0]
        preds[test_idx[0]] = np.expm1(pred) if log_target else pred

    preds = np.clip(preds, 0, None)
    from scipy.stats import spearmanr

    return {
        "r2": r2_score(y, preds),
        "mae": mean_absolute_error(y, preds),
        "rmse": float(np.sqrt(np.mean((y - preds) ** 2))),
        "mape": float(np.mean(np.abs((y - preds) / y.replace(0, np.nan))) * 100),
        # Rank correlation is the metric the deliverable actually depends on:
        # the map sorts stations into classes, so getting the ordering right
        # matters more than matching absolute boardings. R2 and Spearman
        # disagree here - the random forest compresses the range (poor R2) while
        # ordering stations well (good Spearman).
        "spearman": float(spearmanr(y, preds).correlation),
        "predictions": preds,
    }


def morans_i(residuals: np.ndarray, coords: np.ndarray, k: int = 6):
    """Moran's I on residuals - is there spatial structure the model missed?

    Significant autocorrelation means adjacent stations share unexplained
    demand (overlapping catchments, corridor effects) and the standard errors
    are optimistic; the fix is a spatial lag or error model.
    """
    from esda.moran import Moran
    from libpysal.weights import KNN

    weights = KNN.from_array(coords, k=k)
    weights.transform = "r"
    moran = Moran(residuals, weights)
    return moran


# --------------------------------------------------------------------------
# Prediction intervals
# --------------------------------------------------------------------------

def bootstrap_intervals(
    make_model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_predict: pd.DataFrame,
    n_boot: int = C.N_BOOTSTRAP,
    interval: float = C.PREDICTION_INTERVAL,
) -> pd.DataFrame:
    """Per-station prediction intervals by resampling the training stations.

    Reporting a single number for a station that does not exist yet implies
    precision the data cannot support, so every prediction ships with a band.
    """
    rng = np.random.default_rng(C.RANDOM_SEED)
    n = len(X_train)
    draws = np.zeros((n_boot, len(X_predict)))
    log_y = np.log1p(y_train)

    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        model = make_model()
        model.fit(X_train.iloc[idx], log_y.iloc[idx])
        draws[b] = np.expm1(model.predict(X_predict))

    draws = np.clip(draws, 0, None)
    lower = (1 - interval) / 2 * 100
    return pd.DataFrame({
        "prediction": draws.mean(axis=0),
        "lower": np.percentile(draws, lower, axis=0),
        "upper": np.percentile(draws, 100 - lower, axis=0),
    }, index=X_predict.index)


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------

def validate_transfer(phase1: pd.DataFrame, city: str = "chennai",
                      train_city: str | None = None) -> dict:
    """Check the transferred model against the prediction city's own network.

    Two levels, strongest first:

    1. **Station level** - predicted against observed boardings for each Phase 1
       station, where real station-wise data has been collected. This is the
       only test that shows whether the model gets the *distribution* right, not
       merely the total, and it is what the Phase 2 numbers rest on.

    2. **Aggregate** - the summed prediction against the observed system total.

    The benchmark must be like for like. An earlier version compared a *weekday*
    prediction against CMRL's published trailing-12-month average, which
    includes weekends and so runs ~20% lower; that made a well-calibrated model
    look 28% high. The observed weekday total from the collected station data is
    used whenever it is available.
    """
    predicted_total = phase1["prediction"].sum()
    results: dict = {"predicted_total": predicted_total}

    target_path = C.PROCESSED / f"{city}_station_target.csv"
    observed = None

    if target_path.exists():
        obs = pd.read_csv(target_path)
        merged = phase1.merge(
            obs[["station", "boardings_weekday"]],
            left_on="name", right_on="station", how="inner",
        )
        if len(merged) >= 5:
            actual = merged["boardings_weekday"]
            fitted = merged["prediction"]

            ss_res = ((actual - fitted) ** 2).sum()
            ss_tot = ((actual - actual.mean()) ** 2).sum()
            r2 = 1 - ss_res / ss_tot if ss_tot else np.nan
            mae = (actual - fitted).abs().mean()
            mape = ((actual - fitted).abs() / actual.replace(0, np.nan)).mean() * 100
            spearman = actual.corr(fitted, method="spearman")

            in_sample = train_city == city
            header = ("IN-SAMPLE fit - NOT validation" if in_sample
                      else "Transfer validation - station level")
            print(f"\n{'=' * 60}\n{header} ({city})\n{'=' * 60}")
            if in_sample:
                print("  These are the same stations the model was fitted on, so")
                print("  the figures below measure fit, not predictive skill, and")
                print("  will look far better than the model really is.")
                print("  The honest number is the leave-one-out rho above.\n")
            print(f"  stations compared  {len(merged)}")
            print(f"  R2                 {r2:.3f}")
            print(f"  MAE                {mae:,.0f} boardings/day")
            print(f"  MAPE               {mape:.0f}%")
            print(f"  Spearman rank      {spearman:.3f}")
            print("  (rank correlation matters most here: the map ranks stations)")

            worst = merged.assign(error=merged["prediction"] - merged["boardings_weekday"])
            worst = worst.reindex(worst["error"].abs().sort_values(ascending=False).index)
            cols = ["name", "boardings_weekday", "prediction", "error"]
            print("\n  largest misses:")
            print(worst.head(6)[cols].round(0).to_string(index=False))

            observed = actual.sum()
            results.update(r2=r2, mae=mae, mape=mape, spearman=spearman,
                           n_compared=len(merged))
            source = f"observed weekday boardings, {len(merged)} stations"

    if observed is None:
        # No station-wise data yet: fall back to the published system total,
        # flagging that it is an all-days average and so not directly
        # comparable with a weekday prediction.
        try:
            from src.cmrl_system import observed_daily
            observed = observed_daily(months=12)
            source = ("CMRL published series, trailing 12 months - ALL DAYS, "
                      "so ~20% below a weekday figure")
        except FileNotFoundError:
            observed = C.CHENNAI_OBSERVED_DAILY_MEAN
            source = "config estimate"
        coverage = len(phase1) / C.CHENNAI_PHASE1_STATIONS
        predicted_total = predicted_total / coverage if coverage else predicted_total

    ratio = predicted_total / observed if observed else np.nan
    print(f"\n{'=' * 60}\nTransfer calibration - system total\n{'=' * 60}")
    print(f"  predicted   {predicted_total:,.0f}/day")
    print(f"  observed    {observed:,.0f}/day  ({source})")
    print(f"  ratio       {ratio:.2f}")
    if abs(ratio - 1) <= C.CALIBRATION_TOLERANCE:
        print(f"  -> within +/-{C.CALIBRATION_TOLERANCE:.0%}: the transfer holds at "
              f"system level")
    else:
        direction = "over" if ratio > 1 else "under"
        print(f"  -> outside +/-{C.CALIBRATION_TOLERANCE:.0%}: {direction}-predicts by "
              f"{abs(ratio - 1):.0%}. Treat Phase 2 as relative rankings.")

    results.update(observed_total=observed, ratio=ratio)
    return results


def predict_city(
    make_model,
    features: list[str],
    train: pd.DataFrame,
    city: str = "chennai",
    logged: list[str] | None = None,
    train_city: str | None = None,
    save: bool = True,
) -> pd.DataFrame:
    """Apply the fitted model to another city and calibrate the transfer.

    The calibration step is the heart of the method's credibility. Chennai has
    no station-level ridership to validate against, but it does have a known
    *system* total. Predicting all Phase 1 stations and summing gives a number
    that can be checked against reality; if the sum is far off, the transfer
    does not hold and the Phase 2 numbers should not be believed.
    """
    path = C.PROCESSED / f"{city}_features.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing. Run: python -m src.features --city {city}")

    target_df = pd.read_csv(path)

    missing = [f for f in features if f not in target_df.columns]
    if missing:
        raise ValueError(
            f"{city} is missing model features {missing}. "
            "Both cities must be built through the same feature pipeline."
        )

    # Impute with *training* medians so the two cities are treated identically
    # and a gap in the Chennai data cannot shift its own baseline.
    train_medians = train[features].replace([np.inf, -np.inf], np.nan).median()
    X_train = train[features].replace([np.inf, -np.inf], np.nan).fillna(train_medians)
    y_train = train[TARGET]

    X_target = target_df[features].replace([np.inf, -np.inf], np.nan).fillna(train_medians)

    # The prediction city must go through the identical transform, using the
    # column list decided on the training data - re-deriving skew from Chennai
    # would log a different set of columns and silently score the model on
    # inputs it was never fitted to.
    if logged:
        X_train, _ = log_transform_skewed(X_train, columns=logged)
        X_target, _ = log_transform_skewed(X_target, columns=logged)

    print(f"\nPredicting {len(target_df)} {city} stations")
    intervals = bootstrap_intervals(make_model, X_train, y_train, X_target)
    out = pd.concat([target_df.reset_index(drop=True), intervals.reset_index(drop=True)], axis=1)

    # ---- Validate the transfer against Chennai's own Phase 1 ----
    phase1 = out[out["phase"] == "phase1"]
    if len(phase1):
        validate_transfer(phase1, city, train_city=train_city)

    phase2 = out[out["phase"] == "phase2"]
    if len(phase2):
        print(f"\nPhase 2: {len(phase2)} stations, "
              f"predicted total {phase2['prediction'].sum():,.0f}/day")
        top = phase2.nlargest(15, "prediction")
        cols = [c for c in ("name", "Line", "prediction", "lower", "upper", "confidence")
                if c in top.columns]
        print("\ntop 15 by predicted ridership:")
        print(top[cols].round(0).to_string(index=False))

    if save:
        out_path = C.OUT_TABLES / f"{city}_predictions.csv"
        try:
            out.to_csv(out_path, index=False)
            print(f"\n-> {out_path}")
        except PermissionError:
            # Excel holds an exclusive lock on an open CSV, and losing a full
            # model run to that is absurd. Write beside it and say so.
            alt = out_path.with_name(f"{out_path.stem}_new{out_path.suffix}")
            out.to_csv(alt, index=False)
            print(f"\n-> {alt}")
            print(f"   ({out_path.name} is locked - close it in Excel, then "
                  f"rename {alt.name} over it)")
    return out


def load_training_frame(city: str) -> pd.DataFrame:
    """Feature table for a city, with the ridership target attached.

    Bengaluru carries its target already, because BMRCL ridership is joined
    when the station layer is built. Chennai does not - CMRL publishes no
    station-wise figures - so if a target file has been produced by
    `src.chennai_target` it is merged on here.

    This is the switch between the two modes of the project: with Chennai
    ridership in hand the model trains on Chennai directly and every cross-city
    assumption disappears; without it, Bengaluru trains and the result
    transfers.
    """
    path = C.PROCESSED / f"{city}_features.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing. Run: python -m src.features --city {city}"
        )
    frame = pd.read_csv(path)

    if TARGET in frame.columns and frame[TARGET].notna().any():
        return frame

    target_path = C.PROCESSED / f"{city}_station_target.csv"
    if not target_path.exists():
        raise FileNotFoundError(
            f"{city} has no ridership target.\n"
            f"  Either train on a city that has one (--train-city bengaluru),\n"
            f"  or drop a CMRL station-wise file in {C.RAW_RIDERSHIP / city} "
            f"and run: python -m src.chennai_target"
        )

    target = pd.read_csv(target_path)
    merged = frame.merge(
        target[["station", TARGET] + [c for c in ("is_mature", "target_kind") if c in target]],
        left_on="name", right_on="station", how="left",
    )
    matched = int(merged[TARGET].notna().sum())
    print(f"joined {matched}/{len(frame)} {city} stations to observed ridership "
          f"from {target_path.name}")
    if matched == 0:
        raise ValueError(
            f"no station names matched between {path.name} and {target_path.name}"
        )
    return merged


def run(train_city: str = "chennai", target_city: str = "chennai", save: bool = True):
    # NB: the parameter is `target_city`, not `predict_city` - naming it after
    # the module-level `predict_city` function shadowed it and made the call
    # below fail with "'str' object is not callable".
    train = load_training_frame(train_city)
    train = train[train[TARGET].notna()]

    # Stations whose line opened during the observation window report ramp-up
    # ridership, not potential; see src.ridership.flag_maturity.
    if "is_mature" in train.columns:
        immature = int((~train["is_mature"].astype(bool)).sum())
        train = train[train["is_mature"].astype(bool)]
        if immature:
            print(f"excluded {immature} station(s) whose line opened mid-window")

    train = train.reset_index(drop=True)
    print(f"training on {len(train)} {train_city} stations")

    print("\nScreening features")
    y = train[TARGET]

    # Transform first, then screen: correlation and VIF must be measured in the
    # same space the model is fitted in.
    candidates = candidate_features(train)
    X_all = train[candidates].replace([np.inf, -np.inf], np.nan)
    X_all = X_all.fillna(X_all.median())
    X_all, logged = log_transform_skewed(X_all)
    if logged:
        print(f"  log1p applied to {len(logged)} skewed predictor(s): {logged}")

    features = screen_features(X_all, y)
    X = X_all[features]
    logged = [c for c in logged if c in features]

    print("\nLeave-one-out cross-validation")
    results = {}
    for label, factory in (
        ("linear_regression", linear_regression),
        ("elastic_net", elastic_net),
        ("random_forest", random_forest),
        ("gradient_boosting", gradient_boosting),
    ):
        scores = loocv(factory, X, y)
        results[label] = scores
        print(f"  {label:<18} R2={scores['r2']:>6.3f}  rho={scores['spearman']:>5.3f}  "
              f"MAE={scores['mae']:,.0f}  MAPE={scores['mape']:.0f}%")

    print("\nNegative binomial GLM (elasticities)")
    nb = fit_negative_binomial(X, y)
    # np.asarray first: nb.params is a Series indexed 0..k (the design matrix is
    # a bare ndarray), and pd.Series(series, index=features) *reindexes by
    # label* rather than positionally - integer index against string labels
    # matches nothing and silently yields an all-NaN elasticity table.
    coefs = pd.Series(np.asarray(nb.params)[1:], index=features)
    coefs = coefs.sort_values(key=abs, ascending=False)
    print(coefs.to_string())

    # Chosen on rank correlation, not R2: the map sorts stations into classes,
    # so the ordering is what has to be right. The random forest compresses the
    # range (weaker R2) while ordering stations better - which is the trade the
    # deliverable wants.
    best_rho = max(r["spearman"] for r in results.values())
    # Among models whose rank correlation is within noise of the best
    # (0.05), take the one that also fits magnitude. Selecting on rho
    # alone picked gradient boosting at rho=0.549/R2=-0.093 over a
    # random forest at rho=0.532/R2=+0.103 - a worthless trade.
    contenders = {k: v for k, v in results.items()
                  if v["spearman"] >= best_rho - 0.05}
    best_label = max(contenders, key=lambda k: contenders[k]["r2"])
    print(f"\nbest by LOOCV rank correlation: {best_label} "
          f"(rho={results[best_label]['spearman']:.3f}, "
          f"R2={results[best_label]['r2']:.3f})")

    if results[best_label]["r2"] > 0.95:
        print("  WARNING: R2 above 0.95 on ~83 rows suggests target leakage - "
              "check candidate_features()")

    print("\nSpatial autocorrelation of residuals")
    import geopandas as gpd
    stations = gpd.read_file(C.PROCESSED / f"{train_city}_stations.gpkg").to_crs(C.city_crs(train_city))
    coords = np.column_stack([stations.geometry.x, stations.geometry.y])[: len(train)]
    residuals = y.to_numpy() - results[best_label]["predictions"]
    try:
        moran = morans_i(residuals, coords)
        print(f"  Moran's I = {moran.I:.3f}, p = {moran.p_sim:.3f}")
        if moran.p_sim < 0.05:
            print("  -> residuals are spatially clustered; escalate to a spatial "
                  "lag/error model before trusting the standard errors")
        else:
            print("  -> no significant spatial structure left in the residuals")
    except Exception as exc:  # noqa: BLE001
        print(f"  skipped: {type(exc).__name__}: {exc}")

    if save:
        out = C.OUT_TABLES / "model_performance.csv"
        pd.DataFrame({
            label: {k: v for k, v in scores.items() if k != "predictions"}
            for label, scores in results.items()
        }).T.to_csv(out)
        coefs.to_csv(C.OUT_TABLES / "nb_elasticities.csv")
        print(f"\n-> {out}")

    factories = {
        "linear_regression": linear_regression,
        "elastic_net": elastic_net,
        "random_forest": random_forest,
        "gradient_boosting": gradient_boosting,
    }
    factory = factories[best_label]
    predictions = predict_city(factory, features, train, city=target_city,
                               logged=logged, train_city=train_city, save=save)

    return {"features": features, "results": results, "nb": nb,
            "train": train, "predictions": predictions}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-city", default="chennai", choices=list(C.CITIES),
                        help="Chennai by default: its demand structure differs "
                             "sharply from Bengaluru's, so the transfer cannot rank")
    parser.add_argument("--predict-city", default="chennai", choices=list(C.CITIES))
    args = parser.parse_args()
    run(args.train_city, args.predict_city)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
