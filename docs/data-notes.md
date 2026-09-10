# Data availability notes

What is actually obtainable, what is not, and the decisions that follow.
Written as the pipeline was built, so the methodology page can be drafted from
evidence rather than recollection.

---

## 1. Ridership — the dependent variable

**Chennai station-wise ridership is not published.** CMRL releases system-wide
monthly totals only; OpenCity's dataset (Apr 2023 – Jun 2026) is system-level,
and Wikipedia's station list carries no footfall column. This is the single
constraint that shapes the whole method.

**Bengaluru's is.** BMRCL station-wise hourly entries and exits, RTI-sourced,
ODbL-licensed, published at `github.com/Vonter/bmrcl-ridership-hourly`.

| Property | Value |
|---|---|
| Stations | 83 |
| Period | 1 Aug – 30 Sep 2025 (48 of 61 calendar days present) |
| Granularity | hourly entries + exits, plus station-pair OD |
| Complete station-days after cleaning | 3,984 (= 83 × 48, no station has gaps) |
| Mean weekday boardings, system | 754,320 |
| Range across stations | 638 – 33,344 (52×) |

Partial station-days are dropped rather than filled: averaging over an
incomplete day would bias those stations downward without any visible symptom.

Sanity checks passed — Majestic (the interchange hub) tops the list at 33.3k;
MG Road, Indiranagar, Cubbon Park and Trinity follow. Majestic's boarding share
of 0.42 correctly identifies it as destination-dominant.

**Consequence:** the model is trained on Bengaluru and transferred to Chennai.
If an RTI to CMRL returns station-level data, it replaces the training table and
nothing downstream changes.

---

## 2. Station geometry

### Bengaluru — clean

`railway=station` AND `station=subway` returns **exactly 83** features, matching
the ridership file one-for-one. All 83 matched by name with zero misses.

### Chennai Phase 1 — complete but multilingual

40 operational stations per Wikipedia. The strict OSM rule returns 66 features,
because ~25 Phase 2 stations have already been mapped as though operational.

**Roughly a third of Chennai's stations are mapped in OSM with a Tamil name
only** — Guindy (கிண்டி), Saidapet (சைதாப்பேட்டை), Teynampet (தேனாம்பேட்டை),
Nandanam, Thousand Lights, LIC, AG–DMS, Government Estate, Tollgate. Matching on
English names alone recovered just 26 of 40. Supplying the Wikipedia Tamil
column as an alias fixes this.

Two bugs found and covered by tests in `tests/test_matching.py`:

- `normalise()` stripped to `[a-z0-9]`, collapsing every Tamil name to the empty
  string — and two empty strings compare as a *perfect* match, so Tamil names
  matched each other arbitrarily.
- `match_names()` consumed candidates in input order, so an early mediocre match
  ("New Washermanpet") stole the candidate that a later name matched exactly
  ("Washermanpet"). Now resolved strongest-pair-first.

### Chennai Phase 2 — the hard case

CMRL is building **128** stations. Wikipedia names **105** (82%). The remaining
23 exist only in the DPR alignment drawings and would need manual digitising.
**Decision: model the 105 and state the gap, rather than pad the list.**

Positions are assembled from four sources, each recorded per station in
`position_source` with a `confidence` rating:

| Source | Count | Confidence | Accuracy |
|---|---|---|---|
| OSM (mapped, some tagged as operational) | ~23 | high | surveyed |
| Wikipedia article coordinates | ~18 | high | surveyed |
| Nominatim geocoding, snapped to corridor | majority | medium | ~100–300 m |
| Nominatim geocoding, unsnapped | remainder | low | ~300 m – 1 km |

Wikipedia article coordinates cover only 17% of Phase 2 names on their own.
Nominatim fails outright on numbered stations (SIPCOT I/II, Semmancheri I/II,
Sholinganallur Lake I/II); these are retried on a simplified stem.

### Corridor snapping

Phase 2 alignments **are** in OSM — tagged `railway=subway` with "(u/c)" in the
name, not `railway=construction`:

| Corridor | OSM name | Length in OSM |
|---|---|---|
| C3 Madhavaram–SIPCOT (Red) | `Line 3: Madhavaram → SIPCOT (u/c)` | 69.3 km |
| C5 Madhavaram–Sholinganallur (Purple) | `Line 5: Madhavaram → Shozhinganallur (u/c)` | 112.1 km |
| C4 Lighthouse–Poonamallee (Yellow) | `Yellow Line (Elevated / Underground)` | 47.9 + 30.9 km |

Lengths exceed route length because both directions and both tracks are mapped.

Geocoded points are projected onto their own corridor, which removes the error
component perpendicular to the line — most of it — and corrects a systematic
bias: a locality centroid sits in the middle of a settlement, so unsnapped
catchments pull towards settlement cores and **overstate** their population.
Points more than 2.5 km from their corridor are left unsnapped and stay flagged
low-confidence; that far off is a geocoding failure, not a small error.

---

## 2b. Defects found while building (all fixed, most silent)

None of these raised an error. Each produced plausible-looking output that was
wrong, which is the reason they are all listed rather than quietly patched.

| Defect | Symptom | Fix |
|---|---|---|
| `normalise()` stripped to `[a-z0-9]` | Every Tamil name became `""`, and two empty strings compare as a *perfect* match, so Tamil-named stations matched each other at random | Unicode-aware `\w`; empty names score 0 |
| `match_names` consumed candidates in list order | "New Washermanpet" stole the node "Washermanpet" matched exactly | Resolve strongest pair first, one use per candidate |
| **Red/Purple line mapping inverted** | Stations snapped to the *wrong corridor*: median offset 1,218 m, 49 stations >2.5 km from their line | Verified empirically: Purple sits 3 m from Line 3, Red 80 m from Line 5. Offset fell to 190 m |
| `MultiPoint.concave_hull()` does not exist in shapely 2.x | `except Exception: pass` swallowed the `AttributeError` for **every** station, silently substituting convex hulls, which bridge across rail lines and water and erase the severance signal PNR exists to measure | Module-level `shapely.concave_hull`; fallbacks counted and reported |
| Areal stats measured on the network catchment | Hull shape depends on street-node density, so "population" partly measured "how finely the streets are mapped" — Baiyappanahalli returned 143 people. Distribution was visibly bimodal | Measure density on the fixed 800 m buffer; PNR enters separately as its own predictor |
| VIF computed without an intercept | VIFs mostly measured distance of means from zero; **every** informative predictor was pruned | Prepend a constant column |
| Correlation pruning dropped both members of a chain | `dist_cbd_km` (strongest predictor, r = −0.50) dropped against `metro_closeness`, which was then itself dropped | Prune one pair per round, weaker member only, recomputing each time |
| `pd.Series(nb.params[1:], index=features)` | Reindexes *by label*; integer index against string labels matched nothing → an all-NaN elasticity table | `np.asarray(...)` first |
| `poi_total` alongside its components | It is their exact sum: poi_total −0.29 fighting poi_office +0.26 | Components excluded from the model, retained for typology |
| `predict_city` parameter shadowed the function | `TypeError: 'str' object is not callable` | Parameter renamed `target_city` |
| Fuzzy matching ignored qualifiers | Kandanchavadi (Purple, OMR) matched Kumananchavadi's node (Yellow, Poonamallee) — 25 km away, at 0.889 similarity. Anna Nagar East/West likewise | Qualifier guard, plus a **geometric** backstop rejecting any position >3 km from its own corridor |
| Co-location tested by exact coordinates | Elcot and Medavakkam II geocoded 43 m apart — different coordinates, identical catchment | Proximity test at 250 m |

### The one that mattered most: training on a line that had just opened

Fifteen of the 83 BMRCL stations — Bengaluru's **Yellow Line, opened 10 August
2025** — fall inside the 1 Aug – 30 Sep window. They show 38–42 active days
against 48, and were still ramping steeply (Huskur Road +50%, Ragigudda +40%).

Their catchments look like strong performers; their recorded ridership does
not. Central Silk Board has the catchment of a major interchange and recorded
4,316 boardings, so the model was being taught that dense, well-connected
catchments produce *low* ridership. It predicted 21,905 there, and erred by 5x
in the opposite direction at Majestic.

They are now detected from the data (first active date, active-day coverage,
ramp ratio) rather than a hardcoded list, and excluded from training: **n = 68**.
This is the same network-maturity bias the method corrects for in Chennai,
turning up inside the training set itself.

---

## 3. Infrastructure notes

**Overpass throttles hard, and the reason is not obvious.** Asked for their
status, the mirrors answer plainly:

> "Please include a meaningful User-Agent string with your requests to avoid
> rate-limiting."

Without one they return **HTTP 429** in about a second. OSMnx sends its own
default UA, which is what gets throttled — so the Chennai POI and bus-stop
fetches stalled for tens of minutes with no error ever surfacing.
`setup_osmnx()` now sets `ox.settings.http_user_agent` explicitly. With a UA
the requests are accepted; the mirrors may still return 504 when saturated,
which is a separate and transient problem.

`src/acquire.py` also rotates through three mirrors, and distinguishes "server
answered, nothing matched" from a transport failure — conflating the two
produces empty layers that look like genuine absence of data.

**Large bbox queries are refused; tiled ones are not.** A walk network or an
unfiltered `shop=True` query over a whole metropolitan bbox never returned for
Chennai. The same area in a 3x4 grid of tiles goes through, composing on OSM
node ids (graphs) or de-duplicating on element id (features). Both fetchers
fall back to tiling automatically.

**Place-name queries are wrong for this study area.** Phase 2 extends well
beyond the Greater Chennai Corporation boundary (Poonamallee, Madhavaram,
SIPCOT Siruseri), so all queries use an explicit bbox of
`(79.95, 12.75, 80.40, 13.35)`.

**pandas 3.0** removed literal HTML strings in `read_html`; the Wikipedia
scraper wraps the response in `io.StringIO`.

---

## 4. Still to acquire

| Dataset | Purpose | Status |
|---|---|---|
| Bhuvan LULC 50K / SIS-DP 10K / AMRUT | Land-use entropy, employment ratio | manual download from Bhuvan |
| Bhoonidhi Resourcesat LISS-III/IV | Own built-up classification (rubric weight) | manual download |
| GHS-POP R2023 100 m (2025 + 2030) | Catchment population, future scenarios | tile download |
| Census 2011 ward + household assets | Vehicle ownership, official baseline | censusindia |
| GCC 200 ward boundaries | Census join geometry | OpenCity, ODbL |
| CMDA Master Plan zoning | Upzoning scenario | CMDA |
| Survey of India OpenSeries | Mandated base map | SoI portal |

`src/features.py` computes what it can and reports what it skipped, so the
OSM-derived features can be built and checked before these land.

---

## 5. Honest limitations to carry into the methodology page

1. **83 training stations.** Feature set capped at 12, screened by correlation
   and VIF, validated leave-one-out, reported with bootstrap intervals.
2. **Cross-city transfer.** Bengaluru and Chennai differ in income, bus
   competition and network maturity. Validated by predicting Chennai Phase 1 and
   comparing the total against CMRL's observed system ridership.
3. **Network maturity bias.** A model trained on a small system under-predicts a
   large one. Centrality is therefore computed on the *completed 173 km* graph,
   not today's.
4. **105 of 128 stations.** The 23 unnamed ones are absent, not estimated.
5. **Position uncertainty** varies by station and is carried explicitly through
   to the map rather than averaged away.
6. **Scenarios are associational.** A cross-sectional model shows what
   correlates with ridership, not what causes it. Scenario deltas are indicative
   magnitudes, not guarantees.
