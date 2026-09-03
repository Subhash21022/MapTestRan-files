# Chennai Metro Phase 2 — Station Catchment & Ridership Potential

Predicting ridership potential for the 128 under-construction stations of Chennai
Metro Phase 2 from population density, land use and road connectivity in each
station's walkable catchment — and turning those predictions into planning
actions (feeder-bus priorities, TOD upzoning candidates, phasing sequence).

Submission for the **IIT Bombay FOSSEE Geospatial Mapathon 2026 (Edition VI)**.
Aligned to **UN SDG 11.2** — access to safe, affordable and sustainable transport.

Built entirely with free/libre open source software (QGIS, Python, GDAL) and
Indian satellite data (ISRO/NRSC Bhuvan and Bhoonidhi).

---

## The problem this solves

Chennai Metro Phase 2 adds 118.9 km and 128 stations across three corridors,
at a cost of ₹63,246 crore, taking the network to 173 km. Decisions about
feeder buses, station-area land use and opening sequence are being made now,
for stations that do not yet exist and therefore have no ridership history.

Phase 1 has under-performed its forecasts. A station-level, data-driven estimate
of *where the riders actually are* — and what would have to change for them to
use the metro — is worth more than a corridor-level projection.

## Method in one paragraph

A **transfer-learned direct demand model**. Chennai publishes no station-wise
ridership, so the model is trained where observed station-level ridership does
exist (Bengaluru's BMRCL network, RTI-sourced and openly licensed), using
features built identically for both cities under the **5D framework**
(Density, Diversity, Design, Destination accessibility, Distance to transit).
The fitted model is then applied to Chennai's Phase 2 stations. Because a model
trained on a small network would systematically under-predict a large one,
network centrality is computed on the **completed 173 km future graph**, not
today's. Transfer is validated by predicting all 41 Phase 1 stations and
comparing the total against CMRL's observed system-wide ridership.

## Repository layout

```
data/raw/          Source data, by provider (not version controlled)
data/interim/      Intermediate artefacts
data/processed/    Analysis-ready layers (stations, catchments, features)
src/               Pipeline modules
  config.py        All paths, CRS, constants
  acquire.py       OSM / ridership / boundary downloads
  catchment.py     Buffers, network service areas, PNR
  features.py      5D feature extraction
  model.py         GLM / elastic-net / GBM, validation, intervals
  scenarios.py     Prescriptive layer
notebooks/         Exploratory, run in order 01 -> 04
qgis/              QGIS project + print layouts
outputs/           Maps, figures, tables
docs/              Methodology page, data licences
```

## Data sources

### Ridership (dependent variable)

| Source | Use | Licence |
|---|---|---|
| [BMRCL station-wise hourly ridership](https://github.com/Vonter/bmrcl-ridership-hourly) | Training target (~70 stations) | ODbL-1.0 |
| [CMRL monthly ridership, OpenCity](https://data.opencity.in/dataset/chennai-metro-monthly-usage-data) | Aggregate calibration (system-wide, Apr 2023–Jun 2026) | Public Domain |
| [CMRL Phase 2 DPR](https://chennaimetrorail.org/wp-content/uploads/2025/07/Project-DPR-for-Chennai-Metro-Rail-Phase-II.pdf) | Official projections — benchmark | Public document |

Chennai station-wise ridership is **not** published. An RTI request to CMRL is
the route to it; if it arrives, it replaces Bengaluru as the training target and
the rest of the pipeline is unchanged.

### Satellite & land use (ISRO / NRSC)

| Source | Use |
|---|---|
| [Bhuvan LULC 50K](https://bhuvan.nrsc.gov.in/wiki/index.php/Thematic_Data) (2005-06, 2011-12, 2015-16) | Land-use classes; multi-date change detection |
| Bhuvan SIS-DP LULC 10K (Resourcesat-2 LISS-IV, 5.8 m) | Urban-grade land-use detail |
| Bhuvan AMRUT urban land use (1:4000) | Residential/commercial/industrial split for mix entropy |
| [Bhoonidhi](https://bhoonidhi.nrsc.gov.in) — Resourcesat-2/2A LISS-III/IV, Cartosat | Own supervised built-up classification (NDBI/NDVI) |

### Population & boundaries

| Source | Use |
|---|---|
| GHS-POP R2023 (100 m, incl. 2025/2030 projections) | Catchment population, future-year scenarios |
| WorldPop constrained UN-adjusted (100 m) | Cross-check |
| Census of India 2011 (ward/town, household assets) | Vehicle ownership, official baseline |
| [GCC 200 ward boundaries](https://data.opencity.in/dataset/gcc-ward-information) | Census join geometry (ODbL) |
| Survey of India OpenSeries | Mandated base map |

### Network & activity

OpenStreetMap (walk/drive graphs, POIs, bus stops, metro alignments) via OSMnx;
Microsoft/Google building footprints; GHS-BUILT-S/V; VIIRS nighttime lights;
CMDA Master Plan land-use zoning for the upzoning scenario.

## Running the pipeline

```bash
python -m pip install -r requirements.txt
```

```bash
python -m src.acquire --city chennai --city bengaluru
```

Then run the notebooks in order: `01_catchments`, `02_features`, `03_model`,
`04_scenarios`. Final cartography is assembled in `qgis/chennai_ridership.qgz`.

## Honest limitations

- Training on ~70 stations; the model is regularised and reported with
  bootstrap prediction intervals rather than point estimates.
- Cross-city transfer assumes the density–ridership relationship is portable
  between Bengaluru and Chennai. Income, bus competition and network maturity
  differ; sensitivity analysis quantifies the exposure.
- Phase 2 station coordinates are reconciled across OSM, Wikipedia and the DPR;
  low-confidence positions are flagged on the map rather than hidden.
- Census 2011 is dated; GHS-POP 2025/2030 carries the density estimates.

## Licence

Analysis, code and map outputs released under **CC-BY-SA 4.0**, per Mapathon
rules. Input datasets remain under their own licences, listed above.
