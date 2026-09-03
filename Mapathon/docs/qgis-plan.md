# QGIS workplan — Chennai Metro Phase 2 ridership map

## The scoring reality, first

This is a **map-making competition**, not a modelling competition. On the last
published rubric:

| Component | Marks |
|---|---|
| Methodology / **ISRO data** / GIS steps / complexity | **60** |
| Potential application of the map | 40 |

Two consequences that should drive every decision from here:

1. **The map is the deliverable.** A JPG/PDF plus one page of methodology. The
   model is what makes our map worth looking at, but nobody scores the repo.
   Budget more time for cartography than feels reasonable.
2. **ISRO data is the single biggest scoring lever, and it is our biggest
   gap.** Right now the pipeline uses zero Bhuvan/Bhoonidhi data. The
   Diversity dimension of the 5D model is empty for exactly this reason. Fixing
   this raises the model *and* the marks simultaneously — it is the highest
   value work available.

## Where the project actually stands

Working and verified:

- Station layers for both cities — Chennai 133 (39 Phase 1 + 94 Phase 2)
- 400/800/1200 m buffers, network service areas, PNR
- GHS-POP 2025/2030, GHS-BUILT-S/V zonal statistics
- Training target: 83 BMRCL stations → 68 after excluding a line that opened
  mid-window
- Model: OLS / elastic net / random forest / gradient boosting, LOOCV,
  Moran's I, bootstrap intervals

Honest gaps:

- **LOOCV R² = 0.183** (linear regression, the best of the four). Low. The
  missing land-use dimension is the most likely cause.
- **No ISRO data yet** — 60% of the marks ride on this
- **27 of 94 Phase 2 station positions are unreliable** (co-located stem
  geocodes such as SIPCOT I/II)
- Chennai POI/bus layers pending an Overpass fetch

## Division of labour

A previous winning entry (Geo-Visionaries, "Heat Stroke Fatalities and Impacts")
printed its tools table directly on the poster:

| Tool | Their stated role |
|---|---|
| QGIS 3.34.10 | Linking data with the properties of map |
| qgis2web | Creation of webmap |
| GitHub | Transformation of webmap to URL |
| **Power BI** | **Creation of graphs** |
| MS Excel | Data organization |

So proprietary tools alongside FOSS GIS are accepted practice, not a
compliance risk - the requirement bites on the geospatial work, and the tools
are declared openly rather than hidden. Ours:

| Tool | Role |
|---|---|
| Python | Catchments, zonal stats, features, model, reproducibility |
| **QGIS** | ISRO imagery classification, LULC change detection, station position correction, **the map** |
| **Power BI** | **The charts**, exported as PNG into the layout |
| qgis2web + GitHub Pages | Interactive webmap, URL printed on the poster |
| Excel | Eyeballing the collected data (`src/to_excel.py`) |

Rule of thumb: anything done 134 times goes in Python; anything needing eyes on
imagery goes in QGIS; anything that is a *chart* goes in Power BI.

Run `python -m src.to_powerbi` to produce the star schema Power BI expects;
`outputs/powerbi/README.md` has the import steps and the relationships to set.

---

## The five QGIS jobs

### 1. Bhuvan LULC → the missing Diversity dimension

**Why:** `landuse_entropy` is the empty slot in the 5D specification and the
cheapest ISRO win available.

- Add Bhuvan WMS directly: *Layer → Add Layer → Add WMS/WMTS Layer*
  (`https://bhuvan-vec1.nrsc.gov.in/bhuvan/wms`) to browse first
- Download LULC 50K tiles for Chennai from Bhuvan (free registration).
  Prefer **AMRUT urban land use (1:4000)** if available for Chennai — it has
  the residential/commercial/industrial/institutional split we actually want
- Reclassify Level-II classes into the six groups already defined in
  `src/config.py: LULC_GROUPS` (*Raster → Reclassify by table*)
- *Processing → Zonal statistics* against
  `data/processed/chennai_catchments_buffer.gpkg` (filter `radius_m = 800`)
- Export the class fractions; Python computes Shannon entropy and the
  jobs-housing ratio from them

**Drop the resulting raster into `data/raw/bhuvan/` and the feature pipeline
picks it up** — `landuse_features()` already looks for it.

### 2. Bhoonidhi Resourcesat → own classification (the complexity item)

**Why:** "Complexity" is named explicitly in the rubric. Draping a downloaded
LULC layer is not complex; classifying imagery yourself is. This is the single
strongest thing we can show.

- Register at `bhoonidhi.nrsc.gov.in`, download a recent cloud-free
  **Resourcesat-2/2A LISS-III (23.5 m)** scene over Chennai — LISS-IV (5.8 m)
  if the footprint covers the corridors
- Install the **Semi-Automatic Classification Plugin (SCP)**
- Collect training ROIs → built-up / vegetation / water / open land
- Classify (Maximum Likelihood or Random Forest inside SCP)
- **Run the accuracy assessment and keep the confusion matrix + kappa.**
  Reporting classification accuracy is exactly the kind of rigour that scores
- Also compute NDBI and NDVI in the Raster Calculator as supporting evidence

### 3. LULC change detection 2005 → 2015

**Why:** Turns a static map into a story about where Chennai is actually
growing, and it uses ISRO multi-date data — two rubric boxes at once.

- Bhuvan LULC 50K exists for 2005-06, 2011-12 and 2015-16
- Raster Calculator or *Processing → Cross-tabulation* for a change matrix
- Per-catchment built-up growth rate becomes a feature **and** a headline
  finding: *"Phase 2 corridors pass through the fastest-urbanising land in the
  metropolitan area"* — if true, that is a strong claim for the application
  section

### 4. Fix the 27 unreliable station positions

**Why:** This is real science, not busywork. Stem geocoding cannot separate
"SIPCOT I" from "SIPCOT II" — both resolve to "SIPCOT" — so they currently sit
on one point with an identical catchment.

- Open `data/processed/chennai_stations.gpkg`, filter `confidence = 'low'`
- Backdrop: Bhuvan high-resolution imagery + `chennai_alignments.gpkg`
- Move each station to its actual position on the corridor, toggle editing off
- Set `confidence = 'high'` and `position_source = 'manual_bhuvan'` for the
  ones you correct

Then re-run catchments → features → model. This should measurably improve the
fit, and "stations digitised from Bhuvan imagery" is a legitimate methodology
line.

### 5. The map

Detailed below — this is where most of the remaining time goes.

---

## Map layout specification

**Use the official template**: `templates/Mapathon_Submission_Template.pptx`.
It is **A1 landscape, 841 × 594 mm**, one slide.

This changes the workflow from what we assumed. The poster is **assembled in
PowerPoint**, not in QGIS Print Layout - which is exactly why the winning entry
could list Power BI and Excel among its tools. QGIS exports the map as an
image; PowerPoint is the layout surface.

```
   QGIS  ──► map PNG (6840 × 5550 px)  ─┐
                                         ├─► template.pptx ──► PDF + JPG
   Power BI ──► chart PNGs  ────────────┘
```

### The template's fixed geometry

| Region | Position (inches) | Contents |
|---|---|---|
| Header banner | L0.4 T0.5 W22.8 | "IIT Bombay FOSSEE Geospatial Mapathon 2026 (Edition VI)" - leave as is |
| Title | L0.4 T2.2 W22.8 | your problem statement |
| **Map area** | **L0.4 T3.4 → 23.2 × 21.9** | **your map** |
| Map title | L10.5 T22.0 | caption under the map |
| Panel header | L23.3 T0.5 W9.5 | "Map description and analysis" |
| **Text panel** | **L23.4 T2.2 W8.9 H12.3** | the written sections, below |
| Team block | L23.3 T15.0 W9.3 H3.6 | Team Lead, Members 01-04, Mentor - **with Roll/Emp. No.** |
| Institute logo | L23.5 T19.7 | |
| Team name / Topic | L26.2 T19.1 / T19.9 | |
| Organization | L23.2 T21.6 | |
| District & State | L23.1 T22.4 | |

**Export the QGIS map at 22.8 × 18.5 in — 6840 × 5550 px at 300 DPI, aspect
ratio 1.23:1.** Match that ratio in the print layout or the image will be
letterboxed or cropped when placed.

### The topic is already ours

The template pre-fills **"Topic: Transport Network Map"**. A metro ridership
map sits squarely inside that, so the idea needs no bending to fit the brief.

### Required text sections

The template names these - use its headings, not the ones we invented earlier:

1. **Introduction**
2. **Problem Statement**
3. **Theme** — SDG 11.2, access to safe, affordable, sustainable transport
4. **Study Area** — Chennai Metropolitan Area; the 128 Phase 2 stations
5. **Data Source** — Bhuvan/Bhoonidhi first, then CMRL, GHSL, OSM, SoI
6. **Methodology**
7. **Description**

The panel is 8.9 × 12.3 inches - roughly 226 × 312 mm. That is a lot of space
and it is 60% of the marks; fill it properly. The winning entry also added a
**Software/Tools table** beyond the required headings, which is worth copying:
it makes the toolchain legible at a glance and is where QGIS version, Bhuvan
and Bhoonidhi get named explicitly.

### Main map (fills the map area)

Layer order, bottom to top:

1. Bhuvan LULC classified raster, 45% opacity, muted palette
2. Roads (subtle grey, thin)
3. Water bodies
4. 800 m catchments — **outline only**, no fill
5. Phase 1 lines — grey, thin (context, not subject)
6. Phase 2 corridors — coloured by line (Purple/Red/Yellow, matching CMRL)
7. Stations — the subject

**Station symbology — graduated, and honest:**

- **Size** = predicted daily ridership (5 classes, natural breaks)
- **Fill** = the same value on a sequential ramp
- **Outline style = confidence.** Solid outline for high-confidence positions,
  dashed or hollow for low. Showing uncertainty on the face of the map is
  unusual in student submissions and reads as rigour rather than weakness
- **Labels:** rule-based, top 15 by predicted ridership only. Labelling 128
  stations destroys the map

### Charts, inside the map area

The right-hand column is reserved for text, so charts go **inside the map
area** — the winning entry put a row of them along the bottom and a small
multiples grid in the middle. Build them in Power BI from
`outputs/powerbi/` (`python -m src.to_powerbi`), export PNG, place in
PowerPoint:

1. **Top 15 stations** by predicted potential, with prediction intervals as
   error bars
2. **Hourly profile** — the 08:00 peak carrying 11.9% of the day
3. **Weekday vs weekend** — ~377k against ~193k
4. **Quadrant gap analysis** — potential against feeder access, four labelled
   action types
5. **Elasticities** — what actually drives ridership

**Small multiples are worth stealing.** The winning entry ran a 2×2 grid
(2015-Male, 2015-Female, 2022-Male, 2022-Female) which made its comparison
instantly legible. Our equivalent: **Phase 1 observed | Phase 2 predicted**,
or one small map per corridor.

### The webmap — a genuine differentiator

The winning entry published an interactive webmap with **qgis2web**, hosted it
on **GitHub Pages**, and printed the URL across the top of the poster. Few
entries do this and it costs little: QGIS → *Web → qgis2web → Export to
Leaflet*, push to a `gh-pages` branch, print the link.

We already have the repo and the layers, so this is close to free.

### Mandatory furniture

North arrow · scale bar · legend · graticules on each map frame (the winning
entry had them on all six) · **CRS note (EPSG:32644 / UTM 44N)** · a
**"Data Courtesy:"** line along the bottom naming every source ·
**CC-BY-SA 4.0**.

Team details go in the template's own boxes — note it asks for **Roll/Employee
numbers** alongside names, and for **District & State**.

---

## Day plan

**Day 1 — data day (tomorrow)**

- Drop the CMRL ridership file into `data/raw/ridership/chennai/`, run
  `python -m src.chennai_target`. If it has per-station Phase 1 numbers we
  train on Chennai directly and the whole Bengaluru transfer becomes a
  robustness check rather than the backbone — a much stronger position
- Register on Bhoonidhi and start the scene download (slow; start early)
- Finish the Chennai POI/bus fetch

**Day 2 — ISRO raster work**

- Bhuvan LULC download, reclassify, zonal stats
- SCP classification + accuracy assessment

**Day 3 — corrections and re-run**

- Fix the 27 station positions in QGIS
- Re-run catchments → features → model → scenarios
- This is the point where we learn the real model performance

**Day 4 — analysis outputs**

- Prescriptive layer: quadrants, typology, scenarios, phasing
- Export all inset figures

**Days 5–6 — cartography**

- Build the print layout. Iterate. Print an A3 test and read it from a metre
  away — if the symbology is not legible at that distance it is not finished

**Day 7 — methodology page and submission**

- One page, weighted 60/40 to match the rubric: ISRO data + GIS steps +
  complexity first, application second

---

## If time runs short, cut in this order

1. Scenario simulation (nice, not essential)
2. Change detection
3. Station position corrections — but say in the methodology that low-confidence
   positions are flagged on the map
4. **Never cut:** the Bhuvan/Bhoonidhi work or the cartography. They are 60% of
   the marks and the entire deliverable respectively.

## What actually wins this

Most entries will show a thematic map of something. Ours predicts something
that does not exist yet, validates the prediction against a known system total,
and turns it into named actions for specific stations. The differentiators to
put on the page, in order:

1. **Predictive, not descriptive** — a trained model with reported error, not a
   weighted overlay
2. **Honest uncertainty** — prediction intervals and confidence-coded symbols
3. **The network-maturity correction** — centrality computed on the completed
   173 km network, because a model trained on a small system under-predicts a
   large one. Few student entries will think of this
4. **Actionable output** — "these six stations need feeder buses before
   opening" beats "here is a map of ridership potential"
