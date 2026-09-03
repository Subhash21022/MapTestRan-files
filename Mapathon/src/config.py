"""Central configuration for the Chennai Metro Phase 2 ridership potential model.

Every path, CRS and analysis constant lives here so the pipeline stays
reproducible and the methodology page can be written straight off this file.
"""

from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"

RAW_BHUVAN = RAW / "bhuvan"
RAW_BHOONIDHI = RAW / "bhoonidhi"
RAW_WORLDPOP = RAW / "worldpop"
RAW_GHSL = RAW / "ghsl"
RAW_OSM = RAW / "osm"
RAW_RIDERSHIP = RAW / "ridership"
RAW_CENSUS = RAW / "census"
RAW_SOI = RAW / "soi"
RAW_BOUNDARIES = RAW / "boundaries"

OUTPUTS = ROOT / "outputs"
OUT_MAPS = OUTPUTS / "maps"
OUT_FIGURES = OUTPUTS / "figures"
OUT_TABLES = OUTPUTS / "tables"

for _p in (
    RAW_BHUVAN, RAW_BHOONIDHI, RAW_WORLDPOP, RAW_GHSL, RAW_OSM,
    RAW_RIDERSHIP, RAW_CENSUS, RAW_SOI, RAW_BOUNDARIES,
    INTERIM, PROCESSED, OUT_MAPS, OUT_FIGURES, OUT_TABLES,
):
    _p.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# Coordinate reference systems
# --------------------------------------------------------------------------
# Geographic CRS for storage and web display.
CRS_GEO = "EPSG:4326"

# Projected CRS per city, used for every distance/area computation.
# Chennai  ~80.27E -> UTM 44N;  Bengaluru ~77.59E -> UTM 43N.
CRS_CHENNAI = "EPSG:32644"
CRS_BENGALURU = "EPSG:32643"


# --------------------------------------------------------------------------
# Catchment geometry
# --------------------------------------------------------------------------
# Core / standard / extended walk-and-feeder rings, in metres.
# 800 m is the conventional 10-minute walk used throughout the TOD literature.
CATCHMENT_CORE_M = 400
CATCHMENT_STANDARD_M = 800
CATCHMENT_EXTENDED_M = 1200

# Radius used for the network service area and therefore for PNR.
CATCHMENT_NETWORK_M = 800

# Ratio for the concave hull wrapped around reachable network nodes, passed to
# shapely.concave_hull. 0 = maximally concave (hugs the points), 1 = convex.
#
# 0.03 was far too tight: sparse peripheral catchments collapsed into spidery
# shapes covering a fraction of their real area while dense grids filled in
# normally, so hull area became a proxy for street-node density. 0.30 follows
# the reachable extent without bridging across barriers.
#
# Note this only affects PNR now - areal statistics are measured on the
# Euclidean buffer, so hull shape cannot contaminate population or POI counts.
CONCAVE_HULL_RATIO = 0.30

# Feeder-access radius for counting bus stops.
FEEDER_RADIUS_M = 500

# A station within this distance of suburban rail / MRTS is treated as
# facing modal competition (and, if same operator, as an interchange).
COMPETING_RAIL_M = 1000


# --------------------------------------------------------------------------
# Cities
# --------------------------------------------------------------------------

CITIES = {
    "chennai": {
        "name": "Chennai",
        "crs": CRS_CHENNAI,
        "osm_place": "Chennai, Tamil Nadu, India",
        # Bounding box (west, south, east, north) in EPSG:4326.
        # Deliberately wider than the GCC municipal boundary: Phase 2 runs out
        # to Poonamallee (west), Madhavaram (north) and SIPCOT Siruseri
        # (south), all of which fall outside the corporation limit.
        "bbox": (79.95, 12.75, 80.40, 13.35),
        # Parry's Corner / George Town - the historic CBD.
        "cbd": (80.2874, 13.0937),
        # Secondary activity centres used for a min-distance-to-any-centre feature.
        "secondary_centres": {
            "T. Nagar": (80.2340, 13.0418),
            "Guindy": (80.2206, 13.0067),
            "Tidel Park / OMR": (80.2483, 12.9897),
            "Ambattur": (80.1548, 13.1143),
        },
        "intermodal_hubs": [
            "Puratchi Thalaivar Dr. M.G. Ramachandran Central",
            "Chennai International Airport",
            "Guindy",
            "Egmore",
            "St. Thomas Mount",
            "Koyambedu",
            "CMBT",
            "Kilambakkam",
            "Thirumangalam"
        ],
        "role": "predict",
    },
    "bengaluru": {
        "name": "Bengaluru",
        "crs": CRS_BENGALURU,
        "osm_place": "Bengaluru, Karnataka, India",
        # Spans Namma Metro end to end: Madavara (west), Nagasandra (north),
        # Whitefield (east), Silk Institute (south).
        "bbox": (77.40, 12.75, 77.80, 13.15),
        # MG Road / Trinity - the CBD anchor of Namma Metro's Purple Line.
        "cbd": (77.6068, 12.9756),
        "secondary_centres": {
            "Majestic": (77.5713, 12.9767),
            "Whitefield": (77.7500, 12.9698),
            "Electronic City": (77.6600, 12.8452),
            "Koramangala": (77.6245, 12.9352),
        },
        "intermodal_hubs": [
            "Nadaprabhu Kempegowda Stn., Majestic",
            "KSR Bengaluru",
            "Yeshwanthpur",
            "KR Puram",
            "Baiyappanahalli",
            "Kengeri Bus Terminal",
            "Peenya Industry",
            "Silk Board"
        ],
        "role": "train",
    },
}


# --------------------------------------------------------------------------
# Chennai Metro network facts (sourced, for validation and reporting)
# --------------------------------------------------------------------------

CHENNAI_PHASE1_STATIONS = 41          # operational
CHENNAI_PHASE2_STATIONS = 128         # under construction
CHENNAI_PHASE2_KM = 118.9
CHENNAI_NETWORK_KM_COMPLETE = 173.0   # Phase 1 + Phase 2
CHENNAI_PHASE2_COST_CR = 63246        # INR crore, Cabinet approval Oct 2024

PHASE2_CORRIDORS = {
    "C3": {"name": "Madhavaram - SIPCOT", "km": 45.8, "stations": 50},
    "C4": {"name": "Lighthouse - Poonamallee Bypass", "km": 26.1, "stations": 30},
    "C5": {"name": "Madhavaram - Sholinganallur", "km": 47.0, "stations": 48},
}

# CMRL observed system-wide daily ridership, used for aggregate calibration.
# Sources: OpenCity CMRL monthly series (Apr 2023 - Jun 2026); Apr 2026 =
# 9,018,069 riders over 30 days; single-day peak 365,807 on 2 Apr 2026.
# Fallback only. src.cmrl_system computes this from the published monthly
# series instead, which gives 316,121/day on a trailing 12-month window
# (Jul 2025 - Jun 2026). The old 300,000 estimate was ~5% low.
CHENNAI_OBSERVED_DAILY_MEAN = 316_121
CHENNAI_OBSERVED_DAILY_PEAK = 365_807
# Transfer is considered to hold if predicted Phase 1 total lands within this
# fractional band of the observed mean.
CALIBRATION_TOLERANCE = 0.15


# --------------------------------------------------------------------------
# Modelling
# --------------------------------------------------------------------------

RANDOM_SEED = 42

# With ~68 usable training stations, cap the design matrix hard. This was 12
# and it demonstrably overfitted: leave-one-out R2 on the same data was 0.038
# at 12 features and 0.214 at 6. Roughly 8 observations per predictor is the
# most this sample supports.
MAX_FEATURES = 8

# Bootstrap replicates for per-station prediction intervals.
N_BOOTSTRAP = 1000
PREDICTION_INTERVAL = 0.90

# Number of station typologies for the k-means prescriptive clustering.
N_TYPOLOGIES = 5


# --------------------------------------------------------------------------
# OSM tag queries
# --------------------------------------------------------------------------

OSM_POI_CATEGORIES = {
    "education": {"amenity": ["school", "college", "university", "library"]},
    "healthcare": {"amenity": ["hospital", "clinic", "doctors"]},
    "retail": {"shop": True, "amenity": ["marketplace"]},
    "office": {"office": True},
    "government": {"amenity": ["townhall", "courthouse", "public_building"]},
    "transport": {"amenity": ["bus_station"], "public_transport": ["station"]},
    "leisure": {"leisure": ["park", "stadium", "sports_centre"]},
}

OSM_BUS_STOP_TAGS = {"highway": ["bus_stop"], "amenity": ["bus_station"]}
OSM_PARKING_TAGS = {"amenity": ["parking"]}

# Under-construction metro alignments carry construction=* rather than the
# operational railway=subway/light_rail tags.
OSM_METRO_OPERATIONAL_TAGS = {"railway": ["station", "halt"], "station": ["subway"]}
OSM_METRO_CONSTRUCTION_TAGS = {"railway": ["construction"], "construction": ["subway", "light_rail"]}


# --------------------------------------------------------------------------
# Land-use class grouping
# --------------------------------------------------------------------------
# Bhuvan LULC Level-II classes collapse into these groups for the Shannon
# entropy (land-use mix) feature. Populate CLASS_MAP once the actual raster
# legend is inspected -- codes differ between LULC-50K and SIS-DP 10K.
LULC_GROUPS = ["residential", "commercial", "industrial", "institutional", "recreation", "other"]

# Groups counted as "activity generating" for the jobs-housing balance proxy.
LULC_EMPLOYMENT_GROUPS = ["commercial", "industrial", "institutional"]


def city_crs(city: str) -> str:
    """Projected CRS for a configured city."""
    try:
        return CITIES[city]["crs"]
    except KeyError as exc:
        raise KeyError(f"Unknown city {city!r}; expected one of {list(CITIES)}") from exc
