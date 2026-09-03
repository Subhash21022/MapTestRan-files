"""Build the canonical station layer for all three station sets.

Three sets, three different data situations:

  Bengaluru (83, training)
      OSM tags them cleanly. `railway=station` + `station=subway` returns
      exactly the 83 stations that appear in the BMRCL ridership file, so the
      join is a straight name match.

  Chennai Phase 1 (40, validation)
      All present in OSM, but with duplicate nodes per station - separate
      Tamil-script entries, stop positions, and platform ways - which have to
      be collapsed before use.

  Chennai Phase 2 (105 named, 128 built)
      The hard case. OSM has only a fraction, Wikipedia's article coordinates
      cover 18, and the rest need geocoding. Every station therefore carries a
      `position_source` and `confidence` column, and low-confidence positions
      are reported rather than quietly averaged into the results.

    python -m src.stations --city bengaluru
    python -m src.stations --city chennai
"""

from __future__ import annotations

import argparse
import json
import re
import time
import unicodedata
from difflib import SequenceMatcher

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from shapely.geometry import Point

from src import config as C

WIKI_API = "https://en.wikipedia.org/w/api.php"
NOMINATIM = "https://nominatim.openstreetmap.org/search"
USER_AGENT = "mapathon-chennai-metro/0.1 (IITB FOSSEE Mapathon student project)"

# Confidence bands, used for both weighting and honest map symbology.
CONF_HIGH = "high"        # surveyed position: OSM or a Wikipedia article
CONF_MEDIUM = "medium"    # geocoded then snapped to the corridor alignment
CONF_LOW = "low"          # geocoded locality centroid, unsnapped

# Stations officially renamed after political figures, which OSM still carries
# under the short working name. Fuzzy matching cannot bridge these - the
# strings share almost nothing - so they are stated explicitly.
MANUAL_ALIASES = {
    "Puratchi Thalaivar Dr. M.G. Ramachandran Central": ["Central Metro", "Chennai Central"],
    "Puratchi Thalaivi Dr. J. Jayalalithaa CMBT": ["CMBT", "Koyambedu"],
    "Arignar Anna Alandur": ["Alandur"],
}


# --------------------------------------------------------------------------
# Name handling
# --------------------------------------------------------------------------

# Wikipedia marks interchange and transfer stations with a variety of symbols.
# ¤ was missing and survived into the canonical station names, so downstream
# lookups keyed on the clean name ("...Ramachandran Central") failed to find
# the stored one ("...Ramachandran Central¤").
FOOTNOTES = re.compile(r"[*†‡§¶#¤°·]+")
PARENS = re.compile(r"\([^)]*\)")
# "chennai" is noise in a list of Chennai stations - external sources often
# write "Chennai Egmore Metro" where our canonical name is just "Egmore".
# No station is named solely "Chennai", so dropping it is safe.
NOISE_WORDS = ("metro station", "metro", "station", "railway station", "chennai")


def normalise(name: str) -> str:
    """Reduce a station name to a comparable key.

    Handles the footnote daggers in Wikipedia tables, the "Metro" suffixes
    mappers add in OSM, honorific prefixes, and punctuation/spacing drift.

    Non-Latin scripts are preserved: many Chennai stations are mapped in OSM
    only in Tamil, and stripping to [a-z0-9] would collapse every Tamil name to
    the empty string - which then compares as identical to every other one.
    """
    if not isinstance(name, str):
        return ""
    text = unicodedata.normalize("NFKD", name)
    text = FOOTNOTES.sub("", text)
    text = PARENS.sub(" ", text)
    text = text.lower()
    for word in NOISE_WORDS:
        text = text.replace(word, " ")
    text = re.sub(r"\b(dr|sri|shri|smt|mahatma|nadaprabhu|arignar)\b", " ", text)
    # \w is Unicode-aware, so this drops punctuation without touching Tamil.
    text = re.sub(r"[^\w\s]+", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


# Tokens that are the *entire* distinction between two stations, even though
# they are a tiny fraction of the string. "Anna Nagar East" vs "Anna Nagar
# West" scores about 0.93 on raw character similarity, and "SIPCOT I" vs
# "SIPCOT II" higher still - high enough to match each other and place a
# station on the wrong corridor.
QUALIFIERS = {
    "east", "west", "north", "south", "central",
    "i", "ii", "iii", "iv", "v", "1", "2", "3", "4", "5",
    "depot", "market", "junction", "warehouse", "lake", "bypass", "extension",
}


def _qualifiers(name: str) -> frozenset[str]:
    return frozenset(t for t in normalise(name).split() if t in QUALIFIERS)


def similarity(a: str, b: str) -> float:
    """Normalised-name similarity, 0-1.

    Two names that both normalise to nothing are not a match - without this
    guard SequenceMatcher scores a pair of empty strings as a perfect 1.0.

    Names carrying different qualifiers never match, however similar the rest
    of the string. This is what stopped Kandanchavadi (Purple Line, on OMR)
    being matched to Kumananchavadi's OSM node (Yellow Line, near Poonamallee)
    some 25 km away.
    """
    left, right = normalise(a), normalise(b)
    if not left or not right:
        return 0.0
    if _qualifiers(a) != _qualifiers(b):
        return 0.0
    return SequenceMatcher(None, left, right).ratio()


def match_names(
    left: list[str],
    right: list[str],
    threshold: float = 0.82,
    aliases: dict[str, list[str]] | None = None,
) -> tuple[dict[str, str], list[str]]:
    """Best-match between two name lists, strongest pairs first.

    Returns (mapping from left->right, unmatched left names).

    Pairs are consumed in descending order of similarity rather than in list
    order. That matters: matching greedily down the input list lets an early,
    mediocre match steal a candidate that a later name matches exactly - which
    is what silently cost 'Washermanpet' its perfect match on the first run.

    `aliases` supplies extra spellings for a left-hand name, which is how the
    Tamil-script station names are matched: many Chennai stations are mapped in
    OSM only in Tamil, so the Wikipedia Tamil column is passed in here.
    """
    # Alias keys are matched on their normalised form, so a footnote mark on
    # the source name ("...Central¤") cannot cause a silent lookup miss.
    #
    # Merge on collision rather than overwrite. Two entries routinely normalise
    # to the same key - the manual rename keyed "Puratchi ... Central" and the
    # Tamil alias keyed "Puratchi ... Central¤" - and a dict comprehension kept
    # only the last, dropping "Central Metro" and leaving Chennai's busiest
    # interchange unmatched.
    alias_lookup: dict[str, list[str]] = {}
    for key, values in (aliases or {}).items():
        alias_lookup.setdefault(normalise(key), []).extend(values)

    scored = []
    for name in left:
        variants = [name] + alias_lookup.get(normalise(name), [])
        for candidate in right:
            score = max(similarity(v, candidate) for v in variants)
            if score >= threshold:
                scored.append((score, name, candidate))

    scored.sort(reverse=True, key=lambda t: t[0])

    mapping: dict[str, str] = {}
    used_right: set[str] = set()
    for score, name, candidate in scored:
        if name in mapping or candidate in used_right:
            continue
        mapping[name] = candidate
        used_right.add(candidate)

    unmatched = [n for n in left if n not in mapping]
    return mapping, unmatched


# --------------------------------------------------------------------------
# Cleaning the OSM dump
# --------------------------------------------------------------------------

def _dedupe_by_proximity(
    gdf: gpd.GeoDataFrame,
    tolerance_m: float = 150,
) -> gpd.GeoDataFrame:
    """Collapse multiple OSM nodes describing the same station into one point.

    A single station commonly appears as an English-named node, a Tamil-named
    node, one or more `stop=subway` stopping positions and a platform way.
    Clustering on proximity and keeping the longest Latin-script name is more
    robust than trying to reason about OSM's tagging.
    """
    if gdf.empty:
        return gdf

    # Buffer-dissolve-explode gives connected clusters without a clustering dep.
    clusters = gdf.geometry.buffer(tolerance_m / 2).union_all()
    clusters = gpd.GeoSeries(
        [clusters] if clusters.geom_type == "Polygon" else list(clusters.geoms),
        crs=gdf.crs,
    ).explode(index_parts=False).reset_index(drop=True)
    cluster_gdf = gpd.GeoDataFrame({"cluster": range(len(clusters))}, geometry=clusters, crs=gdf.crs)

    joined = gpd.sjoin(gdf, cluster_gdf, how="left", predicate="within")

    def pick(group: pd.DataFrame) -> pd.Series:
        latin = group[group["name"].fillna("").str.contains(r"[A-Za-z]", regex=True)]
        pool = latin if len(latin) else group
        row = pool.loc[pool["name"].fillna("").str.len().idxmax()].copy()
        # Cluster centroid is a steadier position than any single node.
        row["geometry"] = group.geometry.union_all().centroid
        row["n_osm_nodes"] = len(group)
        return row

    out = joined.groupby("cluster", dropna=True).apply(pick, include_groups=False)
    out = gpd.GeoDataFrame(out.reset_index(drop=True), geometry="geometry", crs=gdf.crs)
    return out


def load_osm_metro(city: str, operational: bool = True) -> gpd.GeoDataFrame:
    """Clean metro-station points from the raw OSM dump.

    The strict rule (`railway=station` AND `station=subway`) is what isolates
    metro stations from suburban rail, bus stations and stopping positions.
    """
    path = C.RAW_OSM / f"{city}_stations_raw.gpkg"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing. Run: python -m src.acquire --city {city} --what stations"
        )

    gdf = gpd.read_file(path)
    if operational:
        mask = (gdf.get("railway") == "station") & (gdf.get("station") == "subway")
    else:
        mask = gdf["category"].isin(["metro_construction"])
    subset = gdf[mask].copy()

    if "name" not in subset.columns:
        subset["name"] = None

    deduped = _dedupe_by_proximity(subset)
    print(f"[{city}] OSM metro ({'operational' if operational else 'construction'}): "
          f"{len(subset)} raw -> {len(deduped)} after dedupe")
    return deduped


# --------------------------------------------------------------------------
# External position lookups
# --------------------------------------------------------------------------

def wikipedia_coordinates(names: list[str], suffix: str = " metro station") -> dict[str, tuple[float, float]]:
    """Coordinates from Wikipedia articles, where the station has one.

    Covers only a minority of Phase 2 stations but those it does cover are
    accurate, so it is tried before geocoding.
    """
    titles = [f"{n}{suffix}" for n in names]
    found: dict[str, tuple[float, float]] = {}

    for i in range(0, len(titles), 40):
        batch = titles[i:i + 40]
        resp = requests.get(WIKI_API, headers={"User-Agent": USER_AGENT}, timeout=60, params={
            "action": "query", "prop": "coordinates", "titles": "|".join(batch),
            "format": "json", "redirects": 1, "colimit": "max",
        })
        resp.raise_for_status()
        for page in resp.json().get("query", {}).get("pages", {}).values():
            if "coordinates" in page:
                coord = page["coordinates"][0]
                stem = page["title"].replace(suffix, "").strip()
                found[stem] = (coord["lon"], coord["lat"])
        time.sleep(0.2)

    print(f"  Wikipedia coordinates: {len(found)}/{len(names)}")
    return found


def _geocode_cache_path(city: str):
    return C.RAW_BOUNDARIES / f"{city}_geocode_cache.json"


def _load_geocode_cache(city: str) -> dict:
    path = _geocode_cache_path(city)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def geocode(
    name: str,
    city: str,
    bbox: tuple[float, float, float, float],
    cache: dict | None = None,
) -> tuple[float, float] | None:
    """Nominatim lookup, constrained to the city bounding box.

    Phase 2 station names are almost all existing locality names, so this
    usually lands within a few hundred metres - but it returns a locality
    centroid, not a station, hence the medium/low confidence rating.

    Results are cached on disk, including misses: Nominatim asks for no more
    than one request per second, so a full rebuild otherwise costs two minutes
    of waiting to fetch answers we already had.
    """
    if cache is not None and name in cache:
        hit = cache[name]
        return tuple(hit) if hit else None

    west, south, east, north = bbox
    result = None
    try:
        resp = requests.get(NOMINATIM, headers={"User-Agent": USER_AGENT}, timeout=30, params={
            "q": f"{name}, {C.CITIES[city]['name']}, India",
            "format": "json", "limit": 1,
            "viewbox": f"{west},{north},{east},{south}", "bounded": 1,
        })
        resp.raise_for_status()
        results = resp.json()
        if results:
            result = (float(results[0]["lon"]), float(results[0]["lat"]))
    except Exception as exc:  # noqa: BLE001
        print(f"    geocode failed for {name!r}: {type(exc).__name__}")
        return None  # transient: do not cache a network failure as a miss

    if cache is not None:
        cache[name] = list(result) if result else None
    return result


# --------------------------------------------------------------------------
# Corridor snapping
# --------------------------------------------------------------------------

# Wikipedia names Phase 2 corridors by colour; OSM names them by line number.
#
# Red and Purple do NOT map to Line 3 and Line 5 in the order the names
# suggest. Verified against the high-confidence stations by
# `verify_line_mapping()`: Purple sits 3 m from Line 3 and 4,427 m from Line 5;
# Red sits 80 m from Line 5 and 2,649 m from Line 3. Getting this backwards
# snapped stations onto the wrong corridor with a ~1.2 km median error and
# produced plausible-looking but meaningless catchments.
#
#   Purple Line = Corridor 3, Madhavaram - SIPCOT
#   Red Line    = Corridor 5, Madhavaram - Sholinganallur
#   Yellow Line = Corridor 4, Lighthouse - Poonamallee
#
# Patterns are colon-anchored because a bare "Line 5" also matches
# "Line 5B: Koyambedu -> Pattabiram", a separate proposed branch.
LINE_TO_ALIGNMENT = {
    "Purple Line": ["Line 3:"],
    "Red Line": ["Line 5:"],
    "Yellow Line": ["Yellow Line ("],
    # Wikipedia renders one interchange row with both colours concatenated.
    "Red LinePurple Line": ["Line 3:", "Line 5:"],
    "Blue Line": ["Chennai Metro Line 1"],
    "Green Line": ["Chennai Metro Line 2"],
}

# Alignments that exist in OSM but are not part of this project: proposed
# extensions, depot leads and test tracks.
EXCLUDE_ALIGNMENTS = re.compile(r"proposed|test track|switch", re.IGNORECASE)

# Beyond this, the geocoded point is too far from its corridor to be a
# plausible position for that station, and snapping would invent a location.
MAX_SNAP_M = 2500

# Consecutive metro stations sit roughly a kilometre apart. Two "stations"
# closer than this are one positioning error, not a tight pair.
MIN_STATION_SPACING_M = 250


def load_alignments(city: str) -> gpd.GeoDataFrame:
    path = C.RAW_OSM / f"{city}_alignments.gpkg"
    if not path.exists():
        print(f"  alignments missing ({path.name}); run "
              f"src.acquire --city {city} --what alignments")
        return gpd.GeoDataFrame(geometry=[], crs=C.city_crs(city))
    return gpd.read_file(path)


# OSM labels Namma Metro's alignments by construction reach, not by operating
# line, which would split three lines into fifteen fragments and leave the
# centrality graph disconnected. Reaches consolidate as follows (station counts
# from the assignment confirm it: 36 + 32 + 15 = 83):
#
#   Purple Line, Challaghatta - Whitefield : Reach 1*, Reach 2*, UG1  (36)
#   Green Line,  Madavara - Silk Institute : Reach 3*, Reach 4*, UG2  (32)
#   Yellow Line, RV Road - Bommasandra     : already named            (15)
REACH_TO_LINE = {
    "bengaluru": [
        (re.compile(r"Reach\s*1|Reach\s*2|UG1", re.IGNORECASE), "Purple Line"),
        (re.compile(r"Reach\s*3|Reach\s*4|UG2", re.IGNORECASE), "Green Line"),
        (re.compile(r"Yellow", re.IGNORECASE), "Yellow Line"),
        (re.compile(r"Pink", re.IGNORECASE), "Pink Line"),
    ],
}


def consolidate_lines(stations: gpd.GeoDataFrame, city: str,
                      line_col: str = "Line") -> gpd.GeoDataFrame:
    """Collapse construction-reach labels into operating lines."""
    rules = REACH_TO_LINE.get(city)
    if not rules or line_col not in stations.columns:
        return stations

    def to_line(value):
        if not isinstance(value, str):
            return value
        for pattern, line in rules:
            if pattern.search(value):
                return line
        return value

    out = stations.copy()
    out[line_col] = out[line_col].map(to_line)
    print(f"  consolidated reaches into operating lines:")
    print(out[line_col].value_counts(dropna=False).to_string())
    return out


def assign_lines_from_alignments(
    stations: gpd.GeoDataFrame,
    alignments: gpd.GeoDataFrame,
    max_dist_m: float = 400,
) -> gpd.GeoDataFrame:
    """Label each station with the corridor it sits on, from the OSM linework.

    Bengaluru's stations carry no line attribute in OSM, but the network
    centrality feature needs to know which stations are consecutive on which
    corridor. Assigning each station to its nearest alignment recovers that
    without hand-coding the network.

    Route relations are mapped in both directions, so the direction suffix is
    stripped and the two halves collapse to one line.
    """
    if alignments.empty:
        print("  no alignments - cannot assign lines")
        return stations

    usable = alignments[~alignments["name"].str.contains(EXCLUDE_ALIGNMENTS, na=False)].copy()
    if usable.empty:
        return stations

    # "Purple Line: A -> B" and "Purple Line: B -> A" are the same corridor.
    usable["line_label"] = (
        usable["name"]
        .str.split(":").str[0]
        .str.replace(r"\s*\((u/c|Elevated|Underground\s*\d*)\)\s*", "", regex=True)
        .str.strip()
    )

    out = stations.to_crs(usable.crs).copy()
    merged = usable.dissolve(by="line_label")

    labels, distances = [], []
    for geom in out.geometry:
        dists = merged.geometry.distance(geom)
        nearest = dists.idxmin()
        labels.append(nearest if dists.min() <= max_dist_m else None)
        distances.append(dists.min())

    out["Line"] = labels
    out["line_dist_m"] = distances

    assigned = out["Line"].notna().sum()
    print(f"  assigned lines to {assigned}/{len(out)} stations "
          f"(median distance {np.median(distances):.0f} m)")
    print(out["Line"].value_counts(dropna=False).to_string())
    return out.to_crs(stations.crs)


def reject_off_corridor(
    positions: dict,
    line_of: dict[str, str],
    alignments: gpd.GeoDataFrame,
    crs: str,
    max_offset_m: float = 3000,
) -> list[str]:
    """Discard positions that are implausibly far from the station's own line.

    The geometric backstop against name-matching errors. Kandanchavadi and
    Kumananchavadi score 0.889 on string similarity - close enough to swap -
    but they sit on different corridors 25 km apart, so the wrong one is
    immediately obvious once measured against its own alignment.

    Returns the names whose positions were rejected, so the caller can fall
    back to geocoding them.
    """
    if alignments.empty or not positions:
        return []

    usable = alignments[~alignments["name"].str.contains(EXCLUDE_ALIGNMENTS, na=False)]
    merged = {}
    for line, patterns in LINE_TO_ALIGNMENT.items():
        pattern = "|".join(re.escape(p) for p in patterns)
        sel = usable[usable["name"].str.contains(pattern, case=False, na=False, regex=True)]
        if not sel.empty:
            merged[line] = sel.geometry.union_all()

    rejected = []
    for name, ((lon, lat), source, _conf) in list(positions.items()):
        line = line_of.get(name)
        if line not in merged:
            continue
        point = gpd.GeoSeries([Point(lon, lat)], crs=C.CRS_GEO).to_crs(crs).iloc[0]
        offset = point.distance(merged[line])
        if offset > max_offset_m:
            print(f"    rejected {name!r} ({source}): {offset / 1000:.1f} km from "
                  f"the {line} alignment")
            positions.pop(name)
            rejected.append(name)

    return rejected


def verify_line_mapping(
    stations: gpd.GeoDataFrame,
    alignments: gpd.GeoDataFrame,
    line_col: str = "Line",
    warn_above_m: float = 500,
) -> pd.DataFrame:
    """Check LINE_TO_ALIGNMENT against the trustworthy station positions.

    Only high-confidence stations are used, since their positions come from
    OSM or Wikipedia rather than geocoding. If a line's configured alignment is
    not also its nearest one, the mapping is wrong and snapping would move
    stations onto a different corridor - which is exactly what happened when
    Red and Purple were transposed.
    """
    trusted = stations[stations["confidence"] == CONF_HIGH]
    if trusted.empty or alignments.empty:
        return pd.DataFrame()

    usable = alignments[~alignments["name"].str.contains(EXCLUDE_ALIGNMENTS, na=False)]
    merged = {}
    for line, patterns in LINE_TO_ALIGNMENT.items():
        pattern = "|".join(re.escape(p) for p in patterns)
        sel = usable[usable["name"].str.contains(pattern, case=False, na=False, regex=True)]
        if not sel.empty:
            merged[line] = sel.geometry.union_all()

    rows = []
    for line, group in trusted.groupby(line_col):
        if line not in merged:
            continue
        distances = {other: group.geometry.distance(geom).median()
                     for other, geom in merged.items()}
        nearest = min(distances, key=distances.get)
        rows.append({
            "line": line, "n": len(group),
            "configured_m": distances[line],
            "nearest_line": nearest, "nearest_m": distances[nearest],
            "ok": nearest == line or distances[line] <= warn_above_m,
        })

    report = pd.DataFrame(rows)
    if report.empty:
        return report

    print("\n  line/alignment check (median distance, high-confidence stations):")
    print(report.round(0).to_string(index=False))
    bad = report[~report["ok"]]
    if len(bad):
        print("  WARNING: these lines are closer to a different alignment than the "
              "one configured in LINE_TO_ALIGNMENT:")
        for _, row in bad.iterrows():
            print(f"    {row['line']}: configured {row['configured_m']:.0f} m vs "
                  f"{row['nearest_line']} at {row['nearest_m']:.0f} m")
    return report


def snap_to_corridor(
    stations: gpd.GeoDataFrame,
    alignments: gpd.GeoDataFrame,
    line_col: str = "Line",
) -> gpd.GeoDataFrame:
    """Project each station onto the linework of its own corridor.

    A geocoded locality centroid sits wherever the settlement's centre is; the
    station sits on the alignment. Snapping removes the component of that error
    perpendicular to the line, which is most of it, and also corrects the
    systematic bias whereby centroids pull catchments towards settlement cores
    and overstate their population.

    Only low-confidence positions are moved: surveyed OSM and Wikipedia
    positions are already on the alignment and are left untouched.
    """
    if alignments.empty or line_col not in stations.columns:
        print("  snapping skipped (no alignments or no line column)")
        return stations

    out = stations.to_crs(alignments.crs).copy()
    out["snap_offset_m"] = np.nan

    usable = alignments[~alignments["name"].str.contains(EXCLUDE_ALIGNMENTS, na=False)]
    dropped = len(alignments) - len(usable)
    if dropped:
        print(f"  excluded {dropped} proposed/test-track alignment features")

    for line, patterns in LINE_TO_ALIGNMENT.items():
        mask = (out[line_col] == line) & (out["confidence"] == CONF_LOW)
        if not mask.any():
            continue

        pattern = "|".join(re.escape(p) for p in patterns)
        corridor = usable[usable["name"].str.contains(pattern, case=False, na=False, regex=True)]
        if corridor.empty:
            print(f"  {line}: no matching alignment for {patterns} - not snapped")
            continue
        print(f"  {line}: snapping against {len(corridor)} feature(s), "
              f"{corridor['length_km'].sum():.0f} km")

        merged = corridor.geometry.union_all()
        for idx in out.index[mask]:
            point = out.at[idx, "geometry"]
            offset = point.distance(merged)
            if offset > MAX_SNAP_M:
                # Leave it, and leave it flagged low - a point this far from
                # its corridor is a geocoding failure, not a small error.
                continue
            out.at[idx, "geometry"] = merged.interpolate(merged.project(point))
            out.at[idx, "snap_offset_m"] = offset
            out.at[idx, "confidence"] = CONF_MEDIUM

    snapped = out["snap_offset_m"].notna()
    if snapped.any():
        print(f"  snapped {int(snapped.sum())} stations to their corridor "
              f"(median offset {out.loc[snapped, 'snap_offset_m'].median():.0f} m, "
              f"max {out.loc[snapped, 'snap_offset_m'].max():.0f} m)")
    unsnapped = (out["confidence"] == CONF_LOW).sum()
    if unsnapped:
        print(f"  {unsnapped} still low-confidence (>{MAX_SNAP_M} m from corridor "
              f"or no alignment)")
    return out.to_crs(stations.crs)


# --------------------------------------------------------------------------
# Bengaluru: the training set
# --------------------------------------------------------------------------

def build_bengaluru(save: bool = True) -> gpd.GeoDataFrame:
    """83 operational Namma Metro stations joined to their observed ridership."""
    city = "bengaluru"
    stations = load_osm_metro(city, operational=True)

    target_path = C.PROCESSED / "bmrcl_station_target.csv"
    if not target_path.exists():
        raise FileNotFoundError(f"{target_path} missing. Run: python -m src.ridership")
    target = pd.read_csv(target_path)

    osm_names = stations["name"].fillna("").tolist()
    mapping, unmatched = match_names(target["station"].tolist(), osm_names)

    print(f"[{city}] matched {len(mapping)}/{len(target)} ridership stations to OSM")
    if unmatched:
        print(f"[{city}] UNMATCHED ridership stations ({len(unmatched)}):")
        for name in unmatched:
            best = max(osm_names, key=lambda c: similarity(name, c)) if osm_names else ""
            print(f"    {name!r} -> closest OSM {best!r} ({similarity(name, best):.2f})")

    target["osm_name"] = target["station"].map(mapping)
    merged = stations.merge(
        target, left_on="name", right_on="osm_name", how="inner", suffixes=("_osm", "")
    )

    merged["station_id"] = [f"BLR{i:03d}" for i in range(len(merged))]
    merged["city"] = city
    merged["phase"] = "operational"
    merged["position_source"] = "osm"
    merged["confidence"] = CONF_HIGH

    # OSM carries no line attribute on Bengaluru's stations, but the network
    # centrality feature needs corridor membership, so derive it from the
    # alignment linework exactly as Chennai does.
    print(f"[{city}] assigning corridors from alignments")
    merged = assign_lines_from_alignments(merged, load_alignments(city))
    merged = consolidate_lines(merged, city)

    print(f"[{city}] final training layer: {len(merged)} stations with ridership")

    if save:
        out = C.PROCESSED / f"{city}_stations.gpkg"
        merged.to_file(out, driver="GPKG")
        print(f"[{city}] -> {out}")
    return merged


# --------------------------------------------------------------------------
# Chennai
# --------------------------------------------------------------------------

def _wiki_table(filename: str, name_col: str) -> pd.DataFrame:
    path = C.RAW_BOUNDARIES / filename
    if not path.exists():
        raise FileNotFoundError(f"{path} missing. Run: python -m src.acquire --what wiki")
    table = pd.read_csv(path)
    # The under-construction table repeats its header as the first data row.
    table = table[table.iloc[:, 0].astype(str) != "Sr no."]
    table = table.rename(columns={name_col: "name"})
    table["name"] = table["name"].astype(str).str.replace(FOOTNOTES, "", regex=True).str.strip()
    return table


def build_chennai(save: bool = True) -> gpd.GeoDataFrame:
    """Phase 1 (validation) and Phase 2 (prediction) stations, with provenance."""
    city = "chennai"
    crs = C.city_crs(city)
    bbox = C.CITIES[city]["bbox"]

    # ---- Phase 1: OSM has them all, Wikipedia gives the canonical list ----
    print(f"\n[{city}] Phase 1")
    p1_wiki = _wiki_table("wiki_chennai_operational.csv", "Station name in English")
    p1_names = sorted(p1_wiki["name"].unique())
    print(f"  Wikipedia lists {len(p1_names)} operational stations")

    # Around a third of Chennai's stations carry only a Tamil name in OSM, so
    # the Wikipedia Tamil column is supplied as an alias for each station.
    tamil_col = next((c for c in p1_wiki.columns if "Tamil" in str(c)), None)
    aliases: dict[str, list[str]] = {k: list(v) for k, v in MANUAL_ALIASES.items()}
    if tamil_col:
        for _, row in p1_wiki.iterrows():
            tamil = row.get(tamil_col)
            if isinstance(tamil, str) and tamil.strip():
                aliases.setdefault(row["name"], []).append(tamil.strip())
        print(f"  using Tamil aliases for {len(aliases)} stations "
              f"({len(MANUAL_ALIASES)} manual renames)")

    osm_all = load_osm_metro(city, operational=True)
    osm_names = osm_all["name"].fillna("").tolist()
    mapping, unmatched = match_names(p1_names, osm_names, threshold=0.75, aliases=aliases)
    print(f"  matched {len(mapping)}/{len(p1_names)} to OSM")
    for name in unmatched:
        variants = [name] + alias_lookup.get(normalise(name), [])
        best, score = "", 0.0
        for cand in osm_names:
            s = max(similarity(v, cand) for v in variants)
            if s > score:
                best, score = cand, s
        print(f"    unmatched: {name!r} -> closest {best!r} ({score:.2f})")

    p1 = osm_all[osm_all["name"].isin(mapping.values())].copy()
    inverse = {v: k for k, v in mapping.items()}
    p1["name"] = p1["name"].map(inverse)
    p1["phase"] = "phase1"
    p1["position_source"] = "osm"
    p1["confidence"] = CONF_HIGH
    p1 = p1.merge(
        p1_wiki[["name", "Line", "Layout"]].drop_duplicates("name"), on="name", how="left"
    )

    # ---- Phase 2: assemble positions from whatever is available ----
    print(f"\n[{city}] Phase 2")
    p2_wiki = _wiki_table("wiki_chennai_under_construction.csv", "Station name")
    p2_wiki = p2_wiki.drop_duplicates("name")
    p2_names = sorted(p2_wiki["name"].unique())
    print(f"  Wikipedia lists {len(p2_names)} of the {C.CHENNAI_PHASE2_STATIONS} built stations "
          f"({100 * len(p2_names) / C.CHENNAI_PHASE2_STATIONS:.0f}%)")

    positions: dict[str, tuple[tuple[float, float], str, str]] = {}

    # 1. OSM construction-tagged stations, plus operational-tagged Phase 2
    #    stations that mappers have already added.
    osm_constr = load_osm_metro(city, operational=False)
    osm_pool = pd.concat([osm_all, osm_constr], ignore_index=True)
    osm_pool = gpd.GeoDataFrame(osm_pool, geometry="geometry", crs=osm_all.crs).to_crs(C.CRS_GEO)

    # Nodes already claimed by a Phase 1 station are off the table: without
    # this, "Anna Nagar West" (Phase 2) matched the "Anna Nagar East" node.
    claimed = set(mapping.values())
    available = osm_pool[~osm_pool["name"].isin(claimed)]
    pool_names = available["name"].fillna("").tolist()

    # match_names enforces one station per node, strongest pair first; the old
    # per-name argmax let several Phase 2 stations share a single OSM node.
    p2_mapping, _ = match_names(p2_names, pool_names, threshold=0.88)
    for name, osm_name in p2_mapping.items():
        geom = available.loc[available["name"] == osm_name].geometry.iloc[0]
        positions[name] = ((geom.x, geom.y), "osm", CONF_HIGH)
    print(f"  from OSM: {len(positions)}")

    # Geometric check on those matches before trusting them.
    alignments = load_alignments(city)
    line_of = dict(zip(p2_wiki["name"], p2_wiki["Line"]))
    rejected = reject_off_corridor(positions, line_of, alignments, crs)
    if rejected:
        print(f"  {len(rejected)} OSM match(es) rejected as off-corridor; "
              f"they fall through to geocoding")

    # 2. Wikipedia article coordinates for the rest.
    todo = [n for n in p2_names if n not in positions]
    for name, lonlat in wikipedia_coordinates(todo).items():
        if name in todo:
            positions[name] = (lonlat, "wikipedia", CONF_HIGH)

    # 3. Geocode whatever remains. Nominatim asks for <=1 request/second.
    cache = _load_geocode_cache(city)
    cached_before = len(cache)
    todo = [n for n in p2_names if n not in positions]
    fresh = [n for n in todo if n not in cache]
    print(f"  geocoding remaining {len(todo)} "
          f"({len(cache)} cached, {len(fresh)} to fetch, ~{len(fresh)}s)...")
    for name in todo:
        was_cached = name in cache
        lonlat = geocode(name, city, bbox, cache)
        if lonlat:
            positions[name] = (lonlat, "nominatim", CONF_LOW)
        if not was_cached:
            time.sleep(1.1)

    # 4. Retry the failures on a simplified name. Numbered and qualified
    #    stations ("SIPCOT I", "Semmancheri II", "Sholinganallur Lake I")
    #    are not places Nominatim knows, but their stems are.
    todo = [n for n in p2_names if n not in positions]
    if todo:
        print(f"  retrying {len(todo)} with simplified names...")
        for name in todo:
            stem = re.sub(r"\s+(I{1,3}|IV|V|\d+)$", "", name).strip()
            stem = re.sub(r"\s+(Depot|Market|Junction|Warehouse|Lake)$", "", stem).strip()
            if stem == name or not stem:
                continue
            was_cached = stem in cache
            lonlat = geocode(stem, city, bbox, cache)
            if lonlat:
                # Stem-geocoded: the right neighbourhood, not the right point.
                positions[name] = (lonlat, "nominatim_stem", CONF_LOW)
                print(f"    {name!r} via {stem!r}")
            if not was_cached:
                time.sleep(1.1)

    if len(cache) != cached_before:
        _geocode_cache_path(city).write_text(json.dumps(cache, indent=2), encoding="utf-8")
        print(f"  geocode cache: {len(cache)} entries")

    missing = [n for n in p2_names if n not in positions]
    print(f"  positioned {len(positions)}/{len(p2_names)}; {len(missing)} still missing")
    if missing:
        print(f"  MISSING: {missing}")

    rows = []
    for name, ((lon, lat), source, conf) in positions.items():
        rows.append({
            "name": name, "geometry": Point(lon, lat),
            "position_source": source, "confidence": conf,
        })
    p2 = gpd.GeoDataFrame(rows, geometry="geometry", crs=C.CRS_GEO).to_crs(crs)
    p2["phase"] = "phase2"
    p2 = p2.merge(p2_wiki[["name", "Line", "Layout"]], on="name", how="left")

    # 5. Pull the geocoded points onto their actual corridor.
    print("\n[chennai] snapping approximate positions to corridor alignments")
    verify_line_mapping(p2, alignments)
    p2 = snap_to_corridor(p2, alignments)

    # 6. Flag co-located stations. Stem geocoding cannot separate "SIPCOT I"
    #    from "SIPCOT II" - both resolve to "SIPCOT" - so such pairs sit on
    #    one point and share an identical catchment. They stay in the model,
    #    but as low confidence, and are listed rather than left to be found
    #    later as a pair of suspiciously identical rows.
    #    Tested by proximity, not exact equality: Elcot and Medavakkam II
    #    geocoded to points 43 m apart - different coordinates, but the same
    #    catchment for every practical purpose. Real metro stations sit about
    #    a kilometre apart, so anything under MIN_STATION_SPACING_M is a
    #    positioning failure rather than a tight pair.
    too_close = set()
    for i, geom in zip(p2.index, p2.geometry):
        distances = p2.geometry.distance(geom)
        near = distances[(distances < MIN_STATION_SPACING_M) & (distances.index != i)]
        if len(near):
            too_close.add(i)
            too_close.update(near.index)

    if too_close:
        idx = sorted(too_close)
        p2.loc[idx, "confidence"] = CONF_LOW
        print(f"  {len(idx)} station(s) within {MIN_STATION_SPACING_M} m of another "
              f"- positions unreliable, marked low confidence:")
        for _, row in p2.loc[idx].sort_values("name").iterrows():
            print(f"    {row['name']} ({row['Line']}, via {row['position_source']})")

    # ---- Combine ----
    combined = pd.concat([p1.to_crs(crs), p2], ignore_index=True)
    combined = gpd.GeoDataFrame(combined, geometry="geometry", crs=crs)
    combined["city"] = city
    combined["station_id"] = [f"MAA{i:03d}" for i in range(len(combined))]

    print(f"\n[{city}] combined layer: {len(combined)} stations")
    print(combined.groupby(["phase", "confidence"]).size().to_string())
    if "Line" in combined.columns:
        print(f"\nby line:\n{combined['Line'].value_counts(dropna=False).to_string()}")

    if save:
        out = C.PROCESSED / f"{city}_stations.gpkg"
        combined.to_file(out, driver="GPKG")
        print(f"\n[{city}] -> {out}")
    return combined


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--city", action="append", choices=list(C.CITIES), default=None)
    args = parser.parse_args()

    for city in (args.city or list(C.CITIES)):
        print(f"\n{'=' * 60}\n{C.CITIES[city]['name']}\n{'=' * 60}")
        if city == "bengaluru":
            build_bengaluru()
        else:
            build_chennai()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
