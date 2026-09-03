"""Tests for station name matching.

These cover the two bugs that quietly cost 14 of Chennai's 40 Phase 1 stations
on the first pipeline run:

  1. `normalise` stripped everything outside [a-z0-9], collapsing every
     Tamil-script name to the empty string - and two empty strings compare as a
     perfect match, so Tamil names matched each other at random.
  2. `match_names` consumed candidates in input order, letting an early
     mediocre match steal a station that a later name matched exactly.

    python -m pytest tests/ -v      (or: python tests/test_matching.py)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.stations import match_names, normalise, similarity  # noqa: E402


# --------------------------------------------------------------------------
# normalise
# --------------------------------------------------------------------------

def test_normalise_strips_footnotes_and_suffixes():
    assert normalise("Washermanpet†*") == normalise("Washermanpet")
    assert normalise("Alwarpet Metro") == normalise("Alwarpet")
    assert normalise("Sholinganallur metro station") == normalise("Sholinganallur")


def test_normalise_drops_honorifics():
    assert "arignar" not in normalise("Arignar Anna Alandur")
    assert "alandur" in normalise("Arignar Anna Alandur")


def test_normalise_preserves_tamil_script():
    """The core bug: Tamil names must survive normalisation."""
    tamil = "வண்ணாரப்பேட்டை"
    assert normalise(tamil) != "", "Tamil name collapsed to empty string"


def test_normalise_handles_non_strings():
    assert normalise(None) == ""
    assert normalise(float("nan")) == ""


# --------------------------------------------------------------------------
# similarity
# --------------------------------------------------------------------------

def test_distinct_tamil_names_do_not_match():
    """Two different Tamil names must not score as identical."""
    assert similarity("வண்ணாரப்பேட்டை", "ஆயிரம் விளக்கு") < 0.8


def test_identical_tamil_names_match():
    assert similarity("கிண்டி", "கிண்டி") == 1.0


def test_empty_names_are_not_a_match():
    assert similarity("", "") == 0.0
    assert similarity("Guindy", "") == 0.0


# --------------------------------------------------------------------------
# match_names
# --------------------------------------------------------------------------

def test_exact_match_is_not_stolen_by_earlier_weaker_match():
    """The Washermanpet case.

    'New Washermanpet' is processed first and is a decent partial match for
    'Washermanpet'. Order-based greedy matching would consume it, leaving the
    exact match unmatched.
    """
    left = ["New Washermanpet", "Washermanpet"]
    right = ["Washermanpet"]

    mapping, unmatched = match_names(left, right, threshold=0.75)

    assert mapping.get("Washermanpet") == "Washermanpet"
    assert "New Washermanpet" in unmatched


def test_tamil_aliases_recover_untranslated_stations():
    """OSM maps some Chennai stations only in Tamil; aliases bridge that."""
    left = ["Guindy", "Thousand Lights"]
    right = ["கிண்டி", "ஆயிரம் விளக்கு"]
    aliases = {"Guindy": ["கிண்டி"], "Thousand Lights": ["ஆயிரம் விளக்கு"]}

    mapping, unmatched = match_names(left, right, threshold=0.75, aliases=aliases)

    assert unmatched == []
    assert mapping["Guindy"] == "கிண்டி"
    assert mapping["Thousand Lights"] == "ஆயிரம் விளக்கு"


def test_no_candidate_is_used_twice():
    left = ["Alandur", "Alandur Metro", "Ashok Nagar"]
    right = ["Alandur", "Ashok Nagar"]

    mapping, _ = match_names(left, right, threshold=0.75)

    assert len(set(mapping.values())) == len(mapping), "a candidate was reused"


def test_colliding_alias_keys_are_merged_not_overwritten():
    """Two alias keys that normalise identically must both survive.

    "Puratchi ... Central" (a manual rename) and "Puratchi ... Central¤" (the
    same row carrying a Wikipedia footnote mark, holding the Tamil alias)
    normalise to the same key. Overwriting instead of merging dropped
    "Central Metro" and left Chennai's busiest interchange unmatched.
    """
    aliases = {
        "Puratchi Thalaivar Dr. M.G. Ramachandran Central": ["Central Metro"],
        "Puratchi Thalaivar Dr. M.G. Ramachandran Central¤": ["மத்திய"],
    }
    mapping, unmatched = match_names(
        ["Puratchi Thalaivar Dr. M.G. Ramachandran Central¤"],
        ["Central Metro", "Mannadi"],
        threshold=0.80, aliases=aliases,
    )
    assert mapping.get("Puratchi Thalaivar Dr. M.G. Ramachandran Central¤") == "Central Metro"
    assert not unmatched


def test_directional_qualifiers_never_match():
    """Anna Nagar East and West are different stations on different lines."""
    assert similarity("Anna Nagar East", "Anna Nagar West") == 0.0
    assert similarity("Medavakkam North", "Medavakkam South") == 0.0


def test_ordinal_qualifiers_never_match():
    """SIPCOT I and SIPCOT II are consecutive but distinct stations."""
    assert similarity("SIPCOT I", "SIPCOT II") == 0.0
    assert similarity("Semmancheri I", "Semmancheri II") == 0.0


def test_role_qualifiers_never_match():
    assert similarity("Perambur", "Perambur Market") == 0.0
    assert similarity("Saligramam", "Saligramam Warehouse") == 0.0


def test_qualifier_guard_does_not_break_plain_names():
    """Names without qualifiers must still match normally."""
    assert similarity("Alwarpet", "Alwarpet Metro") > 0.9
    assert similarity("Sholinganallur", "Sholinganallur metro station") > 0.9


def test_confusable_stems_are_not_separable_by_string_similarity():
    """Kandanchavadi / Kumananchavadi score 0.889 - above any usable threshold.

    Chennai has many "-anchavadi" place names, and these two sit ~25 km apart
    on different corridors. Neither carries a qualifier, so the qualifier guard
    does not apply and no similarity cut-off separates them without also
    rejecting legitimate matches. This is documented here deliberately: the
    defence against this class of error is geometric, not textual - see
    `reject_off_corridor`, which discards any position far from the station's
    own line.
    """
    score = similarity("Kandanchavadi", "Kumananchavadi")
    assert score > 0.85, "if this drops, the geometric guard may be redundant"


def test_below_threshold_stays_unmatched():
    mapping, unmatched = match_names(["Guindy"], ["Koyambedu"], threshold=0.82)
    assert mapping == {}
    assert unmatched == ["Guindy"]


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            print(f"  pass  {name}")
        except AssertionError as exc:
            failures += 1
            print(f"  FAIL  {name}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"  ERROR {name}: {type(exc).__name__}: {exc}")

    print(f"\n{failures} failure(s)")
    raise SystemExit(1 if failures else 0)
