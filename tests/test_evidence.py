"""Rule-based transcript evidence: concepts, negation, fields and missing-information questions."""

from incident_reporter.evidence import extract
from incident_reporter.text_clean import clean_post, normalize_for_wer, speakable, wer


def test_concepts_with_spans():
    t = "There is a fire on Main Street and the power lines are down."
    ev = extract(t)
    assert {"fire_smoke", "power_utility"} <= set(ev["asserted"])
    for m in ev["mentions"]:
        assert t[m["start"]:m["end"]] == m["quote"]
    assert any(m["term"] == "power lines" for m in ev["mentions"])  # longest term wins


def test_negation_is_detected_within_clause():
    ev = extract("Smoke is everywhere but no one is injured.")
    assert "fire_smoke" in ev["asserted"] and "injured_people" in ev["negated"]
    ev = extract("Nobody is hurt. A car crashed into the wall.")
    assert "vehicle_damage" in ev["asserted"]


def test_fields_location_time_people():
    ev = extract("Two people are trapped near the corner of Elm Street at 10:30 pm, water is still rising.")
    assert ev["fields"]["location"] and "Elm Street" in ev["fields"]["location"][0]["quote"]
    assert ev["fields"]["time"] and ev["fields"]["people_count"]
    assert "flood_water" in ev["asserted"]
    assert ev["missing"] == []


def test_missing_information_becomes_questions():
    ev = extract("Something bad happened, please send help.")
    assert set(ev["missing"]) == {"location", "time", "people_count", "hazard"}
    assert len(ev["questions"]) == 4


def test_text_cleaning():
    assert clean_post("RT @news: Flood in town https://t.co/x @mayor") == "Flood in town @user"
    assert speakable("#Harvey flood &amp; rain @user https://t.co/a") == "Harvey flood and rain"


def test_wer_normalisation():
    assert normalize_for_wer("Fire, on Main-Street!") == ["fire", "on", "main", "street"]
    assert wer("a fire on main street", "a fly on main street") == (1, 5)
    assert wer("", "") == (0, 0)


def test_lowercase_streets_and_spelled_times_from_asr_output():
    ev = extract("A gas leak was reported outside the bakery on station road at nine pm.")
    assert [f["quote"] for f in ev["fields"]["location"]] == ["on station road"]
    assert ev["fields"]["time"] and "nine pm" in ev["fields"]["time"][0]["quote"]
    assert extract("A building collapsed at seven fifteen this morning.")["fields"]["time"]
    # a bare "the road" is not a location
    assert extract("A tree is blocking the road and we are on the road now.")["fields"]["location"] == []


def test_implicit_people_count():
    assert extract("An older couple is stranded upstairs.")["fields"]["people_count"]
