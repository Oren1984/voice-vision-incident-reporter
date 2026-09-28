"""Incident concepts shared by transcript evidence and image observations.

Each concept has (a) transcript terms matched literally, with simple negation handling, and
(b) CLIP prompts for image scoring. The comparison step lines the two up per concept. These are
our own design choices for the POC; they are not taken from the papers and were not tuned on test data.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Concept:
    id: str
    label: str
    terms: tuple[str, ...]  # lower-case; multi-word allowed; matched on word boundaries
    prompts: tuple[str, ...]  # CLIP prompts; empty = not observable in an image
    hazard: bool = False


CONCEPTS: tuple[Concept, ...] = (
    Concept("fire_smoke", "Fire or smoke",
            ("fire", "fires", "burning", "burned", "smoke", "smoking", "flames", "blaze", "wildfire", "on fire"),
            ("a photo of fire and flames", "a photo of thick smoke from a fire"), hazard=True),
    Concept("flood_water", "Flood water",
            ("flood", "flooded", "flooding", "floodwater", "floodwaters", "submerged", "underwater", "inundated", "water is rising", "water is still rising", "water rising", "rising water", "high water"),
            ("a photo of flood water covering a street", "a photo of flooded houses and cars"), hazard=True),
    Concept("structural_damage", "Damaged or collapsed structure",
            ("collapsed", "collapse", "destroyed", "rubble", "cracked", "caved in", "roof", "damaged building", "damaged house", "wall came down"),
            ("a photo of a collapsed or badly damaged building", "a photo of rubble from destroyed houses")),
    Concept("debris", "Debris or fallen trees",
            ("debris", "fallen tree", "fallen trees", "tree fell", "trees down", "downed tree", "blocking the road", "road is blocked"),
            ("a photo of debris and fallen trees on a road",)),
    Concept("power_utility", "Power lines or utilities",
            ("power line", "power lines", "power is out", "power outage", "no power", "electricity", "gas leak", "transformer"),
            ("a photo of downed power lines and broken utility poles",)),
    Concept("vehicle_damage", "Vehicles involved or damaged",
            ("car", "cars", "vehicle", "vehicles", "truck", "bus", "overturned", "crashed"),
            ("a photo of a damaged or overturned car",)),
    Concept("injured_people", "Injured or trapped people",
            ("injured", "injuries", "hurt", "wounded", "bleeding", "casualties", "dead", "killed", "trapped", "unconscious"),
            ("a photo of injured people receiving first aid",)),
    Concept("displaced_people", "Affected or displaced people",
            ("evacuated", "evacuating", "evacuation", "displaced", "stranded", "homeless", "shelter", "residents", "families"),
            ("a photo of evacuees and displaced families", "a photo of a crowd of people affected by a disaster")),
    Concept("responders", "Responders or rescue activity",
            ("firefighters", "firefighter", "rescue", "rescuers", "paramedics", "ambulance", "police", "responders", "fire department", "crews", "volunteers"),
            ("a photo of firefighters or rescue workers at work", "a photo of an ambulance and emergency responders")),
    Concept("relief_supplies", "Relief supplies or donations",
            ("donations", "donate", "supplies", "food", "bottled water", "blankets", "relief"),
            ("a photo of donated relief supplies, water bottles and boxes",)),
    Concept("non_photo", "Map, screenshot or text graphic",
            (),
            ("a screenshot of text or a social media post", "a weather map or satellite image of a storm")),
)

BY_ID = {c.id: c for c in CONCEPTS}
HAZARDS = tuple(c.id for c in CONCEPTS if c.hazard)

# Background prompts: an observation's score is its probability against these ordinary scenes.
BACKGROUND_PROMPTS: tuple[str, ...] = (
    "an ordinary photo of a street",
    "an ordinary photo of a house",
    "an ordinary indoor photo",
    "an ordinary landscape photo",
    "an ordinary photo of people",
)

# Image observation bands (probability against background). "not detected" does NOT mean absent.
LIKELY = 0.60
POSSIBLE = 0.35

NEGATIONS = ("no", "not", "without", "none", "nobody", "never", "isn't", "wasn't", "aren't", "weren't", "don't", "didn't", "no one")
