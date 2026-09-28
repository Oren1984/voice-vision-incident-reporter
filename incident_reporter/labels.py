"""The five CrisisMMD humanitarian categories (Ofli et al., 2020 agreed-label split).

The application proposes one of these as the incident category. They are the labels the
experiment was trained and evaluated on; they are not a complete incident taxonomy.
"""

LABELS: tuple[str, ...] = (
    "affected_individuals",
    "infrastructure_and_utility_damage",
    "not_humanitarian",
    "other_relevant_information",
    "rescue_volunteering_or_donation_effort",
)

DISPLAY: dict[str, str] = {
    "affected_individuals": "Affected individuals",
    "infrastructure_and_utility_damage": "Infrastructure / utility damage",
    "not_humanitarian": "Not humanitarian",
    "other_relevant_information": "Other relevant information",
    "rescue_volunteering_or_donation_effort": "Rescue, volunteering or donation",
}


def pct(x: float | None) -> str:
    """Model score as shown to people: never "100%" or "0%", which would read as certainty."""
    if x is None:
        return "—"
    x = float(x)
    return ">99%" if x > 0.99 else "<1%" if x < 0.01 else f"{100 * x:.0f}%"


# Zero-shot prompts (protocol S2). Fixed before evaluation; do not tune on test results.
ZERO_SHOT_PROMPTS: dict[str, tuple[str, ...]] = {
    "affected_individuals": (
        "people injured, displaced or stranded by a natural disaster",
        "a photo of disaster victims and evacuees",
    ),
    "infrastructure_and_utility_damage": (
        "damaged buildings, roads, bridges or power lines after a disaster",
        "a photo of destroyed houses and infrastructure damage",
    ),
    "not_humanitarian": (
        "a social media post unrelated to disaster response",
        "a photo of something not related to a disaster",
    ),
    "other_relevant_information": (
        "news, weather maps or warnings about a natural disaster",
        "a photo of a hurricane satellite map or disaster news",
    ),
    "rescue_volunteering_or_donation_effort": (
        "rescue teams, volunteers or donations helping after a disaster",
        "a photo of rescuers and relief supplies",
    ),
}
