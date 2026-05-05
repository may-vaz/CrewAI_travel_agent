"""
validators.py — Sanity checks on the LLM output before displaying.

Catches absurd prices (₹35 flight, ₹5 hotel) that happen when the LLM
hallucinates or misreads search snippets. Flags them with a warning
rather than silently showing garbage data.
"""

import re


# Reasonable minimums in INR for India domestic travel
MINIMUMS = {
    "flight_one_way":    800,    # cheapest domestic flight one-way
    "hotel_per_night":   300,    # cheapest budget room per night
    "activity_per_person": 0,    # free is valid (beaches, temples etc.)
    "meal_per_person":   50,     # cheapest meal
}

MAXIMUMS = {
    "flight_one_way":    80_000,  # business class long-haul domestic
    "hotel_per_night":   50_000,  # luxury suite
}


def _find_prices(text: str) -> list[int]:
    """Extract all ₹ prices from text as integers."""
    prices = []
    for m in re.finditer(r"₹\s*([\d,]+)", text):
        try:
            prices.append(int(m.group(1).replace(",", "")))
        except ValueError:
            pass
    return prices


def check_flight_prices(flight_section: str) -> list[str]:
    """Return list of warning strings if flight prices look wrong."""
    warnings = []
    prices = _find_prices(flight_section)
    for p in prices:
        if 0 < p < MINIMUMS["flight_one_way"]:
            warnings.append(
                f"⚠️ Suspicious flight price detected: ₹{p:,}. "
                "This is likely a search result error (e.g. a per-km rate or typo). "
                "Please verify on MakeMyTrip or Cleartrip before booking."
            )
        if p > MAXIMUMS["flight_one_way"]:
            warnings.append(
                f"⚠️ Very high flight price detected: ₹{p:,}. "
                "Please verify this is a return fare for all passengers."
            )
    return warnings


def check_hotel_prices(hotel_section: str) -> list[str]:
    warnings = []
    prices = _find_prices(hotel_section)
    for p in prices:
        if 0 < p < MINIMUMS["hotel_per_night"]:
            warnings.append(
                f"⚠️ Suspicious hotel price: ₹{p:,}/night. "
                "This seems too low — please verify on Booking.com or MakeMyTrip."
            )
    return warnings


def validate_output(full_output: str) -> list[str]:
    """
    Run all sanity checks on the final itinerary text.
    Returns a list of warning strings (empty = all good).
    """
    warnings = []

    # Split into sections roughly
    lines_lower = full_output.lower()

    flight_section = ""
    hotel_section  = ""

    # Extract flight and hotel sections for targeted checks
    if "your flight" in lines_lower:
        start = lines_lower.find("your flight")
        end   = lines_lower.find("your hotel", start) if "your hotel" in lines_lower else start + 500
        flight_section = full_output[start:end]

    if "your hotel" in lines_lower:
        start = lines_lower.find("your hotel")
        end   = lines_lower.find("day-by-day", start) if "day-by-day" in lines_lower else start + 500
        hotel_section = full_output[start:end]

    warnings.extend(check_flight_prices(flight_section))
    warnings.extend(check_hotel_prices(hotel_section))

    # Check if output looks like JSON (model failure)
    stripped = full_output.strip()
    if stripped.startswith("{") or stripped.startswith("["):
        warnings.append(
            "🔴 The model returned JSON instead of readable text. "
            "Switch to llama3.1:8b: `ollama pull llama3.1:8b`"
        )

    return warnings