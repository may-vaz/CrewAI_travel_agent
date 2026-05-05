"""
search.py — Parallel Tavily pre-fetch layer.

All searches run simultaneously via ThreadPoolExecutor BEFORE any agent starts.
This is the single biggest speed win: agents read pre-fetched text instead of
looping through tool-call round-trips.
"""

import os
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from tavily import TavilyClient
from dotenv import load_dotenv

load_dotenv()


def rooms_needed(group_size: int) -> int:
    """2 adults per room, round up."""
    return math.ceil(group_size / 2)


def _search(client: TavilyClient, query: str, max_results: int = 4) -> list[dict]:
    try:
        resp = client.search(
            query=query,
            search_depth="advanced",
            max_results=max_results,
            include_answer=True,
        )
        out = []
        if resp.get("answer"):
            out.append({"title": "Summary", "content": resp["answer"], "url": ""})
        for r in resp.get("results", [])[:max_results]:
            out.append({
                "title":   r.get("title", ""),
                "content": r.get("content", "")[:400],
                "url":     r.get("url", ""),
            })
        return out
    except Exception as e:
        return [{"title": "Search failed", "content": str(e), "url": ""}]


def _fmt(results: list[dict]) -> str:
    lines = []
    for r in results:
        if r["title"]:
            lines.append(f"• {r['title']}")
        if r["content"]:
            lines.append(f"  {r['content'][:350]}")
        if r["url"]:
            lines.append(f"  URL: {r['url']}")
        lines.append("")
    return "\n".join(lines).strip()


def fetch_all_research(
    origin: str,
    destination: str,
    dates: str,
    budget: str,
    style: str,
    interests: str,
    duration: int,
    group_size: int,
) -> dict[str, str]:
    """
    Run all Tavily searches in parallel. Returns a dict of domain → text.
    Also includes metadata keys used by agents for pricing calculations.
    """
    client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    rooms  = rooms_needed(group_size)

    searches = {
        "flights_rt": (
            f"{origin} to {destination} round trip flight {dates} "
            f"{group_size} passenger price INR cheapest", 5
        ),
        "flights_book": (
            f"MakeMyTrip Cleartrip {origin} {destination} return flight booking {dates}", 4
        ),
        "hotels": (
            f"best {style.lower()} hotel {destination} {rooms} room "
            f"{group_size} guest price per night INR book 2025", 5
        ),
        "hotels2": (
            f"top rated hotel {destination} budget INR booking price per night", 4
        ),
        "activities": (
            f"top things to do {destination} {style.lower()} entry fee per person INR 2025", 6
        ),
        "interests": (
            f"{interests} {destination} experience cost INR", 4
        ),
        "restaurants": (
            f"best local restaurants {destination} {style.lower()} must visit", 5
        ),
        "transport": (
            f"local transport {destination} tourist cab auto bike rental fare INR per day", 4
        ),
        "tips": (
            f"{destination} travel tips {dates} what to know visitor guide", 3
        ),
    }

    raw: dict[str, list] = {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = {ex.submit(_search, client, q, n): k for k, (q, n) in searches.items()}
        for f in as_completed(futures):
            raw[futures[f]] = f.result()

    return {
        "flights":     _fmt(raw.get("flights_rt", []) + raw.get("flights_book", [])),
        "hotels":      _fmt(raw.get("hotels", []) + raw.get("hotels2", [])),
        "activities":  _fmt(raw.get("activities", []) + raw.get("interests", [])),
        "restaurants": _fmt(raw.get("restaurants", [])),
        "transport":   _fmt(raw.get("transport", [])),
        "tips":        _fmt(raw.get("tips", [])),
        # Metadata — passed through to agents for accurate pricing
        "group_size":  str(group_size),
        "rooms":       str(rooms),
        "duration":    str(duration),
        "budget":      budget,
        "origin":      origin,
        "destination": destination,
        "dates":       dates,
    }