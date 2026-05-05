"""tools.py — Utility helpers. Pre-fetching is done by search.py."""

import re
from typing import List, Dict


def normalize_and_dedup_results(results: List[Dict]) -> List[Dict]:
    seen, cleaned = set(), []
    for item in results:
        title = item.get("title", "").strip()
        if title and title not in seen:
            seen.add(title)
            snippet     = item.get("content", item.get("snippet", ""))[:400]
            price_match = re.search(r"₹\s*[\d,]+", str(snippet))
            cleaned.append({
                "title":      title,
                "url":        item.get("url", ""),
                "snippet":    snippet,
                "price_info": price_match.group(0) if price_match else "Check website",
            })
    return cleaned[:8]


def extract_price_inr(text: str) -> str:
    match = re.search(r"₹\s*[\d,]+", text)
    return match.group(0) if match else "Not found"