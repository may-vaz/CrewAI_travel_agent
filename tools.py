from typing import List, Dict
import re


def normalize_and_dedup_results(results: List[Dict]) -> List[Dict]:
    """
    Deduplicate and clean Tavily search results.
    Used optionally if you want to post-process raw Tavily output
    before passing to agents. Not required for TavilySearchTool usage.
    """
    seen = set()
    cleaned = []
    for item in results:
        title = item.get("title", "").strip()
        if title and title not in seen:
            seen.add(title)
            snippet = item.get("content", item.get("snippet", ""))[:400]
            price_match = re.search(r"₹\s*[\d,]+", str(snippet))
            cleaned.append({
                "title": title,
                "url": item.get("url", ""),
                "snippet": snippet,
                "price_info": price_match.group(0) if price_match else "Check website",
            })
    return cleaned[:8]


def extract_price_inr(text: str) -> str:
    """Extract first ₹ price found in text."""
    match = re.search(r"₹\s*[\d,]+", text)
    return match.group(0) if match else "Price not found"


def format_budget_table(categories: dict) -> str:
    """Format a budget dictionary as a markdown table."""
    lines = ["| Category | Estimated Cost |", "|----------|---------------|"]
    total = 0
    for category, amount in categories.items():
        lines.append(f"| {category} | ₹{amount:,} |")
        total += amount
    lines.append(f"| **TOTAL** | **₹{total:,}** |")
    return "\n".join(lines)