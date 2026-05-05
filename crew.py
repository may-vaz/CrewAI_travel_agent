"""
crew.py — Optimised multi-agent travel planner

Speed vs previous version:
  - max_tokens cut: research agents 400-600 (was 700-900), itinerary 2000 (was 2800)
  - Prompts shortened — less text for LLM to process before it starts writing
  - max_iter kept at 2 — pre-fetched data means no looping needed
  - temperature 0.2 — more deterministic = less rambling = faster

Cost display rules (enforced in every prompt):
  - Show TOTAL price only, with per-person in brackets: ₹300 (₹150/person)
  - NO multiplication strings like "₹150 × 2 = ₹300"
  - Summary table: one row per category, total only, no working shown
"""

import os
from crewai import Agent, Task, Crew, Process, LLM
from dotenv import load_dotenv

load_dotenv()

if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = "dummy-not-used"

MODEL    = os.environ.get("OLLAMA_MODEL",    "ollama/llama3.1:8b")
BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")


def make_llm(max_tokens: int) -> LLM:
    return LLM(
        model=MODEL,
        base_url=BASE_URL,
        temperature=0.2,
        max_tokens=max_tokens,
    )


# ── Cost formatting rule injected into every agent ────────────────────────
COST_RULE = (
    "COST FORMAT RULE (mandatory): "
    "Always show the TOTAL price first, then per-person in brackets. "
    "Example: ₹300 (₹150/person). "
    "NEVER write multiplication like '₹150 × 2 = ₹300'. Just write ₹300 (₹150/person)."
)

# ── AGENTS ────────────────────────────────────────────────────────────────

flight_agent = Agent(
    role="Flight Analyst",
    goal="Extract 2 round-trip flight options from search data with correct group pricing.",
    backstory=(
        "You read flight search results and extract real options. "
        "You always calculate: total = one-way fare × 2 × number of people. "
        f"{COST_RULE} "
        "No JSON, no code. Clean markdown only."
    ),
    llm=make_llm(500),
    verbose=False,
    allow_delegation=False,
    max_iter=2,
)

hotel_agent = Agent(
    role="Hotel Analyst",
    goal="Extract 3 real hotels from search data with correct room and night pricing.",
    backstory=(
        "You read hotel search results and find real options for groups. "
        "Total = per-room-per-night × rooms × nights. "
        f"{COST_RULE} "
        "No JSON, no code. Clean markdown only."
    ),
    llm=make_llm(600),
    verbose=False,
    allow_delegation=False,
    max_iter=2,
)

activity_agent = Agent(
    role="Activities Analyst",
    goal="Extract real attractions from search data, organised by day, with group-total entry fees.",
    backstory=(
        "You read search results and extract named attractions with practical details. "
        "Total entry fee = per-person × group size. "
        f"{COST_RULE} "
        "No JSON, no code. Clean markdown only."
    ),
    llm=make_llm(700),
    verbose=False,
    allow_delegation=False,
    max_iter=2,
)

logistics_agent = Agent(
    role="Food & Transport Analyst",
    goal=(
        "Extract named restaurants for each meal slot and the best local transport option. "
        "For restaurants: name and area only — never suggest dishes. "
        "Show group-total food costs."
    ),
    backstory=(
        "You read restaurant and transport search results. "
        "For restaurants: only the name, area, and group-total cost — NO dish suggestions. "
        f"{COST_RULE} "
        "No JSON, no code. Clean markdown only."
    ),
    llm=make_llm(500),
    verbose=False,
    allow_delegation=False,
    max_iter=2,
)

itinerary_agent = Agent(
    role="Master Itinerary Planner",
    goal=(
        "Write the complete day-by-day itinerary using all research. "
        "Show only total prices (per-person in brackets). "
        "End with a clean cost-summary table — one row per category, no arithmetic shown."
    ),
    backstory=(
        "You are a master travel writer. You use only real places from the research. "
        "You write beautiful, specific itineraries. "
        f"{COST_RULE} "
        "For the final cost table: one row per category, total only, no multiplication shown. "
        "For restaurants: name and area only — never suggest dishes. "
        "No JSON, no code."
    ),
    llm=make_llm(2000),
    verbose=True,
    allow_delegation=False,
    max_iter=2,
)


# ── CREW FACTORY ──────────────────────────────────────────────────────────
def create_crew(user_input: str, research: dict) -> Crew:

    g  = research["group_size"]   # "2"
    r  = research["rooms"]        # "1"
    d  = research["duration"]     # "3"
    b  = research["budget"]
    o  = research["origin"]
    ds = research["destination"]
    dt = research["dates"]

    # Shared cost format reminder for task descriptions
    cf = f"Show costs as: ₹TOTAL (₹X/person). Never write multiplication. Group size = {g}."

    # ── Task 1: Flights ───────────────────────────────────────────────────
    task_flights = Task(
        description=(
            f"Trip: {user_input}\n"
            f"Route: {o} → {ds} (return) | {g} people | Dates: {dt}\n\n"
            f"SEARCH DATA:\n{research['flights']}\n\n"
            f"Extract 2 round-trip flight options. {cf}\n"
            f"Total flight cost = one-way fare × 2 (return) × {g} people.\n\n"
            "### ✈️ Flight Option 1\n"
            f"- **Airline**: \n- **Route**: {o} ↔ {ds}\n"
            "- **One-way fare**: ₹X (per person)\n"
            "- **Return total**: ₹TOTAL (₹X/person return)\n"
            f"- **Group total ({g} people)**: ₹TOTAL\n"
            "- **Book at**: URL\n\n"
            "### ✈️ Flight Option 2\n(same format)\n\n"
            "### 💡 Booking Tip: (one sentence)\n\n"
            "If real prices not in data, state the best search URL. No invented prices."
        ),
        expected_output=f"2 round-trip flight options with group total for {g} people. Clean markdown.",
        agent=flight_agent,
    )

    # ── Task 2: Hotels ────────────────────────────────────────────────────
    task_hotels = Task(
        description=(
            f"Trip: {user_input}\n"
            f"Destination: {ds} | {g} people | {r} room(s) | {d} nights\n\n"
            f"SEARCH DATA:\n{research['hotels']}\n\n"
            f"Extract 3 real hotels. {cf}\n"
            f"Total = per-room-per-night × {r} room(s) × {d} nights.\n\n"
            "### 🏨 Hotel 1\n"
            "- **Name**: \n- **Area**: \n- **Category**: \n"
            f"- **Per room/night**: ₹X\n"
            f"- **Stay total ({r} room × {d} nights)**: ₹TOTAL\n"
            "- **Book at**: URL\n\n"
            "### 🏨 Hotel 2\n(same)\n\n### 🏨 Hotel 3\n(same)\n\n"
            "### ⭐ Top Pick: [name — one sentence reason]"
        ),
        expected_output=f"3 real hotels with stay total for {r} room(s) × {d} nights. Clean markdown.",
        agent=hotel_agent,
    )

    # ── Task 3: Activities ────────────────────────────────────────────────
    task_activities = Task(
        description=(
            f"Trip: {user_input}\n"
            f"Destination: {ds} | {g} people | {d} days\n\n"
            f"SEARCH DATA:\n{research['activities']}\n\n"
            f"Organise real attractions into Day 1–{d}. {cf}\n"
            f"Entry fee shown as: ₹TOTAL (₹X/person) where TOTAL = X × {g}.\n\n"
            "For each attraction:\n"
            "- **Name**: \n- **Entry**: ₹TOTAL (₹X/person)\n"
            "- **Hours**: \n- **Time needed**: \n- **Tip**: \n\n"
            "Cluster attractions geographically per day. No invented places."
        ),
        expected_output=f"Day-by-day attractions with group-total entry fees in format ₹TOTAL (₹X/person). Clean markdown.",
        agent=activity_agent,
    )

    # ── Task 4: Food & Transport ──────────────────────────────────────────
    task_logistics = Task(
        description=(
            f"Trip: {user_input}\n"
            f"Destination: {ds} | {g} people | {d} days\n\n"
            f"RESTAURANT DATA:\n{research['restaurants']}\n\n"
            f"TRANSPORT DATA:\n{research['transport']}\n\n"
            f"RESTAURANTS: Assign named restaurants to breakfast/lunch/dinner for each of {d} days.\n"
            f"Format: **[Name]** | Area: [area] | ₹TOTAL (₹X/person)\n"
            "CRITICAL: Restaurant name and area ONLY. Do NOT suggest dishes.\n\n"
            f"TRANSPORT: Best option for {g} people with daily group total. {cf}"
        ),
        expected_output=f"Named restaurants per meal slot for {d} days (name + area + group cost only, no dishes). Transport option. Clean markdown.",
        agent=logistics_agent,
    )

    # ── Task 5: Final Itinerary ───────────────────────────────────────────
    task_itinerary = Task(
        description=(
            f"USER REQUEST: {user_input}\n"
            f"{g} people | {r} room(s) | {d} nights | Budget: ₹{b} | Dates: {dt}\n\n"
            f"TIPS:\n{research['tips']}\n\n"
            "Write the complete itinerary using all research. Follow this exact structure:\n\n"

            "---\n"
            f"# 🌏 {ds} — {d}-Day Trip\n"
            f"*{g} travellers | {dt} | Budget ₹{b}*\n\n"

            "---\n"
            "## ✈️ Your Flight\n"
            "| | |\n|---|---|\n"
            "| Airline | |\n"
            f"| Route | {o} ↔ {ds} (return) |\n"
            "| Per person (return) | ₹ |\n"
            f"| **Total ({g} people)** | **₹** |\n"
            "| Book at | URL |\n\n"

            "---\n"
            "## 🏨 Your Hotel\n"
            "| | |\n|---|---|\n"
            "| Hotel | |\n"
            "| Area | |\n"
            "| Per room/night | ₹ |\n"
            f"| **Total ({r} room × {d} nights)** | **₹** |\n"
            "| Book at | URL |\n\n"

            "---\n"
            "## 📅 Day-by-Day Plan\n\n"
            f"Repeat this block for each of the {d} days:\n\n"
            "### Day N — [Theme]\n"
            "| Time | Activity | Venue | Cost |\n"
            "|------|----------|-------|------|\n"
            "| 08:00 | Breakfast | Restaurant Name — Area | ₹TOTAL (₹X/person) |\n"
            "| 09:30 | Activity | Venue Name | ₹TOTAL (₹X/person) |\n"
            "| 13:00 | Lunch | Restaurant Name — Area | ₹TOTAL (₹X/person) |\n"
            "| 15:00 | Activity | Venue Name | ₹TOTAL (₹X/person) |\n"
            "| 18:00 | Activity | Venue Name | ₹TOTAL (₹X/person) |\n"
            "| 20:00 | Dinner | Restaurant Name — Area | ₹TOTAL (₹X/person) |\n\n"
            "🚗 **Transport**: [mode — daily group total ₹TOTAL]\n\n"

            "COST FORMAT RULES (strictly enforced):\n"
            "- Write ₹TOTAL (₹X/person). Example: ₹300 (₹150/person).\n"
            "- NEVER write '₹150 × 2 = ₹300' or any multiplication string.\n"
            "- Restaurants: name and area only. NO dish suggestions.\n\n"

            "---\n"
            "## 💰 Trip Cost Summary\n"
            "One row per category. Show total only — no arithmetic, no working.\n\n"
            "| Category | Details | Total Cost |\n"
            "|----------|---------|------------|\n"
            "| ✈️ Flights | [Airline], return | ₹ |\n"
            "| 🏨 Hotel | [Name], N nights | ₹ |\n"
            "| 🍽️ Food | N days, 3 meals/day | ₹ |\n"
            "| 🎯 Activities | All entry fees | ₹ |\n"
            "| 🚗 Transport | N days local | ₹ |\n"
            "| 🛍️ Misc | Shopping & tips | ₹ |\n"
            "| **GRAND TOTAL** | **Full trip, all people** | **₹** |\n"
            "| **Per person** | Grand total ÷ people | **₹** |\n\n"
            f"✅ Budget check: state if within or over ₹{b}.\n\n"

            "---\n"
            "## 🎒 Packing List\n"
            "(Destination-specific, not generic)\n\n"
            "---\n"
            "## ⚠️ Key Tips\n"
            "5 specific tips for this destination.\n\n"
            "---\n"
            "## 🔗 Booking Links\n"
            "- Flight: URL\n- Hotel: URL\n"
        ),
        expected_output=(
            f"Complete itinerary for {g} people: flight table, hotel table, "
            f"{d}-day schedule with costs as ₹TOTAL (₹X/person), "
            "clean cost-summary table (no arithmetic shown), packing list, tips, links."
        ),
        agent=itinerary_agent,
        context=[task_flights, task_hotels, task_activities, task_logistics],
    )

    return Crew(
        agents=[flight_agent, hotel_agent, activity_agent, logistics_agent, itinerary_agent],
        tasks=[task_flights, task_hotels, task_activities, task_logistics, task_itinerary],
        process=Process.sequential,
        verbose=True,
        memory=False,
        planning=False,
    )