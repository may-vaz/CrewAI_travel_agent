import os
from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import TavilySearchTool
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────────────────────
# CRITICAL: CrewAI 1.x has internal memory/analyze hooks that
# call OpenAI even when memory=False. A dummy key stops crashes.
# Your LLM stays 100% local via Ollama — no data leaves your machine.
# ─────────────────────────────────────────────────────────────
if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = "dummy-not-used"

# ─────────────────────────────────────────────────────────────
# LOCAL LLM — Ollama
# Correct prefix: "ollama/<model>" (NOT "openai/")
# No /v1 in base_url when using ollama/ prefix
#
# ⚠️  llama3.2 (3B) is too small for complex multi-agent tasks.
#     It hallucinates JSON tool calls instead of readable text.
#     STRONGLY recommend upgrading:
#       ollama pull llama3.1:8b   ← best balance of speed + quality
#       ollama pull mistral       ← good alternative
#     Then set OLLAMA_MODEL in .env or change the default below.
# ─────────────────────────────────────────────────────────────
MODEL    = os.environ.get("OLLAMA_MODEL",    "ollama/llama3.1:8b")
BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

llm = LLM(
    model=MODEL,
    base_url=BASE_URL,
    temperature=0.4,   # Lower = more factual, less hallucination
    max_tokens=3000,
)

# ─────────────────────────────────────────────────────────────
# SEARCH TOOLS
# Separate TavilySearchTool instances per domain so each agent
# runs its own targeted searches without polluting others.
# ─────────────────────────────────────────────────────────────
def make_search(n: int = 5) -> TavilySearchTool:
    return TavilySearchTool(max_results=n, search_depth="advanced", include_answer=True)

flight_search    = make_search(6)
hotel_search     = make_search(6)
activity_search  = make_search(8)
food_search      = make_search(5)
transport_search = make_search(5)


# ─────────────────────────────────────────────────────────────
# AGENT DEFINITIONS
# Six specialist agents, each with a narrow domain.
# ─────────────────────────────────────────────────────────────

analyst = Agent(
    role="Trip Analyst",
    goal=(
        "Parse the user's travel request and produce a detailed structured brief "
        "covering origin city, destination, dates, per-category budget split, "
        "travel style, and key interests."
    ),
    backstory=(
        "You are a senior travel consultant who specialises in understanding client needs. "
        "You produce clear structured briefs that other specialists can act on. "
        "You never output JSON or code — only clean formatted markdown."
    ),
    llm=llm,
    verbose=True,
    allow_delegation=False,
    max_iter=3,
)

flight_agent = Agent(
    role="Flight Research Specialist",
    goal=(
        "Find specific real flight options for the trip using live web search. "
        "Search for flights on the actual route and dates. "
        "Report real airline names, prices in ₹, times, and booking links."
    ),
    backstory=(
        "You are an expert at finding flights using MakeMyTrip, Cleartrip, "
        "EaseMyTrip, Google Flights, and Skyscanner. "
        "You only report actual named airlines with real prices — never vague ranges. "
        "You write clean markdown bullet points. No JSON, no code."
    ),
    tools=[flight_search],
    llm=llm,
    verbose=True,
    allow_delegation=False,
    max_iter=5,
)

hotel_agent = Agent(
    role="Hotel Research Specialist",
    goal=(
        "Find 3 specific real hotels at the destination that fit the budget. "
        "Search and report real hotel names, per-night prices in ₹, "
        "star ratings, neighbourhoods, and booking URLs."
    ),
    backstory=(
        "You are an expert at finding best-value hotels using Booking.com, "
        "MakeMyTrip Hotels, Goibibo, Agoda, and OYO. "
        "You always name real, specific hotels — never generic placeholders. "
        "Clean markdown only. No JSON, no code."
    ),
    tools=[hotel_search],
    llm=llm,
    verbose=True,
    allow_delegation=False,
    max_iter=5,
)

activity_agent = Agent(
    role="Activities & Attractions Specialist",
    goal=(
        "Find specific things to do and see at the destination that match "
        "the travel style and interests. Include real entry fees, opening hours, "
        "exact names, and insider tips for each activity."
    ),
    backstory=(
        "You are a local travel guide expert using TripAdvisor, Google Travel, "
        "Thrillophilia, and local tourism boards. "
        "You are hyper-specific — real place names, real fees, real hours. "
        "Clean markdown. No JSON."
    ),
    tools=[activity_search],
    llm=llm,
    verbose=True,
    allow_delegation=False,
    max_iter=5,
)

logistics_agent = Agent(
    role="Food & Local Transport Specialist",
    goal=(
        "Find specific named restaurants matching the budget and food interests. "
        "Find the best local transport options with real prices. "
        "Assign restaurants to meal slots across the trip."
    ),
    backstory=(
        "You are a foodie and logistics expert. You find restaurants using Zomato, "
        "Swiggy, and Google Maps. For transport you check local cab fares, "
        "bus routes, and vehicle rental prices. Real names, real prices. "
        "Clean markdown. No JSON."
    ),
    tools=[food_search, transport_search],
    llm=llm,
    verbose=True,
    allow_delegation=False,
    max_iter=5,
)

itinerary_agent = Agent(
    role="Master Itinerary Planner",
    goal=(
        "Synthesise ALL research into one complete, hyper-specific day-by-day itinerary. "
        "Use real hotel names, real flight names, real restaurants, real attractions "
        "with actual prices in ₹. Every day must have a time-slotted schedule. "
        "End with a precise cost breakdown table."
    ),
    backstory=(
        "You are a master travel writer with 20 years of experience. "
        "You write hyper-specific, actionable itineraries. Every line is something "
        "the traveller can act on immediately. You format in structured markdown "
        "with tables, headings, and time slots. "
        "No generic fillers, no 'or' options — you pick one and justify it. "
        "No JSON, no code — only clean professional travel writing."
    ),
    llm=llm,
    verbose=True,
    allow_delegation=False,
    max_iter=5,
)


# ─────────────────────────────────────────────────────────────
# CREW FACTORY
# ─────────────────────────────────────────────────────────────
def create_crew(user_input: str) -> Crew:

    task_analyse = Task(
        description=(
            f"Analyse this travel request:\n\nREQUEST: {user_input}\n\n"
            "Produce a structured trip brief with ALL of these sections:\n"
            "- **Origin City**: (infer from context; if unclear write 'Not stated — assume nearest metro')\n"
            "- **Destination**: full name + state/country\n"
            "- **Duration & Dates**: number of days and specific dates\n"
            "- **Total Budget**: ₹ amount, then split as: Flights ~X%, Hotel ~Y%, Food ~Z%, Activities ~W%, Misc ~5%\n"
            "- **Travel Style**: e.g. Family budget / Solo adventure / Couple luxury\n"
            "- **Key Interests**: bullet list\n"
            "- **Special Requirements**: dietary, accessibility, group size, etc.\n"
            "- **Season & Weather Note**: what to expect at the destination on those dates\n\n"
            "Write in clean markdown. No JSON."
        ),
        expected_output=(
            "Structured markdown trip brief with all 8 sections filled. "
            "Budget split must include percentages. No JSON."
        ),
        agent=analyst,
    )

    task_flights = Task(
        description=(
            "Use the trip brief above to search for REAL flights.\n\n"
            "Run web searches with queries such as:\n"
            "  '[origin] to [destination] flights [month year]'\n"
            "  'cheapest flight [origin] to [destination] [dates]'\n"
            "  'MakeMyTrip [origin] [destination] flight price'\n\n"
            "You MUST report:\n"
            "### ✈️ Flight Option 1 (Recommended)\n"
            "- **Airline**: [real airline name]\n"
            "- **Route**: [origin airport code] → [destination airport code]\n"
            "- **Departure / Arrival**: [times]\n"
            "- **Price**: ₹[exact figure] (one-way or return — specify)\n"
            "- **Book at**: [URL]\n\n"
            "### ✈️ Flight Option 2 (Alternative)\n"
            "(Same format)\n\n"
            "### 💡 Flight Booking Tip\n"
            "(Specific advice: which platform is cheapest, any promo codes, best day to book)\n\n"
            "Do NOT say 'flights cost around ₹X-Y'. Name actual airlines with actual prices. "
            "If you truly cannot find live prices, say so and give the best search URL for the user."
        ),
        expected_output=(
            "Two specific named flight options with airline, route, times, ₹ price, "
            "and booking URL. One booking tip. Clean markdown."
        ),
        agent=flight_agent,
        context=[task_analyse],
    )

    task_hotels = Task(
        description=(
            "Use the trip brief to search for REAL hotels at the destination.\n\n"
            "Run web searches with queries such as:\n"
            "  'best budget hotels [destination] [travel style] 2025'\n"
            "  '[destination] hotels near beach under ₹[budget/nights] per night'\n"
            "  'Booking.com [destination] [star] hotel'\n\n"
            "Report EXACTLY 3 real hotels:\n\n"
            "### 🏨 Hotel Option 1\n"
            "- **Name**: [exact hotel name]\n"
            "- **Area / Neighbourhood**: [specific area]\n"
            "- **Price per night**: ₹[X]\n"
            "- **Total for stay**: ₹[X × nights]\n"
            "- **Category**: Budget / Mid-range / Luxury\n"
            "- **Why recommended**: [2 sentences]\n"
            "- **Book at**: [URL]\n\n"
            "### 🏨 Hotel Option 2\n(same format)\n\n"
            "### 🏨 Hotel Option 3\n(same format)\n\n"
            "### ⭐ Top Pick: [Hotel Name]\n"
            "[Why this is the best fit for this traveller's budget and style]\n\n"
            "Do NOT use placeholder names. Write clean markdown. No JSON."
        ),
        expected_output=(
            "3 real named hotels with neighbourhood, ₹/night price, total cost, "
            "category, justification, and booking URL. One clear top pick. Clean markdown."
        ),
        agent=hotel_agent,
        context=[task_analyse],
    )

    task_activities = Task(
        description=(
            "Use the trip brief to find REAL activities and attractions at the destination.\n\n"
            "Run web searches with queries such as:\n"
            "  'best things to do in [destination] [travel style]'\n"
            "  '[destination] top attractions entry fee 2025'\n"
            "  '[specific interest] activities [destination]'\n\n"
            "For EACH activity:\n"
            "- **Name**: [exact name of place or activity]\n"
            "- **Type**: [beach / fort / museum / water sport / market / temple / etc.]\n"
            "- **Entry Fee**: ₹[X] per person (or 'Free')\n"
            "- **Hours**: [opening - closing time]\n"
            "- **Time needed**: [X hours]\n"
            "- **Best time of day**: [morning/afternoon/evening — and why]\n"
            "- **Insider tip**: [one very specific practical tip]\n\n"
            "Organise activities by day (Day 1, Day 2, etc.) based on geographical "
            "proximity so the traveller doesn't criss-cross the destination. "
            "Include enough for EVERY day. Write clean markdown. No JSON."
        ),
        expected_output=(
            "Day-by-day organised list of real named attractions with entry fees, "
            "hours, time needed, best time of day, and insider tips. "
            "Geographically clustered by day. Clean markdown."
        ),
        agent=activity_agent,
        context=[task_analyse],
    )

    task_logistics = Task(
        description=(
            "Use the trip brief to find REAL restaurants and local transport.\n\n"
            "RESTAURANTS — search:\n"
            "  'best [cuisine type] restaurants in [destination]'\n"
            "  'top rated local food [destination] Zomato'\n"
            "  'must try dishes [destination]'\n\n"
            "Assign specific restaurants to meal slots. For each:\n"
            "- **Restaurant Name**: [real name]\n"
            "- **Dish to Order**: [specific dish]\n"
            "- **Cost per person**: ₹[X]\n"
            "- **Area**: [neighbourhood/location]\n\n"
            "Cover breakfast, lunch, and dinner for EACH day of the trip.\n\n"
            "LOCAL TRANSPORT — search:\n"
            "  'how to get around [destination] as a tourist'\n"
            "  '[destination] auto / cab fare'\n"
            "  '[destination] bike / scooter rental price per day'\n\n"
            "Report the single best transport strategy for this trip with real prices. "
            "Clean markdown. No JSON."
        ),
        expected_output=(
            "Meal plan with real restaurant names, dishes, ₹ prices assigned to "
            "each day's breakfast/lunch/dinner slots. Plus the best local transport "
            "option with real fares. Clean markdown."
        ),
        agent=logistics_agent,
        context=[task_analyse],
    )

    task_itinerary = Task(
        description=(
            "You are the final planner. Synthesise ALL the research above into one "
            "complete, beautiful, hyper-specific travel itinerary.\n\n"
            "Use EXACTLY this structure:\n\n"
            "---\n"
            "# 🌏 [Destination] Trip Plan — [Duration] days\n"
            "> *[One-line description of the trip vibe]*\n\n"
            "---\n"
            "## ✈️ Your Flight\n"
            "Pick ONE flight from the research. Format:\n"
            "| Detail | Info |\n"
            "|--------|------|\n"
            "| Airline | [name] |\n"
            "| Route | [origin] → [destination] |\n"
            "| Departure | [time] |\n"
            "| Price | ₹[X] return |\n"
            "| Book at | [URL] |\n\n"
            "---\n"
            "## 🏨 Your Hotel\n"
            "Pick ONE hotel from the research. Format:\n"
            "| Detail | Info |\n"
            "|--------|------|\n"
            "| Hotel | [name] |\n"
            "| Area | [neighbourhood] |\n"
            "| Per night | ₹[X] |\n"
            "| Total ([N] nights) | ₹[X×N] |\n"
            "| Book at | [URL] |\n\n"
            "---\n"
            "## 📅 Day-by-Day Plan\n"
            "For EACH day:\n"
            "### Day [N] — [Catchy Theme Title]\n"
            "| Time | Activity | Venue | Cost |\n"
            "|------|----------|-------|------|\n"
            "| 07:30 | Breakfast | [restaurant name] — [dish] | ₹[X] |\n"
            "| 09:00 | [Activity] | [venue name] | ₹[entry fee] |\n"
            "| 12:30 | Lunch | [restaurant name] — [dish] | ₹[X] |\n"
            "| 14:00 | [Activity] | [venue name] | ₹[X] |\n"
            "| 17:00 | [Activity] | [venue name] | ₹[X] |\n"
            "| 19:30 | Dinner | [restaurant name] — [dish] | ₹[X] |\n\n"
            "🚗 **Getting around today**: [specific transport mode + fare]\n"
            "💡 **Today's tip**: [one specific actionable tip]\n\n"
            "---\n"
            "## 💰 Full Cost Breakdown\n"
            "| Category | Details | Cost |\n"
            "|----------|---------|------|\n"
            "| Flights | [airline] return | ₹[X] |\n"
            "| Hotel | [name] × [N] nights | ₹[X] |\n"
            "| Food | [N] days, 3 meals/day avg ₹[X]/meal | ₹[X] |\n"
            "| Activities | [list] | ₹[X] |\n"
            "| Local Transport | [mode] | ₹[X] |\n"
            "| Miscellaneous | Shopping, tips, etc. | ₹[X] |\n"
            "| **TOTAL** | | **₹[X]** |\n\n"
            "✅ **Budget status**: [Within budget / Over by ₹X — suggestion to cut]\n\n"
            "---\n"
            "## 🎒 Packing List\n"
            "(Specific to this destination and season — not generic)\n\n"
            "---\n"
            "## ⚠️ Essential Tips\n"
            "- [5 specific, actionable tips for this exact destination]\n\n"
            "---\n"
            "## 🔗 Key Booking Links\n"
            "- Flight: [URL]\n"
            "- Hotel: [URL]\n"
            "- [Any activity pre-booking links found]\n\n"
            "RULES:\n"
            "- Every hotel, flight, restaurant, and attraction MUST be a real named entity from the research.\n"
            "- Every price MUST be a specific ₹ number.\n"
            "- Do NOT write 'you could visit X or Y' — pick one and justify it.\n"
            "- Do NOT leave any table cell blank — if unknown write 'Check on arrival'.\n"
            "- Write as if you personally planned and vetted this trip."
        ),
        expected_output=(
            "Complete hyper-specific travel itinerary in clean markdown with: "
            "flight table, hotel table, day-by-day tables with time slots, "
            "full cost breakdown table, packing list, 5 tips, and booking links. "
            "All venues and prices must be real and specific."
        ),
        agent=itinerary_agent,
        context=[task_analyse, task_flights, task_hotels, task_activities, task_logistics],
    )

    return Crew(
        agents=[analyst, flight_agent, hotel_agent, activity_agent, logistics_agent, itinerary_agent],
        tasks=[task_analyse, task_flights, task_hotels, task_activities, task_logistics, task_itinerary],
        process=Process.sequential,
        verbose=True,
        memory=False,    # Keep off — avoids OpenAI embedder/Chroma dependency
        planning=False,  # Keep off — planning agent calls OpenAI by default
    )