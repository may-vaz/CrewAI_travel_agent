"""
app.py — Streamlit UI for AI Travel Agent

Speed optimisations:
  - Parallel Tavily pre-fetch (ThreadPoolExecutor, 8 workers)
  - Agents read pre-fetched data — no tool-calling loops
  - Tighter max_tokens per agent
  - Sanity-check warnings shown above the itinerary

Display fixes:
  - Costs shown as ₹TOTAL (₹X/person) — no multiplication strings
  - Cost summary is a clean table, no arithmetic working shown
  - Price sanity warnings (e.g. ₹35 flight) shown prominently
"""

import streamlit as st
import time
import os
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="AI Travel Agent",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🌍 AI Travel Agent")
st.markdown(
    "**5 Specialist AI Agents** • Ollama (local) + Tavily (live search)  \n"
    "Return flights for full group • Correct rooms • Group activity costs • Clean cost summary"
)

# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🗺️ Plan Your Trip")

    origin      = st.text_input("Origin City", "Mumbai")
    destination = st.text_input("Destination", "Goa, India")

    col1, col2 = st.columns(2)
    with col1:
        duration   = st.number_input("Days", min_value=1, max_value=30, value=3)
    with col2:
        group_size = st.number_input("People", min_value=1, max_value=20, value=2)

    budget    = st.text_input("Total Budget (₹)", "20000",
                               help="Total for all people, whole trip")
    dates     = st.text_input("Travel Dates", "June 5-8, 2025")
    style     = st.selectbox("Trip Style", [
        "Family Budget", "Relaxed Beach", "Adventure", "Cultural",
        "Foodie", "Solo Backpacker", "Romantic Couple", "Luxury", "Pilgrimage",
    ])
    interests = st.text_area("Interests", "beaches, seafood, local markets")
    dietary   = st.text_input("Dietary Preferences", "",
                               help="e.g. vegetarian, halal")

    st.markdown("---")

    # Live pricing preview
    from search import rooms_needed
    rooms = rooms_needed(group_size)
    st.info(
        f"**Trip summary:**\n"
        f"- 👥 {group_size} traveller(s)\n"
        f"- 🛏️ {rooms} room(s) needed\n"
        f"- 📅 {duration} night(s)\n"
        f"- ✈️ Return fares will be calculated"
    )

    st.markdown("---")
    model_choice = st.selectbox(
        "Ollama Model",
        ["llama3.1:8b", "mistral", "gemma2:9b", "llama3.2"],
        index=0,
        help="llama3.1:8b recommended. Pull: ollama pull llama3.1:8b"
    )
    os.environ["OLLAMA_MODEL"] = f"ollama/{model_choice}"

    st.caption("🔒 Reasoning is local. Search queries go to Tavily.")

# ── Generate ──────────────────────────────────────────────────────────────
if st.button("🚀 Generate Trip Plan", type="primary", use_container_width=True):

    # Validation
    errors = []
    if not destination.strip():
        errors.append("Please enter a destination.")
    if not origin.strip():
        errors.append("Please enter an origin city.")
    if origin.strip().lower() == destination.strip().lower():
        errors.append("Origin and destination cannot be the same city.")
    if not os.environ.get("TAVILY_API_KEY"):
        errors.append("TAVILY_API_KEY missing from .env — get a free key at https://app.tavily.com")
    try:
        budget_num = int(budget.replace(",", "").replace("₹", "").strip())
        if budget_num < 500:
            errors.append(f"Budget ₹{budget_num} seems too low. Please check.")
    except ValueError:
        errors.append("Budget must be a number (e.g. 20000).")

    if errors:
        for e in errors:
            st.error(e)
        st.stop()

    dietary_note = f" Dietary: {dietary}." if dietary.strip() else ""
    user_input = (
        f"{duration}-day {style} trip from {origin} to {destination}. "
        f"Dates: {dates}. Group of {group_size} people. "
        f"Total budget ₹{budget} for all {group_size} people. "
        f"Interests: {interests}.{dietary_note}"
    )

    st.info(f"**Planning**: {user_input}")

    progress = st.progress(0)
    status   = st.empty()

    # ── Stage 1: Parallel search ──────────────────────────────────────────
    status.info(
        f"🔍 **Stage 1/2** — Running {9} web searches in parallel...  \n"
        f"Return flights ({group_size} pax) • {rooms} room hotel • "
        f"Activities • Restaurants • Transport"
    )

    from search import fetch_all_research
    t0 = time.time()
    try:
        research = fetch_all_research(
            origin=origin,
            destination=destination,
            dates=dates,
            budget=budget,
            style=style,
            interests=interests,
            duration=duration,
            group_size=group_size,
        )
        search_time = time.time() - t0
        progress.progress(25)
        status.success(f"✅ All searches done in {search_time:.0f}s")
    except Exception as e:
        st.error(f"❌ Search failed: {e}")
        st.stop()

    # ── Stage 2: Agents ───────────────────────────────────────────────────
    status.info(
        "🤖 **Stage 2/2** — AI agents working...  \n"
        "Flight → Hotel → Activities → Food/Transport → Itinerary"
    )
    progress.progress(30)

    from crew import create_crew
    from validators import validate_output

    t1 = time.time()
    try:
        crew   = create_crew(user_input, research)
        result = crew.kickoff()
        agent_time = time.time() - t1
        total_time = time.time() - t0 + search_time

        progress.progress(100)
        status.success(
            f"✅ Done — Search: {search_time:.0f}s | "
            f"Agents: {agent_time:.0f}s | "
            f"**Total: {total_time:.0f}s ({total_time/60:.1f} min)**"
        )

        # Extract text
        if hasattr(result, "raw") and result.raw:
            output = result.raw
        elif hasattr(result, "output") and result.output:
            output = result.output
        else:
            output = str(result)

        # ── Sanity checks ─────────────────────────────────────────────────
        warnings = validate_output(output)
        if warnings:
            st.markdown("---")
            st.markdown("### ⚠️ Price Verification Needed")
            for w in warnings:
                st.warning(w)
            st.markdown(
                "The itinerary below may contain inaccurate prices pulled from search snippets. "
                "Always verify flight and hotel prices on the booking site before purchasing."
            )

        # ── JSON guard ────────────────────────────────────────────────────
        if output.strip().startswith("{") or output.strip().startswith("["):
            st.error(
                "The model returned raw JSON instead of readable text.  \n"
                "Switch to **llama3.1:8b** in the sidebar: `ollama pull llama3.1:8b`"
            )
            with st.expander("Raw debug output"):
                st.code(output)
            st.stop()

        # ── Display itinerary ─────────────────────────────────────────────
        st.markdown("---")
        st.markdown(output)
        st.markdown("---")

        # Download
        fname = f"trip_{destination.replace(',','').replace(' ','_')}_{group_size}pax_{duration}d.md"
        st.download_button(
            "📥 Download Plan (Markdown)",
            data=output,
            file_name=fname,
            mime="text/markdown",
        )

    except Exception as e:
        progress.progress(0)
        st.error(f"❌ Agent error: {e}")
        with st.expander("Troubleshooting"):
            st.markdown(
                "- Ollama running? → `ollama serve`\n"
                "- Model pulled? → `ollama pull llama3.1:8b`\n"
                "- `.env` has `TAVILY_API_KEY` and `OPENAI_API_KEY=dummy-not-used`?\n"
                "- Try `mistral` if getting timeouts\n"
            )
            st.code(str(e))

st.markdown("---")
st.caption(
    "Portfolio Project • Ollama local LLM • Tavily live search • No paid LLM API  \n"
    "Costs: ₹TOTAL (₹X/person) format • Return flights • Correct rooms • Group totals"
)