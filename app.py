import streamlit as st
from crew import create_crew
from dotenv import load_dotenv
import os

load_dotenv()

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Travel Agent",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🌍 AI Travel Agent")
st.markdown(
    "**6 Specialist AI Agents** • CrewAI + Ollama (local) + Tavily live search  \n"
    "*Flight finder • Hotel researcher • Activities planner • Food & transport expert • Itinerary writer*"
)

# ─────────────────────────────────────────────────────────────
# SIDEBAR — TRIP INPUTS
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🗺️ Plan Your Trip")

    origin = st.text_input("Origin City", "Mumbai", help="Where are you flying/travelling from?")
    destination = st.text_input("Destination", "Goa, India")
    col1, col2 = st.columns(2)
    with col1:
        duration = st.number_input("Days", min_value=1, max_value=30, value=3)
    with col2:
        budget = st.text_input("Budget (₹)", "20000")

    dates = st.text_input("Travel Dates", "May 20-23, 2025")

    style = st.selectbox(
        "Trip Style",
        ["Family Budget", "Relaxed Beach", "Adventure", "Cultural", "Foodie",
         "Solo Backpacker", "Romantic Couple", "Luxury", "Pilgrimage"],
    )

    group_size = st.number_input("Group Size", min_value=1, max_value=20, value=2)

    interests = st.text_area(
        "Interests & Preferences",
        "beaches, seafood, local markets, budget stays",
        help="The more specific you are, the better the recommendations."
    )

    dietary = st.text_input(
        "Dietary Preferences", "",
        help="e.g. vegetarian, vegan, no pork, halal, etc."
    )

    st.markdown("---")
    st.markdown("**Model Config**")
    model_hint = st.selectbox(
        "Ollama Model",
        ["llama3.1:8b", "llama3.2", "mistral", "mixtral", "gemma2:9b"],
        index=0,
        help="Must be pulled in Ollama first. llama3.1:8b strongly recommended."
    )
    os.environ["OLLAMA_MODEL"] = f"ollama/{model_hint}"

    st.caption("🔒 All reasoning is local via Ollama. Only Tavily search queries leave your machine.")

# ─────────────────────────────────────────────────────────────
# MAIN — GENERATE PLAN
# ─────────────────────────────────────────────────────────────
if st.button("🚀 Generate My Trip Plan", type="primary", use_container_width=True):
    if not destination:
        st.error("Please enter a destination.")
    elif not os.environ.get("TAVILY_API_KEY"):
        st.error(
            "TAVILY_API_KEY not found in your .env file. "
            "Get a free key at https://app.tavily.com and add it to .env."
        )
    else:
        # Build the natural language request
        dietary_note = f" Dietary: {dietary}." if dietary else ""
        user_input = (
            f"{duration}-day {style} trip from {origin} to {destination}. "
            f"Dates: {dates}. Group size: {group_size} people. "
            f"Total budget: ₹{budget} for {group_size} people. "
            f"Interests: {interests}.{dietary_note}"
        )

        # Show what we're planning
        st.info(f"**Planning**: {user_input}")

        # Progress indicators
        progress_container = st.empty()
        stage_messages = [
            "🔍 Analysing your trip request...",
            "✈️ Searching for real flights...",
            "🏨 Finding hotels that match your budget...",
            "🎯 Researching activities and attractions...",
            "🍽️ Finding restaurants and local transport...",
            "📝 Writing your personalised itinerary...",
        ]

        with st.spinner("Working through 6 specialist agents... (this takes 3–8 minutes with a local LLM)"):
            try:
                # Show which agent is working
                for msg in stage_messages:
                    progress_container.info(msg)

                crew = create_crew(user_input)
                result = crew.kickoff()

                progress_container.empty()

                # Extract output
                if hasattr(result, "raw") and result.raw:
                    output_text = result.raw
                elif hasattr(result, "output") and result.output:
                    output_text = result.output
                else:
                    output_text = str(result)

                # Guard against JSON tool-call output (small model problem)
                stripped = output_text.strip()
                if stripped.startswith("{") or stripped.startswith("["):
                    st.warning(
                        "⚠️ The local model returned a tool-call JSON response instead of readable text.  \n"
                        "This is caused by using a model that's too small (e.g. llama3.2 3B).  \n\n"
                        "**Fix**: In the sidebar, switch to **llama3.1:8b** and pull it first:  \n"
                        "```\nollama pull llama3.1:8b\n```"
                    )
                    with st.expander("Raw output (for debugging)"):
                        st.code(output_text)
                else:
                    st.success("✅ Your trip plan is ready!")
                    st.markdown("---")
                    st.markdown(output_text)
                    st.markdown("---")

                    # Download button
                    st.download_button(
                        label="📥 Download Trip Plan (Markdown)",
                        data=output_text,
                        file_name=f"trip_plan_{destination.replace(', ', '_').replace(' ', '_')}.md",
                        mime="text/markdown",
                    )

            except Exception as e:
                progress_container.empty()
                st.error(f"❌ Error: {e}")

                with st.expander("Troubleshooting"):
                    st.markdown(
                        "**Common fixes:**\n"
                        "- Make sure Ollama is running: `ollama serve`\n"
                        "- Make sure your model is pulled: `ollama pull llama3.1:8b`\n"
                        "- Check `TAVILY_API_KEY` is set in your `.env` file\n"
                        "- Check `OPENAI_API_KEY=dummy-not-used` is in your `.env` file\n"
                        "- Try a different/smaller model if getting timeout errors\n"
                    )
                    st.code(str(e))

# ─────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Portfolio Project • Local LLM via Ollama • Live search via Tavily • "
    "No paid LLM API required  \n"
    "Architecture: 6 specialist CrewAI agents — Analyst → Flight → Hotel → Activities → Food/Transport → Itinerary"
)