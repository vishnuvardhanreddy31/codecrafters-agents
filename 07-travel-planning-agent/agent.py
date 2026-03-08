"""
Travel Planning Agent
Uses CrewAI with specialized agents (Destination Researcher, Itinerary Planner,
Budget Analyst) to create comprehensive personalized travel plans.
"""

import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)


# ── Agents ─────────────────────────────────────────────────────────────────────
destination_researcher = Agent(
    role="Destination Expert",
    goal=(
        "Research the destination thoroughly: attractions, culture, climate, "
        "visa requirements, local customs, safety tips, and hidden gems."
    ),
    backstory=(
        "You are a seasoned travel journalist who has visited over 100 countries. "
        "You know every destination's must-see landmarks, local secrets, best "
        "neighborhoods, and cultural nuances that make each trip memorable."
    ),
    llm=llm,
    verbose=True,
    allow_delegation=False,
)

itinerary_planner = Agent(
    role="Itinerary Specialist",
    goal=(
        "Create a detailed day-by-day travel itinerary that maximizes experiences "
        "while allowing for relaxation and spontaneity."
    ),
    backstory=(
        "You are a professional travel planner with expertise in crafting "
        "seamless itineraries. You consider travel time, opening hours, energy "
        "levels, and logical routing to create perfectly paced trips."
    ),
    llm=llm,
    verbose=True,
    allow_delegation=False,
)

budget_analyst = Agent(
    role="Travel Budget Analyst",
    goal=(
        "Analyze travel costs and create a realistic budget breakdown including "
        "accommodation, transport, food, activities, and contingency."
    ),
    backstory=(
        "You are a financial travel expert who helps travelers maximize their "
        "experiences within their budget. You know money-saving tips, booking "
        "strategies, and the true cost of travel in different destinations."
    ),
    llm=llm,
    verbose=True,
    allow_delegation=False,
)


# ── Tasks factory ──────────────────────────────────────────────────────────────
def build_travel_crew(
    destination: str,
    duration_days: int,
    budget_usd: int,
    interests: str,
    travel_style: str,
) -> Crew:
    research_task = Task(
        description=(
            f"Research {destination} for a {duration_days}-day trip.\n"
            f"Traveler interests: {interests}\n"
            f"Travel style: {travel_style}\n\n"
            "Provide: top attractions, best neighborhoods to stay, local cuisine, "
            "cultural tips, visa/entry requirements, best time to visit, "
            "transportation options within the destination, and safety information."
        ),
        agent=destination_researcher,
        expected_output=(
            "Comprehensive destination guide covering: top 10 attractions, "
            "3 recommended areas to stay, local food highlights, "
            "entry requirements, and practical travel tips."
        ),
    )

    itinerary_task = Task(
        description=(
            f"Create a detailed {duration_days}-day itinerary for {destination}.\n"
            f"Travel style: {travel_style}\n"
            f"Interests: {interests}\n\n"
            "Structure each day with: morning/afternoon/evening activities, "
            "meal recommendations, travel times between locations, "
            "booking requirements, and alternatives for bad weather."
        ),
        agent=itinerary_planner,
        expected_output=(
            f"Complete day-by-day itinerary for {duration_days} days with "
            "morning/afternoon/evening plans, dining recommendations, "
            "and practical logistics."
        ),
        context=[research_task],
    )

    budget_task = Task(
        description=(
            f"Create a budget breakdown for the {destination} trip.\n"
            f"Duration: {duration_days} days\n"
            f"Total budget: ${budget_usd} USD\n"
            f"Travel style: {travel_style}\n\n"
            "Provide: accommodation costs per night, daily food budget, "
            "activity costs, transportation estimate, visa/entry fees, "
            "travel insurance estimate, shopping allowance, and contingency fund. "
            "Include money-saving tips and highlight the best value experiences."
        ),
        agent=budget_analyst,
        expected_output=(
            f"Detailed budget breakdown totaling ~${budget_usd} USD with "
            "per-category estimates, daily spending guide, and 5 money-saving tips."
        ),
        context=[research_task, itinerary_task],
    )

    return Crew(
        agents=[destination_researcher, itinerary_planner, budget_analyst],
        tasks=[research_task, itinerary_task, budget_task],
        process=Process.sequential,
        verbose=True,
    )


# ── Runner ─────────────────────────────────────────────────────────────────────
def plan_trip(
    destination: str,
    duration_days: int = 7,
    budget_usd: int = 3000,
    interests: str = "culture, food, history",
    travel_style: str = "balanced (mix of budget and comfort)",
) -> str:
    crew = build_travel_crew(destination, duration_days, budget_usd, interests, travel_style)
    result = crew.kickoff()
    return str(result)


def main():
    print("✈️  Travel Planning Agent\n" + "=" * 60)
    config = {
        "destination": "Tokyo, Japan",
        "duration_days": 7,
        "budget_usd": 3000,
        "interests": "technology, anime, traditional culture, street food",
        "travel_style": "mid-range (comfortable but value-conscious)",
    }
    print("Trip Details:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    print("\nGenerating your travel plan...\n")
    plan = plan_trip(**config)
    print("\n" + "=" * 60)
    print("🗺️  Your Complete Travel Plan:")
    print("=" * 60)
    print(plan)
    print("\n✅ Travel planning complete!")


if __name__ == "__main__":
    main()
