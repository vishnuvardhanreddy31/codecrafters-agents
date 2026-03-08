"""
Social Media Agent
Uses CrewAI with specialized agents (Strategist, Content Creator, Analytics Expert)
to create platform-specific social media content and strategy.
"""

import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.8)

# ── Platform specifications ────────────────────────────────────────────────────
PLATFORM_SPECS = {
    "twitter": {
        "char_limit": 280,
        "best_practices": "Short, punchy, use 1-2 hashtags, ask questions, include CTAs",
        "content_types": "text, images, polls, threads",
        "best_times": "9am, 12pm, 5pm on weekdays",
    },
    "instagram": {
        "char_limit": 2200,
        "best_practices": "Visual-first, story-driven captions, 5-10 hashtags, emojis",
        "content_types": "images, reels, stories, carousels",
        "best_times": "11am, 2pm, 5pm | Wednesday & Friday",
    },
    "linkedin": {
        "char_limit": 3000,
        "best_practices": "Professional tone, value-driven, industry insights, personal stories",
        "content_types": "articles, posts, documents, videos",
        "best_times": "Tuesday-Thursday, 8-10am or 5-6pm",
    },
    "tiktok": {
        "char_limit": 2200,
        "best_practices": "Trending audio, hooks in first 3 seconds, authentic, educational",
        "content_types": "short videos (15-60 sec), duets, trends",
        "best_times": "7pm-9pm weekdays, 9am-11am weekends",
    },
    "facebook": {
        "char_limit": 63206,
        "best_practices": "Community-focused, longer narratives ok, events, groups",
        "content_types": "posts, videos, stories, events, groups",
        "best_times": "1pm-3pm Wednesday, Thursday-Friday 1-4pm",
    },
}


# ── Agents ─────────────────────────────────────────────────────────────────────
social_strategist = Agent(
    role="Social Media Strategist",
    goal=(
        "Develop a data-driven social media strategy that maximizes engagement, "
        "brand awareness, and audience growth for the specific business and platforms."
    ),
    backstory=(
        "You are a seasoned social media strategist with a track record of growing "
        "brands from 0 to millions of followers. You understand platform algorithms, "
        "audience psychology, and the business impact of social media."
    ),
    llm=llm,
    verbose=True,
    allow_delegation=False,
)

content_creator = Agent(
    role="Social Media Content Creator",
    goal=(
        "Create engaging, platform-optimized content including captions, hashtag sets, "
        "content ideas, and a posting calendar that resonates with the target audience."
    ),
    backstory=(
        "You are a creative content creator who has helped brands go viral multiple times. "
        "You know how to craft hooks, tell stories, and create content that stops the scroll "
        "and drives meaningful engagement on every platform."
    ),
    llm=llm,
    verbose=True,
    allow_delegation=False,
)

analytics_expert = Agent(
    role="Social Media Analytics Expert",
    goal=(
        "Define KPIs, set benchmarks, and create a measurement framework to track "
        "the success of the social media strategy."
    ),
    backstory=(
        "You are a data-driven social media analyst who transforms raw metrics into "
        "actionable insights. You understand vanity metrics vs. business metrics and "
        "help brands focus on what actually drives ROI."
    ),
    llm=llm,
    verbose=True,
    allow_delegation=False,
)


# ── Crew factory ───────────────────────────────────────────────────────────────
def build_social_media_crew(
    brand_name: str,
    industry: str,
    target_audience: str,
    goals: str,
    platforms: List[str],
    tone: str,
) -> Crew:
    platform_info = "\n".join(
        f"  {p}: {PLATFORM_SPECS[p]['best_practices']}"
        for p in platforms
        if p in PLATFORM_SPECS
    )

    strategy_task = Task(
        description=(
            f"Develop a 30-day social media strategy for {brand_name}:\n\n"
            f"Industry: {industry}\n"
            f"Target Audience: {target_audience}\n"
            f"Business Goals: {goals}\n"
            f"Platforms: {', '.join(platforms)}\n"
            f"Brand Tone: {tone}\n\n"
            f"Platform Best Practices:\n{platform_info}\n\n"
            "Provide:\n"
            "1. Content pillars (5 themes to post about)\n"
            "2. Posting frequency per platform\n"
            "3. Content mix (educational/entertaining/promotional ratio)\n"
            "4. Hashtag strategy\n"
            "5. Community engagement tactics\n"
            "6. Competitive positioning"
        ),
        agent=social_strategist,
        expected_output=(
            "Complete 30-day social media strategy with: 5 content pillars, "
            "posting schedule, content mix ratios, hashtag strategy, and engagement tactics."
        ),
    )

    content_task = Task(
        description=(
            f"Create content for {brand_name}'s social media:\n\n"
            f"Platforms: {', '.join(platforms)}\n"
            f"Audience: {target_audience}\n"
            f"Tone: {tone}\n\n"
            "For each platform, create:\n"
            "1. 5 ready-to-post captions with optimal hashtags\n"
            "2. 3 content ideas for this week (specify format)\n"
            "3. 1 engaging question/poll idea\n"
            "4. 1 viral content concept\n"
            "5. A 7-day posting calendar\n\n"
            f"Platform character limits:\n"
            + "\n".join(
                f"  {p}: {PLATFORM_SPECS[p]['char_limit']} chars"
                for p in platforms
                if p in PLATFORM_SPECS
            )
        ),
        agent=content_creator,
        expected_output=(
            "Platform-specific content package with: 5 captions per platform, "
            "content ideas, engagement hooks, and 7-day calendar."
        ),
        context=[strategy_task],
    )

    analytics_task = Task(
        description=(
            f"Create measurement framework for {brand_name}'s social media:\n\n"
            f"Platforms: {', '.join(platforms)}\n"
            f"Business Goals: {goals}\n\n"
            "Define:\n"
            "1. Primary KPIs for each platform (3-5 metrics)\n"
            "2. Target benchmarks for months 1, 3, and 6\n"
            "3. Daily/weekly tracking checklist\n"
            "4. Red flags to watch for\n"
            "5. Monthly reporting template\n"
            "6. Tools recommended for analytics"
        ),
        agent=analytics_expert,
        expected_output=(
            "Measurement framework with: platform KPIs, growth targets, "
            "tracking checklist, and reporting template."
        ),
        context=[strategy_task],
    )

    return Crew(
        agents=[social_strategist, content_creator, analytics_expert],
        tasks=[strategy_task, content_task, analytics_task],
        process=Process.sequential,
        verbose=True,
    )


from typing import List


def create_social_strategy(
    brand_name: str,
    industry: str,
    target_audience: str,
    goals: str,
    platforms: List[str] = None,
    tone: str = "professional yet approachable",
) -> str:
    if platforms is None:
        platforms = ["instagram", "linkedin", "twitter"]
    crew = build_social_media_crew(brand_name, industry, target_audience, goals, platforms, tone)
    result = crew.kickoff()
    return str(result)


def main():
    print("📱 Social Media Agent\n" + "=" * 60)
    config = {
        "brand_name": "EcoBloom",
        "industry": "Sustainable home goods / eco-friendly products",
        "target_audience": "Millennials and Gen Z, environmentally conscious, ages 25-40",
        "goals": "Increase brand awareness, grow Instagram following by 5K, drive 20% more website traffic",
        "platforms": ["instagram", "tiktok", "linkedin"],
        "tone": "inspiring, authentic, educational, and community-driven",
    }
    print("Brand Details:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    print("\nGenerating social media strategy and content...\n")
    result = create_social_strategy(**config)
    print("\n" + "=" * 60)
    print("📊 Your Social Media Package:")
    print("=" * 60)
    print(result)
    print("\n✅ Social media strategy complete!")


if __name__ == "__main__":
    main()
