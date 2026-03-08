"""
Content Creation Agent
Uses CrewAI with specialized agents (Researcher, Writer, Editor)
to produce high-quality blog posts and articles.
"""

import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool
from langchain_openai import ChatOpenAI

load_dotenv()

# ── LLM ────────────────────────────────────────────────────────────────────────
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# ── Tools ──────────────────────────────────────────────────────────────────────
# Optional: use SerperDevTool if SERPER_API_KEY is set
search_tool = SerperDevTool() if os.getenv("SERPER_API_KEY") else None
tool_list = [search_tool] if search_tool else []


# ── Agents ─────────────────────────────────────────────────────────────────────
content_researcher = Agent(
    role="Content Researcher",
    goal=(
        "Research the given topic thoroughly to gather accurate facts, "
        "recent developments, statistics, and expert insights."
    ),
    backstory=(
        "You are an experienced researcher with expertise in fact-checking "
        "and finding the most relevant, up-to-date information. You excel at "
        "identifying key themes, trends, and supporting data for content creation."
    ),
    tools=tool_list,
    llm=llm,
    verbose=True,
    allow_delegation=False,
)

content_writer = Agent(
    role="Content Writer",
    goal=(
        "Write engaging, informative, and well-structured blog posts based on "
        "research findings, targeting the specified audience."
    ),
    backstory=(
        "You are a talented content writer with years of experience creating "
        "compelling blog posts, articles, and digital content. You know how to "
        "craft narratives that educate, engage, and inspire readers."
    ),
    llm=llm,
    verbose=True,
    allow_delegation=False,
)

content_editor = Agent(
    role="Content Editor",
    goal=(
        "Review and polish the drafted content for clarity, accuracy, grammar, "
        "SEO optimization, and overall quality."
    ),
    backstory=(
        "You are a meticulous editor with a sharp eye for detail. You ensure "
        "all content meets high editorial standards, flows naturally, and is "
        "optimized for both readers and search engines."
    ),
    llm=llm,
    verbose=True,
    allow_delegation=False,
)


# ── Tasks ──────────────────────────────────────────────────────────────────────
def create_content_crew(topic: str, audience: str, word_count: int = 800) -> Crew:
    research_task = Task(
        description=(
            f"Research the topic: '{topic}'\n"
            f"Target audience: {audience}\n\n"
            "Gather: key facts, recent developments (2024-2025), statistics, "
            "expert opinions, and 3-5 main talking points. "
            "Organize findings in a structured outline."
        ),
        agent=content_researcher,
        expected_output=(
            "A structured research brief with: topic overview, 5+ key facts, "
            "recent trends, statistics, and content outline."
        ),
    )

    writing_task = Task(
        description=(
            f"Write a {word_count}-word blog post about: '{topic}'\n"
            f"Audience: {audience}\n\n"
            "Use the research brief to write an engaging article. Include:\n"
            "- Compelling headline and introduction\n"
            "- Well-structured sections with subheadings\n"
            "- Supporting data and examples\n"
            "- Clear conclusion with key takeaways\n"
            "- Conversational yet authoritative tone"
        ),
        agent=content_writer,
        expected_output=f"A complete {word_count}-word blog post with title, sections, and conclusion.",
        context=[research_task],
    )

    editing_task = Task(
        description=(
            "Edit and finalize the blog post:\n"
            "1. Fix grammar, spelling, and punctuation\n"
            "2. Improve sentence structure and flow\n"
            "3. Add/suggest SEO keywords naturally\n"
            "4. Ensure consistent tone throughout\n"
            "5. Add a meta description (150-160 chars)\n"
            "6. Format with proper markdown headings"
        ),
        agent=content_editor,
        expected_output=(
            "Final polished blog post in markdown format with meta description, "
            "proper headings, and publication-ready content."
        ),
        context=[writing_task],
    )

    return Crew(
        agents=[content_researcher, content_writer, content_editor],
        tasks=[research_task, writing_task, editing_task],
        process=Process.sequential,
        verbose=True,
    )


# ── Runner ─────────────────────────────────────────────────────────────────────
def generate_content(topic: str, audience: str = "general readers", word_count: int = 800) -> str:
    crew = create_content_crew(topic, audience, word_count)
    result = crew.kickoff()
    return str(result)


def main():
    print("✍️  Content Creation Agent\n" + "=" * 60)
    topic = "The Impact of Artificial Intelligence on Healthcare in 2025"
    audience = "healthcare professionals and tech enthusiasts"
    print(f"Topic: {topic}")
    print(f"Audience: {audience}")
    print(f"Word count target: 800\n")
    print("Starting content creation pipeline...\n")
    result = generate_content(topic, audience, 800)
    print("\n" + "=" * 60)
    print("📄 Final Article:")
    print("=" * 60)
    print(result)
    print("\n✅ Content creation complete!")


if __name__ == "__main__":
    main()
