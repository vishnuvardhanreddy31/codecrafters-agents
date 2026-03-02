"""
News Aggregator Agent
Uses CrewAI with specialized agents (Collector, Analyst, Editor)
to gather, analyze, and summarize news on any topic.
"""

import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from crewai.tools import BaseTool
import requests
from pydantic import BaseModel, Field

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)


# ── Tools ──────────────────────────────────────────────────────────────────────
class NewsSearchInput(BaseModel):
    topic: str = Field(description="The news topic to search for")
    max_results: int = Field(default=5, description="Maximum number of results")


class NewsSearchTool(BaseTool):
    name: str = "news_search"
    description: str = "Search for recent news articles on a given topic using RSS feeds."
    args_schema: type[BaseModel] = NewsSearchInput

    def _run(self, topic: str, max_results: int = 5) -> str:
        try:
            import feedparser
            # Use Google News RSS as a public news source
            url = f"https://news.google.com/rss/search?q={topic.replace(' ', '+')}&hl=en-US&gl=US&ceid=US:en"
            feed = feedparser.parse(url)
            if not feed.entries:
                return f"No news articles found for '{topic}'."
            articles = []
            for entry in feed.entries[:max_results]:
                articles.append(
                    f"Title: {entry.title}\n"
                    f"Published: {entry.get('published', 'N/A')}\n"
                    f"Summary: {entry.get('summary', 'N/A')[:200]}\n"
                    f"Link: {entry.link}"
                )
            return f"News articles about '{topic}':\n\n" + "\n\n---\n\n".join(articles)
        except ImportError:
            return (
                f"feedparser not installed. Sample news for '{topic}':\n"
                f"1. Breaking: New developments in {topic} - Major advances reported\n"
                f"2. Analysis: What experts say about {topic}\n"
                f"3. {topic}: Key trends and future outlook"
            )
        except Exception as e:
            return f"Error fetching news: {e}"


class SentimentAnalysisInput(BaseModel):
    text: str = Field(description="Text to analyze for sentiment")


class SentimentAnalysisTool(BaseTool):
    name: str = "analyze_sentiment"
    description: str = "Analyze the overall sentiment and tone of news content."
    args_schema: type[BaseModel] = SentimentAnalysisInput

    def _run(self, text: str) -> str:
        text_lower = text.lower()
        positive_words = {
            "breakthrough", "growth", "success", "improvement", "advance",
            "positive", "rising", "gain", "benefit", "innovation",
        }
        negative_words = {
            "crisis", "decline", "fail", "loss", "threat", "problem",
            "risk", "concern", "drop", "warning", "negative",
        }
        words = set(text_lower.split())
        pos_count = len(words & positive_words)
        neg_count = len(words & negative_words)

        if pos_count > neg_count:
            sentiment = "Positive 📈"
        elif neg_count > pos_count:
            sentiment = "Negative 📉"
        else:
            sentiment = "Neutral/Mixed ↔️"

        return (
            f"Sentiment Analysis:\n"
            f"  Overall tone: {sentiment}\n"
            f"  Positive indicators: {pos_count}\n"
            f"  Negative indicators: {neg_count}\n"
            f"  Word count: {len(text.split())}"
        )


news_search = NewsSearchTool()
sentiment_tool = SentimentAnalysisTool()


# ── Agents ─────────────────────────────────────────────────────────────────────
news_collector = Agent(
    role="News Collector",
    goal="Gather comprehensive news articles from multiple angles on the given topic.",
    backstory=(
        "You are an expert news researcher who knows how to find relevant, "
        "credible news from diverse sources. You gather articles that cover "
        "different perspectives: mainstream, technical, and analytical views."
    ),
    tools=[news_search],
    llm=llm,
    verbose=True,
    allow_delegation=False,
)

news_analyst = Agent(
    role="News Analyst",
    goal=(
        "Analyze collected news to identify key themes, trends, sentiment, "
        "and the most important developments."
    ),
    backstory=(
        "You are a seasoned journalist and news analyst with expertise in "
        "identifying patterns, biases, and significance in news coverage. "
        "You excel at separating signal from noise."
    ),
    tools=[sentiment_tool],
    llm=llm,
    verbose=True,
    allow_delegation=False,
)

news_editor = Agent(
    role="News Editor",
    goal=(
        "Compile the analysis into a clear, well-structured news briefing "
        "that keeps readers informed and engaged."
    ),
    backstory=(
        "You are an experienced news editor who crafts compelling briefings "
        "that are accurate, balanced, and easy to read. You prioritize "
        "clarity and relevance for your audience."
    ),
    llm=llm,
    verbose=True,
    allow_delegation=False,
)


# ── Crew factory ───────────────────────────────────────────────────────────────
def build_news_crew(topic: str, num_articles: int = 5) -> Crew:
    collection_task = Task(
        description=(
            f"Search for and collect {num_articles} recent news articles about: '{topic}'\n"
            "Use the news_search tool to gather articles. Try multiple search variations "
            "if needed to get comprehensive coverage. Compile all articles with titles, "
            "publication dates, and summaries."
        ),
        agent=news_collector,
        expected_output=f"Collection of {num_articles}+ news articles about {topic} with titles, dates, and summaries.",
    )

    analysis_task = Task(
        description=(
            f"Analyze the collected news articles about '{topic}':\n"
            "1. Use analyze_sentiment on the collected content\n"
            "2. Identify the 3-5 most significant developments\n"
            "3. Find common themes across articles\n"
            "4. Note any conflicting perspectives or controversies\n"
            "5. Assess the overall news cycle direction for this topic"
        ),
        agent=news_analyst,
        expected_output=(
            "News analysis report with: sentiment assessment, top 5 developments, "
            "key themes, notable controversies, and trend direction."
        ),
        context=[collection_task],
    )

    editorial_task = Task(
        description=(
            f"Write a professional news briefing about '{topic}':\n"
            "Structure:\n"
            "1. **Executive Summary** (2-3 sentences)\n"
            "2. **Top Stories** (3-5 bullet points with key facts)\n"
            "3. **Analysis & Trends** (2-3 paragraphs)\n"
            "4. **Key Quotes or Data Points** (if available)\n"
            "5. **What to Watch** (upcoming developments to monitor)\n\n"
            "Keep it concise, factual, and engaging. 400-500 words total."
        ),
        agent=news_editor,
        expected_output="Complete news briefing in structured markdown format, 400-500 words.",
        context=[collection_task, analysis_task],
    )

    return Crew(
        agents=[news_collector, news_analyst, news_editor],
        tasks=[collection_task, analysis_task, editorial_task],
        process=Process.sequential,
        verbose=True,
    )


def aggregate_news(topic: str, num_articles: int = 5) -> str:
    crew = build_news_crew(topic, num_articles)
    result = crew.kickoff()
    return str(result)


def main():
    print("📰 News Aggregator Agent\n" + "=" * 60)
    topic = input("Enter a news topic (e.g., 'artificial intelligence', 'climate change'): ").strip()
    if not topic:
        topic = "artificial intelligence"
    print(f"\nGathering news about: {topic}\n")
    briefing = aggregate_news(topic)
    print("\n" + "=" * 60)
    print(f"📋 News Briefing: {topic}")
    print("=" * 60)
    print(briefing)
    print("\n✅ News aggregation complete!")


if __name__ == "__main__":
    main()
