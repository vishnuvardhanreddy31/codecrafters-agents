# 📰 News Aggregator Agent

A multi-agent news gathering and analysis system built with **CrewAI** featuring a News Collector, Analyst, and Editor working together to produce concise, insightful news briefings on any topic.

## Features

- 🌐 **Multi-source news collection** from RSS/Google News feeds
- 📊 **Sentiment analysis**: Positive/negative/neutral tone detection
- 🔍 **Theme identification**: Key patterns and trends across articles
- 📝 **Professional briefings**: Executive summary + analysis + what to watch
- ⚡ **Any topic**: Politics, technology, science, business, sports, and more

## Architecture

```
Topic Input
    ↓
[News Collector] → Raw Articles (5+ sources)
    ↓
[News Analyst] → Sentiment + Key Developments + Themes
    ↓
[News Editor] → Professional News Briefing
```

## Tech Stack

- **Framework**: CrewAI 0.80+
- **LLM**: OpenAI GPT-4o-mini (OpenAI-compatible API)
- **Data Sources**: Google News RSS, feedparser
- **Tools**: Custom NewsSearchTool, SentimentAnalysisTool
- **Pattern**: Sequential multi-agent pipeline

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file:
```env
OPENAI_API_KEY=your_openai_api_key
```

## Usage

```bash
python agent.py
```

Or use programmatically:
```python
from agent import aggregate_news

# Get a briefing on any topic
briefing = aggregate_news("renewable energy 2025", num_articles=7)
print(briefing)
```

## Output Format

```markdown
**Executive Summary**
[2-3 sentence overview of the topic's current state]

**Top Stories**
- [Key development 1]
- [Key development 2]
- [Key development 3]

**Analysis & Trends**
[In-depth analysis paragraphs]

**What to Watch**
[Upcoming developments to monitor]
```

## Sentiment Analysis

The agent evaluates news tone across articles:
- 📈 **Positive**: Growth, breakthrough, success stories
- 📉 **Negative**: Crisis, decline, conflict
- ↔️ **Neutral/Mixed**: Balanced reporting

## Scheduled Runs

For automated daily briefings, add to cron:
```bash
# Daily 8am news briefing
0 8 * * * cd /path/to/project && python agent.py
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | ✅ | OpenAI API key |
