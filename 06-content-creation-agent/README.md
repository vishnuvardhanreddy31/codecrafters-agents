# ✍️ Content Creation Agent

A multi-agent content creation pipeline built with **CrewAI** featuring a Researcher, Writer, and Editor working together to produce high-quality, publication-ready blog posts and articles.

## Features

- 🔍 **Research Agent**: Gathers facts, trends, statistics, and expert insights
- ✍️ **Writer Agent**: Crafts engaging, well-structured content targeting your audience
- 📝 **Editor Agent**: Polishes for grammar, flow, SEO, and readability
- 🎯 **SEO optimization**: Natural keyword integration and meta descriptions
- 📊 **Customizable**: Adjust topic, audience, tone, and word count

## Architecture

```
Topic + Audience
     ↓
[Content Researcher] → Research Brief
     ↓
[Content Writer] → Draft Article (word-count target)
     ↓
[Content Editor] → Publication-Ready Article + Meta Description
```

## Tech Stack

- **Framework**: CrewAI 0.80+
- **LLM**: OpenAI GPT-4o-mini (OpenAI-compatible API)
- **Pattern**: Sequential multi-agent pipeline
- **Optional**: SerperDev search API for real-time research

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file:
```env
OPENAI_API_KEY=your_openai_api_key
SERPER_API_KEY=your_serper_api_key  # Optional: enables web search
```

## Usage

```bash
python agent.py
```

Or use programmatically:
```python
from agent import generate_content

article = generate_content(
    topic="How to Build a Sustainable Remote Work Culture in 2025",
    audience="HR professionals and team managers",
    word_count=1000
)
print(article)
```

## Agent Roles

| Agent | Role | Specialization |
|-------|------|----------------|
| **Content Researcher** | Research lead | Fact-finding, current developments, statistics |
| **Content Writer** | Primary writer | Narrative, engagement, structure |
| **Content Editor** | Quality assurance | Grammar, SEO, meta descriptions |

## Output Format

The final article includes:
- **Meta description** (150-160 characters, SEO-optimized)
- **Headline** (H1)
- **Introduction** with a compelling hook
- **Body sections** with H2/H3 headings
- **Conclusion** with key takeaways
- **Markdown formatting** ready for any CMS

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | ✅ | OpenAI API key |
| `SERPER_API_KEY` | Optional | Web search for research agent |
