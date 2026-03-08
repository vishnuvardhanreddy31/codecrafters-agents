# 📱 Social Media Agent

A comprehensive social media strategy and content creation system built with **CrewAI** featuring a Strategist, Content Creator, and Analytics Expert to help brands dominate social media.

## Features

- 📊 **Platform strategy**: Tailored approach for each social network
- ✍️ **Ready-to-post content**: 5 captions per platform with hashtags
- 📅 **7-day content calendar**: Organized posting schedule
- 📈 **KPI framework**: Metrics, targets, and measurement methodology
- 🎯 **Audience insights**: Content pillars aligned with audience interests

## Architecture

```
Brand + Goals + Platforms
        ↓
[Social Strategist] → 30-Day Strategy + Content Pillars
        ↓
[Content Creator] → Platform-Specific Content + Calendar
        ↓
[Analytics Expert] → KPIs + Measurement Framework
```

## Tech Stack

- **Framework**: CrewAI 0.80+
- **LLM**: OpenAI GPT-4o-mini (OpenAI-compatible API)
- **Supported Platforms**: Instagram, LinkedIn, Twitter/X, TikTok, Facebook
- **Pattern**: Sequential multi-agent pipeline with context passing

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
from agent import create_social_strategy

package = create_social_strategy(
    brand_name="TechStartup",
    industry="B2B SaaS / productivity tools",
    target_audience="startup founders and product managers",
    goals="Generate 500 leads/month, grow LinkedIn to 10K followers",
    platforms=["linkedin", "twitter"],
    tone="thought leadership, data-driven, approachable"
)
print(package)
```

## Platform Specifications

| Platform | Char Limit | Best Content | Optimal Times |
|----------|-----------|--------------|---------------|
| Instagram | 2,200 | Visual + reels | 11am, 5pm |
| LinkedIn | 3,000 | Professional + insights | Tue-Thu 9am |
| Twitter/X | 280 | Short + punchy | 9am, 5pm |
| TikTok | 2,200 | Short video | 7-9pm weekdays |
| Facebook | 63,206 | Community | Wed 1-3pm |

## Output Package

1. **Strategy**
   - 5 content pillars
   - Platform-specific posting frequency
   - Content mix (educational/entertaining/promotional)
   - Hashtag strategy

2. **Content**
   - 5 ready-to-post captions per platform
   - 3 content ideas for this week
   - 1 viral concept
   - 7-day posting calendar

3. **Analytics**
   - Primary KPIs per platform
   - Month 1/3/6 growth targets
   - Weekly tracking checklist
   - Recommended tools

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | ✅ | OpenAI API key |
