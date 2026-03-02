# ✈️ Travel Planning Agent

A comprehensive travel planning system built with **CrewAI** featuring a Destination Expert, Itinerary Specialist, and Budget Analyst working together to create personalized, detailed travel plans.

## Features

- 🗺️ **Destination Research**: Attractions, culture, visa requirements, safety, hidden gems
- 📅 **Day-by-Day Itinerary**: Morning/afternoon/evening plans with timing and logistics
- 💰 **Budget Breakdown**: Detailed cost estimates per category with money-saving tips
- 🌮 **Local Cuisine**: Restaurant recommendations and must-try dishes
- ⚡ **Weather alternatives**: Backup plans for each activity

## Architecture

```
Destination + Preferences
      ↓
[Destination Expert] → Research Brief (attractions, culture, logistics)
      ↓
[Itinerary Specialist] → Day-by-Day Schedule
      ↓
[Budget Analyst] → Complete Budget Breakdown + Saving Tips
```

## Tech Stack

- **Framework**: CrewAI 0.80+
- **LLM**: OpenAI GPT-4o-mini (OpenAI-compatible API)
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
from agent import plan_trip

plan = plan_trip(
    destination="Barcelona, Spain",
    duration_days=5,
    budget_usd=2000,
    interests="architecture, food, beaches, nightlife",
    travel_style="mid-range"
)
print(plan)
```

## Configuration Options

| Parameter | Options | Default |
|-----------|---------|---------|
| `destination` | Any city/country | Tokyo, Japan |
| `duration_days` | 1-30 | 7 |
| `budget_usd` | Any amount | 3000 |
| `interests` | Comma-separated | culture, food, history |
| `travel_style` | budget/mid-range/luxury | mid-range |

## Output Sections

1. **Destination Guide** - Top attractions, neighborhoods, practical tips
2. **Daily Itinerary** - Hour-by-hour schedule for each day
3. **Budget Breakdown** - Per-category cost estimates
4. **Money-Saving Tips** - Specific actionable savings strategies
5. **What to Pack** - Weather and activity-appropriate packing list

## Example Output

```
Day 1 - Arrival & Shibuya
  Morning: Arrive at Narita Airport, check into hotel in Shinjuku
  Afternoon: Explore Shibuya Crossing, shopping in Harajuku
  Evening: Ramen dinner at Ichiran, explore Kabukicho...

Budget Breakdown:
  Flights (est.): $800
  Accommodation (7 nights): $700
  Food (daily $50): $350
  Transport (IC Card): $100
  Activities: $250
  TOTAL: $2,200
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | ✅ | OpenAI API key |
