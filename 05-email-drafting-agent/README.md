# 📧 Email Drafting Agent

An AI-powered email drafting agent built with **LangGraph** that crafts professional, tone-appropriate emails with structure validation, subject line suggestions, and iterative refinement.

## Features

- ✍️ **Context-aware drafting** based on purpose, recipient, and tone
- 📊 **Tone analysis**: detects and validates formal/informal/professional tone
- 🏗️ **Structure checking**: ensures all email components are present
- 📌 **Subject line generation**: 5 options tailored to the email purpose
- 🔄 **Iterative refinement** until quality standards are met

## Architecture

```
Email Context → Email Agent → [Subject Lines] → [Draft] → [Structure Check] → [Tone Analysis]
                    ↑___________________________|
                    (refines until quality is achieved)
                              ↓
                  Polished, Professional Email
```

## Tech Stack

- **Framework**: LangGraph 0.2+
- **LLM**: OpenAI GPT-4o-mini (OpenAI-compatible API)
- **Tools**: Tone Analyzer, Structure Checker, Subject Line Generator
- **Pattern**: ReAct Agent with quality validation loop

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
from agent import draft_email

context = {
    "purpose": "requesting a meeting with a potential investor",
    "recipient": "Sarah Chen, Partner at Venture Capital Firm",
    "tone": "professional and confident",
    "key_points": "introduce startup, highlight $2M ARR growth, request 30-min call",
    "sender": "Jordan Smith, CEO"
}

email = draft_email(context)
print(email)
```

## Email Context Fields

| Field | Description | Example |
|-------|-------------|---------|
| `purpose` | Email goal | "follow-up after job interview" |
| `recipient` | Who you're writing to | "Hiring Manager, John Doe" |
| `tone` | Desired tone | "professional and enthusiastic" |
| `key_points` | Main content to cover | "express gratitude, reiterate interest" |
| `sender` | Your name | "Alex Chen" |

## Tone Options

- `formal` - Legal/official communications
- `professional` - Business correspondence
- `friendly` - Casual professional communications
- `persuasive` - Sales/pitch emails
- `empathetic` - Customer service/apologies

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | ✅ | OpenAI API key |
