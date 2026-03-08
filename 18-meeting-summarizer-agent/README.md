# 📋 Meeting Summarizer Agent

An intelligent meeting analysis agent built with **LangGraph** that processes meeting transcripts to extract action items, identify decisions, and generate structured summaries with follow-up priorities.

## Features

- ✅ **Action item extraction**: Who does what by when
- 🎯 **Decision identification**: Key outcomes and agreements
- 👥 **Attendee detection**: Participants and their roles
- 📊 **Structured summaries**: Executive format with sections
- 🚨 **Blocker identification**: Escalation items flagged
- 📧 **Meeting minutes**: Ready-to-share formatted output

## Architecture

```
Meeting Transcript → Summarizer Agent → [Extract Attendees] → [Identify Decisions]
                           ↓
                   [Extract Action Items] → [Generate Summary]
                           ↓
                Formatted Meeting Minutes
```

## Tech Stack

- **Framework**: LangGraph 0.2+
- **LLM**: OpenAI GPT-4o-mini (OpenAI-compatible API)
- **Tools**: Attendee Extractor, Decision Identifier, Action Item Extractor, Summary Generator
- **Pattern**: Document analysis agent with structured extraction

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
from agent import summarize_meeting

# From a file
with open("meeting_transcript.txt", "r") as f:
    transcript = f.read()

summary = summarize_meeting(
    transcript=transcript,
    meeting_title="Q1 Planning Session"
)
print(summary)
```

## Input Format

The agent accepts any transcript format:
- Raw conversation logs
- Zoom/Teams auto-transcripts
- Manual meeting notes
- Voice-to-text outputs

Ideal format:
```
[Meeting Title - Date]
Attendees: Name (Role), Name (Role)

Name: Their statement here...
Name: Response here...
```

## Output Format

```
==================================================
       MEETING SUMMARY
==================================================

📅 Meeting: Q1 Product Review
📍 Date: January 15, 2025
👥 Attendees: Sarah (PM), Mike (Eng), Lisa (Design)

KEY DECISIONS
-----------
• API optimization approved for Q1
• Design review scheduled for Wednesday

ACTION ITEMS
-----------
1. Mike to provide estimates by EOD tomorrow
2. Lisa to schedule design review by Wednesday
3. Sarah to escalate server access to IT director

NEXT STEPS
-----------
• Follow up on all action items
• Share summary with all attendees
```

## Integration Ideas

- **Slack bot**: Post summaries to meeting channels automatically
- **Email automation**: Send minutes to attendees after each meeting
- **Project management**: Create Jira/Asana tasks from action items
- **Calendar**: Auto-schedule follow-up meetings mentioned in transcript

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | ✅ | OpenAI API key |
