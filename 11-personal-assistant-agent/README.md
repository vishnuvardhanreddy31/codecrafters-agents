# 🤖 Personal Assistant Agent

An intelligent personal productivity assistant built with **LangGraph** that manages tasks, schedules events, saves notes, and helps you stay organized through natural conversation.

## Features

- ✅ **Task Management**: Create, prioritize, and complete tasks with deadlines
- 📅 **Calendar Events**: Schedule meetings and events with duration tracking
- 📝 **Notes**: Save and search ideas, meeting notes, and reminders
- ⏰ **Time Awareness**: Always knows current date/time for smart scheduling
- 💬 **Natural conversation**: Understands requests like "remind me to call John tomorrow"

## Architecture

```
User Request → Assistant Agent → [Add Task] | [Schedule Event] | [Save Note] | [Search Notes]
                    ↑___________________________|
                    (multi-tool orchestration)
                          ↓
                Confirmation + Proactive Suggestions
```

## Tech Stack

- **Framework**: LangGraph 0.2+
- **LLM**: OpenAI GPT-4o-mini (OpenAI-compatible API)
- **Tools**: Task Manager, Calendar, Notes, DateTime
- **Pattern**: Conversational multi-tool agent with session state
- **Storage**: In-memory (session-based) — easily extensible to SQLite/PostgreSQL

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

Interactive session examples:
```
Alex: Add a high-priority task to prepare Q1 report due Friday
Assistant: ✅ Task added: 'Prepare Q1 report' (Priority: high, Due: Friday)

Alex: Schedule a team standup tomorrow at 9am for 30 minutes
Assistant: 📅 Event scheduled: 'Team standup' - tomorrow at 09:00 - 09:30

Alex: Save a note about the new product idea I had
Assistant: What would you like to note? [continues conversation]

Alex: What tasks do I have pending?
Assistant: [Lists all pending tasks with priorities]
```

## Available Commands

| Intent | Example |
|--------|---------|
| Add task | "Add task to review Q1 report by Friday" |
| Complete task | "Mark task 3 as done" |
| View tasks | "Show my pending tasks" |
| Schedule event | "Schedule meeting at 2pm tomorrow" |
| View schedule | "What's on my calendar today?" |
| Save note | "Note: great idea about feature X" |
| Search notes | "Find my notes about the budget" |
| Current time | "What day is it?" |

## Extending Storage

The agent uses in-memory lists for simplicity. To persist data:
```python
# Replace in-memory lists with database calls
import sqlite3
# Or use SQLAlchemy, MongoDB, etc.
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | ✅ | OpenAI API key |
