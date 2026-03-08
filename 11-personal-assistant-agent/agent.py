"""
Personal Assistant Agent
Uses LangGraph with OpenAI to manage tasks, schedule events,
set reminders, and help with productivity through natural conversation.
"""

import os
import json
from datetime import datetime, timedelta
from typing import Annotated, TypedDict, List, Optional
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()

# ── In-memory storage ──────────────────────────────────────────────────────────
_tasks: List[dict] = []
_events: List[dict] = []
_notes: List[dict] = []


# ── State ──────────────────────────────────────────────────────────────────────
class AssistantState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    user_name: str
    current_datetime: str


# ── Tools ──────────────────────────────────────────────────────────────────────
@tool
def add_task(title: str, priority: str = "medium", due_date: Optional[str] = None) -> str:
    """Add a new task to the to-do list.
    priority: low | medium | high
    due_date: optional date string (e.g., 'tomorrow', '2025-03-15')"""
    task = {
        "id": len(_tasks) + 1,
        "title": title,
        "priority": priority,
        "due_date": due_date,
        "status": "pending",
        "created_at": datetime.now().isoformat(),
    }
    _tasks.append(task)
    return f"✅ Task added: '{title}' (Priority: {priority}" + (f", Due: {due_date})" if due_date else ")")


@tool
def list_tasks(filter_status: str = "pending") -> str:
    """List tasks from the to-do list.
    filter_status: pending | completed | all"""
    if filter_status == "all":
        tasks = _tasks
    else:
        tasks = [t for t in _tasks if t["status"] == filter_status]

    if not tasks:
        return f"No {filter_status} tasks found."

    result = f"Tasks ({filter_status}):\n"
    priority_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}
    for task in tasks:
        emoji = priority_emoji.get(task["priority"], "⚪")
        due = f" | Due: {task['due_date']}" if task["due_date"] else ""
        status = "✅" if task["status"] == "completed" else "⬜"
        result += f"  {status} [{task['id']}] {emoji} {task['title']}{due}\n"
    return result


@tool
def complete_task(task_id: int) -> str:
    """Mark a task as completed by its ID."""
    for task in _tasks:
        if task["id"] == task_id:
            task["status"] = "completed"
            task["completed_at"] = datetime.now().isoformat()
            return f"✅ Task {task_id} '{task['title']}' marked as complete!"
    return f"Task {task_id} not found."


@tool
def schedule_event(
    title: str,
    date: str,
    time: str = "09:00",
    duration_minutes: int = 60,
    notes: str = "",
) -> str:
    """Schedule a calendar event.
    date: date string (e.g., '2025-03-15' or 'tomorrow')
    time: 24-hour format (e.g., '14:30')
    duration_minutes: length of the event"""
    event = {
        "id": len(_events) + 1,
        "title": title,
        "date": date,
        "time": time,
        "duration_minutes": duration_minutes,
        "notes": notes,
        "created_at": datetime.now().isoformat(),
    }
    _events.append(event)
    end_time_hour = int(time.split(":")[0]) + duration_minutes // 60
    end_time_min = (int(time.split(":")[1]) + duration_minutes % 60) % 60
    return (
        f"📅 Event scheduled: '{title}'\n"
        f"   Date: {date} at {time} - {end_time_hour:02d}:{end_time_min:02d}\n"
        f"   Duration: {duration_minutes} minutes"
        + (f"\n   Notes: {notes}" if notes else "")
    )


@tool
def get_schedule(date: str = "today") -> str:
    """Get the schedule for a specific date.
    date: 'today', 'tomorrow', or a date string like '2025-03-15'"""
    if not _events:
        return f"No events scheduled for {date}."
    events_for_date = [e for e in _events if e["date"] == date or date == "all"]
    if not events_for_date:
        return f"No events found for '{date}'."
    result = f"Schedule for {date}:\n"
    for event in sorted(events_for_date, key=lambda x: x["time"]):
        result += f"  🕐 {event['time']} - {event['title']} ({event['duration_minutes']}min)\n"
        if event.get("notes"):
            result += f"     📝 {event['notes']}\n"
    return result


@tool
def save_note(title: str, content: str, tags: str = "") -> str:
    """Save a quick note or idea.
    tags: comma-separated tags for organization"""
    note = {
        "id": len(_notes) + 1,
        "title": title,
        "content": content,
        "tags": [t.strip() for t in tags.split(",") if t.strip()],
        "created_at": datetime.now().isoformat(),
    }
    _notes.append(note)
    return f"📝 Note saved: '{title}'" + (f" [Tags: {tags}]" if tags else "")


@tool
def search_notes(query: str) -> str:
    """Search through saved notes by keyword."""
    if not _notes:
        return "No notes saved yet."
    results = [
        n for n in _notes
        if query.lower() in n["title"].lower() or query.lower() in n["content"].lower()
    ]
    if not results:
        return f"No notes found matching '{query}'."
    output = f"Notes matching '{query}':\n"
    for note in results:
        output += f"\n  📝 [{note['id']}] {note['title']}\n     {note['content'][:100]}...\n"
    return output


@tool
def get_current_datetime() -> str:
    """Get the current date and time."""
    now = datetime.now()
    return (
        f"Current date/time: {now.strftime('%A, %B %d, %Y at %I:%M %p')}\n"
        f"Week number: {now.isocalendar()[1]}\n"
        f"Days until weekend: {5 - now.weekday() if now.weekday() < 5 else 0}"
    )


tools = [
    add_task, list_tasks, complete_task,
    schedule_event, get_schedule,
    save_note, search_notes,
    get_current_datetime,
]

SYSTEM_PROMPT = """You are a helpful personal assistant helping manage tasks, schedule,
and notes for maximum productivity.

Capabilities:
- Task management: add, list, and complete tasks with priorities
- Calendar: schedule events and view upcoming schedule
- Notes: save and search notes and ideas
- Time awareness: always be aware of the current date/time

Be proactive, organized, and supportive. When the user asks to do something, use the
appropriate tool immediately. For complex requests, break them into multiple actions.
Always confirm what was done and suggest next steps."""


def build_assistant_graph():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5).bind_tools(tools)

    def assistant(state: AssistantState) -> dict:
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(state["messages"])
        return {"messages": [llm.invoke(messages)]}

    tool_node = ToolNode(tools)
    graph = StateGraph(AssistantState)
    graph.add_node("assistant", assistant)
    graph.add_node("tools", tool_node)
    graph.add_edge(START, "assistant")
    graph.add_conditional_edges("assistant", tools_condition)
    graph.add_edge("tools", "assistant")
    return graph.compile()


def main():
    app = build_assistant_graph()
    user_name = input("What's your name? ").strip() or "User"
    state: AssistantState = {
        "messages": [],
        "user_name": user_name,
        "current_datetime": datetime.now().isoformat(),
    }

    print(f"\n🤖 Personal Assistant Agent")
    print("=" * 60)
    print(f"Hello {user_name}! I'm your personal assistant. How can I help you today?")
    print("Try: 'Add a task to review quarterly report by Friday'")
    print("Or:  'Schedule a team meeting tomorrow at 2pm for 1 hour'")
    print("Type 'quit' to exit.\n")

    while True:
        user_input = input(f"{user_name}: ").strip()
        if user_input.lower() in ("quit", "exit", "bye"):
            print("Assistant: Goodbye! Have a productive day! 👋")
            break
        if not user_input:
            continue

        state["messages"].append(HumanMessage(content=user_input))
        result = app.invoke(state)
        state["messages"] = result["messages"]
        print(f"\nAssistant: {result['messages'][-1].content}\n")


if __name__ == "__main__":
    main()
