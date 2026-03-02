"""
Meeting Summarizer Agent
Uses LangGraph to analyze meeting transcripts, extract action items,
identify decisions, and generate structured meeting summaries.
"""

import os
import re
from typing import Annotated, TypedDict, List, Optional
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()

# ── Sample transcript ──────────────────────────────────────────────────────────
SAMPLE_TRANSCRIPT = """
[Q1 Product Review Meeting - January 15, 2025, 2:00 PM]
Attendees: Sarah (Product Manager), Mike (Engineering Lead), Lisa (Design), Tom (Marketing)

Sarah: Let's get started. We have a lot to cover today. First, let's review Q4 performance.

Mike: The API response times improved by 40% after the optimization work. We also fixed
15 critical bugs from the backlog. The team shipped 3 major features on schedule.

Sarah: That's great news. What about the user authentication issues we had?

Mike: We resolved those. Two-factor authentication is now working correctly for 99.8%
of users. We need to follow up on the remaining 0.2% - I'll assign that to Dave by Friday.

Lisa: On the design side, we completed the new onboarding flow. User testing showed a
25% improvement in completion rates. However, we need to revisit the mobile dashboard -
several users reported confusion with the navigation. Can we schedule a design review?

Sarah: Absolutely. Lisa, please organize a design review session by next Wednesday.
Include the mobile team. Mike, can you have the engineering team join?

Mike: Yes, I'll make sure they're available. We should probably also discuss the
Q1 roadmap priorities.

Sarah: Right. Tom, what are the marketing priorities?

Tom: We're planning a major campaign for the new enterprise tier launch in March.
We need the feature complete by February 28th. Also, we need better analytics
capabilities - our customers are asking for custom dashboards.

Sarah: Custom dashboards should be on the Q2 roadmap. Mike, can you estimate effort
for the enterprise tier features?

Mike: I'll have estimates ready by tomorrow EOD. I'd say it's roughly 3-4 weeks of work.

Sarah: Perfect. Let's also talk about the customer feedback system. We need to implement
NPS surveys. Tom, can you own that initiative?

Tom: Sure. I'll research NPS tools and have a recommendation by end of next week.

Sarah: Great. Before we close, any blockers?

Mike: We need DevOps access to the new production servers. I've been waiting 2 weeks.
Sarah, can you escalate this?

Sarah: I'll handle that today - will email the IT director. Any other blockers?

Lisa: I need budget approval for the new design tools - about $500/month. Can you
approve that Sarah?

Sarah: Yes, I'll approve that. Send me the details. Okay, let's wrap up. Good meeting everyone.
"""


# ── State ──────────────────────────────────────────────────────────────────────
class MeetingState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    transcript: str
    meeting_title: str


# ── Tools ──────────────────────────────────────────────────────────────────────
@tool
def extract_action_items(transcript: str) -> str:
    """Extract all action items from a meeting transcript.
    Returns a list of tasks with owners, deadlines, and descriptions."""
    action_indicators = [
        r"(?:will|shall|should|need to|needs to|going to|must)\s+(.+?)(?:\.|$)",
        r"(?:action:|todo:|follow up:|follow-up:)\s*(.+?)(?:\.|$)",
        r"(?:please|kindly)\s+(.+?)(?:\.|$)",
        r"(?:by|before|until)\s+(?:Monday|Tuesday|Wednesday|Thursday|Friday|"
        r"next week|tomorrow|EOD|end of day|end of week)\s+(.+?)(?:\.|$)",
        r"([A-Z][a-z]+),?\s+(?:can you|could you|please)\s+(.+?)(?:\?|$)",
    ]

    lines = transcript.split("\n")
    action_items = []

    for line in lines:
        for pattern in action_indicators:
            matches = re.findall(pattern, line, re.IGNORECASE)
            for match in matches:
                item = match if isinstance(match, str) else " ".join(match)
                if len(item.strip()) > 10 and len(item.strip()) < 200:
                    action_items.append(item.strip())

    # Deduplicate
    seen = set()
    unique_items = []
    for item in action_items:
        key = item[:50].lower()
        if key not in seen:
            seen.add(key)
            unique_items.append(item)

    if not unique_items:
        return "No specific action items detected (transcript may need manual review)."

    result = f"Action Items Found ({len(unique_items)}):\n"
    for i, item in enumerate(unique_items[:15], 1):
        result += f"  {i}. {item}\n"
    return result


@tool
def identify_decisions(transcript: str) -> str:
    """Identify key decisions made during the meeting."""
    decision_patterns = [
        r"(?:decided|agreed|approved|confirmed|resolved|will proceed|going with)\s+(.+?)(?:\.|$)",
        r"(?:decision:|resolution:|agreed:)\s*(.+?)(?:\.|$)",
        r"(?:let's|we'll|we will)\s+(.+?)(?:\.|$)",
    ]

    decisions = []
    for line in transcript.split("\n"):
        for pattern in decision_patterns:
            matches = re.findall(pattern, line, re.IGNORECASE)
            for match in matches:
                if len(match.strip()) > 10:
                    decisions.append(match.strip())

    seen = set()
    unique = []
    for d in decisions:
        key = d[:40].lower()
        if key not in seen:
            seen.add(key)
            unique.append(d)

    if not unique:
        return "No formal decisions explicitly detected."
    return f"Key Decisions ({len(unique)}):\n" + "\n".join(f"  • {d}" for d in unique[:10])


@tool
def extract_attendees_and_roles(transcript: str) -> str:
    """Extract meeting attendees and their roles from the transcript header."""
    lines = transcript.split("\n")
    for line in lines:
        if "attendees:" in line.lower() or "participants:" in line.lower():
            return f"Meeting Participants: {line.strip()}"

    # Try to extract from dialogue
    speakers = set()
    for line in lines:
        match = re.match(r"^([A-Z][a-z]+):", line.strip())
        if match:
            speakers.add(match.group(1))

    if speakers:
        return f"Identified Speakers: {', '.join(sorted(speakers))}"
    return "Could not extract attendee list from transcript."


@tool
def generate_meeting_summary(
    transcript: str,
    action_items: str,
    decisions: str,
    attendees: str,
) -> str:
    """Compile all extracted information into a professional meeting summary."""
    meeting_date = "N/A"
    meeting_title = "Meeting"
    duration = "N/A"

    # Extract meeting info from first lines
    lines = [l.strip() for l in transcript.split("\n") if l.strip()][:5]
    for line in lines:
        if any(word in line.lower() for word in ["meeting", "review", "standup", "sync"]):
            meeting_title = line.strip("[]")
            break
        date_match = re.search(r"\b\w+ \d{1,2},?\s+\d{4}\b", line)
        if date_match:
            meeting_date = date_match.group()

    summary = (
        "=" * 50 + "\n"
        "       MEETING SUMMARY\n"
        "=" * 50 + "\n\n"
        f"📅 Meeting: {meeting_title}\n"
        f"📍 Date: {meeting_date}\n"
        f"👥 {attendees}\n\n"
        "KEY DECISIONS\n"
        "-" * 30 + "\n"
        f"{decisions}\n\n"
        "ACTION ITEMS\n"
        "-" * 30 + "\n"
        f"{action_items}\n\n"
        "NEXT STEPS\n"
        "-" * 30 + "\n"
        "• Follow up on all action items before next meeting\n"
        "• Share this summary with all attendees\n"
        "• Schedule follow-up meetings as needed\n\n"
        "=" * 50
    )
    return summary


tools = [
    extract_action_items,
    identify_decisions,
    extract_attendees_and_roles,
    generate_meeting_summary,
]

SYSTEM_PROMPT = """You are a professional meeting analyst and executive assistant.

Your process for every meeting transcript:
1. Use extract_attendees_and_roles to identify participants
2. Use identify_decisions to capture key decisions made
3. Use extract_action_items to list all tasks with owners and deadlines
4. Use generate_meeting_summary to create the final summary

Then provide additional insights:
- Key themes and topics discussed
- Any risks or concerns mentioned
- Suggested follow-up meeting agenda items
- Blockers or escalations needed

Format everything clearly for easy reference and sharing."""


def build_meeting_graph():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1).bind_tools(tools)

    def summarizer(state: MeetingState) -> dict:
        return {"messages": [llm.invoke(state["messages"])]}

    tool_node = ToolNode(tools)
    graph = StateGraph(MeetingState)
    graph.add_node("summarizer", summarizer)
    graph.add_node("tools", tool_node)
    graph.add_edge(START, "summarizer")
    graph.add_conditional_edges("summarizer", tools_condition)
    graph.add_edge("tools", "summarizer")
    return graph.compile()


def summarize_meeting(transcript: str, meeting_title: str = "Meeting") -> str:
    app = build_meeting_graph()
    prompt = (
        f"System: {SYSTEM_PROMPT}\n\n"
        f"Analyze this meeting transcript and generate a comprehensive summary:\n\n"
        f"{transcript}"
    )
    state: MeetingState = {
        "messages": [HumanMessage(content=prompt)],
        "transcript": transcript,
        "meeting_title": meeting_title,
    }
    result = app.invoke(state)
    return result["messages"][-1].content


def main():
    print("📋 Meeting Summarizer Agent\n" + "=" * 60)
    print("Analyzing Q1 Product Review Meeting transcript...\n")
    summary = summarize_meeting(SAMPLE_TRANSCRIPT, "Q1 Product Review Meeting")
    print(summary)
    print("\n✅ Meeting summarization complete!")


if __name__ == "__main__":
    main()
