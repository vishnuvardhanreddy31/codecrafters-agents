"""
Email Drafting Agent
Uses LangGraph to draft professional emails with context awareness,
tone adjustment, and iterative refinement.
"""

import os
from typing import Annotated, TypedDict, List, Optional
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()


# ── State ──────────────────────────────────────────────────────────────────────
class EmailState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    email_context: dict
    draft: Optional[str]
    tone: str
    iteration: int


# ── Tools ──────────────────────────────────────────────────────────────────────
@tool
def analyze_email_tone(email_text: str) -> str:
    """Analyze the tone and professionalism of an email draft.
    Returns a report with tone assessment and improvement suggestions."""
    words = email_text.lower().split()
    word_count = len(words)
    sentence_count = email_text.count(".") + email_text.count("!") + email_text.count("?")

    informal_words = {"hey", "yeah", "nope", "cool", "gonna", "wanna", "kinda", "sorta"}
    informal_count = sum(1 for w in words if w.rstrip(".,!?") in informal_words)

    formal_words = {"sincerely", "regards", "hereby", "pursuant", "aforementioned"}
    formal_count = sum(1 for w in words if w.rstrip(".,!?") in formal_words)

    tone = "neutral"
    if informal_count > 2:
        tone = "informal"
    elif formal_count > 2:
        tone = "highly formal"
    else:
        tone = "professional"

    return (
        f"Email Analysis:\n"
        f"  Word count: {word_count}\n"
        f"  Sentences: {sentence_count}\n"
        f"  Detected tone: {tone}\n"
        f"  Informal words: {informal_count}\n"
        f"  Formal indicators: {formal_count}\n"
        f"  Avg sentence length: {word_count / max(sentence_count, 1):.1f} words"
    )


@tool
def check_email_structure(email_text: str) -> str:
    """Check if an email has proper structure: subject line, greeting,
    body, call-to-action, and closing."""
    checks = {
        "Subject line": "subject:" in email_text.lower(),
        "Greeting": any(g in email_text.lower() for g in ["dear", "hello", "hi "]),
        "Body content": len(email_text) > 100,
        "Call to action": any(
            c in email_text.lower()
            for c in ["please", "kindly", "would you", "could you", "let me know"]
        ),
        "Professional closing": any(
            c in email_text.lower()
            for c in ["sincerely", "regards", "best", "thank you", "thanks"]
        ),
    }
    report = "Email Structure Check:\n"
    for element, present in checks.items():
        status = "✅" if present else "❌"
        report += f"  {status} {element}\n"
    score = sum(checks.values())
    report += f"\nStructure score: {score}/{len(checks)}"
    return report


@tool
def suggest_subject_lines(email_purpose: str) -> str:
    """Generate 5 professional subject line options for the given email purpose."""
    templates = {
        "follow_up": [
            "Following Up on Our Recent Discussion",
            "Quick Follow-Up: [Topic]",
            "Checking In: Next Steps for [Project]",
            "Follow-Up: Action Items from [Meeting]",
            "Touching Base on [Subject]",
        ],
        "request": [
            "Request for [Resource/Information]",
            "Seeking Your Input on [Topic]",
            "Request: [Specific Ask] by [Date]",
            "Action Required: [Brief Description]",
            "Your Expertise Needed: [Topic]",
        ],
        "introduction": [
            "Introduction: [Your Name] from [Company]",
            "Nice to Connect, [Recipient Name]",
            "Reaching Out: [Brief Context]",
            "Hello from [Company] – Let's Connect",
            "New Connection: [Mutual Interest/Context]",
        ],
        "default": [
            "Important Update: [Topic]",
            "Quick Note About [Subject]",
            "Regarding [Project/Topic]",
            "[Action Needed]: [Brief Description]",
            "Update on [Topic]",
        ],
    }
    purpose_lower = email_purpose.lower()
    if "follow" in purpose_lower:
        options = templates["follow_up"]
    elif "request" in purpose_lower or "ask" in purpose_lower:
        options = templates["request"]
    elif "intro" in purpose_lower or "meet" in purpose_lower:
        options = templates["introduction"]
    else:
        options = templates["default"]

    return "Suggested subject lines:\n" + "\n".join(f"  {i+1}. {s}" for i, s in enumerate(options))


tools = [analyze_email_tone, check_email_structure, suggest_subject_lines]


# ── System prompt ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert email writing assistant helping craft professional,
clear, and effective emails.

Your process:
1. Understand the email purpose and context
2. Use suggest_subject_lines to generate subject options
3. Draft the email in the requested tone
4. Use check_email_structure to verify completeness
5. Use analyze_email_tone to verify the tone matches the request
6. Refine and present the final polished email

Always produce a complete email with: Subject, Greeting, Body, Call-to-action, and Closing."""


# ── Graph ──────────────────────────────────────────────────────────────────────
def build_email_graph():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7).bind_tools(tools)

    def email_agent(state: EmailState) -> dict:
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
        return {"messages": [llm.invoke(messages)]}

    tool_node = ToolNode(tools)
    graph = StateGraph(EmailState)
    graph.add_node("email_agent", email_agent)
    graph.add_node("tools", tool_node)
    graph.add_edge(START, "email_agent")
    graph.add_conditional_edges("email_agent", tools_condition)
    graph.add_edge("tools", "email_agent")
    return graph.compile()


# ── Runner ─────────────────────────────────────────────────────────────────────
def draft_email(context: dict) -> str:
    app = build_email_graph()
    prompt = (
        f"Draft a professional email with the following details:\n"
        f"  Purpose: {context.get('purpose', 'general communication')}\n"
        f"  Recipient: {context.get('recipient', 'colleague')}\n"
        f"  Tone: {context.get('tone', 'professional')}\n"
        f"  Key points to cover: {context.get('key_points', 'N/A')}\n"
        f"  Sender name: {context.get('sender', 'Your Name')}\n\n"
        "Use the tools to analyze and refine the email, then present the final version."
    )
    state: EmailState = {
        "messages": [HumanMessage(content=prompt)],
        "email_context": context,
        "draft": None,
        "tone": context.get("tone", "professional"),
        "iteration": 0,
    }
    result = app.invoke(state)
    return result["messages"][-1].content


def main():
    print("📧 Email Drafting Agent\n" + "=" * 60)
    context = {
        "purpose": "follow-up after a job interview",
        "recipient": "Hiring Manager, Sarah Johnson",
        "tone": "professional and enthusiastic",
        "key_points": (
            "Express gratitude for the interview opportunity, "
            "reiterate interest in the Senior Developer role, "
            "mention the exciting projects discussed, "
            "request next steps timeline"
        ),
        "sender": "Alex Chen",
    }
    print("Email Context:")
    for k, v in context.items():
        print(f"  {k}: {v}")
    print("\nDrafting email...\n")
    email = draft_email(context)
    print(email)
    print("\n✅ Email drafted!")


if __name__ == "__main__":
    main()
