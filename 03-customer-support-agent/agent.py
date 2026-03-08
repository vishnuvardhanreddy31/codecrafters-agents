"""
Customer Support Agent
Uses LangGraph with persistent conversation state, intent detection,
knowledge base retrieval, and escalation logic.
"""

import os
from typing import Annotated, TypedDict, List, Optional
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()


# ── Knowledge base (mock) ──────────────────────────────────────────────────────
KNOWLEDGE_BASE = {
    "return_policy": (
        "Our return policy allows returns within 30 days of purchase. "
        "Items must be unused and in original packaging. "
        "Refunds are processed within 5-7 business days."
    ),
    "shipping": (
        "Standard shipping takes 5-7 business days. "
        "Express shipping (2-3 days) is available for an additional fee. "
        "Free shipping on orders over $50."
    ),
    "warranty": (
        "All products come with a 1-year manufacturer warranty. "
        "Extended warranty plans are available for purchase."
    ),
    "payment": (
        "We accept Visa, Mastercard, American Express, PayPal, and Apple Pay. "
        "All transactions are secured with SSL encryption."
    ),
    "account": (
        "You can reset your password via the 'Forgot Password' link on the login page. "
        "Account issues can be resolved by contacting support@example.com."
    ),
}


# ── State ──────────────────────────────────────────────────────────────────────
class SupportState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    customer_id: Optional[str]
    intent: Optional[str]
    escalated: bool


# ── Tools ──────────────────────────────────────────────────────────────────────
@tool
def search_knowledge_base(query: str) -> str:
    """Search the support knowledge base for answers to customer questions.
    Query should be a topic keyword like: return_policy, shipping, warranty,
    payment, or account."""
    key = query.lower().replace(" ", "_")
    for kb_key, content in KNOWLEDGE_BASE.items():
        if kb_key in key or key in kb_key:
            return content
    # Fuzzy fallback
    for kb_key, content in KNOWLEDGE_BASE.items():
        if any(word in kb_key for word in key.split("_")):
            return content
    return "I couldn't find specific information. Please contact support@example.com."


@tool
def lookup_order(order_id: str) -> str:
    """Look up the status of a customer order by order ID."""
    mock_orders = {
        "ORD-001": "Shipped - Expected delivery: 2 days",
        "ORD-002": "Processing - Will ship within 24 hours",
        "ORD-003": "Delivered on Jan 15, 2025",
    }
    return mock_orders.get(
        order_id.upper(),
        f"Order {order_id} not found. Please verify the order ID.",
    )


@tool
def create_support_ticket(
    customer_id: str,
    issue_description: str,
    priority: str = "medium",
) -> str:
    """Create a support ticket for issues requiring human agent escalation.
    Priority should be: low, medium, or high."""
    import random
    ticket_id = f"TKT-{random.randint(10000, 99999)}"
    return (
        f"✅ Support ticket created successfully!\n"
        f"Ticket ID: {ticket_id}\n"
        f"Priority: {priority}\n"
        f"Our team will contact you within "
        f"{'2 hours' if priority == 'high' else '24 hours'}."
    )


tools = [search_knowledge_base, lookup_order, create_support_ticket]


# ── Graph ──────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a helpful and empathetic customer support agent for an e-commerce company.

Your responsibilities:
- Answer customer questions using the knowledge base
- Look up order status when provided an order ID
- Create support tickets for complex issues needing human escalation
- Always be polite, clear, and solution-oriented
- If you cannot resolve an issue, create a support ticket

Always use available tools before responding from memory."""


def build_support_graph() -> StateGraph:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3).bind_tools(tools)

    def support_agent(state: SupportState) -> dict:
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
        response = llm.invoke(messages)
        return {"messages": [response]}

    tool_node = ToolNode(tools)

    graph = StateGraph(SupportState)
    graph.add_node("support_agent", support_agent)
    graph.add_node("tools", tool_node)

    graph.add_edge(START, "support_agent")
    graph.add_conditional_edges("support_agent", tools_condition)
    graph.add_edge("tools", "support_agent")

    return graph.compile()


# ── Runner ─────────────────────────────────────────────────────────────────────
def run_support_session():
    app = build_support_graph()
    state: SupportState = {
        "messages": [],
        "customer_id": "CUST-12345",
        "intent": None,
        "escalated": False,
    }

    print("🤖 Customer Support Agent")
    print("=" * 60)
    print("Type your question or 'quit' to exit.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("quit", "exit", "bye"):
            print("Agent: Thank you for contacting support. Have a great day! 👋")
            break
        if not user_input:
            continue

        state["messages"].append(HumanMessage(content=user_input))
        result = app.invoke(state)
        state["messages"] = result["messages"]

        last_message = result["messages"][-1]
        print(f"\nAgent: {last_message.content}\n")


if __name__ == "__main__":
    run_support_session()
