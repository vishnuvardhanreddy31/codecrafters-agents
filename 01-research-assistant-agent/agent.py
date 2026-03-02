"""
Research Assistant Agent
Uses LangGraph with multi-agent architecture to conduct deep research on any topic.
Tools: Web Search (Tavily), Wikipedia, Summarization
"""

import os
from typing import Annotated, TypedDict, List
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()


# ── State ──────────────────────────────────────────────────────────────────────
class ResearchState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    research_topic: str
    findings: List[str]
    final_report: str


# ── Tools ──────────────────────────────────────────────────────────────────────
search_tool = TavilySearchResults(
    max_results=5,
    description="Search the web for current information on a topic.",
)


@tool
def summarize_findings(findings: List[str], topic: str) -> str:
    """Compile and summarize a list of research findings into a coherent report."""
    combined = "\n\n".join(findings)
    return f"Research Summary on '{topic}':\n\n{combined}"


tools = [search_tool, summarize_findings]


# ── Agent node ─────────────────────────────────────────────────────────────────
def build_research_graph() -> StateGraph:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools(tools)

    def research_agent(state: ResearchState) -> dict:
        response = llm.invoke(state["messages"])
        return {"messages": [response]}

    tool_node = ToolNode(tools)

    graph = StateGraph(ResearchState)
    graph.add_node("research_agent", research_agent)
    graph.add_node("tools", tool_node)

    graph.add_edge(START, "research_agent")
    graph.add_conditional_edges("research_agent", tools_condition)
    graph.add_edge("tools", "research_agent")

    return graph.compile()


# ── Runner ─────────────────────────────────────────────────────────────────────
def run_research(topic: str) -> str:
    """Run the research agent on a given topic and return the final report."""
    app = build_research_graph()

    system_prompt = (
        "You are an expert research assistant. Use the search tool to gather "
        "comprehensive information about the given topic. Search multiple angles, "
        "then use summarize_findings to compile your research into a clear, "
        "well-structured report with key insights, current developments, and "
        "important context."
    )

    initial_state: ResearchState = {
        "messages": [
            HumanMessage(content=f"System: {system_prompt}\n\nResearch topic: {topic}"),
        ],
        "research_topic": topic,
        "findings": [],
        "final_report": "",
    }

    final_state = app.invoke(initial_state)
    last_message = final_state["messages"][-1]
    return last_message.content


def main():
    topic = input("Enter a research topic: ").strip()
    if not topic:
        topic = "Latest advances in quantum computing 2024"

    print(f"\n🔍 Researching: {topic}\n{'=' * 60}\n")
    report = run_research(topic)
    print(report)
    print("\n" + "=" * 60)
    print("✅ Research complete!")


if __name__ == "__main__":
    main()
