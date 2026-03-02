"""
Code Review Agent
Uses LangGraph to perform automated, multi-pass code review with actionable feedback.
Analyzes code for bugs, security issues, style, and performance.
"""

import os
from typing import Annotated, TypedDict, List
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()


# ── State ──────────────────────────────────────────────────────────────────────
class ReviewState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    code_snippet: str
    language: str
    issues: List[str]
    score: int


# ── Tools ──────────────────────────────────────────────────────────────────────
@tool
def check_security_issues(code: str) -> str:
    """Analyze code for common security vulnerabilities such as SQL injection,
    XSS, hardcoded secrets, insecure deserialization, and unsafe eval usage."""
    vulnerabilities = []
    code_lower = code.lower()

    patterns = {
        "SQL Injection risk": ["execute(", "raw_input", "f\"select", "f'select"],
        "Hardcoded secret": ["password =", "secret =", "api_key =", "token ="],
        "Unsafe eval": ["eval(", "exec("],
        "Shell injection risk": ["os.system(", "subprocess.call(", "shell=true"],
        "Insecure deserialization": ["pickle.loads(", "yaml.load("],
    }

    for issue, signals in patterns.items():
        if any(s in code_lower for s in signals):
            vulnerabilities.append(f"⚠️  {issue} detected")

    return "\n".join(vulnerabilities) if vulnerabilities else "✅ No obvious security issues found."


@tool
def check_code_style(code: str, language: str) -> str:
    """Check code for common style issues: long lines, missing docstrings,
    inconsistent naming, and deeply nested blocks."""
    issues = []
    lines = code.split("\n")

    long_lines = [i + 1 for i, l in enumerate(lines) if len(l) > 100]
    if long_lines:
        issues.append(f"Lines exceeding 100 chars: {long_lines[:5]}")

    if language.lower() == "python":
        if "def " in code and '"""' not in code and "'''" not in code:
            issues.append("Functions may be missing docstrings")

    max_indent = max((len(l) - len(l.lstrip())) for l in lines if l.strip())
    if max_indent > 20:
        issues.append(f"Deep nesting detected (max indent: {max_indent} spaces)")

    return "\n".join(issues) if issues else "✅ Style looks good."


@tool
def suggest_improvements(code: str) -> str:
    """Return a placeholder for LLM-driven improvement suggestions.
    The agent will synthesize these after tool calls complete."""
    return (
        "Improvements analysis queued. The agent will synthesize specific "
        "refactoring suggestions based on the full review context."
    )


tools = [check_security_issues, check_code_style, suggest_improvements]


# ── Graph ──────────────────────────────────────────────────────────────────────
def build_review_graph() -> StateGraph:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools(tools)

    def reviewer(state: ReviewState) -> dict:
        response = llm.invoke(state["messages"])
        return {"messages": [response]}

    tool_node = ToolNode(tools)

    graph = StateGraph(ReviewState)
    graph.add_node("reviewer", reviewer)
    graph.add_node("tools", tool_node)

    graph.add_edge(START, "reviewer")
    graph.add_conditional_edges("reviewer", tools_condition)
    graph.add_edge("tools", "reviewer")

    return graph.compile()


# ── Runner ─────────────────────────────────────────────────────────────────────
def review_code(code: str, language: str = "python") -> str:
    app = build_review_graph()

    prompt = f"""You are a senior software engineer conducting a thorough code review.

Language: {language}

Code to review:
```{language}
{code}
```

Please:
1. Use check_security_issues to scan for vulnerabilities
2. Use check_code_style to check formatting and style
3. Use suggest_improvements to flag areas for improvement
4. Provide a comprehensive review with a quality score (1-10), specific issues found,
   and actionable refactoring suggestions."""

    state: ReviewState = {
        "messages": [HumanMessage(content=prompt)],
        "code_snippet": code,
        "language": language,
        "issues": [],
        "score": 0,
    }

    final_state = app.invoke(state)
    return final_state["messages"][-1].content


SAMPLE_CODE = '''
import os
import pickle

password = "supersecret123"

def process_user_data(user_input):
    query = f"SELECT * FROM users WHERE name = {user_input}"
    result = eval(query)
    data = pickle.loads(result)
    return data

def calculate(x,y,z,a,b,c,d,e,f):
    if x > 0:
        if y > 0:
            if z > 0:
                if a > 0:
                    return x+y+z+a+b+c+d+e+f
    return 0
'''


def main():
    print("🔍 Code Review Agent\n" + "=" * 60)
    print("Reviewing sample code with security and style issues...\n")
    review = review_code(SAMPLE_CODE, "python")
    print(review)
    print("\n" + "=" * 60 + "\n✅ Review complete!")


if __name__ == "__main__":
    main()
