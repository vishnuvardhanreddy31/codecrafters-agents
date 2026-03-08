"""
Data Analysis Agent
Uses LangGraph with pandas/matplotlib tools to analyze CSV data,
generate statistics, and produce natural language insights.
"""

import os
import io
from typing import Annotated, TypedDict, List
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()

# ── State ──────────────────────────────────────────────────────────────────────
class AnalysisState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    dataset_path: str
    analysis_request: str


# ── Tools ──────────────────────────────────────────────────────────────────────
@tool
def load_and_describe_csv(filepath: str) -> str:
    """Load a CSV file and return a statistical summary with column info,
    data types, missing values, and basic statistics."""
    try:
        import pandas as pd
        df = pd.read_csv(filepath)
        buf = io.StringIO()
        buf.write(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns\n\n")
        buf.write("Column Info:\n")
        for col in df.columns:
            buf.write(
                f"  {col}: {df[col].dtype}, "
                f"{df[col].isna().sum()} nulls, "
                f"{df[col].nunique()} unique values\n"
            )
        buf.write("\nStatistics (numeric columns):\n")
        buf.write(df.describe().to_string())
        return buf.getvalue()
    except FileNotFoundError:
        return f"File not found: {filepath}"
    except Exception as e:
        return f"Error loading file: {e}"


@tool
def compute_correlations(filepath: str) -> str:
    """Compute and return the correlation matrix for numeric columns in the CSV."""
    try:
        import pandas as pd
        df = pd.read_csv(filepath)
        numeric_df = df.select_dtypes(include="number")
        if numeric_df.empty:
            return "No numeric columns found for correlation analysis."
        return numeric_df.corr().round(3).to_string()
    except Exception as e:
        return f"Error: {e}"


@tool
def detect_outliers(filepath: str, column: str) -> str:
    """Detect outliers in a specific column using the IQR method."""
    try:
        import pandas as pd
        df = pd.read_csv(filepath)
        if column not in df.columns:
            return f"Column '{column}' not found. Available: {list(df.columns)}"
        series = df[column].dropna()
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outliers = series[(series < lower) | (series > upper)]
        return (
            f"Column: {column}\n"
            f"IQR bounds: [{lower:.2f}, {upper:.2f}]\n"
            f"Outliers found: {len(outliers)}\n"
            f"Outlier values: {outliers.values[:10].tolist()}"
        )
    except Exception as e:
        return f"Error: {e}"


@tool
def generate_chart(filepath: str, chart_type: str, columns: str) -> str:
    """Generate a chart from CSV data and save it as chart.png.
    chart_type: histogram | scatter | bar | line
    columns: comma-separated column names."""
    try:
        import pandas as pd
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        df = pd.read_csv(filepath)
        cols = [c.strip() for c in columns.split(",")]
        missing = [c for c in cols if c not in df.columns]
        if missing:
            return f"Columns not found: {missing}"

        fig, ax = plt.subplots(figsize=(10, 6))
        if chart_type == "histogram":
            df[cols[0]].hist(ax=ax, bins=30)
            ax.set_title(f"Histogram of {cols[0]}")
        elif chart_type == "scatter" and len(cols) >= 2:
            ax.scatter(df[cols[0]], df[cols[1]], alpha=0.6)
            ax.set_xlabel(cols[0])
            ax.set_ylabel(cols[1])
            ax.set_title(f"{cols[0]} vs {cols[1]}")
        elif chart_type == "bar":
            df[cols[0]].value_counts().head(15).plot(kind="bar", ax=ax)
            ax.set_title(f"Bar chart of {cols[0]}")
        elif chart_type == "line":
            df[cols[0]].plot(ax=ax)
            ax.set_title(f"Line chart of {cols[0]}")
        else:
            return f"Unsupported chart_type '{chart_type}' or insufficient columns."

        output = "chart.png"
        fig.tight_layout()
        fig.savefig(output)
        plt.close(fig)
        return f"✅ Chart saved as '{output}'"
    except Exception as e:
        return f"Error generating chart: {e}"


tools = [load_and_describe_csv, compute_correlations, detect_outliers, generate_chart]


# ── Graph ──────────────────────────────────────────────────────────────────────
def build_analysis_graph():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools(tools)

    def analyst(state: AnalysisState) -> dict:
        return {"messages": [llm.invoke(state["messages"])]}

    tool_node = ToolNode(tools)
    graph = StateGraph(AnalysisState)
    graph.add_node("analyst", analyst)
    graph.add_node("tools", tool_node)
    graph.add_edge(START, "analyst")
    graph.add_conditional_edges("analyst", tools_condition)
    graph.add_edge("tools", "analyst")
    return graph.compile()


# ── Runner ─────────────────────────────────────────────────────────────────────
def analyze_dataset(filepath: str, question: str) -> str:
    app = build_analysis_graph()
    prompt = (
        f"You are a data analyst. Analyze the dataset at '{filepath}'.\n\n"
        f"User request: {question}\n\n"
        "Use the available tools to load the data, compute statistics, detect outliers, "
        "and generate charts as needed. Provide clear, actionable insights."
    )
    state: AnalysisState = {
        "messages": [HumanMessage(content=prompt)],
        "dataset_path": filepath,
        "analysis_request": question,
    }
    result = app.invoke(state)
    return result["messages"][-1].content


def create_sample_dataset(path: str = "sample_data.csv"):
    import csv, random, math
    headers = ["age", "income", "spending_score", "category", "region"]
    categories = ["Electronics", "Clothing", "Food", "Sports", "Books"]
    regions = ["North", "South", "East", "West"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for _ in range(200):
            writer.writerow({
                "age": random.randint(18, 70),
                "income": round(random.gauss(55000, 20000)),
                "spending_score": random.randint(1, 100),
                "category": random.choice(categories),
                "region": random.choice(regions),
            })
    return path


def main():
    print("📊 Data Analysis Agent\n" + "=" * 60)
    csv_path = create_sample_dataset()
    print(f"Created sample dataset: {csv_path}\n")
    question = (
        "Provide a comprehensive analysis: describe the dataset, "
        "find correlations between numeric columns, check for outliers in income, "
        "and create a histogram of the age distribution."
    )
    print(f"Analysis request: {question}\n")
    result = analyze_dataset(csv_path, question)
    print(result)
    print("\n✅ Analysis complete!")


if __name__ == "__main__":
    main()
