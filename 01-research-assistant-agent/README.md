# 🔍 Research Assistant Agent

An autonomous multi-agent research system built with **LangGraph** that conducts deep, comprehensive research on any topic using web search and synthesizes findings into structured reports.

## Features

- 🌐 **Web Search Integration** using Tavily API for real-time information
- 🔄 **Multi-step reasoning** with iterative search refinement
- 📊 **Structured reports** with key insights and current developments
- 🤖 **ReAct agent pattern** (Reason + Act) via LangGraph

## Architecture

```
User Query → Research Agent → [Web Search Tool] → Synthesis → Final Report
                    ↑__________________|
                    (iterates until complete)
```

## Tech Stack

- **Framework**: LangGraph 0.2+
- **LLM**: OpenAI GPT-4o-mini (OpenAI-compatible API)
- **Tools**: Tavily Search, Custom Summarization
- **Pattern**: ReAct Agent with Tool Nodes

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file:
```env
OPENAI_API_KEY=your_openai_api_key
TAVILY_API_KEY=your_tavily_api_key
```

## Usage

```bash
python agent.py
```

You'll be prompted to enter a research topic. The agent will:
1. Search the web for relevant information from multiple angles
2. Iteratively refine searches based on findings
3. Synthesize all information into a comprehensive report

## Example

```
Enter a research topic: Latest advances in quantum computing 2024

🔍 Researching: Latest advances in quantum computing 2024
============================================================

[Agent searches web, iterates, and returns a comprehensive report]
```

## How It Works

The agent uses a **stateful graph** with:
- `research_agent` node: The LLM that decides what to search and how to synthesize
- `tools` node: Executes the actual web searches
- **Conditional edges**: Agent decides when it has enough information

The `ResearchState` tracks conversation history with `add_messages` annotation, enabling multi-turn tool use within a single research session.

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | ✅ | OpenAI API key |
| `TAVILY_API_KEY` | ✅ | Tavily Search API key |
