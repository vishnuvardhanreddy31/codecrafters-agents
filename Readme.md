# 🤖 CodeCrafters AI Agents

A collection of **20 production-ready AI agent projects** built with the latest agentic AI frameworks (LangGraph, CrewAI) and OpenAI-compatible APIs. Each project demonstrates a different real-world use case with full agentic architecture.

## 🚀 Projects Overview

| # | Project | Framework | Use Case | Key Tools |
|---|---------|-----------|----------|-----------|
| 01 | [Research Assistant](./01-research-assistant-agent/) | LangGraph | Web research & synthesis | Tavily Search |
| 02 | [Code Review](./02-code-review-agent/) | LangGraph | Automated code analysis | OWASP scanner, style checker |
| 03 | [Customer Support](./03-customer-support-agent/) | LangGraph | Multi-turn support chatbot | Knowledge base, ticket creation |
| 04 | [Data Analysis](./04-data-analysis-agent/) | LangGraph | CSV insights & visualization | pandas, matplotlib |
| 05 | [Email Drafting](./05-email-drafting-agent/) | LangGraph | Professional email writing | Tone analyzer, structure checker |
| 06 | [Content Creation](./06-content-creation-agent/) | CrewAI | Blog post generation | Researcher + Writer + Editor |
| 07 | [Travel Planning](./07-travel-planning-agent/) | CrewAI | Personalized trip planning | Destination Expert + Planner + Budget |
| 08 | [Financial Analysis](./08-financial-analysis-agent/) | LangGraph | Stock analysis & insights | yfinance, technical indicators |
| 09 | [Medical Information](./09-medical-information-agent/) | LangGraph | Health education with RAG | FAISS vector store, safety guardrails |
| 10 | [Legal Document](./10-legal-document-agent/) | LangGraph | Contract analysis | Risk analyzer, term extractor |
| 11 | [Personal Assistant](./11-personal-assistant-agent/) | LangGraph | Task & calendar management | Task manager, scheduler, notes |
| 12 | [News Aggregator](./12-news-aggregator-agent/) | CrewAI | News briefing generation | RSS collector, sentiment analysis |
| 13 | [Recipe Generator](./13-recipe-generator-agent/) | LangGraph | Personalized recipe creation | Nutrition calculator, dietary subs |
| 14 | [Job Application](./14-job-application-agent/) | CrewAI | Resume & interview prep | Resume Optimizer + Coach |
| 15 | [Education Tutor](./15-education-tutor-agent/) | LangGraph | Adaptive learning system | Quiz generator, progress tracker |
| 16 | [Security Audit](./16-security-audit-agent/) | LangGraph | Code vulnerability scanning | OWASP scanner, CVE checker |
| 17 | [Social Media](./17-social-media-agent/) | CrewAI | Content strategy & creation | Strategist + Creator + Analyst |
| 18 | [Meeting Summarizer](./18-meeting-summarizer-agent/) | LangGraph | Transcript analysis | Action item extractor, decision finder |
| 19 | [E-Commerce](./19-ecommerce-agent/) | LangGraph | Intelligent shopping assistant | Product search, cart management |
| 20 | [DevOps](./20-devops-agent/) | CrewAI | Infrastructure incident response | System metrics, log analyzer, runbooks |

## 🛠️ Frameworks Used

### LangGraph (14 projects)
[LangGraph](https://github.com/langchain-ai/langgraph) enables building stateful, multi-step agent workflows with:
- **StateGraph**: Define agent state and transitions
- **ToolNode**: Automatic tool execution
- **Conditional edges**: Dynamic routing based on agent decisions
- **ReAct pattern**: Reason → Act → Observe cycle

### CrewAI (6 projects)
[CrewAI](https://github.com/joaomdmoura/crewAI) enables multi-agent collaboration with:
- **Specialized agents**: Each with unique role, goal, and backstory
- **Sequential/Parallel tasks**: Agents build on each other's work
- **Context passing**: Agents share findings across the pipeline
- **Custom tools**: Pydantic-validated tool interfaces

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- OpenAI API key (or compatible endpoint)

### Running Any Project

```bash
# 1. Navigate to a project
cd 01-research-assistant-agent

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Edit .env and add your API keys

# 4. Run the agent
python agent.py
```

### Required API Keys

| Key | Required By | Get it |
|-----|-------------|--------|
| `OPENAI_API_KEY` | All projects | [OpenAI Platform](https://platform.openai.com) |
| `TAVILY_API_KEY` | Project 01 | [Tavily](https://tavily.com) |
| `SERPER_API_KEY` | Project 06 (optional) | [Serper.dev](https://serper.dev) |

## 📐 Architecture Patterns

### ReAct Agent (LangGraph)
```
User Input → Agent → Tool Call → Tool Result → Agent → ... → Final Response
```
Used in: Research, Code Review, Data Analysis, Financial Analysis, and more.

### Multi-Agent Pipeline (CrewAI)
```
Task → Agent1 → Result1 → Agent2 (uses Result1) → Result2 → Agent3 → Final
```
Used in: Content Creation, Travel Planning, Job Application, DevOps, and more.

### RAG Agent (LangGraph + FAISS)
```
Query → Embedding → Vector Search → Retrieved Docs → LLM → Response
```
Used in: Medical Information, Legal Document, Customer Support.

## 🔌 OpenAI-Compatible APIs

All projects use the `langchain_openai.ChatOpenAI` client, making them compatible with any OpenAI-compatible API:

```python
from langchain_openai import ChatOpenAI

# OpenAI
llm = ChatOpenAI(model="gpt-4o-mini")

# Local models (Ollama, LM Studio)
llm = ChatOpenAI(base_url="http://localhost:11434/v1", model="llama3.1")

# Other providers (Groq, Together AI)
llm = ChatOpenAI(base_url="https://api.groq.com/openai/v1", model="mixtral-8x7b-32768")
```

## �� Project Structure

Each project follows this structure:
```
XX-project-name/
├── agent.py          # Main agent implementation
├── requirements.txt  # Project dependencies
├── .env.example      # Environment variable template
└── README.md         # Project documentation
```

## 📄 License

MIT License - feel free to use these projects as starting points for your own AI agent applications.
