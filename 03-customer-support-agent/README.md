# 🤖 Customer Support Agent

An intelligent customer support chatbot built with **LangGraph** featuring multi-turn conversation, knowledge base retrieval, order lookup, and automatic escalation to human agents when needed.

## Features

- 💬 **Multi-turn conversations** with persistent state
- 📚 **Knowledge base search** for returns, shipping, warranty, payment, and account questions
- 📦 **Order status lookup** with real-time simulation
- 🎫 **Ticket creation** for complex issues requiring human escalation
- 🔄 **Stateful sessions** maintaining conversation context

## Architecture

```
Customer Message → Support Agent → Knowledge Base | Order Lookup | Ticket Creation
                        ↑__________________________________|
                        (routes to appropriate tool)
                              ↓
                   Response + Context Update
```

## Tech Stack

- **Framework**: LangGraph 0.2+
- **LLM**: OpenAI GPT-4o-mini (OpenAI-compatible API)
- **Tools**: Knowledge Base Search, Order Lookup, Support Ticket Creator
- **Pattern**: Conversational ReAct Agent with session memory

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

Interactive session example:
```
You: What is your return policy?
Agent: Our return policy allows returns within 30 days of purchase...

You: I need to check order ORD-001
Agent: Order ORD-001 Status: Shipped - Expected delivery: 2 days

You: I have a complex billing dispute
Agent: I've created support ticket TKT-45231 for you...
```

## Knowledge Base Topics

| Topic | Description |
|-------|-------------|
| `return_policy` | Return window, conditions, refund timelines |
| `shipping` | Shipping options, times, and fees |
| `warranty` | Product warranty terms and extensions |
| `payment` | Accepted payment methods and security |
| `account` | Password reset, account management |

## Customization

Extend the knowledge base in `KNOWLEDGE_BASE` dict, add order lookup integration with your actual database, and configure escalation rules based on your business logic.

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | ✅ | OpenAI API key |
