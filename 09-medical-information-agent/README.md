# ⚕️ Medical Information Agent

An AI-powered health information assistant built with **LangGraph** featuring a medical knowledge base, emergency symptom detection, and specialist recommendations.

> ⚠️ **DISCLAIMER**: This agent provides **general health education only** and is NOT a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider.

## Features

- 📚 **Medical knowledge base** with RAG (Retrieval Augmented Generation)
- 🚨 **Emergency detection**: Identifies symptoms requiring immediate care
- 👨‍⚕️ **Specialist recommendations**: Routes to appropriate medical specialties
- 🔒 **Safety guardrails**: Always recommends professional consultation
- 💬 **Multi-turn conversations** for follow-up questions

## Architecture

```
Health Question → Medical Agent → [Knowledge Base Search] → [Emergency Check] → [Specialist Rec]
                       ↑___________________________|
                       (safety-first routing)
                              ↓
                   Educational Response + Safety Disclaimer
```

## Tech Stack

- **Framework**: LangGraph 0.2+
- **LLM**: OpenAI GPT-4o-mini (OpenAI-compatible API)
- **Vector Store**: FAISS (in-memory) with OpenAI embeddings
- **Tools**: Knowledge Base Search, Emergency Checker, Specialist Recommender
- **Pattern**: RAG-augmented safety-focused agent

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

Interactive session:
```
Your health question: What are the symptoms of type 2 diabetes?
Agent: [Searches knowledge base and provides educational information]

Your health question: I have chest pain and difficulty breathing
Agent: ⚠️ URGENT: These symptoms may indicate a medical emergency!
       🚨 CALL 911 IMMEDIATELY or go to the nearest emergency room.
```

## Knowledge Base Topics

| Category | Topics |
|----------|--------|
| Conditions | Diabetes, Hypertension, Common Cold |
| Wellness | Nutrition guidelines, Exercise recommendations |
| Mental Health | Depression, anxiety, crisis resources |
| Safety | Medication safety, drug interactions |
| Emergency | First aid procedures |

## Safety Features

1. **Emergency detection**: Scans for 14+ emergency indicators
2. **Crisis resources**: Always mentions 988 Suicide Prevention Line for mental health
3. **Disclaimer**: Every response includes a professional consultation recommendation
4. **No diagnosis**: Agent explicitly cannot and will not diagnose conditions

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | ✅ | OpenAI API key |
