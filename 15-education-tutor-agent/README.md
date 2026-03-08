# 🎓 Education Tutor Agent

An adaptive AI tutoring system built with **LangGraph** that provides personalized explanations, generates quizzes, checks answers with detailed feedback, and tracks student progress across multiple subjects.

## Features

- 📚 **Multi-subject support**: Math, Python, History, Science
- 🧠 **Socratic teaching method**: Guides students to discover answers
- 📝 **Dynamic quizzes**: Generated from a subject question bank
- ✅ **Answer checking**: Immediate feedback with explanations
- 📊 **Progress tracking**: Scores, mastered concepts, weak areas
- 📖 **Resource recommendations**: Books, videos, exercises per topic
- 🎯 **Adaptive difficulty**: Adjusts to student level

## Architecture

```
Student Input → Tutor Agent → [Explain Concept] | [Generate Quiz] | [Check Answer]
                    ↑___________________________|
                    (adapts to learning needs)
                           ↓
                Progress Update + Next Suggestion
```

## Tech Stack

- **Framework**: LangGraph 0.2+
- **LLM**: OpenAI GPT-4o-mini (OpenAI-compatible API)
- **Tools**: Concept Explainer, Quiz Generator, Answer Checker, Resource Recommender, Progress Tracker
- **Pattern**: Adaptive conversational agent with learning state

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
What subject? python
Your level? intermediate

You: Explain list comprehensions
Tutor: [Detailed explanation with examples]

You: Give me a quiz
Tutor: Q1: What does the 'yield' keyword do in Python?

You: It creates a generator function
Tutor: ✅ Correct! [Full explanation of generators]

You: What are my weak areas?
Tutor: [Progress report with recommendations]
```

## Available Subjects

| Subject | Topics | Questions |
|---------|--------|-----------|
| **Python** | Generators, OOP, data structures | 3+ |
| **Math** | Calculus, algebra, integration | 3+ |
| **History** | World events, U.S. History | 2+ |
| **Science** | Chemistry, physics, biology | 2+ |

## Extending the Question Bank

Add questions to `QUIZ_BANK` in `agent.py`:
```python
QUIZ_BANK["chemistry"] = [
    {
        "question": "What is the atomic number of Carbon?",
        "answer": "6",
        "explanation": "Carbon has 6 protons in its nucleus, giving it atomic number 6..."
    }
]
```

## Teaching Philosophy

The tutor uses the **Socratic method**:
1. Ask questions to check understanding
2. Guide rather than just give answers
3. Celebrate correct answers enthusiastically
4. Frame mistakes as learning opportunities
5. Connect new concepts to already-known ones

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | ✅ | OpenAI API key |
