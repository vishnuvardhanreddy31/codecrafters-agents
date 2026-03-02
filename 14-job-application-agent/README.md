# 💼 Job Application Agent

A comprehensive job application assistant built with **CrewAI** featuring a Resume Optimizer, Cover Letter Specialist, and Interview Coach working together to maximize your chances of landing your dream job.

## Features

- 📄 **ATS-optimized resume**: Keyword-rich formatting that passes automated screening
- ✍️ **Personalized cover letters**: Company-specific, story-driven narratives
- 🎯 **Interview preparation**: 10 likely questions with STAR-method answers
- 💰 **Salary negotiation tips**: Role-specific compensation guidance
- 🏆 **Quantified achievements**: Transforms experience into impact statements

## Architecture

```
Candidate Background + Job Description
        ↓
[Resume Optimizer] → ATS-Optimized Resume + Keyword Analysis
        ↓
[Cover Letter Specialist] → Personalized Cover Letter
        ↓
[Interview Coach] → Questions, STAR Answers, Talking Points
```

## Tech Stack

- **Framework**: CrewAI 0.80+
- **LLM**: OpenAI GPT-4o-mini (OpenAI-compatible API)
- **Pattern**: Sequential multi-agent pipeline with context sharing

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

Or use programmatically:
```python
from agent import prepare_application

result = prepare_application(
    candidate_background="""
        5 years software engineering experience
        Skills: Python, React, AWS, PostgreSQL
        Achievements: Led team of 4, improved performance by 40%
    """,
    target_role="Senior Software Engineer",
    target_company="Google",
    job_description="Full job description text here..."
)
print(result)
```

## What You Get

### Optimized Resume
- Professional summary tailored to the role
- Experience bullets with quantified impact
- Skills section with ATS keywords highlighted
- Education and certifications

### Cover Letter
- Compelling opening hook (not the generic "I am writing to apply...")
- 2-3 specific achievement-to-requirement connections
- Company-specific knowledge demonstration
- Strong call-to-action closing

### Interview Preparation
- 10 anticipated questions (behavioral + technical + situational)
- 5 STAR-method answers pre-written
- 5 thoughtful questions to ask the interviewer
- Salary negotiation strategy

## Agent Roles

| Agent | Expertise | Output |
|-------|-----------|--------|
| **Resume Optimizer** | ATS, keywords, impact statements | Optimized resume |
| **Cover Letter Specialist** | Storytelling, persuasion | Personalized letter |
| **Interview Coach** | Behavioral interviewing, STAR | Prep materials |

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | ✅ | OpenAI API key |
