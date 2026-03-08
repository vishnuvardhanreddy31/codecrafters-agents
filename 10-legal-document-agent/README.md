# ⚖️ Legal Document Agent

An AI-powered legal document analysis assistant built with **LangGraph** that reviews contracts, identifies risks, extracts key terms, and provides plain-language explanations.

> ⚠️ **DISCLAIMER**: This agent provides **general legal information only** and is NOT a substitute for advice from a qualified attorney. Always consult a licensed lawyer for legal matters.

## Features

- 📋 **Contract risk analysis**: Identifies missing clauses, red flags, and unusual terms
- 🔑 **Key term extraction**: Extracts dates, amounts, parties, and obligations
- 📚 **Legal knowledge base**: Context on contracts, IP, NDAs, and liability
- 🔍 **Plain-language summaries**: Translates legal jargon into clear explanations
- ⚡ **Clause identification**: Detects 10+ standard contract clause types

## Architecture

```
Contract Document → Legal Agent → [Risk Analysis] → [Key Term Extraction] → [Knowledge Search]
                         ↑___________________________|
                         (comprehensive review)
                                ↓
                  Plain-Language Legal Analysis Report
```

## Tech Stack

- **Framework**: LangGraph 0.2+
- **LLM**: OpenAI GPT-4o-mini (OpenAI-compatible API)
- **Tools**: Risk Analyzer, Term Extractor, Knowledge Base Search, Document Summarizer
- **Pattern**: Document analysis agent with structured extraction

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
from agent import analyze_document

with open("contract.txt", "r") as f:
    contract = f.read()

analysis = analyze_document(
    document_text=contract,
    question="What are my key obligations and any unusual clauses?"
)
print(analysis)
```

## Analyzed Clause Types

| Clause Type | Risk Level | Description |
|------------|------------|-------------|
| Limitation of Liability | High | Caps damages you can recover |
| Arbitration | Medium | Restricts court access |
| Non-compete | High | May restrict future business |
| IP Assignment | High | Who owns work product |
| Termination Rights | Medium | How to exit the agreement |
| Confidentiality | Low | Information protection period |
| Governing Law | Medium | Which state's laws apply |
| Indemnification | High | Who bears third-party claims |

## Output Format

The analysis includes:
1. **Risk Report**: Color-coded severity with specific concerns
2. **Key Terms Summary**: All important dates, amounts, and periods
3. **Obligations**: What each party must do
4. **Plain-Language Summary**: Executive overview in plain English
5. **Recommended Questions**: What to ask an attorney

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | ✅ | OpenAI API key |
