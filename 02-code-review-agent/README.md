# 🔍 Code Review Agent

An automated code review agent built with **LangGraph** that analyzes code for security vulnerabilities, style issues, performance problems, and provides actionable refactoring suggestions.

## Features

- 🔒 **Security scanning**: SQL injection, hardcoded secrets, unsafe eval, insecure deserialization
- 📏 **Style analysis**: Line length, naming conventions, missing docstrings, deep nesting
- ⚡ **Performance tips**: Inefficient patterns and optimization opportunities
- 📊 **Quality scoring**: 1-10 score with detailed breakdown
- 🛠️ **Actionable fixes**: Specific refactoring code examples

## Architecture

```
Code Input → Reviewer Agent → [Security Check] → [Style Check] → [Improvement Suggestions]
                   ↑___________________________|
                   (iterates through all checks)
                         ↓
              Comprehensive Review Report
```

## Tech Stack

- **Framework**: LangGraph 0.2+
- **LLM**: OpenAI GPT-4o-mini (OpenAI-compatible API)
- **Tools**: Custom security scanner, style checker, improvement suggester
- **Pattern**: ReAct Agent with specialized review tools

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

The agent reviews the built-in vulnerable code sample, or modify `agent.py` to review your own code:

```python
from agent import review_code

# Review your code
my_code = """
def process_user(name):
    query = f"SELECT * FROM users WHERE name = {name}"
    return eval(query)
"""

result = review_code(my_code, "python")
print(result)
```

## Detected Vulnerability Categories

| Category | Examples |
|----------|---------|
| SQL Injection | Unsanitized queries, f-string SQL |
| Hardcoded Secrets | Passwords, API keys, tokens in code |
| Unsafe Execution | `eval()`, `exec()` calls |
| Shell Injection | `os.system()` with user input |
| Insecure Deserialization | `pickle.loads()`, `yaml.load()` |
| Style Issues | Long lines, missing docstrings, deep nesting |

## Example Output

```
Security Analysis:
  ⚠️ SQL Injection risk detected
  ⚠️ Hardcoded secret detected
  ⚠️ Unsafe eval detected

Style Analysis:
  Functions may be missing docstrings
  Deep nesting detected (max indent: 24 spaces)

Overall Quality Score: 3/10

Recommended fixes:
  1. Use parameterized queries instead of f-strings
  2. Load secrets from environment variables
  3. Replace eval() with safe alternatives
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | ✅ | OpenAI API key |
