# 🔒 Security Audit Agent

An automated application security auditing agent built with **LangGraph** that scans code for OWASP Top 10 vulnerabilities, dependency issues, authentication flaws, and generates prioritized remediation reports.

## Features

- 🛡️ **OWASP Top 10 scanning**: Injection, crypto failures, misconfiguration, auth failures
- 📦 **Dependency auditing**: Known CVEs in requirements.txt packages
- 🔐 **Authentication analysis**: Password handling, session management, CSRF
- 📊 **Prioritized reports**: Critical/High/Medium/Low severity classification
- 🔧 **Remediation code**: Specific fix examples for each vulnerability

## Architecture

```
Code/Requirements → Auditor Agent → [OWASP Scan] → [Dependency Check] → [Auth Analysis]
                         ↑___________________________|
                         (comprehensive security review)
                                ↓
                  Prioritized Security Report + Fix Examples
```

## Tech Stack

- **Framework**: LangGraph 0.2+
- **LLM**: OpenAI GPT-4o-mini (OpenAI-compatible API)
- **Tools**: OWASP Scanner, Dependency Auditor, Auth Analyzer, Report Generator
- **Standards**: OWASP Top 10, CVE database (limited), security best practices

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

Or scan your own code:
```python
from agent import audit_code

with open("myapp.py", "r") as f:
    code = f.read()

report = audit_code(code, "python")
print(report)
```

## Vulnerability Categories Detected

### OWASP Top 10 Coverage
| Category | Patterns Detected |
|----------|------------------|
| **A01 Injection** | SQL injection, command injection, eval() |
| **A02 Crypto Failures** | MD5/SHA1, hardcoded passwords |
| **A03 Insecure Deserialization** | pickle.loads(), yaml.load() |
| **A05 Misconfiguration** | DEBUG=True, verify=False, open CORS |
| **A07 Auth Failures** | Plain-text comparison, weak session tokens |

### Dependency CVEs (Sample)
| Package | CVE | Fix |
|---------|-----|-----|
| Django < 4.2.0 | CVE-2023-31047 | Upgrade to 4.2+ |
| Flask < 2.3.0 | CVE-2023-30861 | Upgrade to 2.3+ |
| Requests < 2.28.0 | CVE-2023-32681 | Upgrade to 2.28+ |

## Report Format

```
SECURITY AUDIT REPORT
==============================
CRITICAL: 🔴 SQL Injection (Line 12)
HIGH:      🟠 Hardcoded secret (Line 5)
MEDIUM:    🟡 Security misconfiguration (Line 3)

REMEDIATION PRIORITY
1. 🔴 Fix immediately
2. 🟠 Fix within 24 hours
...
```

## CI/CD Integration

```yaml
# .github/workflows/security.yml
- name: Run security audit
  run: python agent.py < code_to_audit.py
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | ✅ | OpenAI API key |
