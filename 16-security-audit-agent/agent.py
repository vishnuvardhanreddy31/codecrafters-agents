"""
Security Audit Agent
Uses LangGraph to perform automated security audits on Python code,
identifying vulnerabilities, insecure patterns, and recommending fixes.
"""

import os
import ast
import re
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
class SecurityState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    code: str
    language: str
    severity_threshold: str


# ── Security rules ─────────────────────────────────────────────────────────────
OWASP_PATTERNS = {
    "A01-Injection": {
        "patterns": [
            r"execute\s*\(",
            r"raw\s*\(\s*['\"]SELECT",
            r"os\.system\s*\(",
            r"subprocess\.call\s*\(.*shell\s*=\s*True",
            r"eval\s*\(",
            r"exec\s*\(",
        ],
        "severity": "critical",
        "description": "Code injection vulnerability - user input used in command/query execution",
    },
    "A02-Cryptographic-Failures": {
        "patterns": [
            r"MD5\s*\(",
            r"hashlib\.md5",
            r"hashlib\.sha1",
            r"DES\s*\(",
            r"password\s*=\s*['\"]",
            r"secret\s*=\s*['\"]",
        ],
        "severity": "high",
        "description": "Weak cryptography or hardcoded secrets",
    },
    "A03-Insecure-Deserialization": {
        "patterns": [
            r"pickle\.loads\s*\(",
            r"pickle\.load\s*\(",
            r"yaml\.load\s*\([^,)]+\)",
            r"marshal\.loads\s*\(",
        ],
        "severity": "critical",
        "description": "Insecure deserialization - can lead to RCE",
    },
    "A05-Security-Misconfiguration": {
        "patterns": [
            r"DEBUG\s*=\s*True",
            r"ALLOWED_HOSTS\s*=\s*\[.*\*.*\]",
            r"verify\s*=\s*False",
            r"ssl\._create_unverified_context",
        ],
        "severity": "medium",
        "description": "Security misconfiguration in application settings",
    },
    "A07-Auth-Failures": {
        "patterns": [
            r"password\s*==\s*",
            r"if.*token.*==.*None",
            r"authentication\s*=\s*False",
        ],
        "severity": "high",
        "description": "Authentication and authorization failures",
    },
}


# ── Tools ──────────────────────────────────────────────────────────────────────
@tool
def scan_owasp_vulnerabilities(code: str) -> str:
    """Scan code against OWASP Top 10 vulnerability patterns.
    Returns all found vulnerabilities with severity levels and line numbers."""
    findings = []
    lines = code.split("\n")

    for vuln_id, vuln_info in OWASP_PATTERNS.items():
        for pattern in vuln_info["patterns"]:
            for line_num, line in enumerate(lines, 1):
                if re.search(pattern, line, re.IGNORECASE):
                    findings.append({
                        "vulnerability": vuln_id,
                        "severity": vuln_info["severity"],
                        "line": line_num,
                        "code": line.strip(),
                        "description": vuln_info["description"],
                    })

    if not findings:
        return "✅ No OWASP Top 10 vulnerabilities detected."

    result = f"🚨 Found {len(findings)} vulnerability(-ies):\n\n"
    for f in sorted(findings, key=lambda x: {"critical": 0, "high": 1, "medium": 2, "low": 3}.get(x["severity"], 4)):
        severity_icon = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}.get(f["severity"], "⚪")
        result += (
            f"{severity_icon} [{f['severity'].upper()}] {f['vulnerability']}\n"
            f"   Line {f['line']}: {f['code'][:80]}\n"
            f"   Issue: {f['description']}\n\n"
        )
    return result


@tool
def check_dependency_vulnerabilities(requirements_text: str) -> str:
    """Analyze requirements.txt content for known vulnerable package versions.
    requirements_text: content of requirements.txt file"""
    known_vulnerabilities = {
        "django": {
            "vulnerable_below": "4.2.0",
            "cve": "CVE-2023-31047",
            "description": "File upload bypass vulnerability",
        },
        "flask": {
            "vulnerable_below": "2.3.0",
            "cve": "CVE-2023-30861",
            "description": "Cookie session security bypass",
        },
        "requests": {
            "vulnerable_below": "2.28.0",
            "cve": "CVE-2023-32681",
            "description": "Proxy-Authorization header leak",
        },
        "pillow": {
            "vulnerable_below": "9.3.0",
            "cve": "CVE-2022-45199",
            "description": "Denial of service vulnerability",
        },
        "cryptography": {
            "vulnerable_below": "41.0.0",
            "cve": "CVE-2023-38325",
            "description": "NULL pointer dereference vulnerability",
        },
    }

    findings = []
    for line in requirements_text.split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.match(r"^([a-zA-Z0-9_-]+)[>=<!\s]*([0-9.]+)?", line)
        if match:
            package = match.group(1).lower()
            version = match.group(2)
            if package in known_vulnerabilities:
                vuln = known_vulnerabilities[package]
                findings.append(
                    f"⚠️  {package} {version or '(unspecified)'}: "
                    f"{vuln['cve']} - {vuln['description']}\n"
                    f"   Fix: Upgrade to >= {vuln['vulnerable_below']}"
                )

    if not findings:
        return "✅ No known vulnerabilities found in dependencies (limited check)."
    return "Dependency Vulnerability Report:\n\n" + "\n\n".join(findings)


@tool
def analyze_authentication_patterns(code: str) -> str:
    """Analyze code for authentication and session management weaknesses."""
    issues = []
    code_lower = code.lower()

    auth_checks = {
        "Plain-text password comparison": r"password\s*==\s*['\"]",
        "Missing input validation": r"request\.(args|form|json)\[",
        "Insecure random for tokens": r"random\.(random|randint|choice)\(",
        "Missing CSRF protection": "csrf" not in code_lower and (
            "flask" in code_lower or "django" in code_lower
        ),
        "SQL without parameterization": r"['\"].*%s.*['\"].*execute",
        "Sensitive data in logs": r"(log|print|logger)\s*.*password",
    }

    for issue_name, pattern in auth_checks.items():
        if isinstance(pattern, bool):
            if pattern:
                issues.append(f"⚠️  {issue_name}")
        elif re.search(pattern, code, re.IGNORECASE):
            issues.append(f"⚠️  {issue_name} detected")

    # Check for security best practices
    good_practices = []
    if "bcrypt" in code_lower or "argon2" in code_lower or "pbkdf2" in code_lower:
        good_practices.append("✅ Using secure password hashing")
    if "secrets" in code_lower or "os.urandom" in code_lower:
        good_practices.append("✅ Using secure random number generation")
    if "https" in code_lower:
        good_practices.append("✅ HTTPS references found")

    result = "Authentication Security Analysis:\n\n"
    if issues:
        result += "Issues Found:\n" + "\n".join(f"  {i}" for i in issues)
    else:
        result += "No authentication issues detected.\n"
    if good_practices:
        result += "\n\nGood Practices:\n" + "\n".join(f"  {p}" for p in good_practices)
    return result


@tool
def generate_security_report(
    code_vulnerabilities: str,
    dependency_issues: str,
    auth_issues: str,
) -> str:
    """Compile all security findings into a prioritized remediation report."""
    report = (
        "=" * 50 + "\n"
        "       SECURITY AUDIT REPORT\n"
        "=" * 50 + "\n\n"
        "EXECUTIVE SUMMARY\n"
        "-" * 30 + "\n"
        "This automated security audit identified the following findings.\n"
        "Immediate action is recommended for critical and high severity issues.\n\n"
        "FINDINGS\n"
        "-" * 30 + "\n"
        f"Code Vulnerabilities:\n{code_vulnerabilities}\n\n"
        f"Dependency Issues:\n{dependency_issues}\n\n"
        f"Authentication Issues:\n{auth_issues}\n\n"
        "REMEDIATION PRIORITY\n"
        "-" * 30 + "\n"
        "1. 🔴 CRITICAL: Fix immediately - active exploitation risk\n"
        "2. 🟠 HIGH: Fix within 24 hours\n"
        "3. 🟡 MEDIUM: Fix within 1 week\n"
        "4. 🟢 LOW: Fix in next sprint\n\n"
        "GENERAL RECOMMENDATIONS\n"
        "-" * 30 + "\n"
        "• Enable automated dependency scanning (Dependabot, Snyk)\n"
        "• Implement SAST in CI/CD pipeline\n"
        "• Conduct regular penetration testing\n"
        "• Follow OWASP Secure Coding Practices\n"
        "• Implement security training for development team\n"
        "=" * 50
    )
    return report


tools = [
    scan_owasp_vulnerabilities,
    check_dependency_vulnerabilities,
    analyze_authentication_patterns,
    generate_security_report,
]

SYSTEM_PROMPT = """You are a senior application security engineer conducting
a comprehensive security audit.

Audit process:
1. Use scan_owasp_vulnerabilities to check for OWASP Top 10 issues
2. Use check_dependency_vulnerabilities if requirements are provided
3. Use analyze_authentication_patterns for auth/session weaknesses
4. Use generate_security_report to compile all findings
5. Provide specific, actionable remediation code examples

Be thorough, precise, and prioritize findings by actual exploitability and impact.
Always provide code examples showing how to fix each vulnerability."""


def build_security_graph():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools(tools)

    def auditor(state: SecurityState) -> dict:
        return {"messages": [llm.invoke(state["messages"])]}

    tool_node = ToolNode(tools)
    graph = StateGraph(SecurityState)
    graph.add_node("auditor", auditor)
    graph.add_node("tools", tool_node)
    graph.add_edge(START, "auditor")
    graph.add_conditional_edges("auditor", tools_condition)
    graph.add_edge("tools", "auditor")
    return graph.compile()


def audit_code(code: str, language: str = "python") -> str:
    app = build_security_graph()
    prompt = (
        f"Perform a comprehensive security audit on this {language} code:\n\n"
        f"```{language}\n{code}\n```\n\n"
        "Use all available tools to scan for vulnerabilities, check authentication "
        "patterns, and compile a complete security report with specific fix recommendations."
    )
    state: SecurityState = {
        "messages": [HumanMessage(content=prompt)],
        "code": code,
        "language": language,
        "severity_threshold": "low",
    }
    result = app.invoke(state)
    return result["messages"][-1].content


VULNERABLE_CODE = '''
import os
import pickle
import hashlib
import sqlite3
from flask import Flask, request, session
import random

app = Flask(__name__)
app.secret_key = "mysecretkey123"
DEBUG = True

def login(username, password):
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    # Vulnerable to SQL injection
    query = f"SELECT * FROM users WHERE username = '{username}' AND password = '{password}'"
    cursor.execute(query)
    user = cursor.fetchone()
    if user:
        session["user"] = username
    return user

def hash_password(password):
    # Weak hashing
    return hashlib.md5(password.encode()).hexdigest()

def run_command(user_input):
    # Command injection
    os.system(f"echo {user_input}")

def load_user_data(data):
    # Insecure deserialization
    return pickle.loads(data)

def generate_token():
    # Insecure random
    return random.randint(100000, 999999)
'''


def main():
    print("🔒 Security Audit Agent\n" + "=" * 60)
    print("Auditing sample vulnerable code...\n")
    result = audit_code(VULNERABLE_CODE, "python")
    print(result)
    print("\n✅ Security audit complete!")


if __name__ == "__main__":
    main()
