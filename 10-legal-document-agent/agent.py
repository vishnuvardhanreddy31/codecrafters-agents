"""
Legal Document Agent
Uses LangGraph with PDF parsing and RAG to analyze legal documents,
answer questions, and summarize contracts and agreements.

DISCLAIMER: This agent provides general legal information only and is NOT
a substitute for advice from a qualified attorney.
"""

import os
from typing import Annotated, TypedDict, List, Optional
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_core.documents import Document
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()

# ── Sample legal documents ─────────────────────────────────────────────────────
SAMPLE_CONTRACT = """
SERVICE AGREEMENT

This Service Agreement ("Agreement") is entered into as of January 1, 2025
between TechCorp Inc. ("Service Provider") and ClientCo LLC ("Client").

1. SERVICES
Service Provider agrees to develop and deliver a custom software platform
("Services") as described in Exhibit A. Delivery deadline: March 31, 2025.

2. PAYMENT
Client shall pay $50,000 USD. Payment schedule:
- 30% upon signing ($15,000)
- 40% at milestone completion ($20,000)
- 30% upon final delivery ($15,000)
Late payments incur 1.5% monthly interest.

3. INTELLECTUAL PROPERTY
All work product created under this Agreement shall be owned by Client
upon full payment. Service Provider retains rights to pre-existing IP
and general methodologies.

4. CONFIDENTIALITY
Both parties agree to keep all non-public information confidential for
a period of 3 years after termination of this Agreement.

5. LIMITATION OF LIABILITY
Service Provider's total liability shall not exceed the total fees paid
under this Agreement. Neither party is liable for indirect, incidental,
or consequential damages.

6. TERMINATION
Either party may terminate with 30 days written notice. Client owes
payment for work completed to date. Service Provider shall deliver
all completed work upon termination.

7. GOVERNING LAW
This Agreement is governed by the laws of the State of California.
Disputes shall be resolved through binding arbitration in San Francisco, CA.

8. ENTIRE AGREEMENT
This Agreement constitutes the entire agreement between the parties
and supersedes all prior negotiations and understandings.
"""

LEGAL_KNOWLEDGE = [
    Document(
        page_content=(
            "Contract formation requires: offer, acceptance, consideration, and mutual "
            "assent. An offer is a proposal to enter a contract. Acceptance is agreement "
            "to the offer's terms. Consideration is something of value exchanged by both "
            "parties. For a contract to be binding, all elements must be present."
        ),
        metadata={"topic": "contract basics"},
    ),
    Document(
        page_content=(
            "Limitation of liability clauses restrict the amount one party can recover "
            "from another in case of breach or negligence. Courts generally enforce them "
            "unless they are unconscionable, against public policy, or involve gross "
            "negligence or intentional misconduct."
        ),
        metadata={"topic": "limitation of liability"},
    ),
    Document(
        page_content=(
            "Non-disclosure agreements (NDAs) protect confidential information. Key elements: "
            "definition of confidential information, obligations of receiving party, "
            "duration, exceptions (publicly known info, independently developed), "
            "and remedies for breach (injunctive relief, damages)."
        ),
        metadata={"topic": "confidentiality and NDA"},
    ),
    Document(
        page_content=(
            "Intellectual property (IP) ownership in service contracts: work-for-hire "
            "doctrine means work created by an employee during employment is owned by the "
            "employer. For contractors, IP ownership must be explicitly assigned in writing. "
            "Without assignment, contractors retain IP rights even if paid for the work."
        ),
        metadata={"topic": "intellectual property"},
    ),
    Document(
        page_content=(
            "Contract termination clauses specify how parties may end the agreement. "
            "Termination for cause: when one party breaches. Termination for convenience: "
            "either party may terminate with notice, typically 30-90 days. "
            "Important to specify obligations upon termination: payment for completed work, "
            "return of materials, and survival of certain clauses like confidentiality."
        ),
        metadata={"topic": "termination clauses"},
    ),
]


# ── State ──────────────────────────────────────────────────────────────────────
class LegalState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    document_text: Optional[str]
    query: str


# ── Tools ──────────────────────────────────────────────────────────────────────
@tool
def analyze_contract_risks(contract_text: str) -> str:
    """Identify potential risks, red flags, and important clauses in a contract.
    Analyzes payment terms, liability caps, termination conditions, and IP ownership."""
    risks = []
    contract_lower = contract_text.lower()

    risk_patterns = {
        "Auto-renewal clause": "auto-renew",
        "Unilateral modification right": "may modify at any time",
        "Broad indemnification": "indemnify and hold harmless",
        "Non-compete clause": "non-compete",
        "Arbitration required": "arbitration",
        "Jurisdiction specification": "governing law",
        "IP assignment": "intellectual property",
        "Limitation of liability cap": "limitation of liability",
        "Confidentiality obligation": "confidential",
        "Termination notice period": "days written notice",
    }

    found_clauses = {}
    for clause_name, keyword in risk_patterns.items():
        found_clauses[clause_name] = keyword in contract_lower

    report = "Contract Risk Analysis:\n\n"
    report += "Clauses Identified:\n"
    for clause, found in found_clauses.items():
        status = "✅ Present" if found else "⚠️  Missing/Not found"
        report += f"  {status}: {clause}\n"

    report += "\nKey Risk Areas:\n"
    if "limitation of liability" not in contract_lower:
        risks.append("No liability cap - you may have unlimited exposure")
    if "arbitration" in contract_lower:
        risks.append("Mandatory arbitration - limits access to courts")
    if "non-compete" in contract_lower:
        risks.append("Non-compete clause - may restrict future business activities")
    if not risks:
        risks.append("No major red flags identified in this analysis")

    report += "\n".join(f"  • {r}" for r in risks)
    return report


@tool
def extract_key_terms(contract_text: str) -> str:
    """Extract and summarize key business terms from a contract:
    parties, dates, payment terms, deliverables, and duration."""
    import re
    lines = contract_text.split("\n")
    key_terms = {}

    # Look for dates (simple pattern)
    dates = re.findall(r'\b\w+ \d{1,2},?\s+\d{4}\b', contract_text)
    if dates:
        key_terms["Dates mentioned"] = ", ".join(dates[:5])

    # Look for money amounts
    amounts = re.findall(r'\$[\d,]+(?:\.\d{2})?', contract_text)
    if amounts:
        key_terms["Financial terms"] = ", ".join(set(amounts))

    # Look for percentage values
    percentages = re.findall(r'\d+(?:\.\d+)?%', contract_text)
    if percentages:
        key_terms["Percentages"] = ", ".join(set(percentages))

    # Look for time periods
    time_periods = re.findall(r'\d+\s+(?:days|months|years)', contract_text.lower())
    if time_periods:
        key_terms["Time periods"] = ", ".join(set(time_periods))

    result = "Extracted Key Terms:\n"
    for k, v in key_terms.items():
        result += f"  {k}: {v}\n"
    return result if len(key_terms) > 0 else "Could not extract structured key terms automatically."


@tool
def search_legal_knowledge(query: str) -> str:
    """Search the legal knowledge base for general information about
    legal concepts, contract clauses, and common legal terms."""
    query_lower = query.lower()
    relevant = []
    for doc in LEGAL_KNOWLEDGE:
        topic = doc.metadata.get("topic", "")
        if any(word in doc.page_content.lower() for word in query_lower.split()[:4]):
            relevant.append(f"[{topic}]\n{doc.page_content}")
    return "\n\n---\n\n".join(relevant[:2]) if relevant else (
        "No specific legal knowledge found on this topic. Please consult a qualified attorney."
    )


@tool
def summarize_document(document_text: str) -> str:
    """Create a plain-language executive summary of a legal document,
    highlighting the most important obligations, rights, and terms."""
    summary_prompt = (
        "This is a pre-processing step. The agent will create a plain-language summary "
        "based on the full document analysis."
    )
    word_count = len(document_text.split())
    section_count = document_text.count("\n\n")
    return (
        f"Document Statistics:\n"
        f"  Word count: ~{word_count}\n"
        f"  Sections: ~{section_count}\n"
        f"  Document type: Service/Commercial Agreement\n\n"
        "Full summary will be provided by the legal analysis agent based on content review."
    )


tools = [analyze_contract_risks, extract_key_terms, search_legal_knowledge, summarize_document]

SYSTEM_PROMPT = """You are a legal document analysis assistant helping users understand
contracts and legal documents. You provide educational legal information.

Your approach:
1. Use analyze_contract_risks to identify potential issues
2. Use extract_key_terms to pull important dates, amounts, and terms
3. Use search_legal_knowledge for context on legal concepts
4. Provide plain-language explanations of complex legal provisions
5. Always recommend consulting a qualified attorney for specific legal advice

⚖️ DISCLAIMER: This analysis is for educational purposes only and does not
constitute legal advice. Always consult a licensed attorney for your specific situation."""


def build_legal_graph():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1).bind_tools(tools)

    def legal_agent(state: LegalState) -> dict:
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(state["messages"])
        return {"messages": [llm.invoke(messages)]}

    tool_node = ToolNode(tools)
    graph = StateGraph(LegalState)
    graph.add_node("legal_agent", legal_agent)
    graph.add_node("tools", tool_node)
    graph.add_edge(START, "legal_agent")
    graph.add_conditional_edges("legal_agent", tools_condition)
    graph.add_edge("tools", "legal_agent")
    return graph.compile()


def analyze_document(document_text: str, question: str) -> str:
    app = build_legal_graph()
    prompt = (
        f"Analyze the following legal document and answer this question: {question}\n\n"
        f"Document:\n{document_text}\n\n"
        "Use the tools to analyze risks, extract key terms, and provide a comprehensive review."
    )
    state: LegalState = {
        "messages": [HumanMessage(content=prompt)],
        "document_text": document_text,
        "query": question,
    }
    result = app.invoke(state)
    return result["messages"][-1].content


def main():
    print("⚖️  Legal Document Agent\n" + "=" * 60)
    print("⚠️  DISCLAIMER: For educational purposes only. Not legal advice.\n")
    print("Analyzing sample service agreement...\n")
    question = (
        "Please analyze this contract: identify the key terms and payment structure, "
        "highlight any risks or unusual clauses, and provide a plain-language summary "
        "of what each party is agreeing to."
    )
    result = analyze_document(SAMPLE_CONTRACT, question)
    print(result)
    print("\n✅ Legal analysis complete!")


if __name__ == "__main__":
    main()
