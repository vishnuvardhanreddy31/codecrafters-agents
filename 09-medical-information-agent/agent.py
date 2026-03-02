"""
Medical Information Agent
Uses LangGraph with RAG (Retrieval Augmented Generation) to answer
medical questions from a knowledge base with safety guardrails.

DISCLAIMER: This agent provides general health information only and is NOT
a substitute for professional medical advice, diagnosis, or treatment.
"""

import os
from typing import Annotated, TypedDict, List
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_core.documents import Document
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()

# ── Medical knowledge base (sample documents) ─────────────────────────────────
MEDICAL_DOCUMENTS = [
    Document(
        page_content=(
            "Diabetes mellitus is a chronic metabolic disease characterized by elevated "
            "blood glucose levels. Type 1 diabetes is an autoimmune condition where the "
            "pancreas produces little or no insulin. Type 2 diabetes occurs when the body "
            "doesn't use insulin effectively. Symptoms include increased thirst, frequent "
            "urination, fatigue, blurred vision, and slow-healing wounds. Management includes "
            "diet, exercise, blood glucose monitoring, and medication or insulin therapy."
        ),
        metadata={"category": "conditions", "topic": "diabetes"},
    ),
    Document(
        page_content=(
            "Hypertension (high blood pressure) is a condition where the force of blood "
            "against artery walls is consistently too high (≥130/80 mmHg). Risk factors "
            "include obesity, high-sodium diet, lack of exercise, smoking, and family history. "
            "It's often called the 'silent killer' as it has few symptoms. Treatment includes "
            "lifestyle changes (DASH diet, exercise, stress reduction) and medications such as "
            "ACE inhibitors, beta-blockers, diuretics, and calcium channel blockers."
        ),
        metadata={"category": "conditions", "topic": "hypertension"},
    ),
    Document(
        page_content=(
            "Common cold is caused by rhinoviruses and other viruses. Symptoms include "
            "runny nose, sore throat, cough, congestion, sneezing, and mild fever. "
            "Treatment is symptomatic: rest, hydration, over-the-counter decongestants, "
            "antihistamines, and pain relievers. Antibiotics are ineffective against viruses. "
            "Prevention includes frequent handwashing and avoiding touching the face."
        ),
        metadata={"category": "conditions", "topic": "common cold"},
    ),
    Document(
        page_content=(
            "Healthy nutrition guidelines recommend: fruits and vegetables (half your plate), "
            "whole grains (quarter of your plate), lean proteins, and low-fat dairy. "
            "Limit saturated fats, trans fats, sodium (<2300mg/day), and added sugars. "
            "Stay hydrated with 8 glasses of water daily. The Mediterranean diet, "
            "DASH diet, and plant-based diets are associated with reduced chronic disease risk."
        ),
        metadata={"category": "wellness", "topic": "nutrition"},
    ),
    Document(
        page_content=(
            "Exercise guidelines for adults: at least 150 minutes of moderate-intensity "
            "aerobic activity per week, or 75 minutes of vigorous activity. Add muscle-"
            "strengthening activities 2+ days per week. Benefits include: reduced risk of "
            "heart disease, stroke, type 2 diabetes, cancer; improved mental health; "
            "better weight management; stronger bones and muscles."
        ),
        metadata={"category": "wellness", "topic": "exercise"},
    ),
    Document(
        page_content=(
            "Mental health encompasses emotional, psychological, and social well-being. "
            "Common mental health conditions include depression, anxiety disorders, bipolar "
            "disorder, and PTSD. Signs of poor mental health: persistent sadness, excessive "
            "worry, mood swings, withdrawal, substance use changes. Treatments include "
            "psychotherapy (CBT, DBT), medication, lifestyle changes, and support groups. "
            "Crisis resources: National Suicide Prevention Lifeline: 988."
        ),
        metadata={"category": "mental health", "topic": "mental health overview"},
    ),
    Document(
        page_content=(
            "Medication safety: Always take medications as prescribed. Do not share medications. "
            "Check for drug interactions before combining medications. Common dangerous "
            "combinations: blood thinners + NSAIDs, MAOIs + many antidepressants. "
            "Store medications as directed. Dispose of unused medications safely. "
            "Inform all healthcare providers about all medications, supplements, and herbal remedies."
        ),
        metadata={"category": "safety", "topic": "medication safety"},
    ),
    Document(
        page_content=(
            "First aid for common emergencies: Burns - cool with running water for 10+ minutes, "
            "do not use ice or butter. Choking - Heimlich maneuver for adults. CPR - 30 chest "
            "compressions to 2 rescue breaths. Bleeding - apply firm pressure with clean cloth. "
            "Seizures - protect from injury, do not restrain, call 911 if >5 minutes. "
            "Allergic reaction - use EpiPen if available, call 911 immediately."
        ),
        metadata={"category": "emergency", "topic": "first aid"},
    ),
]


# ── Vector store (in-memory) ───────────────────────────────────────────────────
def build_vector_store():
    try:
        from langchain_community.vectorstores import FAISS
        embeddings = OpenAIEmbeddings()
        return FAISS.from_documents(MEDICAL_DOCUMENTS, embeddings)
    except Exception:
        return None


# ── State ──────────────────────────────────────────────────────────────────────
class MedicalState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    query: str


# ── Tools ──────────────────────────────────────────────────────────────────────
_vector_store = None


@tool
def search_medical_knowledge(query: str) -> str:
    """Search the medical knowledge base for relevant health information.
    Returns the most relevant documents for the given health query."""
    global _vector_store
    if _vector_store is None:
        _vector_store = build_vector_store()

    if _vector_store:
        docs = _vector_store.similarity_search(query, k=2)
        if docs:
            return "\n\n---\n\n".join(
                f"[{d.metadata.get('topic', 'General')}]\n{d.page_content}"
                for d in docs
            )

    # Fallback: keyword matching
    query_lower = query.lower()
    relevant = []
    for doc in MEDICAL_DOCUMENTS:
        topic = doc.metadata.get("topic", "")
        if topic in query_lower or any(
            word in doc.page_content.lower() for word in query_lower.split()[:3]
        ):
            relevant.append(doc.page_content)
    return "\n\n".join(relevant[:2]) if relevant else "No specific information found on this topic."


@tool
def check_emergency_symptoms(symptoms: str) -> str:
    """Check if described symptoms indicate a medical emergency requiring immediate care."""
    emergency_indicators = [
        "chest pain", "difficulty breathing", "shortness of breath",
        "sudden severe headache", "facial drooping", "arm weakness",
        "speech difficulty", "loss of consciousness", "uncontrolled bleeding",
        "severe allergic reaction", "suicidal thoughts", "overdose",
        "severe abdominal pain", "high fever with stiff neck",
    ]
    symptoms_lower = symptoms.lower()
    found = [e for e in emergency_indicators if e in symptoms_lower]

    if found:
        return (
            "⚠️ URGENT: These symptoms may indicate a medical emergency!\n"
            f"Emergency indicators: {', '.join(found)}\n\n"
            "🚨 CALL 911 IMMEDIATELY or go to the nearest emergency room.\n"
            "Do not wait or self-treat these symptoms."
        )
    return (
        "No immediate emergency indicators detected. However, if symptoms are severe, "
        "worsening, or you're concerned, please consult a healthcare provider."
    )


@tool
def get_specialist_recommendation(condition: str) -> str:
    """Recommend the appropriate medical specialist for a given condition or symptom."""
    specialist_map = {
        "heart": "Cardiologist",
        "skin": "Dermatologist",
        "bone": "Orthopedist",
        "mental": "Psychiatrist or Psychologist",
        "eye": "Ophthalmologist",
        "ear": "ENT (Ear, Nose, Throat) Specialist",
        "stomach": "Gastroenterologist",
        "lung": "Pulmonologist",
        "kidney": "Nephrologist",
        "diabetes": "Endocrinologist",
        "hormone": "Endocrinologist",
        "cancer": "Oncologist",
        "child": "Pediatrician",
        "women": "Gynecologist/OB-GYN",
        "pregnancy": "OB-GYN",
        "nerve": "Neurologist",
        "joint": "Rheumatologist",
        "allergy": "Allergist/Immunologist",
    }
    condition_lower = condition.lower()
    for keyword, specialist in specialist_map.items():
        if keyword in condition_lower:
            return (
                f"Recommended specialist: {specialist}\n"
                "Start with your primary care physician who can provide a referral."
            )
    return (
        "Recommended: Start with your Primary Care Physician (PCP) or General Practitioner. "
        "They can evaluate your condition and refer you to the appropriate specialist."
    )


tools = [search_medical_knowledge, check_emergency_symptoms, get_specialist_recommendation]

SYSTEM_PROMPT = """You are a knowledgeable medical information assistant providing general
health education. You are NOT a doctor and cannot diagnose conditions.

Guidelines:
- Use search_medical_knowledge to find relevant health information
- Use check_emergency_symptoms for any concerning symptom descriptions
- Use get_specialist_recommendation when appropriate
- Always recommend consulting a qualified healthcare provider
- For mental health crises, always mention: National Crisis Line 988
- Never diagnose conditions; only provide educational information
- Be empathetic, clear, and safety-focused

⚕️ DISCLAIMER: Information provided is educational only and not a substitute for
professional medical advice."""


def build_medical_graph():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1).bind_tools(tools)

    def medical_agent(state: MedicalState) -> dict:
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(state["messages"])
        return {"messages": [llm.invoke(messages)]}

    tool_node = ToolNode(tools)
    graph = StateGraph(MedicalState)
    graph.add_node("medical_agent", medical_agent)
    graph.add_node("tools", tool_node)
    graph.add_edge(START, "medical_agent")
    graph.add_conditional_edges("medical_agent", tools_condition)
    graph.add_edge("tools", "medical_agent")
    return graph.compile()


def get_medical_info(query: str) -> str:
    app = build_medical_graph()
    state: MedicalState = {
        "messages": [HumanMessage(content=query)],
        "query": query,
    }
    result = app.invoke(state)
    return result["messages"][-1].content


def main():
    app = build_medical_graph()
    state: MedicalState = {"messages": [], "query": ""}

    print("⚕️  Medical Information Agent")
    print("=" * 60)
    print("⚠️  DISCLAIMER: For educational purposes only. Not medical advice.")
    print("Type 'quit' to exit.\n")

    while True:
        query = input("Your health question: ").strip()
        if query.lower() in ("quit", "exit"):
            print("Stay healthy! Always consult your doctor for medical concerns. 👋")
            break
        if not query:
            continue

        state["messages"].append(HumanMessage(content=query))
        result = app.invoke(state)
        state["messages"] = result["messages"]
        print(f"\nAgent: {result['messages'][-1].content}\n")


if __name__ == "__main__":
    main()
