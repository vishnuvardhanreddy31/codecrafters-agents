"""
Education Tutor Agent
Uses LangGraph to provide personalized tutoring with adaptive learning,
quizzes, explanations, and progress tracking across subjects.
"""

import os
import random
from typing import Annotated, TypedDict, List, Optional
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()

# ── Quiz questions bank ────────────────────────────────────────────────────────
QUIZ_BANK = {
    "math": [
        {
            "question": "What is the derivative of f(x) = x³ + 2x² - 5?",
            "answer": "f'(x) = 3x² + 4x",
            "explanation": "Apply the power rule: d/dx(xⁿ) = nxⁿ⁻¹ to each term. Constants have derivative 0.",
        },
        {
            "question": "Solve for x: 2x² - 8 = 0",
            "answer": "x = ±2",
            "explanation": "2x² = 8 → x² = 4 → x = ±2",
        },
        {
            "question": "What is the integral of 3x² dx?",
            "answer": "x³ + C",
            "explanation": "Apply the power rule for integration: ∫xⁿ dx = xⁿ⁺¹/(n+1) + C",
        },
    ],
    "python": [
        {
            "question": "What does the 'yield' keyword do in Python?",
            "answer": "It creates a generator function that returns values lazily",
            "explanation": "yield pauses function execution and returns a value. The function resumes from where it left off on the next call, making generators memory-efficient for large sequences.",
        },
        {
            "question": "What is the difference between a list and a tuple in Python?",
            "answer": "Lists are mutable (can be changed), tuples are immutable (cannot be changed after creation)",
            "explanation": "Lists use [], tuples use (). Tuples are faster and use less memory. Use tuples for data that shouldn't change.",
        },
        {
            "question": "What does the __init__ method do in a Python class?",
            "answer": "It's the constructor that initializes a new instance of the class",
            "explanation": "__init__ is automatically called when creating a new object. It sets up the initial state by assigning values to instance attributes.",
        },
    ],
    "history": [
        {
            "question": "What year did World War II end?",
            "answer": "1945",
            "explanation": "WWII ended in 1945: Germany surrendered on May 8 (V-E Day) and Japan surrendered on September 2 (V-J Day) after atomic bombs were dropped on Hiroshima and Nagasaki.",
        },
        {
            "question": "Who was the first President of the United States?",
            "answer": "George Washington",
            "explanation": "George Washington served as the first U.S. President from 1789 to 1797. He was unanimously elected by the Electoral College.",
        },
    ],
    "science": [
        {
            "question": "What is the chemical formula for water?",
            "answer": "H₂O",
            "explanation": "Water consists of two hydrogen atoms covalently bonded to one oxygen atom. The formula H₂O represents this molecular structure.",
        },
        {
            "question": "What force keeps planets in orbit around the sun?",
            "answer": "Gravity",
            "explanation": "Gravitational force between the sun and planets keeps them in elliptical orbits. Newton's law of universal gravitation: F = G(m₁m₂)/r²",
        },
    ],
}

# ── Progress tracker ───────────────────────────────────────────────────────────
_student_progress = {
    "topics_studied": [],
    "quiz_scores": [],
    "concepts_mastered": [],
    "weak_areas": [],
}


# ── State ──────────────────────────────────────────────────────────────────────
class TutorState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    subject: str
    student_level: str
    learning_goals: str


# ── Tools ──────────────────────────────────────────────────────────────────────
@tool
def explain_concept(concept: str, detail_level: str = "intermediate") -> str:
    """Explain an educational concept with examples.
    detail_level: beginner | intermediate | advanced"""
    levels = {
        "beginner": "Use very simple language, analogies, and basic examples.",
        "intermediate": "Use clear explanations with moderate technical depth.",
        "advanced": "Provide comprehensive technical depth with nuances.",
    }
    instruction = levels.get(detail_level, levels["intermediate"])
    return (
        f"Concept explanation requested: '{concept}' at {detail_level} level.\n"
        f"Instruction for agent: {instruction}\n"
        "The agent will provide a comprehensive explanation with examples, "
        "key points, and common misconceptions."
    )


@tool
def generate_quiz(subject: str, difficulty: str = "medium", num_questions: int = 3) -> str:
    """Generate a quiz for the student on a given subject.
    subject: math | python | history | science
    difficulty: easy | medium | hard"""
    subject_lower = subject.lower()
    if subject_lower not in QUIZ_BANK:
        available = list(QUIZ_BANK.keys())
        return (
            f"Quiz not available for '{subject}'. "
            f"Available subjects: {', '.join(available)}"
        )

    questions = QUIZ_BANK[subject_lower]
    selected = random.sample(questions, min(num_questions, len(questions)))

    _student_progress["topics_studied"].append(subject_lower)

    quiz = f"📝 Quiz: {subject.title()} ({difficulty} difficulty)\n"
    quiz += "=" * 40 + "\n"
    for i, q in enumerate(selected, 1):
        quiz += f"\nQ{i}: {q['question']}\n"
    quiz += "\n[Answer when ready - type your answers and I'll provide feedback!]"
    quiz += f"\n\n_Answers stored for grading: {[q['answer'] for q in selected]}_"
    return quiz


@tool
def check_answer(subject: str, question: str, student_answer: str) -> str:
    """Check a student's answer and provide detailed feedback.
    Returns correct/incorrect status, the right answer, and full explanation."""
    subject_lower = subject.lower()
    if subject_lower not in QUIZ_BANK:
        return f"Subject '{subject}' not found."

    for q in QUIZ_BANK[subject_lower]:
        if any(word in q["question"].lower() for word in question.lower().split()[:4]):
            is_correct = (
                student_answer.strip().lower() in q["answer"].lower()
                or q["answer"].lower() in student_answer.strip().lower()
            )
            if is_correct:
                _student_progress["quiz_scores"].append(1)
                _student_progress["concepts_mastered"].append(
                    q["question"][:40]
                )
                return (
                    f"✅ Correct! Well done!\n\n"
                    f"Full answer: {q['answer']}\n\n"
                    f"Explanation: {q['explanation']}"
                )
            else:
                _student_progress["quiz_scores"].append(0)
                _student_progress["weak_areas"].append(subject_lower)
                return (
                    f"❌ Not quite. Let's learn from this!\n\n"
                    f"Correct answer: {q['answer']}\n\n"
                    f"Explanation: {q['explanation']}\n\n"
                    f"💡 Tip: Review this concept and try similar problems."
                )
    return "Question not found in the database. The agent will evaluate your answer."


@tool
def get_learning_resources(topic: str, resource_type: str = "all") -> str:
    """Get recommended learning resources for a topic.
    resource_type: books | videos | exercises | all"""
    resources = {
        "python": {
            "books": ["'Python Crash Course' by Eric Matthes", "'Fluent Python' by Luciano Ramalho"],
            "videos": ["CS50P (Harvard's Python course - free)", "Corey Schafer Python Tutorials (YouTube)"],
            "exercises": ["LeetCode", "HackerRank Python track", "Exercism.io"],
        },
        "math": {
            "books": ["'Calculus' by James Stewart", "'Linear Algebra Done Right' by Axler"],
            "videos": ["3Blue1Brown (YouTube)", "Khan Academy Math"],
            "exercises": ["Brilliant.org", "Wolfram Problem Generator", "Art of Problem Solving"],
        },
        "history": {
            "books": ["'Sapiens' by Yuval Noah Harari", "'Guns Germs and Steel' by Jared Diamond"],
            "videos": ["Crash Course History (YouTube)", "History Channel Documentaries"],
            "exercises": ["Quizlet History Flashcards", "Sporcle History Quizzes"],
        },
    }

    topic_lower = topic.lower()
    for key, res in resources.items():
        if key in topic_lower or topic_lower in key:
            if resource_type == "all":
                result = f"Learning Resources for {topic}:\n"
                for rtype, items in res.items():
                    result += f"\n{rtype.title()}:\n"
                    result += "\n".join(f"  • {item}" for item in items)
                return result
            elif resource_type in res:
                return f"{resource_type.title()} for {topic}:\n" + "\n".join(
                    f"  • {item}" for item in res[resource_type]
                )

    return (
        f"General resources for '{topic}':\n"
        f"  • Search on Coursera, edX, or Khan Academy\n"
        f"  • Wikipedia for overviews\n"
        f"  • YouTube for video tutorials\n"
        f"  • Reddit communities for peer help"
    )


@tool
def get_student_progress() -> str:
    """Get the current student's learning progress summary."""
    scores = _student_progress["quiz_scores"]
    avg_score = (sum(scores) / len(scores) * 100) if scores else 0
    return (
        f"Student Progress Report:\n"
        f"  Topics studied: {', '.join(set(_student_progress['topics_studied'])) or 'None yet'}\n"
        f"  Quiz attempts: {len(scores)}\n"
        f"  Average score: {avg_score:.0f}%\n"
        f"  Concepts mastered: {len(_student_progress['concepts_mastered'])}\n"
        f"  Areas needing review: {', '.join(set(_student_progress['weak_areas'])) or 'None identified'}"
    )


tools = [
    explain_concept, generate_quiz, check_answer,
    get_learning_resources, get_student_progress,
]

SYSTEM_PROMPT = """You are an encouraging, patient, and knowledgeable tutor.

Teaching approach:
- Use the Socratic method: guide students to discover answers themselves
- Use check_answer when a student provides an answer to a quiz question
- Use explain_concept for detailed explanations with examples
- Use generate_quiz to test knowledge and reinforce learning
- Use get_learning_resources to suggest further study materials
- Use get_student_progress to monitor and discuss the student's journey

Adapt your language and complexity to the student's level. Always be encouraging,
even when students make mistakes. Celebrate progress and frame errors as learning
opportunities."""


def build_tutor_graph():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5).bind_tools(tools)

    def tutor(state: TutorState) -> dict:
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(state["messages"])
        return {"messages": [llm.invoke(messages)]}

    tool_node = ToolNode(tools)
    graph = StateGraph(TutorState)
    graph.add_node("tutor", tutor)
    graph.add_node("tools", tool_node)
    graph.add_edge(START, "tutor")
    graph.add_conditional_edges("tutor", tools_condition)
    graph.add_edge("tools", "tutor")
    return graph.compile()


def main():
    app = build_tutor_graph()

    print("🎓 Education Tutor Agent\n" + "=" * 60)
    subject = input("What subject would you like to study? (math/python/history/science): ").strip() or "python"
    level = input("Your level (beginner/intermediate/advanced): ").strip() or "intermediate"

    state: TutorState = {
        "messages": [],
        "subject": subject,
        "student_level": level,
        "learning_goals": f"Master {subject} concepts at {level} level",
    }

    print(f"\nTutor: Welcome! I'm your personal {subject.title()} tutor. Let's start learning!")
    print("Try asking: 'Explain recursion', 'Give me a quiz', or 'What are my weak areas?'")
    print("Type 'quit' to exit.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("quit", "exit"):
            app.invoke(
                {**state, "messages": state["messages"] + [HumanMessage(content="Show my progress")]}
            )
            print("\nTutor: Great session! Keep practicing. Learning is a journey! 📚")
            break
        if not user_input:
            continue

        state["messages"].append(HumanMessage(content=user_input))
        result = app.invoke(state)
        state["messages"] = result["messages"]
        print(f"\nTutor: {result['messages'][-1].content}\n")


if __name__ == "__main__":
    main()
