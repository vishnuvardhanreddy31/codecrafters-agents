"""
Job Application Agent
Uses CrewAI with specialized agents (Resume Analyzer, Cover Letter Writer,
Interview Coach) to help candidates with their job search.
"""

import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# ── Agents ─────────────────────────────────────────────────────────────────────
resume_analyzer = Agent(
    role="Resume Optimization Expert",
    goal=(
        "Analyze the candidate's background and job requirements to create "
        "an optimized, ATS-friendly resume that highlights relevant experience."
    ),
    backstory=(
        "You are a certified career coach and resume expert with 15 years of "
        "experience helping professionals land jobs at top companies. You know "
        "exactly what hiring managers and ATS systems look for."
    ),
    llm=llm,
    verbose=True,
    allow_delegation=False,
)

cover_letter_writer = Agent(
    role="Cover Letter Specialist",
    goal=(
        "Write a compelling, personalized cover letter that tells the candidate's "
        "story and demonstrates genuine enthusiasm for the specific role and company."
    ),
    backstory=(
        "You are a professional writer specializing in persuasive career documents. "
        "You craft cover letters that open doors by connecting the candidate's unique "
        "experience to the company's specific needs and culture."
    ),
    llm=llm,
    verbose=True,
    allow_delegation=False,
)

interview_coach = Agent(
    role="Interview Preparation Coach",
    goal=(
        "Prepare the candidate for the interview by anticipating likely questions, "
        "crafting strong answers using the STAR method, and providing confidence tips."
    ),
    backstory=(
        "You are an executive interview coach who has helped thousands of candidates "
        "successfully navigate interviews at Fortune 500 companies. You know exactly "
        "what behavioral questions to expect and how to answer them brilliantly."
    ),
    llm=llm,
    verbose=True,
    allow_delegation=False,
)


# ── Crew factory ───────────────────────────────────────────────────────────────
def build_job_application_crew(
    candidate_background: str,
    target_role: str,
    target_company: str,
    job_description: str,
) -> Crew:
    resume_task = Task(
        description=(
            f"Optimize the resume for this application:\n\n"
            f"Candidate Background:\n{candidate_background}\n\n"
            f"Target Role: {target_role} at {target_company}\n\n"
            f"Job Description:\n{job_description}\n\n"
            "Tasks:\n"
            "1. Identify key skills and requirements from the job description\n"
            "2. Match candidate's experience to requirements\n"
            "3. Write optimized resume sections: Summary, Experience, Skills, Education\n"
            "4. Use action verbs and quantified achievements\n"
            "5. Include relevant ATS keywords from the job description\n"
            "6. Format for maximum impact and readability"
        ),
        agent=resume_analyzer,
        expected_output=(
            "Complete optimized resume with: Professional Summary, Work Experience "
            "(with bullet points and metrics), Skills section, and Education. "
            "Include ATS keyword analysis."
        ),
    )

    cover_letter_task = Task(
        description=(
            f"Write a personalized cover letter for {target_role} at {target_company}.\n\n"
            f"Job Description:\n{job_description}\n\n"
            "The cover letter should:\n"
            "- Open with a compelling hook (not 'I am applying for...')\n"
            "- Demonstrate knowledge of the company and role\n"
            "- Connect 2-3 specific candidate achievements to job requirements\n"
            "- Show genuine enthusiasm and cultural fit\n"
            "- Include a strong call-to-action closing\n"
            "- Be 3-4 paragraphs, 300-400 words\n"
            "- Feel authentic and conversational, not generic"
        ),
        agent=cover_letter_writer,
        expected_output=(
            "Complete cover letter (300-400 words) with compelling opening, "
            "experience connection, company knowledge, and strong closing."
        ),
        context=[resume_task],
    )

    interview_task = Task(
        description=(
            f"Prepare interview materials for {target_role} at {target_company}.\n\n"
            f"Job Description:\n{job_description}\n\n"
            "Provide:\n"
            "1. Top 10 likely interview questions (mix of behavioral, technical, situational)\n"
            "2. STAR-method answers for 5 behavioral questions based on candidate background\n"
            "3. 5 thoughtful questions the candidate should ask the interviewer\n"
            "4. Key talking points to emphasize\n"
            "5. Salary negotiation tips for this role\n"
            "6. Day-of interview tips and confidence boosters"
        ),
        agent=interview_coach,
        expected_output=(
            "Comprehensive interview prep guide with: 10 expected questions, "
            "5 STAR answers, 5 questions to ask, key talking points, and practical tips."
        ),
        context=[resume_task],
    )

    return Crew(
        agents=[resume_analyzer, cover_letter_writer, interview_coach],
        tasks=[resume_task, cover_letter_task, interview_task],
        process=Process.sequential,
        verbose=True,
    )


def prepare_application(
    candidate_background: str,
    target_role: str,
    target_company: str,
    job_description: str,
) -> str:
    crew = build_job_application_crew(
        candidate_background, target_role, target_company, job_description
    )
    result = crew.kickoff()
    return str(result)


SAMPLE_BACKGROUND = """
Name: Jordan Lee
Experience: 5 years as a software engineer
- 3 years at StartupXYZ: Built React/Node.js web applications, led team of 3 developers,
  improved app performance by 40%, reduced load times by 60%
- 2 years at ConsultingCo: Full-stack development, Python/Django backends,
  delivered 8 client projects on time and budget

Skills: Python, JavaScript/TypeScript, React, Node.js, AWS, Docker, PostgreSQL,
Redis, REST APIs, GraphQL, Git, Agile/Scrum

Education: B.S. Computer Science, State University, 2019

Achievements:
- Led migration of monolithic app to microservices, reducing downtime by 90%
- Built internal tooling that saved team 10 hours/week
- Mentored 2 junior developers
"""

SAMPLE_JOB_DESCRIPTION = """
Senior Software Engineer - Platform Team at TechGiant Inc.

We're looking for a Senior Software Engineer to join our Platform team building
the infrastructure and tools that power our products used by 50M+ users.

Requirements:
- 5+ years of software engineering experience
- Strong Python and JavaScript/TypeScript skills
- Experience with cloud platforms (AWS preferred)
- Knowledge of distributed systems and microservices
- Experience leading technical projects
- Strong communication skills

Nice to have:
- Experience with Kubernetes and containerization
- GraphQL API design
- Open source contributions

About TechGiant: We're building the future of productivity software, trusted by
millions of businesses worldwide. Culture: innovative, collaborative, fast-paced.
"""


def main():
    print("💼 Job Application Agent\n" + "=" * 60)
    print("Preparing complete job application package...\n")
    result = prepare_application(
        candidate_background=SAMPLE_BACKGROUND,
        target_role="Senior Software Engineer",
        target_company="TechGiant Inc.",
        job_description=SAMPLE_JOB_DESCRIPTION,
    )
    print("\n" + "=" * 60)
    print("📋 Your Job Application Package:")
    print("=" * 60)
    print(result)
    print("\n✅ Job application preparation complete!")


if __name__ == "__main__":
    main()
