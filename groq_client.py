"""
groq_client.py — Groq AI interface for InterviewAI
Uses llama-3.3-70b-versatile (free tier) via Groq API
"""

import re
import json
import random
from groq import Groq


class GroqInterviewer:
    """Handles all AI interactions with Groq API."""

    MODEL = "llama-3.3-70b-versatile"

    def __init__(self, api_key: str):
        self.client = Groq(api_key=api_key)

    # ── Internal helper ──────────────────────────────────────────────────────
    def _chat(self, system: str, user: str, max_tokens: int = 1024, temperature: float = 0.7) -> str:
        resp = self.client.chat.completions.create(
            model=self.MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return resp.choices[0].message.content.strip()

    # ── Generate Questions ───────────────────────────────────────────────────
    def generate_questions(
        self,
        job_role: str,
        interview_type: str,
        difficulty: str,
        num_questions: int,
        resume_text: str = "",
    ) -> list[str]:
        """Generate tailored interview questions."""

        resume_section = ""
        if resume_text.strip():
            resume_section = f"""
The candidate has provided their resume/background:
---
{resume_text[:2000]}
---
Use this to make questions more personalized and relevant.
"""

        difficulty_guide = {
            "Beginner": "entry-level, foundational concepts, simple scenarios",
            "Intermediate": "mid-level, practical application, moderate complexity",
            "Advanced": "senior-level, complex scenarios, leadership, system design",
            "Expert": "architect/principal level, extremely challenging, deep expertise",
        }.get(difficulty, "intermediate")

        type_guide = {
            "Technical": "focus on technical skills, coding concepts, system design, tools, frameworks",
            "Behavioral": "use STAR-method situations, past experiences, soft skills, teamwork, leadership",
            "HR": "focus on culture fit, motivation, career goals, salary, work style, values",
            "Mixed": "blend of technical, behavioral, and situational questions",
        }.get(interview_type, "mixed")

        system = f"""You are an expert technical interviewer at a top-tier tech company.
Generate exactly {num_questions} interview questions for a {job_role} position.
Interview Type: {interview_type} — {type_guide}
Difficulty: {difficulty} — {difficulty_guide}

Rules:
- Each question on its own line, numbered: 1. Question here
- No sub-bullets, no explanations, just the questions
- Questions should be realistic, challenging, and professionally phrased
- Vary the topics and question styles
- Do not include answers
{resume_section}"""

        user = f"Generate {num_questions} {difficulty} {interview_type} interview questions for a {job_role} role."

        raw = self._chat(system, user, max_tokens=1500, temperature=0.8)

        # Parse numbered list
        questions = []
        for line in raw.split("\n"):
            line = line.strip()
            if not line:
                continue
            # Remove leading number/bullet
            cleaned = re.sub(r"^\d+[\.\)]\s*", "", line).strip()
            if cleaned and len(cleaned) > 15:
                questions.append(cleaned)

        # Fallback
        if len(questions) < num_questions:
            fallback = [
                f"Tell me about your experience as a {job_role}.",
                "Describe a challenging project you worked on.",
                "How do you handle tight deadlines and multiple priorities?",
                "What are your greatest technical strengths?",
                "Where do you see yourself in 5 years?",
            ]
            while len(questions) < num_questions and fallback:
                questions.append(fallback.pop(0))

        return questions[:num_questions]

    # ── Evaluate Answer ──────────────────────────────────────────────────────
    def evaluate_answer(
        self,
        question: str,
        answer: str,
        job_role: str,
        interview_type: str,
        difficulty: str,
    ) -> tuple[str, int]:
        """Evaluate a candidate's answer. Returns (feedback_text, score_0_100)."""

        system = """You are a senior hiring manager evaluating interview answers.
Provide structured, honest, actionable feedback.

Your response MUST follow this exact format:
SCORE: [number 0-100]
STRENGTHS: [2-3 specific things done well]
IMPROVEMENTS: [2-3 specific areas to improve]
OVERALL: [2-3 sentence summary with coaching advice]

Be specific, reference the actual answer content. Be encouraging but honest."""

        user = f"""Job Role: {job_role}
Interview Type: {interview_type}
Difficulty: {difficulty}

Question: {question}

Candidate's Answer: {answer}

Evaluate this answer following the exact format above."""

        raw = self._chat(system, user, max_tokens=800, temperature=0.4)

        # Extract score
        score = 60  # default
        score_match = re.search(r"SCORE:\s*(\d+)", raw, re.IGNORECASE)
        if score_match:
            score = max(0, min(100, int(score_match.group(1))))

        # Format feedback for display
        feedback = raw
        # Clean up the SCORE line from display
        feedback = re.sub(r"SCORE:\s*\d+\s*\n?", "", feedback).strip()

        # Add visual formatting
        feedback = feedback.replace("STRENGTHS:", "✅ **Strengths:**")
        feedback = feedback.replace("IMPROVEMENTS:", "📈 **Areas to Improve:**")
        feedback = feedback.replace("OVERALL:", "🎯 **Overall Assessment:**")

        return feedback, score

    # ── Generate Follow-up ───────────────────────────────────────────────────
    def generate_follow_up(self, question: str, answer: str) -> str:
        """Generate a relevant follow-up question based on the answer."""
        try:
            system = "You are an interviewer. Generate ONE short follow-up question (max 25 words) based on the candidate's answer. Return ONLY the question, nothing else."
            user = f"Original question: {question}\nCandidate answered: {answer[:500]}\nFollow-up question:"
            return self._chat(system, user, max_tokens=80, temperature=0.6)
        except Exception:
            return ""

    # ── Get Hint ────────────────────────────────────────────────────────────
    def get_hint(self, question: str, job_role: str) -> str:
        """Generate a subtle hint for answering the question."""
        system = "You are a career coach. Give a brief, helpful hint (2-3 sentences max) on how to approach answering this interview question. Don't give the answer, just direction."
        user = f"Role: {job_role}\nQuestion: {question}\nHint:"
        return self._chat(system, user, max_tokens=150, temperature=0.5)

    # ── Generate Summary ─────────────────────────────────────────────────────
    def generate_summary(
        self,
        questions: list[str],
        answers: list[str],
        scores: list[int],
        job_role: str,
        interview_type: str,
    ) -> str:
        """Generate a holistic performance summary."""

        qa_pairs = "\n".join([
            f"Q{i+1} [Score: {s}/100]: {q}\nA: {a[:300]}"
            for i, (q, a, s) in enumerate(zip(questions, answers, scores))
        ])

        avg = sum(scores) / len(scores) if scores else 0

        system = """You are a senior hiring manager writing a comprehensive performance review.
Write a detailed, personalized summary that includes:
1. Overall performance assessment
2. Key strengths demonstrated
3. Primary areas for improvement  
4. Specific actionable advice for the next 30 days
5. Hiring recommendation (Strong Yes / Yes / Maybe / No)

Be encouraging, specific, and professional. Use plain text with section headers."""

        user = f"""Role: {job_role} | Type: {interview_type} | Average Score: {avg:.0f}/100

Interview Q&A:
{qa_pairs}

Write a comprehensive performance summary."""

        return self._chat(system, user, max_tokens=1200, temperature=0.5)
