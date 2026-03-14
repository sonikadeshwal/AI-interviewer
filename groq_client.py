"""
groq_client.py — Groq AI client for VoiceCoach AI
Model: llama-3.1-8b-instant (fast free tier)
"""

import re
import json
from groq import Groq


class GroqInterviewer:
    MODEL = "llama-3.1-8b-instant"

    def __init__(self, api_key: str):
        self.client = Groq(api_key=api_key)

    def _chat(self, system: str, user: str, max_tokens: int = 800, temperature: float = 0.5) -> str:
        resp = self.client.chat.completions.create(
            model=self.MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return resp.choices[0].message.content.strip()

    # ── Generate Questions ───────────────────────────────────────────────────
    def generate_questions(self, job_role, interview_type, difficulty, num_questions, resume_text=""):
        resume_ctx = f"\nCandidate background:\n{resume_text[:1500]}" if resume_text.strip() else ""

        difficulty_map = {
            "Beginner": "entry-level, foundational, simple scenarios",
            "Intermediate": "mid-level, practical application, moderate complexity",
            "Advanced": "senior-level, complex scenarios, leadership",
            "Expert": "principal/architect level, extremely challenging",
        }
        type_map = {
            "Technical":  "technical skills, coding, system design, tools, algorithms",
            "Behavioral": "STAR-method situations, past experiences, soft skills, leadership",
            "HR":         "culture fit, motivation, career goals, work style",
            "Mixed":      "blend of technical, behavioral, and situational questions",
        }

        system = f"""You are a senior interviewer at a top company.
Generate exactly {num_questions} interview questions for a {job_role} position.
Type: {interview_type} — {type_map.get(interview_type, 'mixed')}
Difficulty: {difficulty} — {difficulty_map.get(difficulty, 'intermediate')}
{resume_ctx}

Rules:
- Output ONLY a numbered list: 1. Question
- No explanations, no sub-bullets, just the questions
- Each question must be realistic, specific, and professionally worded
- Vary topics across questions"""

        raw = self._chat(system, f"Generate {num_questions} questions.", max_tokens=1200, temperature=0.8)

        questions = []
        for line in raw.split("\n"):
            line = line.strip()
            if not line:
                continue
            cleaned = re.sub(r"^\d+[\.\)]\s*", "", line).strip()
            if cleaned and len(cleaned) > 15:
                questions.append(cleaned)

        fallback = [
            f"Tell me about your experience as a {job_role}.",
            "Describe a challenging project and how you handled it.",
            "How do you manage competing priorities under pressure?",
            "What are your core technical strengths?",
            "Where do you see yourself professionally in 3 years?",
        ]
        while len(questions) < num_questions and fallback:
            questions.append(fallback.pop(0))

        return questions[:num_questions]

    # ── Evaluate Answer ──────────────────────────────────────────────────────
    def evaluate_answer(self, question, answer, job_role, interview_type, difficulty):
        system = """You are a senior hiring manager. Evaluate this interview answer.

Respond in EXACTLY this format (no deviations):
SCORE: [0-100]
STRENGTHS: [2-3 specific things done well, referencing the actual answer]
IMPROVEMENTS: [2-3 specific actionable improvements]
OVERALL: [2-3 sentence coaching summary]

Be specific, honest, and encouraging."""

        user = f"""Role: {job_role} | Type: {interview_type} | Difficulty: {difficulty}
Question: {question}
Answer: {answer}"""

        raw = self._chat(system, user, max_tokens=600, temperature=0.4)

        # Extract score
        score = 60
        m = re.search(r"SCORE:\s*(\d+)", raw, re.IGNORECASE)
        if m:
            score = max(0, min(100, int(m.group(1))))

        # Format feedback
        fb = re.sub(r"SCORE:\s*\d+\s*\n?", "", raw).strip()
        fb = fb.replace("STRENGTHS:", "✅ **Strengths:**")
        fb = fb.replace("IMPROVEMENTS:", "📈 **Areas to Improve:**")
        fb = fb.replace("OVERALL:", "🎯 **Overall:**")

        return fb, score

    # ── Analyze Language: Pronunciation + Communication ──────────────────────
    def analyze_language(self, answer: str, job_role: str) -> tuple[dict, dict]:
        """
        Returns:
          pronunciation_report: dict with corrections, good_phrases, filler_words, language_score, clarity, overall_tip
          communication_scores: dict with clarity, confidence, vocabulary, structure, conciseness, professionalism (0-100 each)
        """
        if not answer.strip() or answer == "[Skipped]":
            return {}, {}

        system = """You are an expert English language coach and communication trainer.
Analyze the given interview answer for language quality, pronunciation patterns (based on word choice/spelling), and communication skills.

Respond ONLY with a valid JSON object in this EXACT structure (no markdown, no extra text):
{
  "pronunciation": {
    "language_score": <0-100>,
    "clarity": "<Excellent|Good|Average|Poor>",
    "corrections": [
      {"word": "<misused/unclear word>", "correct": "<better word/phrase>", "phonetic": "<pronunciation hint>", "tip": "<why this is better>"}
    ],
    "good_phrases": ["<strong phrase used>", "..."],
    "filler_words": ["<filler detected>", "..."],
    "overall_tip": "<one actionable tip for pronunciation and clarity>"
  },
  "communication": {
    "clarity": <0-100>,
    "confidence": <0-100>,
    "vocabulary": <0-100>,
    "structure": <0-100>,
    "conciseness": <0-100>,
    "professionalism": <0-100>
  }
}

Focus on:
- Filler words: um, uh, like, you know, basically, literally, sort of, kind of
- Weak language: "I think maybe", "I guess", "I'm not sure but" → suggest confident alternatives
- Grammar errors or unclear phrasing
- Vocabulary richness for a professional context
- Keep corrections list to max 5 items
- Keep good_phrases to max 4 items
- Keep filler_words to max 5 unique items"""

        user = f"Job Role: {job_role}\n\nAnswer to analyze:\n{answer[:2000]}"

        try:
            raw = self._chat(system, user, max_tokens=900, temperature=0.3)

            # Strip markdown fences if present
            raw = re.sub(r"```json\s*", "", raw)
            raw = re.sub(r"```\s*", "", raw)
            raw = raw.strip()

            # Find JSON block
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start >= 0 and end > start:
                raw = raw[start:end]

            data = json.loads(raw)
            pron = data.get("pronunciation", {})
            comm = data.get("communication", {})
            return pron, comm

        except Exception:
            # Graceful fallback
            pron_fallback = {
                "language_score": 65,
                "clarity": "Good",
                "corrections": [],
                "good_phrases": [],
                "filler_words": [],
                "overall_tip": "Focus on using confident, specific language and avoid filler words."
            }
            comm_fallback = {
                "clarity": 65, "confidence": 60, "vocabulary": 65,
                "structure": 60, "conciseness": 65, "professionalism": 70
            }
            return pron_fallback, comm_fallback

    # ── Follow-up Question ───────────────────────────────────────────────────
    def generate_follow_up(self, question: str, answer: str) -> str:
        try:
            system = "You are an interviewer. Generate ONE sharp follow-up question (max 20 words) based on the candidate's answer. Return ONLY the question."
            user = f"Q: {question}\nA: {answer[:400]}\nFollow-up:"
            return self._chat(system, user, max_tokens=60, temperature=0.6)
        except Exception:
            return ""

    # ── Hint ────────────────────────────────────────────────────────────────
    def get_hint(self, question: str, job_role: str) -> str:
        system = "You are a career coach. Give a brief, helpful hint (2–3 sentences max) on how to approach this interview question. Don't give the full answer, just direction and structure."
        user = f"Role: {job_role}\nQuestion: {question}"
        return self._chat(system, user, max_tokens=140, temperature=0.5)

    # ── Final Summary ────────────────────────────────────────────────────────
    def generate_summary(self, questions, answers, scores, job_role, interview_type):
        avg = sum(scores) / len(scores) if scores else 0
        qa = "\n".join([
            f"Q{i+1} [Score:{s}/100]: {q}\nA: {a[:250]}"
            for i, (q, a, s) in enumerate(zip(questions, answers, scores))
        ])

        system = """You are a senior hiring manager writing a post-interview performance review.
Write a detailed, personalized summary with these sections:
1. Overall Assessment
2. Key Strengths
3. Primary Areas for Improvement
4. Top 3 Actionable Recommendations for the next 30 days
5. Hiring Recommendation (Strong Yes / Yes / Maybe / No — with reason)

Be specific, reference actual answers, and be constructive. Plain text with section headers."""

        user = f"Role: {job_role} | Type: {interview_type} | Avg Score: {avg:.0f}/100\n\n{qa}"
        return self._chat(system, user, max_tokens=1000, temperature=0.5)
