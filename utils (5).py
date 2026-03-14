"""utils.py — Constants and helpers for VoiceCoach AI"""

JOB_ROLES = [
    "Software Engineer", "Frontend Developer", "Backend Developer",
    "Full Stack Developer", "Mobile Developer (iOS/Android)", "DevOps / SRE Engineer",
    "Cloud Architect", "Data Engineer", "Machine Learning Engineer", "AI/LLM Engineer",
    "Embedded Systems Engineer", "QA / Test Engineer", "Security Engineer",
    "Blockchain Developer", "Data Scientist", "Data Analyst",
    "Business Intelligence Analyst", "Research Scientist", "Product Manager",
    "Product Designer / UX Designer", "UI Designer", "UX Researcher",
    "Technical Program Manager", "Engineering Manager", "VP of Engineering",
    "CTO", "Head of Product", "Marketing Manager", "Growth Hacker",
    "Sales Engineer", "Account Executive", "Customer Success Manager",
    "Operations Manager", "Business Analyst", "Financial Analyst",
    "HR Manager", "Recruiter", "Quantitative Analyst", "Investment Banker",
    "Financial Planner", "Biomedical Engineer", "Clinical Data Scientist",
    "Healthcare IT Specialist", "IT Support Specialist", "Systems Administrator",
    "Network Engineer", "Prompt Engineer", "MLOps Engineer",
    "Solutions Architect", "Technical Writer",
]

INTERVIEW_TYPES  = ["Technical", "Behavioral", "HR", "Mixed"]
DIFFICULTY_LEVELS = ["Beginner", "Intermediate", "Advanced", "Expert"]


def get_performance_badge(score: float) -> tuple[str, str, str]:
    if score >= 90: return "🏆", "#34d399", "Outstanding Performance"
    if score >= 80: return "⭐", "#34d399", "Excellent Performance"
    if score >= 70: return "✅", "#6ee7f7", "Good Performance"
    if score >= 60: return "📈", "#a78bfa", "Above Average"
    if score >= 50: return "📊", "#fbbf24", "Average Performance"
    if score >= 40: return "📌", "#fbbf24", "Below Average"
    return "🔄", "#f472b6", "Needs Practice"


def calculate_score(scores: list) -> float:
    if not scores: return 0.0
    weights = [1 + i * 0.05 for i in range(len(scores))]
    return sum(s * w for s, w in zip(scores, weights)) / sum(weights)
