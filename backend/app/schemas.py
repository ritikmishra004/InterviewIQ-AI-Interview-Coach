
"""
schemas.py — API request and response contracts.

FastAPI uses Pydantic models to validate incoming requests
and structure outgoing responses.

The React frontend and Python backend will use these schemas
as a consistent contract, preventing mismatched field names.
"""

from typing import Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------
# Interview configuration
# ---------------------------------------------------------

class StartInterviewRequest(BaseModel):
    """Settings selected by the candidate before an interview."""

    topic: str = Field(min_length=1, max_length=100)
    difficulty: Literal["Easy", "Medium", "Hard"] = "Easy"

    # Web search is optional; the application works without it.
    use_web_search: bool = False


class StartInterviewResponse(BaseModel):
    """Information returned after creating an interview session."""

    session_id: int
    topic: str
    difficulty: str
    total_questions: int
    first_question: dict


# ---------------------------------------------------------
# Answer submission
# ---------------------------------------------------------

class SubmitAnswerRequest(BaseModel):
    """Candidate's answer for the current interview question."""

    session_id: int = Field(gt=0)
    question_id: str = Field(min_length=1, max_length=100)
    answer: str = Field(min_length=1, max_length=20000)


# ---------------------------------------------------------
# Evaluation response
# ---------------------------------------------------------

class CriterionScore(BaseModel):
    """Score and explanation for one evaluation criterion."""

    score: int = Field(ge=0, le=5)
    reason: str


class EvaluationResponse(BaseModel):
    """Evaluation returned by the local LLM workflow."""

    technical_correctness: CriterionScore
    completeness: CriterionScore
    clarity: CriterionScore
    structure: CriterionScore

    strengths: list[str] = Field(default_factory=list)
    missing_points: list[str] = Field(default_factory=list)

    reference_answer: str = ""
    improved_answer: str = ""

    final_score: float = Field(ge=0, le=10)

    sources: list[dict[str, str]] = Field(default_factory=list)
    search_note: str = ""


class SubmitAnswerResponse(BaseModel):
    """Evaluation plus interview progress after an answer is submitted."""

    session_id: int
    question_id: str
    evaluation: EvaluationResponse
    evaluated_count: int
    total_questions: int
    next_question: dict | None = None
    interview_completed: bool = False


# ---------------------------------------------------------
# Dashboard and history
# ---------------------------------------------------------

class DashboardStats(BaseModel):
    """Actual statistics calculated from saved interview records."""

    total_interviews: int
    completed_interviews: int
    total_answers: int
    overall_average: float


class InterviewSummary(BaseModel):
    """Compact representation of a saved interview."""

    id: int
    topic: str
    difficulty: str
    started_at: str
    completed_at: str | None = None
    total_questions: int = 0
    average_score: float = 0
    status: str


class InterviewHistoryResponse(BaseModel):
    """List of interview sessions."""

    interviews: list[InterviewSummary]


# ---------------------------------------------------------
# Generic API message
# ---------------------------------------------------------

class MessageResponse(BaseModel):
    """Standard success message for simple API operations."""

    message: str
