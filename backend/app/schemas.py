"""
schemas.py — API request and response contracts.

FastAPI uses Pydantic models to validate incoming requests
and structure outgoing responses.

Updated for InterviewIQ with Strands Decider 2B (#policy) and Amazon Bedrock:
- DecisionMetadata: Decision, confidence, action, latency, fallback flag.
- Adaptive actions: EVALUATE, ASK_CLARIFICATION, REDIRECT.
- SubmitAnswerResponse: Carries decider metadata, optional clarification question,
  optional redirect guidance, and evaluation when evaluated.
"""

from typing import Any, Literal
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
# Answer submission & clarification
# ---------------------------------------------------------

class SubmitAnswerRequest(BaseModel):
    """Candidate's answer for the current interview question."""

    session_id: int = Field(gt=0)
    question_id: str = Field(min_length=1, max_length=100)
    answer: str = Field(min_length=1, max_length=20000)
    # Optional flag indicating this is a follow-up answer to a clarification question
    is_clarification_response: bool = False


# ---------------------------------------------------------
# Strands Decider 2B Decision Metadata
# ---------------------------------------------------------

class DecisionMetadata(BaseModel):
    """Metadata produced by Strands Decider 2B (#policy triage)."""

    model: str = "StrandsAgents/strands-decider-2B-hobson-v21"
    pattern: str = "#policy"
    decision: Literal["EVALUATE", "ASK_CLARIFICATION", "REDIRECT"]
    model_decision: Literal["EVALUATE", "ASK_CLARIFICATION", "REDIRECT"] | None = None
    final_decision: Literal["EVALUATE", "ASK_CLARIFICATION", "REDIRECT"]
    action_taken: str
    confidence: float = Field(ge=0.0, le=1.0)
    latency_ms: float = Field(ge=0.0)
    is_fallback: bool = False
    fallback_reason: str | None = None
    probabilities: dict[str, float] = Field(default_factory=dict)


# ---------------------------------------------------------
# Evaluation response
# ---------------------------------------------------------

class CriterionScore(BaseModel):
    """Score and explanation for one evaluation criterion."""

    score: int = Field(ge=0, le=5)
    reason: str


class EvaluationResponse(BaseModel):
    """Evaluation returned by Bedrock/LangGraph workflow."""

    technical_correctness: CriterionScore
    completeness: CriterionScore
    clarity: CriterionScore
    structure: CriterionScore

    strengths: list[str] = Field(default_factory=list)
    missing_points: list[str] = Field(default_factory=list)

    reference_answer: str = ""
    improved_answer: str = ""

    final_score: float = Field(ge=0, le=10)

    evaluator_model: str = "Bedrock/google.gemma-3-4b-it"
    evaluator_status: Literal["bedrock_evaluated", "local_fallback", "deterministic_fallback"] = "bedrock_evaluated"

    sources: list[dict[str, str]] = Field(default_factory=list)
    search_note: str = ""
    decider_metadata: DecisionMetadata | None = None


class SubmitAnswerResponse(BaseModel):
    """Adaptive interview response covering EVALUATE, ASK_CLARIFICATION, and REDIRECT."""

    session_id: int
    question_id: str
    action: Literal["EVALUATE", "ASK_CLARIFICATION", "REDIRECT"]
    decider: DecisionMetadata

    # Present when action == 'EVALUATE'
    evaluation: EvaluationResponse | None = None

    # Present when action == 'ASK_CLARIFICATION'
    clarification_question: str | None = None
    clarification_prompt: str | None = None

    # Present when action == 'REDIRECT'
    redirect_message: str | None = None

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
