
"""
routes.py — REST API endpoints for AI Interview Coach.

The React frontend communicates with these endpoints through HTTP.
FastAPI validates requests using Pydantic schemas.

This module connects:
- Question bank
- SQLite persistence
- Local LangChain + LangGraph evaluation
- Interview progress and history

The LLM runs locally through Ollama. Only optional search queries
are sent to the external search provider.
"""

import json
import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from app.config import APPROVED_TOPICS, QUESTION_BANK_PATH
from app.database import (
    complete_interview,
    create_interview,
    get_dashboard_stats,
    get_interview_details,
    get_interview_history,
    save_answer,
)
from app.schemas import (
    DashboardStats,
    InterviewHistoryResponse,
    InterviewSummary,
    MessageResponse,
    StartInterviewRequest,
    StartInterviewResponse,
    SubmitAnswerRequest,
    SubmitAnswerResponse,
    EvaluationResponse,
)
from app.services.interview_agent import execute_interview_turn

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["Interview Coach"])


# ---------------------------------------------------------
# Question-bank helpers
# ---------------------------------------------------------

def load_question_bank() -> list[dict[str, Any]]:
    """Load the locally stored question-only JSON bank."""
    if not QUESTION_BANK_PATH.exists():
        raise HTTPException(
            status_code=500,
            detail=(
                "Question bank not found. Add "
                "backend/data/raw/question_bank.json first."
            ),
        )

    try:
        raw_data = json.loads(QUESTION_BANK_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.exception("Could not read question bank.")
        raise HTTPException(
            status_code=500,
            detail="Question bank could not be read as valid JSON.",
        ) from exc

    # Accept either a JSON list or {"questions": [...]}.
    if isinstance(raw_data, dict):
        raw_data = raw_data.get("questions", [])

    if not isinstance(raw_data, list):
        raise HTTPException(
            status_code=500,
            detail="Question bank must contain a list of questions.",
        )

    valid_questions = []

    for index, item in enumerate(raw_data):
        if not isinstance(item, dict):
            continue

        question = item.get("question")
        topic = item.get("topic")
        difficulty = item.get("difficulty", "Medium")
        question_id = item.get("question_id", f"q{index + 1}")

        if not isinstance(question, str) or not question.strip():
            continue

        if not isinstance(topic, str) or not topic.strip():
            continue

        if difficulty not in ("Easy", "Medium", "Hard"):
            difficulty = "Medium"

        valid_questions.append(
            {
                "question_id": str(question_id),
                "topic": topic.strip(),
                "difficulty": difficulty,
                "question": question.strip(),
                "question_type": item.get("question_type", "conceptual"),
            }
        )

    if not valid_questions:
        raise HTTPException(
            status_code=500,
            detail="Question bank contains no valid questions.",
        )

    return valid_questions


def public_question(item: dict[str, Any]) -> dict[str, str]:
    """Expose only fields required by the interview frontend."""
    return {
        "question_id": str(item["question_id"]),
        "topic": str(item["topic"]),
        "difficulty": str(item["difficulty"]),
        "question": str(item["question"]),
        "question_type": str(item.get("question_type", "conceptual")),
    }


# ---------------------------------------------------------
# In-memory active-session state
# ---------------------------------------------------------
#
# SQLite persists interview records and answers.
# This dictionary tracks which questions belong to an active session.
#
# It is suitable for a single local development server. For a
# multi-worker or production deployment, move session progress into
# persistent storage (or another shared session store).

active_sessions: dict[int, dict[str, Any]] = {}


# ---------------------------------------------------------
# Health and configuration
# ---------------------------------------------------------

@router.get("/health")
def health_check() -> dict[str, str]:
    """Basic API health endpoint; it does not call the LLM."""
    return {"status": "ok", "service": "AI Interview Coach API"}


@router.get("/topics")
def get_topics() -> dict[str, list[str]]:
    """Return the 16 canonical approved topics in the question bank."""
    return {"topics": list(APPROVED_TOPICS)}


@router.get("/question-counts")
def get_question_counts() -> dict[str, Any]:
    """Return available counts for each of the 16 approved topics and difficulties."""
    questions = load_question_bank()
    counts: dict[str, dict[str, int]] = {
        topic: {"Easy": 0, "Medium": 0, "Hard": 0} for topic in APPROVED_TOPICS
    }

    for item in questions:
        topic = item["topic"]
        difficulty = item["difficulty"]
        if topic in counts and difficulty in counts[topic]:
            counts[topic][difficulty] += 1

    return {"counts": counts}


@router.get("/questions")
def get_questions(
    topic: str | None = Query(default=None),
    difficulty: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> dict[str, Any]:
    """Return questions matching filters, with pagination."""
    questions = load_question_bank()

    filtered = questions
    if topic:
        filtered = [q for q in filtered if q["topic"].casefold() == topic.casefold()]
    if difficulty:
        filtered = [q for q in filtered if q["difficulty"].casefold() == difficulty.casefold()]

    total = len(filtered)
    paged = filtered[offset : offset + limit]

    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "questions": paged,
    }


# ---------------------------------------------------------
# Dashboard
# ---------------------------------------------------------

@router.get("/dashboard/stats", response_model=DashboardStats)
def dashboard_stats() -> dict[str, Any]:
    """Return real statistics from SQLite."""
    return get_dashboard_stats()


@router.get("/interviews", response_model=InterviewHistoryResponse)
def list_interviews(
    limit: int = Query(default=50, ge=1, le=200),
) -> dict[str, Any]:
    """Return recent saved interview sessions."""
    rows = get_interview_history(limit=limit)

    interviews = [
        InterviewSummary(
            id=row["id"],
            topic=row["topic"],
            difficulty=row["difficulty"],
            started_at=row["started_at"],
            completed_at=row.get("completed_at"),
            total_questions=row["total_questions"],
            average_score=row["average_score"],
            status=row["status"],
        )
        for row in rows
    ]

    return {"interviews": interviews}


@router.get("/interviews/{interview_id}")
def interview_details(interview_id: int) -> dict[str, Any]:
    """Return a saved interview and its evaluated answers."""
    details = get_interview_details(interview_id)

    if details is None:
        raise HTTPException(status_code=404, detail="Interview not found.")

    return details


# ---------------------------------------------------------
# Start interview
# ---------------------------------------------------------

@router.post("/interviews/start", response_model=StartInterviewResponse)
def start_interview(
    request: StartInterviewRequest,
) -> dict[str, Any]:
    """
    Create an interview using all matching questions.

    The candidate does not select a question count.
    If the chosen difficulty has no questions for the selected topic,
    all questions for that topic are used as a fallback.
    """
    question_bank = load_question_bank()

    topic_questions = [
        item for item in question_bank
        if item["topic"].casefold() == request.topic.casefold()
    ]

    if not topic_questions:
        raise HTTPException(
            status_code=404,
            detail=f"No questions found for topic '{request.topic}'.",
        )

    matching = [
        item for item in topic_questions
        if item["difficulty"].casefold() == request.difficulty.casefold()
    ]

    chosen_questions = matching if matching else topic_questions

    # Randomize question order for varied practice sessions.
    import random
    random.shuffle(chosen_questions)

    session_id = create_interview(
        topic=request.topic,
        difficulty=request.difficulty,
        total_questions=len(chosen_questions),
    )

    active_sessions[session_id] = {
        "topic": request.topic,
        "difficulty": request.difficulty,
        "use_web_search": request.use_web_search,
        "questions": chosen_questions,
        "current_index": 0,
        "answered_question_ids": set(),
        "pending_clarification": {},
    }

    return {
        "session_id": session_id,
        "topic": request.topic,
        "difficulty": request.difficulty,
        "total_questions": len(chosen_questions),
        "first_question": public_question(chosen_questions[0]),
    }


# ---------------------------------------------------------
# Submit answer and return the next question
# ---------------------------------------------------------

@router.post(
    "/interviews/answer",
    response_model=SubmitAnswerResponse,
)
def submit_answer(
    request: SubmitAnswerRequest,
) -> dict[str, Any]:
    """Evaluate one answer, save it, and provide interview progress."""
    session_id = request.session_id
    session = active_sessions.get(session_id)

    if session is None:
        raise HTTPException(
            status_code=404,
            detail=(
                "Active interview not found. The session may have been "
                "completed or the API server restarted."
            ),
        )

    if request.question_id in session["answered_question_ids"]:
        raise HTTPException(
            status_code=409,
            detail="This question has already been evaluated.",
        )

    questions = session["questions"]
    index = session["current_index"]

    if index >= len(questions):
        raise HTTPException(
            status_code=409,
            detail="All questions in this interview have been answered.",
        )

    current_question = questions[index]

    if str(current_question["question_id"]) != request.question_id:
        raise HTTPException(
            status_code=409,
            detail="The submitted question does not match the current question.",
        )

    if not request.answer.strip():
        raise HTTPException(
            status_code=422,
            detail="Please enter an answer before submitting.",
        )

    pending_clarif = session.get("pending_clarification", {}).get(request.question_id)
    is_clarif = request.is_clarification_response or (pending_clarif is not None)
    initial_answer = pending_clarif["initial_answer"] if is_clarif and pending_clarif else request.answer.strip()
    follow_up_answer = request.answer.strip() if is_clarif and pending_clarif else None

    try:
        turn_result = execute_interview_turn(
            question=current_question["question"],
            candidate_answer=initial_answer,
            topic=session["topic"],
            search_enabled=session["use_web_search"],
            is_clarification_response=is_clarif,
            combined_follow_up=follow_up_answer,
        )
    except Exception as exc:
        logger.exception("Interview turn execution failed.")
        raise HTTPException(
            status_code=502,
            detail=f"Interview turn processing failed: {exc}",
        ) from exc

    action = turn_result["action"]
    decider = turn_result["decider"]
    total_questions = len(questions)

    # 1. REDIRECT: off-topic or evasive answer
    if action == "REDIRECT":
        return {
            "session_id": session_id,
            "question_id": request.question_id,
            "action": "REDIRECT",
            "decider": decider,
            "evaluation": None,
            "clarification_question": None,
            "clarification_prompt": None,
            "redirect_message": turn_result["redirect_message"],
            "evaluated_count": session["current_index"],
            "total_questions": total_questions,
            "next_question": None,
            "interview_completed": False,
        }

    # 2. ASK_CLARIFICATION: partial / ambiguous answer needing one follow-up
    if action == "ASK_CLARIFICATION":
        session.setdefault("pending_clarification", {})[request.question_id] = {
            "initial_answer": request.answer.strip(),
            "clarification_question": turn_result["clarification_question"],
        }
        return {
            "session_id": session_id,
            "question_id": request.question_id,
            "action": "ASK_CLARIFICATION",
            "decider": decider,
            "evaluation": None,
            "clarification_question": turn_result["clarification_question"],
            "clarification_prompt": turn_result["clarification_prompt"],
            "redirect_message": None,
            "evaluated_count": session["current_index"],
            "total_questions": total_questions,
            "next_question": None,
            "interview_completed": False,
        }

    # 3. EVALUATE: substantive answer evaluated and scored
    evaluation = turn_result["evaluation"]

    # Combine answers if clarification was submitted
    final_saved_answer = request.answer.strip()
    if is_clarif and pending_clarif:
        final_saved_answer = f"{pending_clarif['initial_answer']}\n[Clarification]: {request.answer.strip()}"
        session["pending_clarification"].pop(request.question_id, None)

    # Persist evaluated answer to SQLite
    save_answer(
        interview_id=session_id,
        question_id=str(current_question["question_id"]),
        question=current_question["question"],
        candidate_answer=final_saved_answer,
        evaluation=evaluation,
    )

    session["answered_question_ids"].add(request.question_id)
    session["current_index"] += 1

    evaluated_count = session["current_index"]
    interview_completed = evaluated_count >= total_questions

    next_question = None
    if interview_completed:
        complete_interview(session_id)
    else:
        next_question = public_question(questions[session["current_index"]])

    return {
        "session_id": session_id,
        "question_id": request.question_id,
        "action": "EVALUATE",
        "decider": decider,
        "evaluation": evaluation,
        "clarification_question": None,
        "clarification_prompt": None,
        "redirect_message": None,
        "evaluated_count": evaluated_count,
        "total_questions": total_questions,
        "next_question": next_question,
        "interview_completed": interview_completed,
    }


# ---------------------------------------------------------
# Finish interview early
# ---------------------------------------------------------

@router.post(
    "/interviews/{interview_id}/finish",
    response_model=MessageResponse,
)
def finish_interview(interview_id: int) -> dict[str, str]:
    """Allow the candidate to finish early and save the partial report."""
    session = active_sessions.get(interview_id)

    if session is None:
        details = get_interview_details(interview_id)

        if details is None:
            raise HTTPException(status_code=404, detail="Interview not found.")

        if details["interview"]["status"] == "completed":
            return {"message": "Interview was already completed."}

        raise HTTPException(
            status_code=404,
            detail="Active interview session not found.",
        )

    complete_interview(interview_id)
    active_sessions.pop(interview_id, None)

    return {"message": "Interview completed and saved."}
