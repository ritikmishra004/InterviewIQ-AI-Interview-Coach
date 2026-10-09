
"""
database.py — SQLite persistence for AI Interview Coach.

SQLite is included with Python, so no database server or paid
service is required.

This module stores:
- Interview sessions and their status.
- Questions asked during each session.
- Candidate answers and evaluation feedback.
- Scores, timestamps, and optional web sources.

All SQL values are passed as parameters to avoid SQL injection.
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from app.config import DATABASE_PATH


def get_connection() -> sqlite3.Connection:
    """Open a local SQLite connection with dictionary-like rows."""
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(str(DATABASE_PATH), timeout=15)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 15000")

    return connection


def initialize_database() -> None:
    """
    Create the database schema if it does not exist.

    CREATE TABLE IF NOT EXISTS makes this function safe to call
    each time the FastAPI application starts.
    """
    with get_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS interviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic TEXT NOT NULL,
                difficulty TEXT NOT NULL,
                started_at TEXT NOT NULL,
                completed_at TEXT,
                total_questions INTEGER NOT NULL DEFAULT 0,
                average_score REAL NOT NULL DEFAULT 0,
                status TEXT NOT NULL DEFAULT 'in_progress'
            )
            """
        )

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS interview_answers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                interview_id INTEGER NOT NULL,
                question_id TEXT NOT NULL,
                question TEXT NOT NULL,
                candidate_answer TEXT NOT NULL,
                technical_score INTEGER NOT NULL,
                completeness_score INTEGER NOT NULL,
                clarity_score INTEGER NOT NULL,
                structure_score INTEGER NOT NULL,
                final_score REAL NOT NULL,
                strengths TEXT NOT NULL DEFAULT '[]',
                missing_points TEXT NOT NULL DEFAULT '[]',
                reference_answer TEXT NOT NULL DEFAULT '',
                improved_answer TEXT NOT NULL DEFAULT '',
                sources TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL,
                UNIQUE(interview_id, question_id),
                FOREIGN KEY (interview_id)
                    REFERENCES interviews(id)
                    ON DELETE CASCADE
            )
            """
        )

        # Indexes make common history queries faster as data grows.
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_interviews_started
            ON interviews(started_at)
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_answers_interview
            ON interview_answers(interview_id)
            """
        )


def create_interview(
    topic: str,
    difficulty: str,
    total_questions: int,
) -> int:
    """Create an interview session and return its database ID."""
    started_at = datetime.now().astimezone().isoformat(timespec="seconds")

    with get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO interviews (
                topic,
                difficulty,
                started_at,
                total_questions,
                status
            )
            VALUES (?, ?, ?, ?, 'in_progress')
            """,
            (topic, difficulty, started_at, total_questions),
        )

        return int(cursor.lastrowid)


def save_answer(
    interview_id: int,
    question_id: str,
    question: str,
    candidate_answer: str,
    evaluation: dict[str, Any],
) -> None:
    """
    Save one successfully evaluated answer.

    The evaluation dictionary must contain the four scoring criteria
    and final_score, as returned by the interview evaluation service.
    """
    created_at = datetime.now().astimezone().isoformat(timespec="seconds")

    with get_connection() as connection:
        session = connection.execute(
            "SELECT status FROM interviews WHERE id = ?",
            (interview_id,),
        ).fetchone()

        if session is None:
            raise ValueError(f"Interview session {interview_id} does not exist.")

        if session["status"] != "in_progress":
            raise ValueError("Cannot save an answer to a completed interview.")

        connection.execute(
            """
            INSERT INTO interview_answers (
                interview_id,
                question_id,
                question,
                candidate_answer,
                technical_score,
                completeness_score,
                clarity_score,
                structure_score,
                final_score,
                strengths,
                missing_points,
                reference_answer,
                improved_answer,
                sources,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                interview_id,
                question_id,
                question,
                candidate_answer,
                evaluation["technical_correctness"]["score"],
                evaluation["completeness"]["score"],
                evaluation["clarity"]["score"],
                evaluation["structure"]["score"],
                evaluation["final_score"],
                json.dumps(
                    evaluation.get("strengths", []),
                    ensure_ascii=False,
                ),
                json.dumps(
                    evaluation.get("missing_points", []),
                    ensure_ascii=False,
                ),
                evaluation.get("reference_answer", ""),
                evaluation.get("improved_answer", ""),
                json.dumps(
                    evaluation.get("sources", []),
                    ensure_ascii=False,
                ),
                created_at,
            ),
        )


def complete_interview(interview_id: int) -> None:
    """
    Mark an interview complete and calculate its average score.

    Only answers that were actually evaluated and saved contribute
    to the average.
    """
    completed_at = datetime.now().astimezone().isoformat(timespec="seconds")

    with get_connection() as connection:
        session = connection.execute(
            "SELECT id FROM interviews WHERE id = ?",
            (interview_id,),
        ).fetchone()

        if session is None:
            raise ValueError(f"Interview session {interview_id} does not exist.")

        connection.execute(
            """
            UPDATE interviews
            SET completed_at = ?,
                status = 'completed',
                total_questions = (
                    SELECT COUNT(*)
                    FROM interview_answers
                    WHERE interview_id = ?
                ),
                average_score = COALESCE(
                    (
                        SELECT ROUND(AVG(final_score), 2)
                        FROM interview_answers
                        WHERE interview_id = ?
                    ),
                    0
                )
            WHERE id = ?
            """,
            (completed_at, interview_id, interview_id, interview_id),
        )


def get_dashboard_stats() -> dict[str, int | float]:
    """Return dashboard metrics calculated from real database records."""
    with get_connection() as connection:
        total_interviews = connection.execute(
            "SELECT COUNT(*) FROM interviews"
        ).fetchone()[0]

        completed_interviews = connection.execute(
            "SELECT COUNT(*) FROM interviews WHERE status = 'completed'"
        ).fetchone()[0]

        total_answers = connection.execute(
            "SELECT COUNT(*) FROM interview_answers"
        ).fetchone()[0]

        average_score = connection.execute(
            "SELECT COALESCE(AVG(final_score), 0) FROM interview_answers"
        ).fetchone()[0]

    return {
        "total_interviews": int(total_interviews),
        "completed_interviews": int(completed_interviews),
        "total_answers": int(total_answers),
        "overall_average": round(float(average_score), 2),
    }


def get_interview_history(limit: int = 100) -> list[dict[str, Any]]:
    """Return recent interview sessions, newest first."""
    if limit < 1:
        raise ValueError("History limit must be at least 1.")

    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT *
            FROM interviews
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    return [dict(row) for row in rows]


def get_interview_details(interview_id: int) -> dict[str, Any] | None:
    """Return a session and all its saved answers, or None if not found."""
    with get_connection() as connection:
        session = connection.execute(
            "SELECT * FROM interviews WHERE id = ?",
            (interview_id,),
        ).fetchone()

        if session is None:
            return None

        answers = connection.execute(
            """
            SELECT *
            FROM interview_answers
            WHERE interview_id = ?
            ORDER BY id
            """,
            (interview_id,),
        ).fetchall()

    # Convert JSON text fields back into Python lists for the API.
    parsed_answers = []

    for row in answers:
        answer = dict(row)

        for field in ("strengths", "missing_points", "sources"):
            try:
                answer[field] = json.loads(answer.get(field) or "[]")
            except json.JSONDecodeError:
                answer[field] = []

        parsed_answers.append(answer)

    return {
        "interview": dict(session),
        "answers": parsed_answers,
    }
