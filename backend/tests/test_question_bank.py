
"""Basic tests for the interview question bank."""

import json
from pathlib import Path

from app.config import QUESTION_BANK_PATH


def test_question_bank_file_exists():
    """The configured question bank file should exist."""
    assert QUESTION_BANK_PATH.exists()


def test_question_bank_contains_questions():
    """The JSON file should contain a non-empty questions list."""
    with QUESTION_BANK_PATH.open("r", encoding="utf-8") as file:
        data = json.load(file)

    assert isinstance(data, dict)
    assert isinstance(data.get("questions"), list)
    assert len(data["questions"]) > 0


def test_questions_have_required_fields():
    """Every question should have an ID, topic, difficulty and text."""
    with QUESTION_BANK_PATH.open("r", encoding="utf-8") as file:
        data = json.load(file)

    required_fields = {
        "question_id",
        "topic",
        "difficulty",
        "question",
    }

    for question in data["questions"]:
        assert required_fields.issubset(question.keys())
        assert question["question_id"]
        assert question["topic"]
        assert question["question"]
        assert question["difficulty"] in {"Easy", "Medium", "Hard"}


def test_question_ids_are_unique():
    """Question IDs must be unique for reliable answer tracking."""
    with QUESTION_BANK_PATH.open("r", encoding="utf-8") as file:
        data = json.load(file)

    question_ids = [
        question["question_id"]
        for question in data["questions"]
    ]

    assert len(question_ids) == len(set(question_ids))

