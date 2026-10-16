
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


def test_questions_belong_to_approved_topics():
    """Every question in the question bank must belong strictly to one of the 16 approved topics."""
    from app.config import APPROVED_TOPICS

    with QUESTION_BANK_PATH.open("r", encoding="utf-8") as file:
        data = json.load(file)

    assert len(APPROVED_TOPICS) == 16
    approved_set = set(APPROVED_TOPICS)

    found_topics = set()
    for question in data["questions"]:
        topic = question["topic"]
        assert topic in approved_set, f"Unexpected topic found: {topic}"
        found_topics.add(topic)

    # All approved topics must be represented
    assert found_topics == approved_set, f"Missing topics: {approved_set - found_topics}"


def test_each_approved_topic_has_valid_questions():
    """Verify each approved topic has questions and no empty/invalid records."""
    with QUESTION_BANK_PATH.open("r", encoding="utf-8") as file:
        data = json.load(file)

    for q in data["questions"]:
        assert len(q["question"].strip()) > 5
        assert q["difficulty"] in ["Easy", "Medium", "Hard"]


def test_api_topics_and_counts_match_16_approved_topics():
    """Verify API routes return strictly the 16 approved topics and accurate counts."""
    from fastapi.testclient import TestClient
    from app.main import app
    from app.config import APPROVED_TOPICS

    client = TestClient(app)

    # 1. /api/topics
    resp = client.get("/api/topics")
    assert resp.status_code == 200
    topics = resp.json().get("topics")
    assert topics == APPROVED_TOPICS
    assert len(topics) == 16

    # 2. /api/question-counts
    counts_resp = client.get("/api/question-counts")
    assert counts_resp.status_code == 200
    counts = counts_resp.json().get("counts")
    assert set(counts.keys()) == set(APPROVED_TOPICS)

    # 3. /api/questions for RAG
    rag_resp = client.get("/api/questions?topic=RAG&limit=500")
    assert rag_resp.status_code == 200
    rag_data = rag_resp.json()
    assert rag_data["total"] == 126
    for q in rag_data["questions"]:
        assert q["topic"] == "RAG"

    # 4. /api/questions for Computer Networks
    net_resp = client.get("/api/questions?topic=Computer%20Networks&limit=500")
    assert net_resp.status_code == 200
    net_data = net_resp.json()
    assert net_data["total"] == 100
    for q in net_data["questions"]:
        assert q["topic"] == "Computer Networks"

