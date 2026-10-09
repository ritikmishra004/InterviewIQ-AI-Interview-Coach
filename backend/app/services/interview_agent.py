
"""
interview_agent.py — Local LLM interview evaluation workflow.

Open-weight model:
    Phi-3 through Ollama, running locally on the user's machine.

Libraries:
    LangChain     -> builds prompts and calls the local chat model.
    LangGraph     -> orchestrates evidence retrieval, evaluation,
                     and deterministic score calculation.
    Pydantic      -> validates model-generated evaluation data.

Important:
    The LLM evaluates answer quality, but Python calculates the final
    weighted score. This avoids relying on the LLM for arithmetic.

    Web snippets are optional supporting evidence, not ground truth.
"""

import json
import logging
import re
from typing import Any, TypedDict

from pydantic import BaseModel, Field, ValidationError

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph

from app.config import LLM_TEMPERATURE, OLLAMA_MODEL, SCORING_WEIGHTS
from app.services.search_service import search_question

logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# 1. Structured output schema
# ---------------------------------------------------------

class CriterionEvaluation(BaseModel):
    """One criterion's score and explanation."""

    score: int = Field(ge=0, le=5)
    reason: str


class ModelEvaluation(BaseModel):
    """Expected evaluation structure returned by the local LLM."""

    technical_correctness: CriterionEvaluation
    completeness: CriterionEvaluation
    clarity: CriterionEvaluation
    structure: CriterionEvaluation

    strengths: list[str] = Field(default_factory=list)
    missing_points: list[str] = Field(default_factory=list)

    reference_answer: str
    improved_answer: str


# ---------------------------------------------------------
# 2. Local open-weight LLM via Ollama
# ---------------------------------------------------------

# ChatOllama uses Ollama's local HTTP service.
# Ensure Ollama is installed and `phi3:latest` is available.
# No cloud LLM API key is used.
llm = ChatOllama(
    model=OLLAMA_MODEL,
    temperature=LLM_TEMPERATURE,
    format="json",
)


# ---------------------------------------------------------
# 3. Evaluation prompt
# ---------------------------------------------------------

evaluation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a fair technical interview evaluator and personal coach.

Evaluate the candidate's answer against the exact question asked.

FAIRNESS RULES:
1. Judge only what the question asks.
2. Do not penalize a short answer merely because it omits extra details.
3. Do not require causes, examples, advantages, disadvantages, or
   solutions unless the question asks for them or they are essential
   to the requested explanation.
4. A technically correct concise definition can receive full marks
   when the question asks only for a definition.
5. Do not invent mistakes or claim the candidate said something
   they did not say.
6. Web snippets may be incomplete or misleading. Treat them as
   optional supporting evidence, not as mandatory checklists.
7. If code is supplied, assess its logic and correctness where
   possible. Do not claim code was executed unless it actually was.
8. Be constructive, specific, and respectful.

SCORING:
- Every criterion receives an integer from 0 to 5.
- Technical correctness: factual accuracy.
- Completeness: coverage of the question's essential requirements.
- Clarity: understandable wording.
- Structure: logical organization appropriate to the answer.

Completeness guide:
5 = all essential requirements of the question are covered.
4 = correct answer with a small relevant omission.
3 = partially answers the question or misses an important requirement.
2 = substantial relevant information is missing.
1 = minimal relevant information.
0 = no meaningful answer.

Return one JSON object only. Do not use Markdown fences.
All required fields must be present.
""",
        ),
        (
            "human",
            """
Topic: {topic}

Question:
{question}

Candidate answer:
{candidate_answer}

Optional web evidence:
{evidence}

Produce the following JSON structure:
{{
  "technical_correctness": {{"score": 0, "reason": "Explain accuracy."}},
  "completeness": {{"score": 0, "reason": "Explain coverage of the question."}},
  "clarity": {{"score": 0, "reason": "Explain clarity."}},
  "structure": {{"score": 0, "reason": "Explain organization."}},
  "strengths": ["Specific strength"],
  "missing_points": ["Only essential missing requirements"],
  "reference_answer": "A technically sound answer to the question.",
  "improved_answer": "An improved version of the candidate answer."
}}

Additional instructions:
- The reference answer must directly answer the actual question.
- The improved answer should preserve correct candidate content.
- Do not add unrelated material just to make an answer longer.
- If no essential point is missing, return an empty missing_points list.
- Keep reasons concise and evidence-based.
""",
        ),
    ]
)


# LangChain Expression Language (LCEL):
# prompt -> local Ollama model -> textual JSON response.
evaluation_chain = evaluation_prompt | llm


# ---------------------------------------------------------
# 4. LangGraph shared state
# ---------------------------------------------------------

class InterviewState(TypedDict, total=False):
    """Data passed between LangGraph nodes."""

    topic: str
    question: str
    candidate_answer: str
    search_enabled: bool

    sources: list[dict[str, str]]
    search_note: str
    evidence: str

    evaluation: dict[str, Any]
    final_score: float
    error: str


# ---------------------------------------------------------
# 5. Helpers: parse and validate model output
# ---------------------------------------------------------

def _message_to_text(message: Any) -> str:
    """Convert a LangChain model response to plain text."""
    content = getattr(message, "content", message)

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        # Some chat models return content as a list of text blocks.
        parts = [
            item.get("text", "")
            for item in content
            if isinstance(item, dict)
        ]
        return "\n".join(parts)

    return str(content)


def parse_model_evaluation(raw_text: str) -> dict[str, Any]:
    """Parse JSON and validate required fields and score ranges."""
    cleaned = raw_text.strip()

    # Be tolerant of Markdown fences if the model adds them.
    cleaned = re.sub(
        r"^\s*```(?:json)?\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\s*```\s*$", "", cleaned)

    start = cleaned.find("{")
    end = cleaned.rfind("}")

    if start < 0 or end < start:
        raise ValueError("The local model did not return a JSON object.")

    try:
        parsed = json.loads(cleaned[start:end + 1])
    except json.JSONDecodeError as exc:
        raise ValueError("The local model returned invalid JSON.") from exc

    try:
        validated = ModelEvaluation.model_validate(parsed)
    except ValidationError as exc:
        raise ValueError(
            f"The local model returned an invalid evaluation: {exc}"
        ) from exc

    return validated.model_dump()


# ---------------------------------------------------------
# 6. LangGraph node: retrieve optional evidence
# ---------------------------------------------------------

def retrieve_evidence_node(state: InterviewState) -> dict[str, Any]:
    """Retrieve search snippets only when web search is enabled."""
    if not state.get("search_enabled", False):
        return {
            "sources": [],
            "search_note": "Web search disabled; local model only.",
            "evidence": "No external evidence supplied.",
        }

    sources, note = search_question(
        topic=state["topic"],
        question=state["question"],
    )

    if not sources:
        evidence = "No usable web evidence was retrieved."
    else:
        evidence = "\n\n".join(
            (
                f"Title: {source['title']}\n"
                f"URL: {source['url']}\n"
                f"Snippet: {source['snippet']}"
            )
            for source in sources
        )

    return {
        "sources": sources,
        "search_note": note,
        "evidence": evidence,
    }


# ---------------------------------------------------------
# 7. LangGraph node: evaluate candidate answer
# ---------------------------------------------------------

def evaluate_answer_node(state: InterviewState) -> dict[str, Any]:
    """Ask Phi-3 to evaluate the answer and produce useful feedback."""
    response = evaluation_chain.invoke(
        {
            "topic": state["topic"],
            "question": state["question"],
            "candidate_answer": state["candidate_answer"],
            "evidence": state.get("evidence", "No external evidence supplied."),
        }
    )

    raw_text = _message_to_text(response)
    evaluation = parse_model_evaluation(raw_text)

    # Keep only valid source data returned by the search service.
    evaluation["sources"] = state.get("sources", [])
    evaluation["search_note"] = state.get("search_note", "")

    return {"evaluation": evaluation}


# ---------------------------------------------------------
# 8. LangGraph node: deterministic weighted scoring
# ---------------------------------------------------------

def calculate_score_node(state: InterviewState) -> dict[str, Any]:
    """
    Calculate the final score in Python.

    Each criterion is scored from 0 to 5.
    Weights sum to 1.0; multiplying the weighted result by 2
    converts the result to a 0-10 scale.
    """
    evaluation = state["evaluation"]

    expected_criteria = {
        "technical_correctness",
        "completeness",
        "clarity",
        "structure",
    }

    if set(SCORING_WEIGHTS) != expected_criteria:
        raise ValueError("SCORING_WEIGHTS must define all four criteria.")

    if abs(sum(SCORING_WEIGHTS.values()) - 1.0) > 1e-9:
        raise ValueError("SCORING_WEIGHTS must sum to 1.0.")

    weighted_score = 0.0

    for criterion, weight in SCORING_WEIGHTS.items():
        score = evaluation[criterion]["score"]

        if not isinstance(score, int) or not 0 <= score <= 5:
            raise ValueError(f"Invalid score for {criterion}: {score!r}")

        weighted_score += score * weight

    final_score = round(weighted_score * 2, 2)

    # Keep the evaluation result and the computed final score together.
    updated_evaluation = dict(evaluation)
    updated_evaluation["final_score"] = final_score

    return {
        "final_score": final_score,
        "evaluation": updated_evaluation,
    }


# ---------------------------------------------------------
# 9. Build and compile the LangGraph workflow
# ---------------------------------------------------------

# Execution order:
# START -> retrieve evidence -> evaluate -> calculate score -> END.
graph_builder = StateGraph(InterviewState)

graph_builder.add_node("retrieve_evidence", retrieve_evidence_node)
graph_builder.add_node("evaluate_answer", evaluate_answer_node)
graph_builder.add_node("calculate_score", calculate_score_node)

graph_builder.add_edge(START, "retrieve_evidence")
graph_builder.add_edge("retrieve_evidence", "evaluate_answer")
graph_builder.add_edge("evaluate_answer", "calculate_score")
graph_builder.add_edge("calculate_score", END)

interview_graph = graph_builder.compile()


# ---------------------------------------------------------
# 10. Public function used by the FastAPI service
# ---------------------------------------------------------

def evaluate_candidate_answer(
    question: str,
    candidate_answer: str,
    topic: str,
    search_enabled: bool = False,
) -> dict[str, Any]:
    """
    Evaluate one answer through the compiled LangGraph workflow.

    This function is the stable entry point for the API layer.
    """
    if not question.strip():
        raise ValueError("Question cannot be empty.")

    if not candidate_answer.strip():
        raise ValueError("Candidate answer cannot be empty.")

    if not topic.strip():
        raise ValueError("Topic cannot be empty.")

    result = interview_graph.invoke(
        {
            "topic": topic.strip(),
            "question": question.strip(),
            "candidate_answer": candidate_answer.strip(),
            "search_enabled": search_enabled,
        }
    )

    # Return a plain dictionary suitable for Pydantic/FastAPI.
    evaluation = dict(result["evaluation"])
    evaluation["final_score"] = result["final_score"]

    return evaluation
