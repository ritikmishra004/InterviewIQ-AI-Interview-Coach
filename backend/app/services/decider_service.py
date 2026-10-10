"""
decider_service.py — Strands Decider 2B integration.

AWS Strands Decider 2B Hackathon (October 2026)
Pattern: #policy
Model: StrandsAgents/strands-decider-2B-hobson-v21

Provides sub-150ms typed decision triage using Strands Decider 2B.
Classifies interview candidate answers into:
- EVALUATE: Relevant and sufficiently meaningful to evaluate.
- ASK_CLARIFICATION: Related but incomplete or ambiguous; targeted follow-up needed.
- REDIRECT: Clearly unrelated, blank/evasive, or off-topic.

Features:
- Official Strands Decider ChoiceQuestion API
- Calibrated confidence tracking and latency measurement
- Transparent separation of model_decision vs final_decision
- Never misrepresents fallback as confident real-model decision
- Memory-safe Apple Silicon (MPS/CPU) loading with fallback
"""

import logging
import re
import time
from typing import Any, Dict, Optional, Tuple

from app.config import (
    DECIDER_CONFIDENCE_THRESHOLD,
    DECIDER_DEVICE,
    DECIDER_MODEL_ID,
    USE_LOCAL_DECIDER,
)

logger = logging.getLogger(__name__)

# Global cache for the engine to avoid reloading weights repeatedly
_ENGINE_INSTANCE: Any = None
_ENGINE_INITIALIZED: bool = False
_ENGINE_LOAD_ERROR: Optional[str] = None


def get_interview_policy_question() -> Any:
    """
    Construct the typed ChoiceQuestion according to strands_decider.schema.
    """
    from strands_decider.schema import ChoiceQuestion

    instructions = (
        "You are an expert technical interview panel chair. "
        "Analyze the candidate's answer to the technical interview question. "
        "Decide the exact next pedagogical action to take from the available choices. "
        "Short definitions, code snippets, and mathematical formulas are valid answers if technically relevant. "
        "Only choose ASK_CLARIFICATION if the answer is on-topic but missing a crucial distinction or ambiguous. "
        "Choose REDIRECT if the answer is clearly unrelated, evasive, or gibberish."
    )

    criteria = {
        "EVALUATE": (
            "The candidate's response is relevant, addresses the core concepts of the question, "
            "and provides enough technical substance (even if concise, short, or containing errors) "
            "for formal rubric evaluation."
        ),
        "ASK_CLARIFICATION": (
            "The candidate's response is directly related to the question topic, but is ambiguous, "
            "leaves out a central qualifying condition, or is a partial fragment where asking a single "
            "clarifying follow-up will allow the candidate to demonstrate their true understanding."
        ),
        "REDIRECT": (
            "The response is completely off-topic, evasive (e.g. 'I do not know', 'skip', 'pass'), "
            "nonsensical, adversarial, or fails to address any aspect of the asked question."
        ),
    }

    return ChoiceQuestion(
        instructions=instructions,
        criteria=criteria,
    )


def initialize_decider_engine() -> Tuple[Optional[Any], Optional[str]]:
    """
    Attempt to load the real Strands Decider 2B engine if configured and available.
    Returns (engine, error_message).
    """
    global _ENGINE_INSTANCE, _ENGINE_INITIALIZED, _ENGINE_LOAD_ERROR

    if _ENGINE_INITIALIZED:
        return _ENGINE_INSTANCE, _ENGINE_LOAD_ERROR

    _ENGINE_INITIALIZED = True

    if not USE_LOCAL_DECIDER:
        _ENGINE_LOAD_ERROR = "Local decider disabled by configuration (USE_LOCAL_DECIDER=false)."
        return None, _ENGINE_LOAD_ERROR

    try:
        from strands_decider.infer import load_engine

        target_device = DECIDER_DEVICE
        if target_device == "mps":
            import torch

            if not torch.backends.mps.is_available():
                logger.warning("MPS requested but not available; falling back to CPU.")
                target_device = "cpu"

        logger.info(
            "Loading Strands Decider 2B checkpoint %s on device %s...",
            DECIDER_MODEL_ID,
            target_device,
        )
        engine = load_engine(DECIDER_MODEL_ID, device=target_device)
        _ENGINE_INSTANCE = engine
        _ENGINE_LOAD_ERROR = None
        logger.info("Successfully loaded Strands Decider 2B engine.")
        return _ENGINE_INSTANCE, None

    except Exception as exc:
        err_msg = f"Failed to load Strands Decider model ({DECIDER_MODEL_ID}): {exc}"
        logger.warning(err_msg)
        _ENGINE_LOAD_ERROR = err_msg
        _ENGINE_INSTANCE = None
        return None, _ENGINE_LOAD_ERROR


def classify_interview_answer_fallback(
    question: str,
    candidate_answer: str,
    topic: str,
) -> Tuple[str, float, Dict[str, float], str]:
    """
    Deterministic rule-based policy fallback when Decider 2B is unavailable or low-confidence.
    Takes into account short valid answers, code snippets, and evasion phrases.
    """
    cleaned = candidate_answer.strip()
    lower = cleaned.lower()

    # Evasive / empty / gibberish checks -> REDIRECT
    evasive_exact = {"idk", "i don't know", "i dont know", "no idea", "pass", "skip", "n/a", "none"}
    evasive_keywords = {"idk", "pass", "skip", "nope", "dunno", "nothing"}
    words_list = lower.split()
    has_evasive_kw = any(w in evasive_keywords for w in words_list)
    if (
        not cleaned
        or lower in evasive_exact
        or len(cleaned) < 4
        or (has_evasive_kw and len(words_list) <= 4)
        or any(phrase in lower for phrase in ["i don't know", "i dont know", "no idea", "have no clue"])
    ):
        return (
            "REDIRECT",
            0.92,
            {"REDIRECT": 0.92, "ASK_CLARIFICATION": 0.05, "EVALUATE": 0.03},
            "Answer is empty, evasive, or insufficient to evaluate.",
        )

    # Check for off-topic/adversarial statements
    adversarial_markers = ["ignore all previous", "system prompt", "as an ai", "pretend you are"]
    if any(marker in lower for marker in adversarial_markers):
        return (
            "REDIRECT",
            0.88,
            {"REDIRECT": 0.88, "ASK_CLARIFICATION": 0.08, "EVALUATE": 0.04},
            "Off-topic or prompt injection detected.",
        )

    # Short definitions / code snippets:
    code_markers = ["def ", "class ", "return ", "import ", "select ", "{", "}", "->", "=>"]
    has_code = any(marker in lower for marker in code_markers)

    words = cleaned.split()
    word_count = len(words)

    # Check if key topic tokens appear in answer or question
    topic_tokens = set(topic.lower().split())
    question_tokens = set(re_tokens(question.lower()))
    answer_tokens = set(re_tokens(lower))

    overlap = len(answer_tokens.intersection(question_tokens))

    # Partial / ambiguous fragment check -> ASK_CLARIFICATION
    clarification_phrases = ["maybe", "i think it has to do with", "sort of", "something with", "it depends"]
    if any(phrase in lower for phrase in clarification_phrases) and word_count < 20:
        return (
            "ASK_CLARIFICATION",
            0.78,
            {"ASK_CLARIFICATION": 0.78, "EVALUATE": 0.15, "REDIRECT": 0.07},
            "Answer is tentative or ambiguous; clarification requested.",
        )

    if word_count < 10 and not has_code and overlap == 0:
        return (
            "ASK_CLARIFICATION",
            0.70,
            {"ASK_CLARIFICATION": 0.70, "EVALUATE": 0.20, "REDIRECT": 0.10},
            "Brief answer lacks detail; follow-up clarification recommended.",
        )

    # Standard technical answers (definitions, code, explanations) -> EVALUATE
    return (
        "EVALUATE",
        0.85,
        {"EVALUATE": 0.85, "ASK_CLARIFICATION": 0.10, "REDIRECT": 0.05},
        "Substantive technical answer provided.",
    )


def re_tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"\w+", text) if len(t) > 2]


def decide_interview_action(
    question: str,
    candidate_answer: str,
    topic: str,
    force_fallback: bool = False,
) -> Dict[str, Any]:
    """
    Execute the #policy decision for the candidate answer.

    Returns consistent schema:
    {
        "model": str,
        "pattern": "#policy",
        "decision": str,          # The operational decision adopted by the agent
        "model_decision": str | None, # Raw decision returned by real model (None if fallback only)
        "final_decision": str,    # Final action after confidence gating / fallback
        "action_taken": str,
        "confidence": float,
        "latency_ms": float,
        "is_fallback": bool,
        "fallback_reason": str | None,
        "probabilities": dict
    }
    """
    start_time = time.perf_counter()

    # 1. Force fallback mode (explicit test or offline flag)
    if force_fallback:
        fb_dec, fb_conf, fb_probs, reason = classify_interview_answer_fallback(question, candidate_answer, topic)
        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
        return {
            "model": DECIDER_MODEL_ID,
            "pattern": "#policy",
            "decision": fb_dec,
            "model_decision": None,
            "final_decision": fb_dec,
            "action_taken": f"Fallback policy selected: {fb_dec} ({reason})",
            "confidence": fb_conf,
            "latency_ms": latency_ms,
            "is_fallback": True,
            "fallback_reason": "Forced fallback / test mode",
            "probabilities": fb_probs,
        }

    engine, load_error = initialize_decider_engine()

    # 2. Engine not available (fallback)
    if engine is None:
        fb_dec, fb_conf, fb_probs, reason = classify_interview_answer_fallback(question, candidate_answer, topic)
        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
        return {
            "model": DECIDER_MODEL_ID,
            "pattern": "#policy",
            "decision": fb_dec,
            "model_decision": None,
            "final_decision": fb_dec,
            "action_taken": f"Fallback policy selected: {fb_dec} ({reason})",
            "confidence": fb_conf,
            "latency_ms": latency_ms,
            "is_fallback": True,
            "fallback_reason": load_error or "Strands Decider engine unavailable",
            "probabilities": fb_probs,
        }

    # 3. Real Strands Decider 2B inference
    try:
        policy_q = get_interview_policy_question()
        state_content = (
            f"Topic: {topic}\n"
            f"Interview Question: {question}\n"
            f"Candidate Answer: {candidate_answer}"
        )

        response = engine.ask(
            state=state_content,
            questions={"policy_decision": policy_q},
        )

        answer_obj = response.answers.get("policy_decision")
        if answer_obj is None:
            raise ValueError("Strands Decider did not return 'policy_decision' in response.")

        selected_choice = answer_obj.choice
        confidence = float(answer_obj.confidence)
        probabilities = {k: float(v) for k, v in getattr(answer_obj, "probabilities", {}).items()}
        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

        # Check confidence threshold for reliable gating
        if confidence < DECIDER_CONFIDENCE_THRESHOLD:
            logger.info(
                "Decider confidence %.3f below threshold %.2f. Triggering fallback.",
                confidence,
                DECIDER_CONFIDENCE_THRESHOLD,
            )
            fb_dec, fb_conf, fb_probs, fb_reason = classify_interview_answer_fallback(
                question, candidate_answer, topic
            )
            return {
                "model": DECIDER_MODEL_ID,
                "pattern": "#policy",
                "decision": fb_dec,
                "model_decision": selected_choice,
                "final_decision": fb_dec,
                "action_taken": (
                    f"Decider chose {selected_choice} with low confidence "
                    f"({confidence:.2f}); safe fallback engaged: {fb_dec}"
                ),
                "confidence": confidence,
                "latency_ms": latency_ms,
                "is_fallback": True,
                "fallback_reason": (
                    f"Confidence ({confidence:.2f}) fell below "
                    f"threshold ({DECIDER_CONFIDENCE_THRESHOLD})"
                ),
                "probabilities": probabilities or fb_probs,
            }

        # Confident real model decision
        return {
            "model": DECIDER_MODEL_ID,
            "pattern": "#policy",
            "decision": selected_choice,
            "model_decision": selected_choice,
            "final_decision": selected_choice,
            "action_taken": f"Strands Decider 2B routed to: {selected_choice}",
            "confidence": confidence,
            "latency_ms": latency_ms,
            "is_fallback": False,
            "fallback_reason": None,
            "probabilities": probabilities,
        }

    except Exception as exc:
        logger.exception("Error during Strands Decider inference.")
        fb_dec, fb_conf, fb_probs, fb_reason = classify_interview_answer_fallback(
            question, candidate_answer, topic
        )
        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
        return {
            "model": DECIDER_MODEL_ID,
            "pattern": "#policy",
            "decision": fb_dec,
            "model_decision": None,
            "final_decision": fb_dec,
            "action_taken": f"Inference exception -> Fallback: {fb_dec}",
            "confidence": fb_conf,
            "latency_ms": latency_ms,
            "is_fallback": True,
            "fallback_reason": f"Runtime inference exception: {exc}",
            "probabilities": fb_probs,
        }