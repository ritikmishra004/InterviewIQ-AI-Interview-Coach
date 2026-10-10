"""
Adaptive Interview Policy Decision Pattern.

Hackathon Pattern: #policy
Model: StrandsAgents/strands-decider-2B-hobson-v21
API: strands_decider.schema.ChoiceQuestion

Provides a standalone, reusable decision pattern for interactive agents
to triage candidate responses before invoking generative models.

Actions:
- EVALUATE: Candidate provided a substantive answer (or definition/code snippet).
- ASK_CLARIFICATION: Candidate provided an ambiguous or incomplete answer.
- REDIRECT: Candidate provided an evasive or off-topic answer.
"""

import logging
import time
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


def create_policy_decision_question() -> Any:
    """Create the typed ChoiceQuestion instance for Strands Decider."""
    from strands_decider.schema import ChoiceQuestion

    return ChoiceQuestion(
        instructions=(
            "You are an expert technical interview panel chair. "
            "Analyze the candidate's answer to the technical interview question. "
            "Decide the exact next pedagogical action to take from the available choices. "
            "Short definitions, code snippets, and mathematical formulas are valid answers if technically relevant. "
            "Only choose ASK_CLARIFICATION if the answer is on-topic but missing a crucial distinction or ambiguous. "
            "Choose REDIRECT if the answer is clearly unrelated, evasive, or gibberish."
        ),
        criteria={
            "EVALUATE": (
                "The candidate's response is relevant, addresses core concepts of the question, "
                "and provides sufficient technical substance for formal rubric evaluation."
            ),
            "ASK_CLARIFICATION": (
                "The candidate's response is related to the question topic, but is ambiguous, "
                "leaves out a central qualifying condition, or is a partial fragment where asking a single "
                "clarifying follow-up will allow the candidate to demonstrate their true understanding."
            ),
            "REDIRECT": (
                "The response is completely off-topic, evasive (e.g. 'I do not know', 'skip', 'pass'), "
                "nonsensical, adversarial, or fails to address any aspect of the asked question."
            ),
        },
    )


class AdaptiveInterviewPolicy:
    """
    Reusable decision pattern that wraps Strands Decider 2B with:
    - Official ChoiceQuestion schema
    - Confidence thresholding
    - Calibrated fallback policy
    - Latency benchmarking
    """

    def __init__(
        self,
        model_id: str = "StrandsAgents/strands-decider-2B-hobson-v21",
        device: str = "mps",
        confidence_threshold: float = 0.60,
        enable_local_engine: bool = True,
    ):
        self.model_id = model_id
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.enable_local_engine = enable_local_engine
        self.engine = None
        self._load_error = None
        self._initialized = False

    def _ensure_engine(self):
        if self._initialized:
            return
        self._initialized = True
        if not self.enable_local_engine:
            self._load_error = "Local engine disabled"
            return

        try:
            from strands_decider.infer import load_engine
            target_device = self.device
            if target_device == "mps":
                import torch
                if not torch.backends.mps.is_available():
                    target_device = "cpu"

            self.engine = load_engine(self.model_id, device=target_device)
            self._load_error = None
        except Exception as exc:
            self._load_error = str(exc)
            self.engine = None

    def decide(
        self,
        question: str,
        candidate_answer: str,
        topic: str = "General",
        force_fallback: bool = False,
    ) -> Dict[str, Any]:
        """
        Classify candidate response into EVALUATE, ASK_CLARIFICATION, or REDIRECT.
        """
        start = time.perf_counter()

        if force_fallback:
            return self._fallback_result(question, candidate_answer, topic, start, "Forced fallback mode")

        self._ensure_engine()

        if self.engine is None:
            return self._fallback_result(
                question, candidate_answer, topic, start, self._load_error or "Engine unavailable"
            )

        try:
            q_schema = create_policy_decision_question()
            state = f"Topic: {topic}\nInterview Question: {question}\nCandidate Answer: {candidate_answer}"
            resp = self.engine.ask(state=state, questions={"decision": q_schema})
            ans = resp.answers["decision"]

            confidence = float(ans.confidence)
            latency_ms = round((time.perf_counter() - start) * 1000, 2)
            choice = ans.choice
            probabilities = {k: float(v) for k, v in getattr(ans, "probabilities", {}).items()}

            if confidence < self.confidence_threshold:
                # Calibrated low-confidence fallback
                fallback_choice, fb_conf, fb_probs, fb_reason = self._rule_fallback(
                    question, candidate_answer, topic
                )
                return {
                    "model": self.model_id,
                    "pattern": "#policy",
                    "decision": fallback_choice,
                    "model_decision": choice,
                    "final_decision": fallback_choice,
                    "confidence": confidence,
                    "latency_ms": latency_ms,
                    "action_taken": f"Low-confidence ({confidence:.2f}) -> Fallback: {fallback_choice}",
                    "is_fallback": True,
                    "fallback_reason": f"Confidence below threshold ({self.confidence_threshold})",
                    "probabilities": probabilities or fb_probs,
                }

            return {
                "model": self.model_id,
                "pattern": "#policy",
                "decision": choice,
                "model_decision": choice,
                "final_decision": choice,
                "confidence": confidence,
                "latency_ms": latency_ms,
                "action_taken": f"Strands Decider 2B routed to: {choice}",
                "is_fallback": False,
                "fallback_reason": None,
                "probabilities": probabilities,
            }

        except Exception as exc:
            return self._fallback_result(
                question, candidate_answer, topic, start, f"Runtime error: {exc}"
            )

    def _fallback_result(self, question, answer, topic, start, reason):
        choice, conf, probs, rule_reason = self._rule_fallback(question, answer, topic)
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        return {
            "model": self.model_id,
            "pattern": "#policy",
            "decision": choice,
            "model_decision": None,
            "final_decision": choice,
            "confidence": conf,
            "latency_ms": latency_ms,
            "action_taken": f"Fallback policy: {choice} ({rule_reason})",
            "is_fallback": True,
            "fallback_reason": reason,
            "probabilities": probs,
        }

    def _rule_fallback(self, question, answer, topic) -> Tuple[str, float, dict, str]:
        cleaned = answer.strip()
        lower = cleaned.lower()
        evasive = {"idk", "pass", "skip", "n/a", "no idea", "i don't know", "i dont know", "none"}
        if not cleaned or len(cleaned) < 4 or any(w in lower for w in evasive):
            return "REDIRECT", 0.92, {"REDIRECT": 0.92, "ASK_CLARIFICATION": 0.05, "EVALUATE": 0.03}, "Evasive answer"

        code_markers = ["def ", "class ", "return ", "import ", "select ", "{", "}", "->", "=>"]
        if any(m in lower for m in code_markers):
            return "EVALUATE", 0.88, {"EVALUATE": 0.88, "ASK_CLARIFICATION": 0.08, "REDIRECT": 0.04}, "Code snippet"

        words = cleaned.split()
        if len(words) < 8 and "it" in lower:
            return (
                "ASK_CLARIFICATION",
                0.75,
                {"ASK_CLARIFICATION": 0.75, "EVALUATE": 0.15, "REDIRECT": 0.10},
                "Ambiguous or brief statement",
            )

        return "EVALUATE", 0.85, {"EVALUATE": 0.85, "ASK_CLARIFICATION": 0.10, "REDIRECT": 0.05}, "Relevant answer"
