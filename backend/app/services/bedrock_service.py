"""
bedrock_service.py — Amazon Bedrock generative evaluation and clarification service.

AWS Strands Decider Agent Hackathon (October 2026)
Region: ap-southeast-2 (or configurable via AWS_REGION)
Primary Model ID: google.gemma-3-4b-it (configurable via BEDROCK_MODEL_ID)

Features:
- Boto3 bedrock-runtime Converse API integration
- Robust JSON extraction & Pydantic validation
- Evaluation feedback generation (4 criteria, strengths, missing points, reference, improved answers)
- Clarification follow-up question generation for ASK_CLARIFICATION decisions
- Graceful error handling for throttling, credentials, timeouts, and local model fallback
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

from app.config import (
    AWS_REGION,
    BEDROCK_MODEL_ID,
    LLM_TEMPERATURE,
    USE_BEDROCK,
)

logger = logging.getLogger(__name__)

# Boto3 client singleton with standard timeouts
_BEDROCK_CLIENT: Optional[Any] = None


def get_bedrock_runtime_client() -> Optional[Any]:
    """Obtain or initialize the Bedrock Runtime client."""
    global _BEDROCK_CLIENT
    if _BEDROCK_CLIENT is not None:
        return _BEDROCK_CLIENT

    try:
        config = Config(
            region_name=AWS_REGION,
            retries={"max_attempts": 3, "mode": "standard"},
            connect_timeout=10,
            read_timeout=60,
        )
        _BEDROCK_CLIENT = boto3.client("bedrock-runtime", config=config)
        return _BEDROCK_CLIENT
    except Exception as exc:
        logger.warning("Could not initialize Bedrock client: %s", exc)
        return None


def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """Robustly extract and parse JSON object from LLM response text."""
    if not text:
        return None

    cleaned = text.strip()

    # Direct parse attempt
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Try extracting inside markdown json code blocks
    code_block_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
    if code_block_match:
        try:
            return json.loads(code_block_match.group(1))
        except json.JSONDecodeError:
            pass

    # Find first '{' and matching outermost '}'
    start_idx = cleaned.find("{")
    end_idx = cleaned.rfind("}")
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        candidate = cleaned[start_idx : end_idx + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    return None


def invoke_bedrock_converse(
    system_prompt: str,
    user_prompt: str,
    model_id: Optional[str] = None,
    temperature: float = LLM_TEMPERATURE,
    max_tokens: int = 1500,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Invoke Bedrock using the Converse API.
    Returns (response_text, error_message).
    """
    client = get_bedrock_runtime_client()
    if client is None:
        return None, "Bedrock client not initialized (check AWS credentials/region)."

    target_model = model_id or BEDROCK_MODEL_ID

    messages = [
        {
            "role": "user",
            "content": [{"text": user_prompt}],
        }
    ]

    system = [{"text": system_prompt}] if system_prompt else []

    try:
        response = client.converse(
            modelId=target_model,
            messages=messages,
            system=system,
            inferenceConfig={
                "temperature": temperature,
                "maxTokens": max_tokens,
            },
        )

        output_message = response.get("output", {}).get("message", {})
        content_list = output_message.get("content", [])
        if content_list and "text" in content_list[0]:
            return content_list[0]["text"], None

        return None, "Bedrock returned empty response content."

    except (ClientError, BotoCoreError) as exc:
        err_msg = f"Bedrock invocation failed ({target_model}): {exc}"
        logger.warning(err_msg)
        return None, err_msg
    except Exception as exc:
        err_msg = f"Unexpected error during Bedrock invocation: {exc}"
        logger.exception(err_msg)
        return None, err_msg


# ---------------------------------------------------------
# Prompts
# ---------------------------------------------------------

EVALUATION_SYSTEM_PROMPT = """You are an expert technical interview evaluator and coach.
Evaluate the candidate's answer against the exact technical question asked.

FAIRNESS & SCORING RULES:
1. Judge only what the question asks.
2. Do not penalize concise answers if they are technically accurate definitions.
3. Code snippets and mathematical expressions are valid responses.
4. Score each criterion from 0 to 5 (integers only):
   - technical_correctness (0-5)
   - completeness (0-5)
   - clarity (0-5)
   - structure (0-5)
5. Return ONLY a valid JSON object matching this exact schema:
{
  "technical_correctness": {"score": 4, "reason": "Explanation..."},
  "completeness": {"score": 3, "reason": "Explanation..."},
  "clarity": {"score": 5, "reason": "Explanation..."},
  "structure": {"score": 4, "reason": "Explanation..."},
  "strengths": ["Clear definition", "Correct terminology"],
  "missing_points": ["Could mention trade-offs"],
  "reference_answer": "Complete, ideal model answer...",
  "improved_answer": "How the candidate's specific answer could be upgraded..."
}
Do not include any conversational preamble or markdown wrap outside the JSON.
"""

CLARIFICATION_SYSTEM_PROMPT = """You are an encouraging technical interviewer conducting a live interview.
The candidate gave an initial response that is partially relevant or ambiguous.
Your task is to ask ONE targeted, concise follow-up question (1-2 sentences) to help them clarify or complete their answer.
Do not give away the solution. Be professional and supportive.
Return ONLY a valid JSON object:
{
  "follow_up_question": "...",
  "clarification_prompt": "..."
}
"""


def evaluate_with_bedrock(
    topic: str,
    question: str,
    candidate_answer: str,
    evidence: str = "No external evidence supplied.",
    combined_follow_up: Optional[str] = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Generate evaluation using Amazon Bedrock.
    Supports evaluating initial answer or combined (initial + clarification) answer.
    """
    answer_text = candidate_answer
    if combined_follow_up:
        answer_text = (
            f"Initial Answer: {candidate_answer}\n"
            f"Clarification / Follow-up Answer: {combined_follow_up}"
        )

    user_prompt = (
        f"Topic: {topic}\n"
        f"Question: {question}\n\n"
        f"Candidate Answer:\n{answer_text}\n\n"
        f"Supporting Evidence:\n{evidence}\n\n"
        f"Evaluate the answer and return the requested JSON."
    )

    raw_response, error = invoke_bedrock_converse(
        system_prompt=EVALUATION_SYSTEM_PROMPT,
        user_prompt=user_prompt,
    )

    if error or not raw_response:
        return None, error or "No response from Bedrock"

    parsed = extract_json_from_text(raw_response)
    if parsed is None:
        return None, f"Failed to parse JSON from Bedrock response: {raw_response[:200]}"

    return parsed, None


def generate_clarification_question(
    topic: str,
    question: str,
    candidate_answer: str,
) -> Tuple[str, str]:
    """
    Generate one targeted clarification question using Amazon Bedrock,
    with deterministic fallback if Bedrock is offline.
    """
    user_prompt = (
        f"Topic: {topic}\n"
        f"Interview Question: {question}\n"
        f"Candidate's initial response: {candidate_answer}\n\n"
        f"Ask one targeted clarifying question."
    )

    if USE_BEDROCK:
        raw_response, error = invoke_bedrock_converse(
            system_prompt=CLARIFICATION_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            max_tokens=300,
        )
        if raw_response and not error:
            parsed = extract_json_from_text(raw_response)
            if parsed and "follow_up_question" in parsed:
                return (
                    parsed["follow_up_question"],
                    parsed.get("clarification_prompt", "Please clarify your response:"),
                )

    # Clean deterministic fallback clarification
    fallback_q = (
        f"Could you elaborate on how your answer directly applies to '{question}'? "
        f"Specifically, what key trade-offs or implementation details would you highlight?"
    )
    fallback_prompt = (
        "Your initial answer was relevant but brief. Please expand on your explanation above."
    )
    return fallback_q, fallback_prompt
