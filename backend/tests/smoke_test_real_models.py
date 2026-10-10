"""
Smoke test runner for real live Strands Decider 2B and Amazon Bedrock inference.

Usage:
    # Run with default environment settings:
    PYTHONPATH=backend python backend/tests/smoke_test_real_models.py

    # Enable real Bedrock execution (requires AWS credentials configured):
    USE_BEDROCK=true AWS_REGION=ap-southeast-2 PYTHONPATH=backend python backend/tests/smoke_test_real_models.py
"""

import os
import sys

from app.services.decider_service import decide_interview_action, initialize_decider_engine
from app.services.bedrock_service import evaluate_with_bedrock, generate_clarification_question
from app.config import USE_BEDROCK, BEDROCK_MODEL_ID, AWS_REGION, DECIDER_MODEL_ID


def main():
    print("=" * 70)
    print("InterviewIQ — Real Model Smoke Test")
    print(f"Decider Target: {DECIDER_MODEL_ID}")
    print(f"Bedrock Target: {BEDROCK_MODEL_ID} in {AWS_REGION} (USE_BEDROCK={USE_BEDROCK})")
    print("=" * 70)

    # 1. Strands Decider 2B Check
    print("\n[1/3] Testing Strands Decider 2B Engine Loading & Inference...")
    engine, err = initialize_decider_engine()
    if engine is None:
        print(f"  [-] Local engine not loaded into memory: {err}")
        print("  [+] Testing graceful calibrated fallback execution...")
    else:
        print("  [+] Strands Decider 2B engine successfully initialized in memory.")

    res = decide_interview_action(
        question="What is the GIL in Python?",
        candidate_answer="The GIL is the Global Interpreter Lock in CPython that restricts thread execution.",
        topic="Python",
    )
    print(f"  [+] Decider Result:")
    print(f"      - Decision:    {res['decision']}")
    print(f"      - Confidence:  {res['confidence']:.2%}")
    print(f"      - Latency:     {res['latency_ms']:.1f} ms")
    print(f"      - Fallback:    {res['is_fallback']} ({res.get('fallback_reason')})")

    # 2. Clarification Generation Check
    print("\n[2/3] Testing Adaptive Clarification Generation...")
    q, p = generate_clarification_question(
        topic="Python",
        question="What is the GIL in Python?",
        candidate_answer="It is a lock for threads.",
    )
    print(f"  [+] Clarification Question Generated:\n      \"{q}\"")

    # 3. Bedrock Generative Evaluation Check
    print("\n[3/3] Testing Generative Evaluation Pipeline...")
    if USE_BEDROCK:
        print(f"  [*] Attempting live Amazon Bedrock invocation ({BEDROCK_MODEL_ID})...")
        eval_dict, eval_err = evaluate_with_bedrock(
            topic="Python",
            question="What is the GIL in Python?",
            candidate_answer="The GIL prevents concurrent multi-threading in CPython by holding a mutex on bytecodes.",
        )
        if eval_dict:
            print("  [+] Amazon Bedrock live invocation succeeded:")
            print(f"      - Technical correctness: {eval_dict.get('technical_correctness')}")
            print(f"      - Completeness:          {eval_dict.get('completeness')}")
        else:
            print(f"  [-] Bedrock call failed ({eval_err}). Fallback pipeline engaged.")
    else:
        print("  [*] USE_BEDROCK is false. Using local fallback evaluation pipeline.")

    print("\n" + "=" * 70)
    print("Smoke test completed.")
    print("=" * 70)


if __name__ == "__main__":
    main()
