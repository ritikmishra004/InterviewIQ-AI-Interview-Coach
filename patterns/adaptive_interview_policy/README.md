# Reusable Decision Pattern: Adaptive Interview Policy

**Hackathon Pattern:** `#policy`  
**Model:** `StrandsAgents/strands-decider-2B-hobson-v21`  
**Primary Tags:** `#strands-decider-2b` `#policy`

---

## 1. Overview
In conversational agents and AI interviewers, calling expensive generative models (such as Amazon Bedrock Claude or Gemma) for every single raw user input leads to:
1. High token costs on blank, evasive, or adversarial attempts.
2. Inappropriate hallucinated feedback on nonsense responses.
3. Inability to clarify ambiguous or partial answers before finalizing a score.

The **Adaptive Interview Policy** pattern uses **Strands Decider 2B** as a fast, sub-150ms "System 1" decision gate. It evaluates the tuple `(question, candidate_answer, topic)` against discrete next-step pedagogical choices before triggering System 2 generative evaluation.

---

## 2. Decision Space & Schema
The pattern defines a typed `ChoiceQuestion` using the official `strands-decider` API:

| Decision | Semantic Meaning | Subsequent Workflow Action |
| :--- | :--- | :--- |
| **`EVALUATE`** | Substantive, relevant answer (definitions, code snippets, or explanations). | Routes to Amazon Bedrock for 4-criterion evaluation and scoring. |
| **`ASK_CLARIFICATION`** | Related but incomplete, tentative, or ambiguous response. | Prompts Bedrock to generate one targeted follow-up question. |
| **`REDIRECT`** | Evasive ("idk", "skip"), blank, or completely off-topic response. | Returns immediate coaching guidance; prompts candidate to retry without scoring. |

---

## 3. Calibrated Confidence & Low-Confidence Fallback
* Strands Decider models provide calibrated probability distributions via a pointer head.
* The policy enforces a configurable **confidence threshold** (default `0.60`).
* When the model confidence falls below this threshold or if local hardware acceleration is unavailable, the policy transitions safely to a **calibrated deterministic fallback**.
* **Zero Misrepresentation:** The pattern explicitly flags `is_fallback: True` and includes the exact `fallback_reason` in the response payload.

---

## 4. Quickstart Integration

```python
from patterns.adaptive_interview_policy.policy import AdaptiveInterviewPolicy

policy = AdaptiveInterviewPolicy(
    model_id="StrandsAgents/strands-decider-2B-hobson-v21",
    device="mps", # Apple Silicon MPS or CUDA / CPU
    confidence_threshold=0.60,
)

decision = policy.decide(
    question="What is the GIL in Python?",
    candidate_answer="GIL is the Global Interpreter Lock in CPython",
    topic="Python",
)

print(decision["decision"])       # "EVALUATE"
print(decision["confidence"])     # 0.85
print(decision["latency_ms"])     # 12.4 ms
print(decision["is_fallback"])    # False
```

---

## 5. How to Adapt to Other Agents
To adapt this pattern to customer support, triage, or code review:
1. Update `criteria` dictionary inside `create_policy_decision_question()` with your domain-specific actions (e.g., `ESCALATE_TO_HUMAN`, `ANSWER_FROM_KB`, `REQUEST_ORDER_ID`).
2. Pass your domain state (e.g., user request + ticket history) as the state prompt.
3. Connect the returned action directly into your LangGraph / agent router.
