# Decision Pattern: Adaptive Interview Policy

**Pattern Tag:** `#policy`  
**Model Tag:** `#strands-decider-2b`  
**Framework:** Strands Decider Official Python API (`strands-decider`)

---

## 1. Problem Definition
Standard conversational agents pass all user inputs to heavy generative LLMs. In an AI interview setting, this creates three critical problems:
1. **Token Cost Waste**: Users submitting blank answers, "idk", or prompt injections trigger costly multi-thousand token generation calls.
2. **Hallucinated Evaluations**: Generative models often fabricate partial marks or hallucinate criteria for evasive or nonsense answers.
3. **Premature Evaluation**: Promising candidates who give brief or slightly ambiguous answers are prematurely penalized instead of prompted for a standard interview clarification.

---

## 2. Decider 2B Solution
Strands Decider 2B is a 1.9B-parameter decision model that uses a pointer head over masked token representations to execute a single, sub-150ms forward pass.

### Typed Schema
```python
from strands_decider.schema import ChoiceQuestion

policy_question = ChoiceQuestion(
    instructions="Analyze the candidate's answer and decide the next pedagogical action.",
    criteria={
        "EVALUATE": "Relevant technical response containing sufficient substance for rubric evaluation.",
        "ASK_CLARIFICATION": "Related to question but ambiguous or missing a central distinction; 1 follow-up needed.",
        "REDIRECT": "Completely off-topic, evasive ('idk', 'pass'), or prompt injection.",
    }
)
```

---

## 3. Calibrated Confidence Handling
- **Calibrated Scores**: Decider 2B outputs genuine probabilities over options.
- **Safety Gate**: If the model's highest probability falls below `0.60`, the policy safely flags a low-confidence decision and routes via calibrated fallback rules.
- **Transparency**: Fallback status is reported to the API response and rendered in the frontend UI badge.
