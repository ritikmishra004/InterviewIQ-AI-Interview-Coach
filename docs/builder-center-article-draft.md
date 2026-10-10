# Building InterviewIQ: Adaptive AI Interviewing with Strands Decider 2B and Amazon Bedrock

**Tags:** `#strands-decider-2b` `#policy` `AmazonBedrock` `LangGraph` `GenerativeAI`

---

## Introduction: The Challenge of Evaluating Technical Answers

Evaluating technical interview answers with generative models presents a key dilemma. Traditional architectures pass every user answer directly into large language models (System 2). When a candidate submits an evasive answer like "idk", an off-topic tangent, or a brief snippet, standard generative models either hallucinate undeserved marks, waste thousands of expensive tokens, or fail to prompt the candidate for clarification.

In a real human interview, a skilled interviewer doesn't grade an empty answer—they redirect. And if an answer is promising but ambiguous, they don't fail the candidate on the spot; they ask a targeted follow-up question.

To bridge this gap, we built **InterviewIQ — Adaptive AI Interview Agent** using AWS's newly released **Strands Decider 2B** and **Amazon Bedrock**.

---

## Architecture: Fast Triage Meets Deep Generative Evaluation

InterviewIQ implements a two-stage decision and evaluation pipeline:

1. **System 1 (Decision Gate)**: **Strands Decider 2B** (`StrandsAgents/strands-decider-2B-hobson-v21`). Operating in sub-150ms latency using a pointer head, it evaluates the tuple `(question, candidate_answer, topic)` to classify the next action using the **`#policy`** pattern:
   - **`EVALUATE`**: The answer contains technical substance (explanations, definitions, or code snippets) ready for rubric scoring.
   - **`ASK_CLARIFICATION`**: The answer touches on the topic but is ambiguous or partial; 1 targeted follow-up question is required.
   - **`REDIRECT`**: The answer is off-topic, evasive, or adversarial; the candidate is coached to retry without grading.

2. **System 2 (Generative Reasoning)**: **Amazon Bedrock** (`google.gemma-3-4b-it` in `ap-southeast-2`).
   - When routed to `EVALUATE`, Bedrock performs multi-factor rubric analysis (Technical Correctness 40%, Completeness 30%, Clarity 15%, Structure 15%), highlights strengths, and provides model answers.
   - When routed to `ASK_CLARIFICATION`, Bedrock generates a targeted clarifying question, allowing the candidate to respond before calculating their final score.

3. **Deterministic Scoring**: Arithmetic is calculated in Python to prevent LLM hallucination in numerical calculations.

---

## Calibrated Confidence and Safety Fallback

Decider 2B provides calibrated output distributions rather than uncalibrated logit guesses. InterviewIQ monitors this probability:
- If confidence drops below `0.60`, the system automatically engages a calibrated fallback policy.
- The UI transparently presents whether a real model or fallback took the action, providing complete auditability for users and interviewers.

---

## Reusable Decision Pattern for the Community

We extracted this policy pattern into a clean, standalone module in `patterns/adaptive_interview_policy/`. Any developer building an interactive coaching or tutoring agent can drop in this pattern to prevent token waste, guard against prompt injection, and add adaptive follow-up behavior in less than 50 lines of code.

---

## Conclusion

Combining fast, specialized decision models like Strands Decider 2B with generative powerhouses like Amazon Bedrock creates agents that feel intuitive, conversational, and computationally efficient. InterviewIQ demonstrates how decision models elevate AI coaches from static grading forms into genuine interactive mentors.
