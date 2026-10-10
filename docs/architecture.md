# InterviewIQ — System Architecture

**Target:** AWS Strands Decider Agent Hackathon (October 2026)  
**Required Tags:** `#strands-decider-2b` `#policy`

---

## 1. System Overview

InterviewIQ is an adaptive AI interview coach combining low-latency decision triage via **Strands Decider 2B** with deep generative evaluation and clarification via **Amazon Bedrock**.

```
                           +--------------------------------------+
                           |            React Frontend            |
                           |    (Vite / Tailwind / Lucide-React)  |
                           |       Configurable API Base URL      |
                           +------------------+-------------------+
                                              | HTTP REST
                                              v
                           +--------------------------------------+
                           |           FastAPI Backend            |
                           |   (Active Sessions & SQLite History) |
                           +------------------+-------------------+
                                              |
                                     LangGraph State Machine
                                              |
               +------------------------------+------------------------------+
               |                                                             |
               v [Sub-150ms Decision]                                        v [Generative Evaluation]
+-------------------------------+                             +-------------------------------+
|      Strands Decider 2B       |                             |        Amazon Bedrock         |
|   (Pointer Head / #policy)    |                             |      google.gemma-3-4b-it     |
|   MPS / Local / Fallback      |                             |        ap-southeast-2         |
+---------------+---------------+                             +---------------+---------------+
                |                                                             ^
                | [EVALUATE] -------------------------------------------------+
                | [ASK_CLARIFICATION] -> Bedrock generates 1 targeted follow-up question
                | [REDIRECT] ----------> Immediate coaching feedback; retry without penalty
```

---

## 2. Core Execution Flow

1. **Question Selection**: Candidate chooses topic (e.g. Python, LLM & GenAI, Agentic AI) and difficulty. 202 curated questions loaded without generative drift.
2. **Candidate Response Ingestion**: The candidate submits their answer.
3. **Strands Decider 2B Triage (`#policy`)**:
   - Classifies input against typed `ChoiceQuestion` criteria into `EVALUATE`, `ASK_CLARIFICATION`, or `REDIRECT`.
   - Records calibrated confidence score and latency.
   - Triggers low-confidence fallback if confidence < 0.60.
4. **Conditional Routing**:
   - `REDIRECT`: Returns coaching notice directly to candidate; avoids token waste and score fabrication.
   - `ASK_CLARIFICATION`: Bedrock generates one concise follow-up question (1-turn limit); candidate clarifies.
   - `EVALUATE`: Bedrock produces structured evaluation across 4 criteria + strengths + improvements.
5. **Deterministic Scoring**: Python math strictly computes weighted score:
   - Technical Correctness: 40%
   - Completeness: 30%
   - Clarity: 15%
   - Structure: 15%
6. **SQLite Persistence & Dashboard**: Saved to `interview_history.db` for session analytics.

---

## 3. Technology Stack

- **Decision Model**: `StrandsAgents/strands-decider-2B-hobson-v21`
- **Generative Model**: Amazon Bedrock (`google.gemma-3-4b-it`)
- **Orchestration**: LangGraph, LangChain Core
- **Backend**: FastAPI, Pydantic v2, Uvicorn, SQLite3, Boto3
- **Frontend**: React 19, Vite, Tailwind CSS, Recharts, Lucide-React
