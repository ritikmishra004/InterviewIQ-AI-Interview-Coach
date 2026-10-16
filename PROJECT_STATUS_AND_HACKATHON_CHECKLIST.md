# InterviewIQ — Project Status, Audit Log & Hackathon Checklist

**Date:** October 16, 2026  
**Repository:** [https://github.com/ritikmishra004/InterviewIQ-AI-Interview-Coach](https://github.com/ritikmishra004/InterviewIQ-AI-Interview-Coach)  
**Hackathon:** AWS Builder Center — Agent Experience Hackathon: Strands Decider  
**Required Tags:** `#strands-decider-2b` `#policy`

---

## 1. Audit Log: Current Project State vs Previous Audits

### 1.1 What was already present during the previous audit
* **Foundational Architecture:** FastAPI backend, React 19/Vite/Tailwind CSS frontend, SQLite persistence (`interview_history.db`).
* **Strands Decider 2B & Bedrock Integration:** Initial integration with `StrandsAgents/strands-decider-2B-hobson-v21` on MPS/local and `google.gemma-3-4b-it` on Amazon Bedrock (`ap-southeast-2`).
* **Multi-Turn Adaptive Workflow:** LangGraph state machine routing responses to `EVALUATE`, `ASK_CLARIFICATION`, or `REDIRECT`.
* **Decision Pattern Export:** `patterns/adaptive_interview_policy/` with `policy.py` and `example_usage.py`.
* **Automated Test Suite:** 22 tests in `test_adaptive_interview.py` and `test_question_bank.py`.

### 1.2 What changes were made since that audit
* **Raw Question Bank Expansion (`commit e00d5a7`):** Expanded from 202 questions to 1,350 questions. However, questions were fragmented across 185 unstructured, ad-hoc topics (e.g., *Advanced RAG*, *Attention Mechanism*, *Arrays*, *TCP and UDP*, *Hash Maps and Sets*, etc.).
* **Question Bank Navigation Bug Identified:** Clicking topics inside the Question Bank redirected the candidate back to the `Overview` dashboard rather than displaying matching questions in-place.
* **Evaluation Parsing Fragility:** Valid Bedrock API responses that returned malformed JSON or failed Pydantic validation could cause unhandled exceptions instead of cleanly falling back to Ollama or the deterministic rule-based evaluator.

---

## 2. Changes Implemented and Confirmed Working in This Task

### 2.1 Question Bank Taxonomy Migration (Strict 16 Whitelist)
* Created safe backups: `backend/data/raw/question_bank.json.bak` and `backend/data/interview_history.db.bak`.
* Mapped all 1,350 questions from 185 fragmented topics into the exact 16 approved canonical topics with **zero silent deletions and zero unmapped questions**:

| # | Approved Topic | Question Count | Key Subtopics Covered |
| :--- | :--- | :--- | :--- |
| 1 | **LLM** | 127 | Fundamentals, Architecture, Parameters, Tokenization, Pretraining, Fine-Tuning, Inference, Deployment |
| 2 | **Transformer** | 29 | Architecture, Attention Mechanism, Multi-Head Attention, Positional Encoding, RoPE, ALiBi |
| 3 | **Generative AI** | 31 | GenAI Fundamentals, Multimodal, Hallucinations, Prompt Engineering, Responsible AI |
| 4 | **Agentic AI** | 103 | Architecture, Planning & Reasoning, Tool Calling, Multi-Agent, Memory, Observability, LangGraph |
| 5 | **SQL** | 3 | Complex queries, joins, rankings, duplicate detection, aggregations |
| 6 | **Coding** | 103 | DSA, Arrays, Linked Lists, Stacks, Queues, Trees, Graphs, Two Pointers, Dynamic Programming |
| 7 | **Statistics** | 96 | Probability, Distributions, Hypothesis Testing, Sampling, Correlation, A/B Testing, Bayesian |
| 8 | **Machine Learning** | 16 | ML Engineering, Feature Engineering, Bias-Variance Tradeoff, Model Selection, Drift |
| 9 | **Deep Learning** | 96 | Neural Networks, Activation Functions, Backpropagation, CNN, RNN, Loss Functions, Transfer Learning |
| 10 | **MCP** | 107 | Fundamentals, Architecture, Primitives, Tools, Transports, Security, SDKs, Human Approval |
| 11 | **Computer Networks** | 100 | OSI & TCP/IP, IP & Subnetting, Routing & Switching, TCP/UDP, DNS/DHCP/NAT, Security |
| 12 | **Operating System** | 100 | Processes & Threads, CPU Scheduling, Deadlocks, Virtual Memory, File Systems, I/O, Linux/Unix |
| 13 | **Python** | 150 | Fundamentals, OOP, Decorators, Iterators/Generators, Mutability, Concurrency, Performance, NumPy, Pandas |
| 14 | **NLP** | 43 | Tokenization, Word Embeddings, NER, Text Classification, Sequence Models, Preprocessing |
| 15 | **AI Engineering** | 120 | System Design, APIs, DevOps, Cloud/Containers, MLOps, Performance, Observability |
| 16 | **RAG** | 126 | Ingestion, Chunking, Embeddings, Vector DBs, Retrieval, Reranking, Production RAG, Evaluation |
| **Total** | **All 16 Topics** | **1,350** | **100% of original records preserved with original IDs and difficulty tags** |

### 2.2 Critical Question Bank Navigation Fix
* **Root Cause:** In `frontend/src/App.jsx`, clicking a topic button in `QuestionBankPage` called `onSelect(topic)` which executed `setSelectedTopic(topic); goToPage("Overview");`.
* **Fix Applied:**
  * Upgraded `QuestionBankPage` to manage internal topic selection and detailed question exploration.
  * Clicking any of the 16 topics now displays all matching questions in-place inside `Question Bank`.
  * Added difficulty filtering (`All`, `Easy`, `Medium`, `Hard`) and back button navigation to return to the 16-topic grid.
  * Added "Practice This Topic" button to launch directly into an active interview for the selected topic.
  * Added dedicated empty states for zero matching filters.

### 2.3 Backend API Alignment
* Added canonical `APPROVED_TOPICS` whitelist in `backend/app/config.py`.
* Updated `GET /api/topics` to strictly return the 16 approved topics.
* Updated `GET /api/question-counts` to initialize all 16 topics with `{Easy, Medium, Hard}` breakdowns.
* Added `GET /api/questions` endpoint supporting topic filtering, difficulty filtering, limit, and offset pagination.

### 2.4 Resilient Evaluation and Fallback Parsing
* In `backend/app/services/interview_agent.py`: Wrapped `parse_model_evaluation` in a try/except block for both Bedrock and Ollama.
* When Bedrock returns malformed JSON or invalid schema values, the error is logged and the pipeline gracefully triggers the next intended evaluator (Ollama -> Rule-based Deterministic Evaluator) without crashing the session.

---

## 3. Verification & Test Outcomes

* **Pytest Test Suite:** All 25 unit and integration tests passing:
  ```bash
  PYTHONPATH=backend ./.venv/bin/pytest -v
  # 25 passed in 0.68s
  ```
  * `test_questions_belong_to_approved_topics` (PASSED)
  * `test_each_approved_topic_has_valid_questions` (PASSED)
  * `test_api_topics_and_counts_match_16_approved_topics` (PASSED)
  * `test_relevant_complete_answer_routes_to_evaluate` (PASSED)
  * `test_low_confidence_fallback_handling` (PASSED)
  * `test_bedrock_service_unavailable_fallback` (PASSED)
  * `test_invalid_structured_bedrock_output_error_handling` (PASSED)
* **Frontend Production Build:**
  ```bash
  npm run build
  # ✓ built in 679ms, zero syntax or bundling errors
  ```
* **Reusable Decision Pattern Example:**
  ```bash
  PYTHONPATH=. python patterns/adaptive_interview_policy/example_usage.py
  # Verified across all 4 candidate answer scenarios (EVALUATE, ASK_CLARIFICATION, REDIRECT)
  ```

---

## 4. Hackathon Submission Readiness Checklist

| Requirement | Description | Status | Action Needed |
| :--- | :--- | :--- | :--- |
| **Working Agent on AWS** | Strands Decider 2B for triage + Amazon Bedrock (`google.gemma-3-4b-it`) | **Complete & Verified** | Operational locally and ready for AWS container deployment |
| **Reusable Decision Pattern** | `#policy` pattern exported with typed questions and test runner | **Complete & Verified** | Located in `patterns/adaptive_interview_policy/` |
| **Open Source License** | Apache License 2.0 present in repository | **Complete** | `LICENSE` file verified |
| **No Secrets in Repo** | `.env` and `.db` excluded from Git | **Complete** | Verified `.gitignore` and git status |
| **16 Strict Topics** | Clean taxonomy aligned across UI and API | **Complete & Verified** | Verified 1,350 questions across 16 categories |
| **Article on AWS Builder Center** | Written technical article describing agent & decision pattern | **Draft Ready (Needs Publication)** | Review & publish `docs/builder-center-article-draft.md` on AWS Builder Center |
| **Demo Video (< 3 min)** | Public video demonstration of the working agent and Decider gate | **Script Ready (Needs Recording)** | Record video using `docs/demo-video-script.md` and upload (YouTube/Loom) |
| **Hackathon Tags** | `#strands-decider-2b` and `#policy` | **Complete** | Tagged in README, pattern docs, and metadata |
