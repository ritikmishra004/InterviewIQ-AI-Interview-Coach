# InterviewIQ — Adaptive AI Interview Agent

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Hackathon](https://img.shields.io/badge/AWS-Strands%20Decider%20Hackathon-orange.svg)](#)
[![Tags](https://img.shields.io/badge/Tags-%23strands--decider--2b%20|%20%23policy-purple.svg)](#)

**InterviewIQ** is an adaptive technical interview coaching agent powered by **Strands Decider 2B** and **Amazon Bedrock**.

Instead of routing every user response directly into heavy generative models, InterviewIQ implements the **`#policy`** decision pattern:
1. **System 1 (Decision Gate)**: **Strands Decider 2B** performs sub-150ms triage using a pointer head, classifying answers into `EVALUATE`, `ASK_CLARIFICATION`, or `REDIRECT`.
2. **System 2 (Generative Reasoning)**: **Amazon Bedrock** (`google.gemma-3-4b-it` in `ap-southeast-2`) produces deep 4-criterion feedback or asks targeted clarifying questions.
3. **Deterministic Scoring**: Strict Python weighted math avoids generative arithmetic errors.

---

## Architecture Flow

```
Candidate Answer 
      │
      ▼
Strands Decider 2B (#policy Decision Gate)
      ├─► REDIRECT --------► Evasive / Off-Topic (Instant Coaching Guidance)
      ├─► ASK_CLARIFICATION► Ambiguous (Bedrock Generates 1 Targeted Follow-up)
      └─► EVALUATE --------► Substantive Answer -> Amazon Bedrock 4-Criteria Evaluation
```

---

## Submission Tags

- `#strands-decider-2b`
- `#policy`

---

## Key Features

- **Sub-150ms Decision Triage**: Powered by `StrandsAgents/strands-decider-2B-hobson-v21` on Apple Silicon MPS or CPU.
- **Calibrated Confidence Tracking**: Every decision reports real model confidence and latency.
- **Fail-Safe Fallback**: If confidence drops below `0.60` or hardware is unavailable, a calibrated fallback engages with full transparency.
- **Adaptive Clarification**: Prevents premature candidate penalties by generating targeted follow-up questions before scoring.
- **Deterministic 4-Criteria Rubric**: Technical Correctness (40%), Completeness (30%), Clarity (15%), Structure (15%).
- **202 Curated Technical Questions**: Spanning 10 domains from Python to RAG and Agentic AI.
- **Modern Full-Stack Experience**: FastAPI backend + React 19 / Vite / Tailwind CSS frontend.
- **Reusable Decision Pattern**: Exported in `patterns/adaptive_interview_policy/`.

---

## Quickstart (Local Development)

### 1. Prerequisites
- Python 3.12+
- Node.js 18+ (for frontend)
- Apple Silicon Mac (M-series) or CUDA GPU / CPU

### 2. Backend Setup
```bash
# Clone the repository
git clone https://github.com/ritikmishra004/PrepMind-AI.git
cd ai-interview-coach-hackathon

# Setup virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
```

### 3. Run Automated Tests
```bash
PYTHONPATH=backend pytest backend/tests/
```

### 4. Run Smoke Test for Live Models
```bash
PYTHONPATH=backend python backend/tests/smoke_test_real_models.py
```

### 5. Start Backend Server
```bash
source .venv/bin/activate
cd backend
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

### 6. Start Frontend Application
```bash
cd frontend
npm install
npm run dev
```
Open your browser at `http://localhost:5173`.

---

## AWS Deployment Architecture

For submission compliance and production hosting on AWS:
- **FastAPI Backend**: Containerized on **AWS App Runner** or **ECS Fargate** (1 vCPU, 2 GB RAM).
- **React Frontend**: Static web build deployed on **Amazon S3 + CloudFront CDN**.
- **Generative Inference**: **Amazon Bedrock** (`google.gemma-3-4b-it` in region `ap-southeast-2`).
- **Decider Inference**: Local/containerized Strands Decider 2B engine.

---

## Reusable Pattern: Adaptive Interview Policy

A standalone implementation of the `#policy` decision pattern is available in `patterns/adaptive_interview_policy/`:
```bash
PYTHONPATH=. python patterns/adaptive_interview_policy/example_usage.py
```

See [patterns/adaptive_interview_policy/README.md](patterns/adaptive_interview_policy/README.md) for full integration details.

---

## Documentation

- [System Architecture](docs/architecture.md)
- [Decision Pattern Guide](docs/decision-pattern.md)
- [AWS Builder Center Article Draft](docs/builder-center-article-draft.md)
- [Demo Video Script (<3 min)](docs/demo-video-script.md)
- [Hackathon Submission Checklist](docs/hackathon-submission-checklist.md)

---

## License

This project is licensed under the [Apache License 2.0](LICENSE).
