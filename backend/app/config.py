"""
Application configuration for InterviewIQ — Adaptive AI Interview Agent.
Supports Strands Decider 2B for #policy triage and Amazon Bedrock for generative evaluation.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env if present
load_dotenv()

# backend/ is the parent of the app/ directory.
BACKEND_DIR = Path(__file__).resolve().parent.parent

# Persistent data stays inside the backend directory.
DATA_DIR = BACKEND_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

QUESTION_BANK_PATH = RAW_DATA_DIR / "question_bank.json"
DATABASE_PATH = DATA_DIR / "interview_history.db"
SEARCH_CACHE_PATH = PROCESSED_DATA_DIR / "search_cache.json"

# Ollama runs the open-weight model on the local machine (fallback).
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "phi3:latest")

# AWS Bedrock configuration
AWS_REGION = os.getenv("AWS_REGION", "ap-southeast-2")
BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "google.gemma-3-4b-it")
USE_BEDROCK = os.getenv("USE_BEDROCK", "false").lower() in ("true", "1", "yes")

# Strands Decider 2B configuration
DECIDER_MODEL_ID = os.getenv("DECIDER_MODEL_ID", "StrandsAgents/strands-decider-2B-hobson-v21")
DECIDER_DEVICE = os.getenv("DECIDER_DEVICE", "mps")
USE_LOCAL_DECIDER = os.getenv("USE_LOCAL_DECIDER", "true").lower() in ("true", "1", "yes")
DECIDER_CONFIDENCE_THRESHOLD = float(os.getenv("DECIDER_CONFIDENCE_THRESHOLD", "0.60"))
DECIDER_ENDPOINT_URL = os.getenv("DECIDER_ENDPOINT_URL", "")

# Keep model generation deterministic for more consistent evaluation.
LLM_TEMPERATURE = 0

# Weighted criteria must sum to exactly 1.0.
SCORING_WEIGHTS = {
    "technical_correctness": 0.40,
    "completeness": 0.30,
    "clarity": 0.15,
    "structure": 0.15,
}

# Optional web search should remain disabled unless the candidate
# explicitly enables it for an interview.
DEFAULT_WEB_SEARCH_ENABLED = os.getenv("DEFAULT_WEB_SEARCH_ENABLED", "false").lower() in ("true", "1", "yes")

# Canonical 16 approved topics whitelist for InterviewIQ
APPROVED_TOPICS = [
    "LLM",
    "Transformer",
    "Generative AI",
    "Agentic AI",
    "SQL",
    "Coding",
    "Statistics",
    "Machine Learning",
    "Deep Learning",
    "MCP",
    "Computer Networks",
    "Operating System",
    "Python",
    "NLP",
    "AI Engineering",
    "RAG",
]

# Search snippets are supporting evidence, not guaranteed ground truth.
MAX_SEARCH_RESULTS = 4

