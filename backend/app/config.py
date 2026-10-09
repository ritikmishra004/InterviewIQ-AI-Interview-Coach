
"""
Application configuration.

The project uses Phi-3 through Ollama for local inference.
No paid LLM API key is needed.

Centralizing configuration prevents model names, file paths,
and scoring weights from drifting between modules.
"""

from pathlib import Path


# backend/ is the parent of the app/ directory.
BACKEND_DIR = Path(__file__).resolve().parent.parent

# Persistent data stays inside the backend directory.
DATA_DIR = BACKEND_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

QUESTION_BANK_PATH = RAW_DATA_DIR / "question_bank.json"
DATABASE_PATH = DATA_DIR / "interview_history.db"
SEARCH_CACHE_PATH = PROCESSED_DATA_DIR / "search_cache.json"

# Ollama runs the open-weight model on the local machine.
OLLAMA_MODEL = "phi3:latest"

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
DEFAULT_WEB_SEARCH_ENABLED = False

# Search snippets are supporting evidence, not guaranteed ground truth.
MAX_SEARCH_RESULTS = 4
