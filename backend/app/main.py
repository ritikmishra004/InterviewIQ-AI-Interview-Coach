
"""
main.py — FastAPI application entry point.

Responsibilities:
- Create the FastAPI application.
- Initialize the local SQLite database on startup.
- Register interview API routes.
- Configure CORS for the React development server.
- Expose health and root endpoints.

Run from the backend directory:
    uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
"""

from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.database import initialize_database


# ---------------------------------------------------------
# Logging
# ---------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# Application lifecycle
# ---------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initialize local resources before accepting requests.

    SQLite setup is synchronous and lightweight, so it is safe
    to initialize it during startup for this local application.
    """
    logger.info("Initializing AI Interview Coach database...")
    initialize_database()
    logger.info("Database initialized.")

    yield

    logger.info("AI Interview Coach API shutting down.")


# ---------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------

app = FastAPI(
    title="AI Interview Coach API",
    description=(
        "A local-first interview practice application powered by "
        "Phi-3 through Ollama, LangChain, LangGraph, and SQLite."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------
# CORS configuration
# ---------------------------------------------------------
#
# Vite's default development URL is http://localhost:5173.
# CORS allows browser requests from the React frontend.
# The API does not require credentials or authentication yet.

allowed_origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)


# ---------------------------------------------------------
# Register API routes
# ---------------------------------------------------------

app.include_router(router)


# ---------------------------------------------------------
# Root endpoint
# ---------------------------------------------------------

@app.get("/", tags=["Health"])
def root() -> dict[str, str]:
    """Return basic API information."""
    return {
        "name": "AI Interview Coach API",
        "status": "running",
        "docs": "/docs",
        "health": "/api/health",
    }
