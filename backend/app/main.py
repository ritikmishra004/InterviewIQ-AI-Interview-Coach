
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
# CORS_ALLOWED_ORIGINS env variable can supply CloudFront or custom domains.
import os

default_origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]
env_origins = os.getenv("CORS_ALLOWED_ORIGINS")
allowed_origins = [o.strip() for o in env_origins.split(",") if o.strip()] if env_origins else default_origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins if allowed_origins != ["*"] else ["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
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
