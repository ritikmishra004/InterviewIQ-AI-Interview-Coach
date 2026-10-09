
"""
search_service.py — Optional web search with local caching.

DDGS provides free web search without a paid API key.
Search snippets are supporting evidence, NOT verified ground truth.

The cache avoids repeating the same query unnecessarily.
If search fails, the interview can continue using the local LLM.
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from ddgs import DDGS

from app.config import MAX_SEARCH_RESULTS, SEARCH_CACHE_PATH

logger = logging.getLogger(__name__)


def _load_cache() -> dict[str, Any]:
    """Read cached search results; use an empty cache if unavailable."""
    SEARCH_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)

    if not SEARCH_CACHE_PATH.exists():
        return {}

    try:
        content = SEARCH_CACHE_PATH.read_text(encoding="utf-8")
        data = json.loads(content)

        # Only dictionaries are valid cache roots.
        return data if isinstance(data, dict) else {}

    except (OSError, json.JSONDecodeError):
        logger.warning("Search cache could not be read; starting with an empty cache.")
        return {}


def _save_cache(cache: dict[str, Any]) -> None:
    """Persist the cache locally using UTF-8 JSON."""
    SEARCH_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)

    SEARCH_CACHE_PATH.write_text(
        json.dumps(cache, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _normalize_result(item: dict[str, Any]) -> dict[str, str] | None:
    """Convert a DDGS result to the stable structure used by the app."""
    url = str(item.get("href") or item.get("url") or "").strip()

    # Ignore entries without a URL.
    if not url.startswith(("https://", "http://")):
        return None

    return {
        "title": str(item.get("title") or "Untitled source").strip(),
        "url": url,
        "snippet": str(item.get("body") or "").strip(),
    }


def search_question(
    topic: str,
    question: str,
) -> tuple[list[dict[str, str]], str]:
    """
    Search for supporting information for a technical question.

    Returns:
        (sources, note)

    `sources` is a list of title/url/snippet dictionaries.
    `note` explains whether results came from cache or live search.
    """
    clean_topic = topic.strip()
    clean_question = question.strip()

    if not clean_question:
        return [], "Search skipped because the question was empty."

    query = f"{clean_topic} {clean_question} technical explanation documentation"
    cache_key = hashlib.sha256(
        query.casefold().encode("utf-8")
    ).hexdigest()

    cache = _load_cache()

    # Reuse previously retrieved evidence for identical queries.
    cached_entry = cache.get(cache_key)
    if isinstance(cached_entry, dict):
        cached_sources = cached_entry.get("sources", [])

        if isinstance(cached_sources, list):
            valid_sources = [
                source
                for source in cached_sources
                if isinstance(source, dict)
                and source.get("url", "").startswith(("https://", "http://"))
            ]

            return valid_sources, "Loaded search results from the local cache."

    try:
        # DDGS may contact an external search provider.
        # Only the search query is sent; the candidate's answer is not.
        raw_results = DDGS().text(
            query,
            max_results=MAX_SEARCH_RESULTS,
        )

        sources: list[dict[str, str]] = []

        for item in raw_results or []:
            if not isinstance(item, dict):
                continue

            normalized = _normalize_result(item)

            if normalized:
                sources.append(normalized)

        # Cache successful results, including an empty result list.
        cache[cache_key] = {
            "query": query,
            "sources": sources,
        }

        try:
            _save_cache(cache)
        except OSError:
            # A cache write failure must not interrupt the interview.
            logger.warning("Search results could not be cached.")

        if sources:
            return sources, "Fresh search results retrieved and cached locally."

        return [], "Search returned no usable results; local-model evaluation will continue."

    except Exception as exc:
        # Search-provider outages, rate limits, and network failures
        # should not prevent the candidate from practising.
        logger.warning("Web search failed: %s", type(exc).__name__)

        return [], (
            "Web search was unavailable. Evaluation can continue "
            "using the local model without external evidence."
        )
