from __future__ import annotations

import re

from loguru import logger
from pydantic import BaseModel, Field

from logic.intent_resolver import TaskIntent
from logic.page_classifier import PageContext
from logic.text_reasoner import TextReasonerProtocol

_STOP_WORDS = frozenset({
    "a", "an", "and", "at", "best", "buy", "compare", "find", "for", "from",
    "get", "give", "good", "great", "how", "in", "is", "it", "its", "me",
    "most", "my", "of", "on", "or", "our", "show", "that", "the", "this",
    "to", "top", "what", "which", "with",
})

_EXTRACT_SYSTEM_PROMPT = (
    "You extract product listings from e-commerce page data.\n"
    "You receive numbered text blocks scraped from a page.\n"
    "Identify which blocks are actual product listings relevant to the user's goal.\n"
    "Skip navigation links, banners, account menus, and unrelated items.\n"
    "Return a JSON object: {\"products\": [{\"index\": N, \"title\": \"...\", "
    "\"price\": \"...\", \"rating\": \"...\", \"is_sponsored\": bool}, ...]}\n"
    "Only include real products. Keep titles concise and accurate."
)


class ExtractedItem(BaseModel):
    title: str = ""
    url: str = ""
    price: str | None = None
    rating: str | None = None
    snippet: str = ""
    fields: dict[str, str] = Field(default_factory=dict)


class ExtractedContent(BaseModel):
    items: list[ExtractedItem] = Field(default_factory=list)
    raw_text: str = ""


class ContentExtractor:
    def __init__(self, text_reasoner: TextReasonerProtocol | None = None) -> None:
        self._text_reasoner = text_reasoner

    async def extract(
        self,
        browser,  # noqa: ANN001
        page_context: PageContext,
        intent: TaskIntent,
        *,
        max_candidates: int = 12,
        max_main_text_chars: int = 15000,
    ) -> ExtractedContent:
        if page_context.page_type == "article":
            return ExtractedContent(raw_text=await browser.get_main_text(limit=max_main_text_chars))

        if page_context.page_type in {"search_results", "product_listing"}:
            raw_blocks = await browser.extract_candidate_items(limit=max_candidates * 3)
            if not raw_blocks:
                return ExtractedContent()

            if self._text_reasoner and self._text_reasoner.available:
                items = await self._extract_via_llm(raw_blocks, intent, max_candidates)
                if items:
                    return ExtractedContent(items=items)

            items = [ExtractedItem.model_validate(b) for b in raw_blocks]
            items = _filter_relevant(items, intent, max_candidates)
            return ExtractedContent(items=items)

        if intent.expected_result == "text":
            return ExtractedContent(raw_text=await browser.get_visible_text(limit=max_main_text_chars))

        return ExtractedContent()

    async def _extract_via_llm(
        self,
        raw_blocks: list[dict],
        intent: TaskIntent,
        limit: int,
    ) -> list[ExtractedItem] | None:
        prompt_lines = [f"Goal: {intent.search_query or 'find products'}\n\nPage blocks:"]
        for idx, block in enumerate(raw_blocks):
            snippet = (block.get("snippet") or block.get("title") or "")[:400]
            prompt_lines.append(f"{idx}. {snippet}")

        payload = await self._text_reasoner.generate_json(
            system_prompt=_EXTRACT_SYSTEM_PROMPT,
            user_prompt="\n".join(prompt_lines),
        )
        if not payload:
            return None

        products = payload.get("products") or payload.get("items") or []
        if isinstance(payload, list):
            products = payload

        items: list[ExtractedItem] = []
        for entry in products:
            try:
                idx = int(entry.get("index", -1))
            except (TypeError, ValueError):
                continue
            if idx < 0 or idx >= len(raw_blocks):
                continue
            if entry.get("is_sponsored"):
                continue
            block = raw_blocks[idx]
            items.append(ExtractedItem(
                title=str(entry.get("title") or block.get("title", "")),
                url=str(block.get("url", "")),
                price=entry.get("price") or block.get("price"),
                rating=entry.get("rating") or block.get("rating"),
                snippet=str(block.get("snippet", "")),
            ))
            if len(items) >= limit:
                break

        logger.info(
            "LLM product extraction",
            raw_blocks=len(raw_blocks),
            identified=len(items),
        )
        return items if items else None


def _filter_relevant(
    items: list[ExtractedItem],
    intent: TaskIntent,
    limit: int,
) -> list[ExtractedItem]:
    """Heuristic fallback when LLM is unavailable."""
    terms = _goal_terms(intent.search_query or "")
    if not terms:
        return items[:limit]

    scored: list[tuple[float, int, ExtractedItem]] = []
    for idx, item in enumerate(items):
        haystack = f"{item.title} {item.snippet}".lower()

        is_sponsored = _is_sponsored(haystack)
        term_hits = sum(1 for t in terms if t in haystack)
        if term_hits == 0:
            continue

        score = term_hits / len(terms)
        if is_sponsored:
            score *= 0.3
        scored.append((score, idx, item))

    scored.sort(key=lambda t: (-t[0], t[1]))
    result = [item for _, _, item in scored[:limit]]
    return result if result else items[:limit]


def _goal_terms(query: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", query.lower())
    return [w for w in words if w not in _STOP_WORDS and len(w) >= 2]


def _is_sponsored(text: str) -> bool:
    return bool(re.search(r"\bsponsored\b", text, flags=re.IGNORECASE))
