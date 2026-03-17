from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field

from logic.text_reasoner import TextReasonerProtocol


class TaskIntent(BaseModel):
    task_type: Literal[
        "navigate",
        "extract_and_return",
        "extract_and_compare",
        "extract_and_summarize",
        "form_fill",
        "search_and_return",
    ] = "navigate"
    criteria: list[str] = Field(default_factory=list)
    expected_result: Literal["link", "text", "list", "count", "confirmation"] = "confirmation"
    target_count: int | None = None
    search_query: str | None = None


class IntentResolver:
    def __init__(self, text_reasoner: TextReasonerProtocol | None = None) -> None:
        self._text_reasoner = text_reasoner

    async def resolve(self, goal: str) -> TaskIntent:
        heuristic = self._resolve_heuristically(goal)
        if heuristic.task_type != "navigate" or not self._text_reasoner or not self._text_reasoner.available:
            return heuristic

        payload = await self._text_reasoner.generate_json(
            system_prompt=(
                "Classify the user goal into a task intent. "
                "Return JSON with task_type, criteria, expected_result, target_count, search_query."
            ),
            user_prompt=goal,
        )
        if not payload:
            return heuristic

        try:
            refined = TaskIntent.model_validate(payload)
            if refined.search_query is None:
                refined.search_query = heuristic.search_query
            if not refined.criteria:
                refined.criteria = heuristic.criteria
            if refined.target_count is None:
                refined.target_count = heuristic.target_count
            return refined
        except Exception:
            return heuristic

    def _resolve_heuristically(self, goal: str) -> TaskIntent:
        text = goal.lower().strip()
        target_count = _extract_count(text)
        criteria = _extract_criteria(text)
        search_query = _extract_search_query(goal)

        if any(token in text for token in ["summarize", "summary", "tl;dr"]):
            return TaskIntent(
                task_type="extract_and_summarize",
                expected_result="text",
                target_count=target_count,
                search_query=search_query,
            )

        if any(token in text for token in ["log in", "login", "sign in", "fill out", "submit form"]):
            return TaskIntent(
                task_type="form_fill",
                expected_result="confirmation",
                target_count=target_count,
                search_query=search_query,
            )

        if any(token in text for token in ["best", "compare", "cheapest", "lowest price", "highest rated"]):
            expected = "link" if "link" in text or "url" in text else "text"
            if target_count and target_count > 1:
                expected = "list" if expected != "link" else "link"
            return TaskIntent(
                task_type="extract_and_compare",
                criteria=criteria,
                expected_result=expected,
                target_count=target_count or 1,
                search_query=search_query,
            )

        if any(token in text for token in ["return the link", "give me the link", "give me a link", "find link"]):
            return TaskIntent(
                task_type="search_and_return",
                criteria=criteria,
                expected_result="link",
                target_count=target_count or 1,
                search_query=search_query,
            )

        if any(token in text for token in ["how many", "count", "number of"]):
            return TaskIntent(
                task_type="extract_and_return",
                criteria=criteria,
                expected_result="count",
                target_count=target_count,
                search_query=search_query,
            )

        if any(token in text for token in ["what is", "which is", "tell me", "give me", "show me"]):
            return TaskIntent(
                task_type="extract_and_return",
                criteria=criteria,
                expected_result="text",
                target_count=target_count,
                search_query=search_query,
            )

        return TaskIntent(
            task_type="navigate",
            criteria=criteria,
            expected_result="confirmation",
            target_count=target_count,
            search_query=search_query,
        )


def _extract_count(text: str) -> int | None:
    patterns = [
        r"\btop\s+(\d+)\b",
        r"\bbest\s+(\d+)\b",
        r"\bcompare\s+(\d+)\b",
        r"\b(\d+)\s+(?:products|options|items|results)\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return max(1, int(match.group(1)))
    return None


def _extract_criteria(text: str) -> list[str]:
    criteria: list[str] = []
    for token in ["price", "cheap", "cheapest", "rating", "reviews", "speed", "fast", "best", "quality"]:
        if token in text and token not in criteria:
            criteria.append(token)
    return criteria


def _extract_search_query(goal: str) -> str | None:
    match = re.search(r"(?:for|about|find)\s+(.+)", goal, flags=re.IGNORECASE)
    if not match:
        return None
    query = match.group(1).strip(" .?")
    return query or None
