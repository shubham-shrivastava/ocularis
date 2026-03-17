from __future__ import annotations

from logic.intent_resolver import TaskIntent
from logic.page_classifier import PageContext


class StrategyRouter:
    def should_extract(
        self,
        *,
        intent: TaskIntent | None,
        page_context: PageContext | None,
        has_pending_recovery: bool,
    ) -> bool:
        if not intent or not page_context or has_pending_recovery:
            return False
        if intent.task_type in {"navigate", "form_fill"}:
            return False
        if page_context.page_type == "article":
            return intent.task_type == "extract_and_summarize"
        if page_context.page_type in {"search_results", "product_listing"}:
            return intent.task_type in {
                "extract_and_compare",
                "search_and_return",
                "extract_and_return",
            }
        return False
