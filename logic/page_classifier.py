from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from core.schemas import AgentState


class PageContext(BaseModel):
    page_type: Literal[
        "unknown",
        "search_results",
        "product_listing",
        "article",
        "form",
        "dashboard",
        "product_detail",
    ] = "unknown"
    structure_hint: str | None = None
    has_actionable_items: bool = False


class PageClassifier:
    async def classify(self, state: AgentState, browser) -> PageContext:  # noqa: ANN001
        url = state.url.lower()
        title = (await browser.get_page_title()).lower()
        text = await browser.get_visible_text(limit=4000)
        hints = await browser.get_page_structure_hints()
        candidate_count = int(hints.get("candidate_count", 0))
        paragraph_count = int(hints.get("paragraph_count", 0))
        form_count = int(hints.get("form_count", 0))
        input_count = int(hints.get("input_count", 0))

        if any(token in url for token in ["/dp/", "/gp/", "/aw/d/"]):
            return PageContext(
                page_type="product_detail",
                structure_hint="detail page",
                has_actionable_items=True,
            )

        if candidate_count >= 3 and ("/s?" in url or "search" in url or "?q=" in url or "&q=" in url):
            return PageContext(
                page_type="search_results",
                structure_hint="repeated result cards",
                has_actionable_items=True,
            )

        if candidate_count >= 3:
            return PageContext(
                page_type="product_listing",
                structure_hint="listing or card grid",
                has_actionable_items=True,
            )

        if ("sign in" in title or "log in" in title or "register" in title) or (
            form_count > 0 and input_count >= 3
        ):
            return PageContext(
                page_type="form",
                structure_hint="form inputs visible",
                has_actionable_items=True,
            )

        if any(token in url for token in ["/dashboard", "/admin"]) or "dashboard" in title:
            return PageContext(
                page_type="dashboard",
                structure_hint="navigation-heavy dashboard",
                has_actionable_items=True,
            )

        if paragraph_count >= 5 and len(text) >= 900:
            return PageContext(
                page_type="article",
                structure_hint="long-form text content",
                has_actionable_items=False,
            )

        return PageContext(
            page_type="unknown",
            structure_hint=None,
            has_actionable_items=candidate_count > 0 or form_count > 0,
        )
