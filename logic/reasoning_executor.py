from __future__ import annotations

from pydantic import BaseModel, Field

from core.schemas import ResultCandidate, RunAnswer
from logic.content_extractor import ExtractedContent, ExtractedItem
from logic.intent_resolver import TaskIntent
from logic.text_reasoner import TextReasonerProtocol


class ReasoningResult(BaseModel):
    selected_item: ExtractedItem | None = None
    answer: RunAnswer | None = None
    compared_items: list[ExtractedItem] = Field(default_factory=list)


class ReasoningExecutor:
    def __init__(self, text_reasoner: TextReasonerProtocol | None = None) -> None:
        self._text_reasoner = text_reasoner

    async def reason(
        self,
        *,
        content: ExtractedContent,
        intent: TaskIntent,
        user_goal: str,
    ) -> ReasoningResult | None:
        if intent.task_type == "extract_and_summarize" and content.raw_text.strip():
            summary = await self._summarize(content.raw_text, user_goal)
            if not summary:
                return None
            return ReasoningResult(
                answer=RunAnswer(
                    result_type="text",
                    text=summary,
                    confidence=0.8,
                )
            )

        if intent.task_type in {"extract_and_compare", "search_and_return", "extract_and_return"} and content.items:
            compared_items = content.items[:12]
            selection = await self._select_item(compared_items, intent, user_goal)
            if not selection:
                return None
            selected = selection[0]
            candidates = [_to_candidate(item) for item in compared_items]
            result_type = intent.expected_result if intent.expected_result in {"link", "list", "count"} else "text"
            answer = RunAnswer(
                result_type=result_type,
                link=selected.url if intent.expected_result == "link" else None,
                text=selection[1],
                items=candidates,
                confidence=selection[2],
            )
            if intent.expected_result == "count":
                answer.text = str(len(compared_items))
            if intent.expected_result == "list":
                answer.text = selection[1]
            return ReasoningResult(
                selected_item=selected,
                answer=answer,
                compared_items=compared_items,
            )

        return None

    async def _summarize(self, text: str, goal: str) -> str | None:
        if self._text_reasoner and self._text_reasoner.available:
            summary = await self._text_reasoner.generate_text(
                system_prompt="Summarize the page text for the user in 2-3 sentences.",
                user_prompt=f"Goal: {goal}\n\nText:\n{text[:12000]}",
            )
            if summary:
                return summary
        return None

    async def _select_item(
        self,
        items: list[ExtractedItem],
        intent: TaskIntent,
        goal: str,
    ) -> tuple[ExtractedItem, str, float] | None:
        if self._text_reasoner and self._text_reasoner.available:
            payload = await self._text_reasoner.generate_json(
                system_prompt=(
                    "Given a user goal and a list of extracted candidates, select the best candidate. "
                    "Return JSON with keys index, answer_text, confidence."
                ),
                user_prompt=_format_items_prompt(goal, intent.criteria, items),
            )
            if payload:
                try:
                    index = int(payload.get("index", 0))
                    if 0 <= index < len(items):
                        return (
                            items[index],
                            str(payload.get("answer_text", "")) or items[index].title,
                            max(0.0, min(1.0, float(payload.get("confidence", 0.75)))),
                        )
                except Exception:
                    pass
        if not items:
            return None
        cheapest = _select_cheapest(items) if any(token in intent.criteria for token in ["cheap", "cheapest", "price"]) else None
        chosen = cheapest or items[0]
        answer = chosen.title if chosen.title else chosen.snippet
        return chosen, answer, 0.55


def _format_items_prompt(goal: str, criteria: list[str], items: list[ExtractedItem]) -> str:
    lines = [f"Goal: {goal}", f"Criteria: {', '.join(criteria) or 'none'}", "Candidates:"]
    for idx, item in enumerate(items):
        lines.append(
            f"{idx}. title={item.title!r} url={item.url!r} price={item.price!r} "
            f"rating={item.rating!r} snippet={item.snippet!r}"
        )
    return "\n".join(lines)


def _select_cheapest(items: list[ExtractedItem]) -> ExtractedItem | None:
    ranked: list[tuple[float, ExtractedItem]] = []
    for item in items:
        if not item.price:
            continue
        digits = "".join(ch for ch in item.price if ch.isdigit() or ch == ".")
        if not digits:
            continue
        try:
            ranked.append((float(digits), item))
        except ValueError:
            continue
    if not ranked:
        return None
    ranked.sort(key=lambda pair: pair[0])
    return ranked[0][1]


def _to_candidate(item: ExtractedItem) -> ResultCandidate:
    return ResultCandidate(
        title=item.title,
        url=item.url,
        price=item.price,
        rating=item.rating,
        snippet=item.snippet,
        fields=item.fields,
    )
