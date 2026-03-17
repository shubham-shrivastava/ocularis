from __future__ import annotations

import base64
import hashlib
from unittest.mock import AsyncMock

import pytest

from core.model_client import MockModelClient
from core.run_session import RunSession
from core.schemas import AgentState, BrowserMode, ComparisonMode, RunRequest, RunStatus
from core.settings import load_settings
from logic.content_extractor import ExtractedContent, ExtractedItem
from logic.intent_resolver import IntentResolver
from logic.page_classifier import PageClassifier
from logic.reasoning_executor import ReasoningExecutor
from logic.text_reasoner import DisabledTextReasoner
from tests.test_orchestration import _make_mock_repo


class _FakeTextReasoner:
    available = True

    async def generate_json(self, *, system_prompt: str, user_prompt: str) -> dict | None:  # noqa: ARG002
        if "best candidate" in system_prompt:
            return {"index": 1, "answer_text": "Second item is the best match", "confidence": 0.91}
        return {
            "task_type": "extract_and_compare",
            "criteria": ["price"],
            "expected_result": "link",
            "target_count": 3,
            "search_query": "1TB SSD",
        }

    async def generate_text(self, *, system_prompt: str, user_prompt: str) -> str | None:  # noqa: ARG002
        return "Short summary"


def _state(url: str) -> AgentState:
    raw = f"state::{url}".encode()
    return AgentState(
        screenshot_b64=base64.b64encode(raw).decode(),
        screenshot_bytes=raw,
        url=url,
        state_hash=hashlib.md5(raw).hexdigest(),
    )


@pytest.mark.asyncio
async def test_intent_resolver_detects_compare_goal():
    resolver = IntentResolver()
    intent = await resolver.resolve("Search for the best 5 Samsung SSDs and give me the best link")
    assert intent.task_type == "extract_and_compare"
    assert intent.target_count == 5
    assert intent.expected_result == "link"


@pytest.mark.asyncio
async def test_page_classifier_detects_search_results():
    browser = AsyncMock()
    browser.get_page_title.return_value = "Search results"
    browser.get_visible_text.return_value = "Lots of products listed here"
    browser.get_page_structure_hints.return_value = {
        "candidate_count": 6,
        "paragraph_count": 1,
        "form_count": 0,
        "input_count": 1,
    }
    classifier = PageClassifier()
    context = await classifier.classify(_state("https://example.com/search?q=ssd"), browser)
    assert context.page_type == "search_results"


@pytest.mark.asyncio
async def test_reasoning_executor_uses_reasoner_selection():
    executor = ReasoningExecutor(_FakeTextReasoner())
    result = await executor.reason(
        content=ExtractedContent(
            items=[
                ExtractedItem(title="Item 1", url="https://example.com/1", price="$50"),
                ExtractedItem(title="Item 2", url="https://example.com/2", price="$40"),
            ]
        ),
        intent=await IntentResolver().resolve("Give me the best link"),
        user_goal="Give me the best link",
    )
    assert result is not None
    assert result.selected_item is not None
    assert result.selected_item.url == "https://example.com/2"
    assert result.answer is not None
    assert result.answer.items[1].url == "https://example.com/2"


@pytest.mark.asyncio
async def test_run_session_completes_via_context_aware_comparison():
    settings = load_settings()
    settings.context_aware.enabled = True

    request = RunRequest(
        goal="Find the best 3 SSDs and give me the best link",
        start_url="https://example.com/search?q=ssd",
        max_steps=2,
        comparison_mode=ComparisonMode.md5,
        use_memory=False,
        browser_mode=BrowserMode.launch,
        ephemeral=True,
    )
    session = RunSession(
        run_id="ctx-run-1",
        request=request,
        settings=settings,
        model_client=MockModelClient(total_steps=99),
        memory=AsyncMock(recall=AsyncMock(return_value=[]), store=AsyncMock()),
        text_reasoner=_FakeTextReasoner(),
    )
    repo = _make_mock_repo()

    states = iter(
        [
            _state("https://example.com/search?q=ssd"),
            _state("https://example.com/search?q=ssd"),
        ]
    )

    session.browser.get_page_title = AsyncMock(return_value="SSD search results")  # type: ignore[method-assign]
    session.browser.get_visible_text = AsyncMock(return_value="SSD search results page")  # type: ignore[method-assign]
    session.browser.get_page_structure_hints = AsyncMock(  # type: ignore[method-assign]
        return_value={"candidate_count": 6, "paragraph_count": 1, "form_count": 0, "input_count": 1}
    )
    session.browser.extract_candidate_items = AsyncMock(  # type: ignore[method-assign]
        return_value=[
            {"title": "SSD A", "url": "https://example.com/a", "price": "$90", "rating": "4.5/5", "snippet": "A"},
            {"title": "SSD B", "url": "https://example.com/b", "price": "$80", "rating": "4.8/5", "snippet": "B"},
            {"title": "SSD C", "url": "https://example.com/c", "price": "$85", "rating": "4.6/5", "snippet": "C"},
        ]
    )

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(session.browser, "launch", AsyncMock())
        mp.setattr(session.browser, "close", AsyncMock())
        mp.setattr(session.browser, "get_state", AsyncMock(side_effect=lambda: next(states)))
        mp.setattr(session.browser, "execute_action", AsyncMock(return_value=True))
        await session.run_loop(repo)

    assert session.status == RunStatus.completed
    result = session.build_result()
    assert result is not None
    assert result.answer is not None
    assert result.answer.link == "https://example.com/b"
    assert len(result.answer.items) == 3
    assert session.build_comparison_state() is not None
