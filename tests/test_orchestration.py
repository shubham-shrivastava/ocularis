"""
Integration test: full 3-step orchestration loop with MockModelClient.

This test mocks the browser and DB layer to keep it pure Python --
no real Playwright or PostgreSQL needed. Verifies:
- Step traces are produced and contain correct data
- Run status transitions from queued -> running -> completed
- Goal evaluation terminates the loop after total_steps
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.model_client import MockModelClient
from core.run_session import RunSession
from core.schemas import (
    ActionRequest,
    ActionType,
    AgentState,
    BrowserMode,
    ComparisonMode,
    RunRequest,
    RunStatus,
    StepOutcome,
)
from core.settings import load_settings
from logic.memory import NullMemory


def _fake_state(seed: bytes, url: str = "https://example.com") -> AgentState:
    raw = seed + b"\x00" * 100  # make it look distinct
    b64 = base64.b64encode(raw).decode()
    return AgentState(
        screenshot_b64=b64,
        screenshot_bytes=raw,
        url=url,
        state_hash=hashlib.md5(raw).hexdigest(),
    )


def _make_mock_repo():
    repo = AsyncMock()
    repo.create_run = AsyncMock()
    repo.update_run_status = AsyncMock()
    repo.complete_run = AsyncMock()
    repo.log_step = AsyncMock()
    repo._session = AsyncMock()
    repo._session.commit = AsyncMock()
    return repo


class _FakePlanner:
    def __init__(self, steps: list[str]) -> None:
        self._steps = steps

    async def decompose(self, goal: str, start_url: str) -> list[str]:  # noqa: ARG002
        return self._steps


class _RecoveryModelClient:
    def __init__(self) -> None:
        self.predict_calls = 0
        self.recovery_calls = 0

    async def predict(self, state, goal, history, episodes):  # noqa: ANN001, ARG002
        self.predict_calls += 1
        return ActionRequest(action_type=ActionType.wait, params={"duration_ms": 200})

    async def predict_recovery_action(
        self,
        state,
        goal,
        sub_step,
        history,
        episodes,
        failure_reason,
        last_action,
    ):  # noqa: ANN001, ARG002
        self.recovery_calls += 1
        return ActionRequest(action_type=ActionType.click, params={"x": 400, "y": 120})

    async def critique(self, pre_state, post_state, action, goal):  # noqa: ANN001, ARG002
        return MagicMock(did_progress=True, blocker_type=None)

    async def evaluate_goal(self, state, goal, steps_taken):  # noqa: ANN001, ARG002
        if steps_taken >= 2:
            return MagicMock(status="achieved", confidence=0.95, reasoning="done")
        return MagicMock(status="not_achieved", confidence=0.2, reasoning="not done")


@pytest.mark.asyncio
async def test_full_3_step_loop():
    settings = load_settings()
    request = RunRequest(
        goal="search for flights",
        start_url="https://example.com",
        max_steps=5,
        comparison_mode=ComparisonMode.md5,
        use_memory=False,
        browser_mode=BrowserMode.launch,
        ephemeral=True,
    )

    # Scripted model: 3 distinct actions
    model = MockModelClient(
        action_sequence=[
            ActionRequest(action_type=ActionType.click, params={"x": 100, "y": 200}),
            ActionRequest(action_type=ActionType.type, params={"text": "Paris"}),
            ActionRequest(action_type=ActionType.click, params={"x": 300, "y": 400}),
        ],
        total_steps=3,
    )

    session = RunSession(
        run_id="test-orch-001",
        request=request,
        settings=settings,
        model_client=model,
        memory=NullMemory(),
    )

    repo = _make_mock_repo()

    state_counter = [0]

    def _next_state():
        state_counter[0] += 1
        return _fake_state(f"step-{state_counter[0]}".encode())

    with patch.object(session.browser, "launch", AsyncMock()):
        with patch.object(session.browser, "close", AsyncMock()):
            with patch.object(
                session.browser,
                "get_state",
                side_effect=_next_state,
            ):
                with patch.object(session.browser, "execute_action", AsyncMock(return_value=True)):
                    await session.run_loop(repo)

    # Run should complete (goal achieved after 3 steps)
    assert session.status == RunStatus.completed
    assert session.goal_verdict is not None
    assert session.goal_verdict.status == "achieved"

    # Verify at least 3 step traces were created
    assert len(session.steps) >= 3

    # Verify steps have correct structure
    first_step = session.steps[0]
    assert first_step.step_number == 1
    assert first_step.action.action_type == ActionType.click


@pytest.mark.asyncio
async def test_max_steps_exceeded():
    settings = load_settings()
    request = RunRequest(
        goal="impossible task",
        start_url="https://example.com",
        max_steps=2,
        ephemeral=True,
    )

    # MockModelClient with total_steps=999 means goal is never achieved in 2 steps
    model = MockModelClient(total_steps=999)
    session = RunSession(
        run_id="test-max-steps",
        request=request,
        settings=settings,
        model_client=model,
        memory=NullMemory(),
    )

    state_counter = [0]

    def _next_state():
        state_counter[0] += 1
        return _fake_state(f"step-{state_counter[0]}".encode())

    repo = _make_mock_repo()

    with patch.object(session.browser, "launch", AsyncMock()):
        with patch.object(session.browser, "close", AsyncMock()):
            with patch.object(session.browser, "get_state", side_effect=_next_state):
                with patch.object(session.browser, "execute_action", AsyncMock(return_value=True)):
                    await session.run_loop(repo)

    assert session.status == RunStatus.max_steps_exceeded


@pytest.mark.asyncio
async def test_dropdown_postcondition_advances_without_goal_verdict():
    settings = load_settings()
    request = RunRequest(
        goal="sort results",
        start_url="https://example.com/results",
        max_steps=2,
        comparison_mode=ComparisonMode.md5,
        use_memory=False,
        browser_mode=BrowserMode.launch,
        ephemeral=True,
    )
    planner = _FakePlanner(
        [
            "Click the 'Sort by: Featured' dropdown menu",
            "Review the sorted results",
        ]
    )
    model = MockModelClient(total_steps=2)
    session = RunSession(
        run_id="test-dropdown-postcondition",
        request=request,
        settings=settings,
        model_client=model,
        memory=NullMemory(),
        planner=planner,
    )
    repo = _make_mock_repo()

    states = iter(
        [
            _fake_state(b"step1-pre", "https://example.com/results"),
            _fake_state(b"step1-post", "https://example.com/results"),
            _fake_state(b"step2-pre", "https://example.com/results"),
            _fake_state(b"step2-post", "https://example.com/results"),
        ]
    )

    with patch.object(session.browser, "launch", AsyncMock()):
        with patch.object(session.browser, "close", AsyncMock()):
            with patch.object(session.browser, "get_state", side_effect=lambda: next(states)):
                with patch.object(session.browser, "execute_action", AsyncMock(return_value=True)):
                    with patch.object(session.browser, "is_dropdown_open", AsyncMock(return_value=True)):
                        await session.run_loop(repo)

    assert session.status == RunStatus.completed
    assert "postcondition=dropdown_opened" in session.steps[0].critic_analysis
    assert session.steps[0].action.params.get("_deterministic") == "dropdown_trigger_open"


@pytest.mark.asyncio
async def test_search_box_focus_step_uses_deterministic_click():
    settings = load_settings()
    request = RunRequest(
        goal="search for SSDs",
        start_url="https://example.com",
        max_steps=2,
        comparison_mode=ComparisonMode.md5,
        use_memory=False,
        browser_mode=BrowserMode.launch,
        ephemeral=True,
    )
    planner = _FakePlanner(["Click the search box at the top of the page"])
    model = MockModelClient(total_steps=2)
    session = RunSession(
        run_id="test-search-focus",
        request=request,
        settings=settings,
        model_client=model,
        memory=NullMemory(),
        planner=planner,
    )
    repo = _make_mock_repo()

    states = iter(
        [
            _fake_state(b"step1-pre", "https://example.com"),
            _fake_state(b"step1-post", "https://example.com"),
        ]
    )

    with patch.object(session.browser, "launch", AsyncMock()):
        with patch.object(session.browser, "close", AsyncMock()):
            with patch.object(session.browser, "get_state", side_effect=lambda: next(states)):
                with patch.object(session.browser, "execute_action", AsyncMock(return_value=True)):
                    with patch.object(session.browser, "get_search_box_selector", AsyncMock(return_value="#search")):
                        with patch.object(session.browser, "is_search_box_focused", AsyncMock(return_value=True)):
                            await session.run_loop(repo)

    assert session.status == RunStatus.completed
    assert session.steps[0].action.params.get("_deterministic") == "search_box_focus"
    assert "postcondition=search_box_focused" in session.steps[0].critic_analysis


@pytest.mark.asyncio
async def test_recovery_model_used_when_enter_does_not_submit_search():
    settings = load_settings()
    request = RunRequest(
        goal="search for 1TB SSD",
        start_url="https://amazon.in",
        max_steps=3,
        comparison_mode=ComparisonMode.md5,
        use_memory=False,
        browser_mode=BrowserMode.launch,
        ephemeral=True,
    )
    planner = _FakePlanner(
        [
            "Type '1TB SSD' in the search box and press Enter",
        ]
    )
    model = _RecoveryModelClient()
    session = RunSession(
        run_id="test-search-recovery",
        request=request,
        settings=settings,
        model_client=model,
        memory=NullMemory(),
        planner=planner,
    )
    repo = _make_mock_repo()

    states = iter(
        [
            _fake_state(b"search-pre", "https://amazon.in"),
            _fake_state(b"search-post-suggestions", "https://amazon.in"),
            _fake_state(b"search-pre-recovery", "https://amazon.in"),
            _fake_state(b"search-post-results", "https://amazon.in/s?k=1TB+SSD"),
        ]
    )

    with patch.object(session.browser, "launch", AsyncMock()):
        with patch.object(session.browser, "close", AsyncMock()):
            with patch.object(session.browser, "get_state", side_effect=lambda: next(states)):
                with patch.object(session.browser, "execute_action", AsyncMock(return_value=True)):
                    await session.run_loop(repo)

    assert session.status == RunStatus.completed
    assert model.recovery_calls == 1
    assert session.steps[0].action.action_type == ActionType.type
    assert session.steps[1].action.action_type == ActionType.click
