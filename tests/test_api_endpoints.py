from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
import pytest_asyncio

from api.main import _get_repo, app
from core.model_client import MockModelClient
from core.schemas import (
    ComparisonState,
    ResultCandidate,
    RunAnswer,
    RunDetailResponse,
    RunResult,
    RunStatus,
)
from core.settings import load_settings
from logic.text_reasoner import DisabledTextReasoner


class _FakeRepo:
    def __init__(self) -> None:
        self.create_run = AsyncMock()
        self.list_runs = AsyncMock(return_value=[])
        self.get_run_detail = AsyncMock()
        self.get_step_screenshot_path = AsyncMock()
        self._session = SimpleNamespace(commit=AsyncMock())


@pytest.fixture
def fake_repo():
    return _FakeRepo()


@pytest_asyncio.fixture
async def api_client(fake_repo):
    settings = load_settings()
    app.state.settings = settings
    app.state.model_client = MockModelClient()
    app.state.planner = None
    app.state.text_reasoner = DisabledTextReasoner()
    app.state.memory_enabled = False
    app.dependency_overrides[_get_repo] = lambda: fake_repo
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_create_run_persists_and_returns_run_id(api_client, fake_repo, monkeypatch):
    async def _register(session):  # noqa: ANN001
        return None

    monkeypatch.setattr("api.main.RunRegistry.new_run_id", staticmethod(lambda: "run-123"))
    monkeypatch.setattr("api.main.registry.register", AsyncMock(side_effect=_register))
    monkeypatch.setattr("api.main._run_with_new_session", AsyncMock())

    def _fake_create_task(coro, name=None):  # noqa: ANN001, ARG001
        coro.close()
        return None

    monkeypatch.setattr("api.main.asyncio.create_task", _fake_create_task)

    response = await api_client.post(
        "/run",
        json={
            "goal": "find the best link",
            "start_url": "https://example.com",
            "ephemeral": True,
        },
    )

    assert response.status_code == 202
    assert response.json()["run_id"] == "run-123"
    fake_repo.create_run.assert_awaited_once()
    fake_repo._session.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_get_run_returns_live_result_and_comparison_state(api_client, fake_repo, monkeypatch):
    fake_repo.get_run_detail.return_value = RunDetailResponse(
        run_id="run-abc",
        goal="compare ssds",
        start_url="https://example.com",
        status=RunStatus.running,
        comparison_mode="md5",
        ephemeral=True,
        steps=[],
        goal_verdict=None,
    )
    session = SimpleNamespace(
        sub_steps=["search", "compare"],
        _current_sub_step=1,
        waiting_reason=None,
        build_result=lambda: RunResult(
            summary="selected link",
            final_url="https://example.com/results",
            collected_count=3,
            candidates=[],
            answer=RunAnswer(
                result_type="link",
                link="https://example.com/best",
                text="Best SSD",
                items=[ResultCandidate(title="SSD B", url="https://example.com/best")],
                confidence=0.92,
            ),
        ),
        build_comparison_state=lambda: ComparisonState(
            target_count=3,
            collected_count=3,
            collected_items=[ResultCandidate(title="SSD A", url="https://example.com/a")],
            compared_items=[ResultCandidate(title="SSD B", url="https://example.com/best")],
            selected_item=ResultCandidate(title="SSD B", url="https://example.com/best"),
            status="answered",
        ),
    )
    monkeypatch.setattr("api.main.registry.get", AsyncMock(return_value=session))

    response = await api_client.get("/runs/run-abc")

    assert response.status_code == 200
    payload = response.json()
    assert payload["result"]["answer"]["link"] == "https://example.com/best"
    assert payload["comparison_state"]["target_count"] == 3
    assert payload["current_sub_step"] == 1


@pytest.mark.asyncio
async def test_screenshot_endpoint_serves_jpeg(api_client, fake_repo, tmp_path: Path):
    screenshot = tmp_path / "step_0001_post.jpg"
    screenshot.write_bytes(b"jpeg-bytes")
    fake_repo.get_step_screenshot_path.return_value = screenshot

    response = await api_client.get("/runs/run-1/steps/1/screenshot?phase=post")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/jpeg")
    assert response.content == b"jpeg-bytes"
