"""
Tests for core/run_session.py

Covers:
- Two concurrent RunSessions don't share state
- Pause/resume works correctly
- Intervention queue delivers actions
- RunRegistry register/get/remove
"""
from __future__ import annotations

import asyncio

import pytest

from core.model_client import MockModelClient
from core.run_session import RunRegistry, RunSession
from core.schemas import ActionRequest, ActionType, BrowserMode, ComparisonMode, RunRequest, RunStatus
from core.settings import load_settings
from logic.memory import NullMemory


def _make_request(**overrides) -> RunRequest:
    data = dict(
        goal="test goal",
        start_url="https://example.com",
        max_steps=3,
        comparison_mode=ComparisonMode.md5,
        use_memory=False,
        browser_mode=BrowserMode.launch,
        ephemeral=True,
    )
    data.update(overrides)
    return RunRequest.model_validate(data)


def _make_session(run_id: str = "test-run-1") -> RunSession:
    settings = load_settings()
    return RunSession(
        run_id=run_id,
        request=_make_request(),
        settings=settings,
        model_client=MockModelClient(total_steps=3),
        memory=NullMemory(),
    )


class TestRunSessionIsolation:
    def test_two_sessions_have_independent_state(self):
        s1 = _make_session("run-a")
        s2 = _make_session("run-b")

        s1.steps.append(object())  # type: ignore[arg-type]
        assert len(s2.steps) == 0

    def test_supervisors_are_independent(self):
        s1 = _make_session("run-a")
        s2 = _make_session("run-b")
        assert s1.supervisor is not s2.supervisor

    def test_browsers_are_independent(self):
        s1 = _make_session("run-a")
        s2 = _make_session("run-b")
        assert s1.browser is not s2.browser


class TestPauseResume:
    @pytest.mark.asyncio
    async def test_pause_sets_status_and_clears_event(self):
        s = _make_session()
        assert s._pause_event.is_set()
        await s.pause()
        assert s.status == RunStatus.paused
        assert not s._pause_event.is_set()

    @pytest.mark.asyncio
    async def test_resume_sets_status_and_event(self):
        s = _make_session()
        await s.pause()
        await s.resume()
        assert s.status == RunStatus.running
        assert s._pause_event.is_set()


class TestInterventionQueue:
    @pytest.mark.asyncio
    async def test_intervene_delivers_action(self):
        s = _make_session()
        action = ActionRequest(action_type=ActionType.click, params={"x": 10, "y": 20})
        await s.intervene(action)
        received = await asyncio.wait_for(s._intervention_queue.get(), timeout=1.0)
        assert received.action_type == ActionType.click

    @pytest.mark.asyncio
    async def test_wait_for_intervention_returns_action(self):
        s = _make_session()
        action = ActionRequest(action_type=ActionType.key_press, params={"key": "Enter"})

        async def _put():
            await asyncio.sleep(0.05)
            await s.intervene(action)

        asyncio.create_task(_put())
        result = await s._wait_for_intervention(timeout=2.0)
        assert result is not None
        assert result.action_type == ActionType.key_press

    @pytest.mark.asyncio
    async def test_wait_for_intervention_timeout(self):
        s = _make_session()
        result = await s._wait_for_intervention(timeout=0.05)
        assert result is None
        assert s.status == RunStatus.failed


class TestRunRegistry:
    @pytest.mark.asyncio
    async def test_register_and_get(self):
        reg = RunRegistry()
        s = _make_session("reg-test")
        await reg.register(s)
        assert await reg.get("reg-test") is s

    @pytest.mark.asyncio
    async def test_remove(self):
        reg = RunRegistry()
        s = _make_session("reg-rm")
        await reg.register(s)
        await reg.remove("reg-rm")
        assert await reg.get("reg-rm") is None

    @pytest.mark.asyncio
    async def test_active_count(self):
        reg = RunRegistry()
        s1 = _make_session("r1")
        s2 = _make_session("r2")
        await reg.register(s1)
        await reg.register(s2)
        assert reg.active_count() == 2
        await reg.remove("r1")
        assert reg.active_count() == 1

    def test_new_run_id_is_unique(self):
        ids = {RunRegistry.new_run_id() for _ in range(100)}
        assert len(ids) == 100


class TestStrictValidation:
    def test_extract_requested_product_count(self):
        s = _make_session()
        assert s._extract_requested_product_count("give me best 3 samsung ssds") == 3
        assert s._extract_requested_product_count("find top 5 laptops") == 5
        assert s._extract_requested_product_count("buy one product") == 1

    @pytest.mark.asyncio
    async def test_irrelevant_hdd_product_page_is_rejected_for_ssd_goal(self):
        s = _make_session()
        s.request.goal = "Search for a branded 1TB SSD and open the product details page"
        state = type("State", (), {"url": "https://www.amazon.in/gp/aw/d/B08ZJFYB9J"})()

        async def _title():
            return "Seagate Expansion 5TB External HDD"

        async def _text():
            return "Seagate Expansion 5TB External HDD portable hard drive"

        s.browser.get_page_title = _title  # type: ignore[method-assign]
        s.browser.get_visible_text = _text  # type: ignore[method-assign]

        valid, reason = await s._strict_sub_step_validation(
            "Click on the first product listed to view details",
            state,  # type: ignore[arg-type]
        )
        assert valid is False
        assert reason == "irrelevant_product_page"

    @pytest.mark.asyncio
    async def test_duplicate_product_page_is_rejected(self):
        s = _make_session()
        s.request.goal = "Search for the best 3 Samsung SSDs and inspect product details"
        s._target_product_count = 3
        sig = "https://www.amazon.in/dp/demo::samsung 990 pro ssd 1tb"
        s._collected_product_signatures.add(sig)
        s._collected_product_titles.append("Samsung 990 PRO SSD 1TB")
        s._current_product_signature = None
        state = type("State", (), {"url": "https://www.amazon.in/dp/demo"})()

        async def _title():
            return "Samsung 990 PRO SSD 1TB"

        async def _text():
            return "Samsung 990 PRO SSD 1TB PCIe 4.0 Internal Solid State Drive"

        s.browser.get_page_title = _title  # type: ignore[method-assign]
        s.browser.get_visible_text = _text  # type: ignore[method-assign]

        valid, reason = await s._strict_sub_step_validation(
            "Click a different relevant product title that has not been opened yet",
            state,  # type: ignore[arg-type]
        )
        assert valid is False
        assert reason == "duplicate_product_page"

    def test_build_result_includes_collected_candidates(self):
        s = _make_session()
        s.request.goal = "Give me the best 3 Samsung SSDs"
        s._target_product_count = 3
        s._collected_products = [
            {"title": "Samsung 990 PRO SSD 1TB", "url": "https://example.com/p1", "signature": "a"},
            {"title": "Samsung T7 SSD 1TB", "url": "https://example.com/p2", "signature": "b"},
        ]
        result = s.build_result()
        assert result is not None
        assert result.requested_count == 3
        assert result.collected_count == 2
        assert [c.title for c in result.candidates] == [
            "Samsung 990 PRO SSD 1TB",
            "Samsung T7 SSD 1TB",
        ]
