"""
Tests for logic/memory.py

Covers:
- NullMemory always returns empty list, store is no-op
- EpisodicMemory.store is no-op in ephemeral mode
- EpisodicMemory._build_summary produces expected text
- EpisodicMemory.recall handles OpenAI errors gracefully
"""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock

from core.schemas import ActionRequest, ActionType, StepOutcome
from logic.memory import EpisodicMemory, NullMemory
from tests.conftest import make_state


# ---------------------------------------------------------------------------
# NullMemory
# ---------------------------------------------------------------------------


class TestNullMemory:
    @pytest.mark.asyncio
    async def test_recall_returns_empty(self):
        mem = NullMemory()
        result = await mem.recall("https://example.com", "buy ticket")
        assert result == []

    @pytest.mark.asyncio
    async def test_store_is_noop(self):
        from core.schemas import StepTrace, StepOutcome  # noqa: PLC0415

        mem = NullMemory()
        trace = MagicMock()
        # Should not raise
        await mem.store(trace, "some goal")


# ---------------------------------------------------------------------------
# EpisodicMemory
# ---------------------------------------------------------------------------


def _make_step_trace():
    from core.schemas import StepTrace, StepOutcome  # noqa: PLC0415

    pre = make_state(b"pre-state", url="https://example.com/search")
    post = make_state(b"post-state", url="https://example.com/results")
    action = ActionRequest(action_type=ActionType.click, params={"x": 100, "y": 200})
    return StepTrace(
        step_number=1,
        pre_state=pre,
        post_state=post,
        action=action,
        outcome=StepOutcome.success,
        duration_ms=500,
        critic_analysis="did_progress=True blocker=None",
    )


def _make_memory(ephemeral: bool = False) -> tuple[EpisodicMemory, MagicMock]:
    repo = MagicMock()
    repo._session = AsyncMock()
    repo.find_similar_steps = AsyncMock(return_value=[])

    openai_client = AsyncMock()
    openai_client.embeddings = AsyncMock()
    openai_client.embeddings.create = AsyncMock(
        return_value=MagicMock(data=[MagicMock(embedding=[0.1] * 1536)])
    )

    cfg = MagicMock()
    cfg.embedding_model = "text-embedding-3-small"
    cfg.top_k = 5
    cfg.similarity_threshold = 0.75

    mem = EpisodicMemory(repo, openai_client, cfg, ephemeral=ephemeral)
    return mem, openai_client


class TestEpisodicMemory:
    def test_build_summary_contains_key_fields(self):
        trace = _make_step_trace()
        summary = EpisodicMemory._build_summary(trace, "search for flights")
        assert "search for flights" in summary
        assert "https://example.com/search" in summary
        assert "click" in summary
        assert "success" in summary

    @pytest.mark.asyncio
    async def test_store_is_noop_in_ephemeral_mode(self):
        mem, openai_client = _make_memory(ephemeral=True)
        trace = _make_step_trace()
        await mem.store(trace, "test goal")
        # Should not call OpenAI at all
        openai_client.embeddings.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_recall_calls_embed_and_repo(self):
        mem, openai_client = _make_memory()
        result = await mem.recall("https://example.com", "find hotels")
        openai_client.embeddings.create.assert_called_once()
        mem._repo.find_similar_steps.assert_called_once()
        assert result == []  # repo returned empty list

    @pytest.mark.asyncio
    async def test_recall_returns_empty_on_openai_error(self):
        mem, openai_client = _make_memory()
        openai_client.embeddings.create.side_effect = RuntimeError("API down")
        result = await mem.recall("https://example.com", "goal")
        assert result == []

    @pytest.mark.asyncio
    async def test_store_is_noop_in_ephemeral_mode(self):
        mem, openai_client = _make_memory(ephemeral=True)
        trace = _make_step_trace()
        await mem.store(trace, "goal", run_id="run-1", step_number=1)
        openai_client.embeddings.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_store_skips_without_run_id(self):
        mem, openai_client = _make_memory(ephemeral=False)
        trace = _make_step_trace()
        await mem.store(trace, "goal")
        openai_client.embeddings.create.assert_not_called()
