"""Tests for logic/goal_evaluator.py"""
from __future__ import annotations

import pytest

from core.model_client import MockModelClient
from core.settings import GoalEvaluatorSettings
from logic.goal_evaluator import GoalEvaluator
from tests.conftest import make_state


@pytest.fixture
def evaluator():
    model = MockModelClient(total_steps=3)
    cfg = GoalEvaluatorSettings(confidence_threshold=0.8)
    return GoalEvaluator(model, cfg)


@pytest.mark.asyncio
async def test_not_achieved_before_final_step(evaluator):
    state = make_state()
    verdict = await evaluator.evaluate(state, "buy a ticket", steps_taken=1)
    assert verdict.status == "not_achieved"
    assert not evaluator.is_achieved(verdict)


@pytest.mark.asyncio
async def test_achieved_at_final_step(evaluator):
    state = make_state()
    verdict = await evaluator.evaluate(state, "buy a ticket", steps_taken=3)
    assert verdict.status == "achieved"
    assert evaluator.is_achieved(verdict)


@pytest.mark.asyncio
async def test_is_achieved_respects_confidence_threshold():
    from core.schemas import GoalVerdict  # noqa: PLC0415

    model = MockModelClient(total_steps=1)
    cfg = GoalEvaluatorSettings(confidence_threshold=0.9)
    ev = GoalEvaluator(model, cfg)

    low_confidence = GoalVerdict(status="achieved", confidence=0.5, reasoning="low")
    assert not ev.is_achieved(low_confidence)

    high_confidence = GoalVerdict(status="achieved", confidence=0.95, reasoning="high")
    assert ev.is_achieved(high_confidence)


@pytest.mark.asyncio
async def test_evaluate_returns_uncertain_on_error():
    """If model raises, GoalEvaluator should return uncertain instead of propagating."""
    from unittest.mock import AsyncMock  # noqa: PLC0415

    mock_model = AsyncMock()
    mock_model.evaluate_goal.side_effect = RuntimeError("model offline")
    cfg = GoalEvaluatorSettings(confidence_threshold=0.8)
    ev = GoalEvaluator(mock_model, cfg)

    verdict = await ev.evaluate(make_state(), "search for flights", steps_taken=2)
    assert verdict.status == "uncertain"
    assert verdict.confidence == 0.0
