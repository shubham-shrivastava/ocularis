"""
Tests for logic/supervisor.py

Covers:
- MD5 stuck detection (threshold-based)
- SSIM stuck detection returning (bool, float)
- Recovery strategy cycling
- Circuit breaker (should_halt)
- Reset on success
- confirm_patterns check
"""
from __future__ import annotations

import io

import pytest
from PIL import Image

from core.schemas import ActionType, ComparisonMode, CriticResult, ActionRequest
from logic.supervisor import Supervisor
from tests.conftest import default_security_cfg, default_supervisor_cfg, make_state


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_supervisor(
    mode: ComparisonMode = ComparisonMode.md5,
    stuck_threshold: int = 3,
    max_retries: int = 3,
    confirm_patterns: list[str] | None = None,
) -> Supervisor:
    return Supervisor(
        comparison_mode=mode,
        supervisor_cfg=default_supervisor_cfg(
            stuck_threshold=stuck_threshold,
            max_retries=max_retries,
        ),
        security_cfg=default_security_cfg(confirm_patterns=confirm_patterns or []),
    )


def _solid_png(color=(100, 100, 100), size=(64, 64)) -> bytes:
    img = Image.new("RGB", size, color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# MD5 stuck detection
# ---------------------------------------------------------------------------


class TestMD5StuckDetection:
    def test_not_stuck_on_first_appearance(self):
        sup = _make_supervisor(stuck_threshold=3)
        state = make_state(b"page-a")
        stuck, _ = sup.pre_action_check(state)
        assert not stuck

    def test_stuck_at_threshold(self):
        sup = _make_supervisor(stuck_threshold=3)
        state = make_state(b"same-page")
        # History after each call: count(hash) before append
        # Call 1: count=0 (< 3) → not stuck, append → [h]
        # Call 2: count=1 (< 3) → not stuck, append → [h, h]
        # Call 3: count=2 (< 3) → not stuck, append → [h, h, h]
        # Call 4: count=3 (>= 3) → stuck!
        for _ in range(3):
            stuck, _ = sup.pre_action_check(state)
            assert not stuck
        stuck, reason = sup.pre_action_check(state)
        assert stuck
        assert reason == "stuck_md5"

    def test_different_states_not_stuck(self):
        sup = _make_supervisor(stuck_threshold=3)
        for i in range(10):
            stuck, _ = sup.pre_action_check(make_state(f"page-{i}".encode()))
            assert not stuck

    def test_stuck_reason_is_stuck_md5(self):
        sup = _make_supervisor(stuck_threshold=2)
        state = make_state(b"frozen-page")
        sup.pre_action_check(state)
        sup.pre_action_check(state)
        stuck, reason = sup.pre_action_check(state)
        assert stuck
        assert reason == "stuck_md5"


# ---------------------------------------------------------------------------
# SSIM stuck detection
# ---------------------------------------------------------------------------


class TestSSIMStuckDetection:
    def test_first_call_never_stuck(self):
        sup = _make_supervisor(mode=ComparisonMode.ssim)
        raw = _solid_png((100, 100, 100))
        state = make_state(raw)
        stuck, reason = sup.pre_action_check(state)
        assert not stuck
        assert reason is None

    def test_identical_screenshots_return_stuck_and_score(self):
        sup = _make_supervisor(mode=ComparisonMode.ssim)
        raw = _solid_png((100, 100, 100))
        state = make_state(raw)
        sup.pre_action_check(state)
        stuck, reason = sup.pre_action_check(state)
        assert stuck
        assert reason == "stuck_ssim"

    def test_different_screenshots_not_stuck(self):
        sup = _make_supervisor(mode=ComparisonMode.ssim)
        state_a = make_state(_solid_png((10, 10, 10)))
        state_b = make_state(_solid_png((200, 200, 200)))
        sup.pre_action_check(state_a)
        stuck, _ = sup.pre_action_check(state_b)
        assert not stuck

    def test_ssim_internal_returns_tuple_bool_float(self):
        sup = _make_supervisor(mode=ComparisonMode.ssim)
        raw = _solid_png((50, 50, 50))
        result = sup._is_stuck_ssim(raw)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], bool)
        assert isinstance(result[1], float)

    def test_ssim_score_is_high_for_identical(self):
        sup = _make_supervisor(mode=ComparisonMode.ssim)
        raw = _solid_png((100, 100, 100))
        sup._is_stuck_ssim(raw)
        stuck, score = sup._is_stuck_ssim(raw)
        assert stuck
        assert score >= 0.98


# ---------------------------------------------------------------------------
# Recovery strategy cycling
# ---------------------------------------------------------------------------


class TestRecovery:
    def test_cycles_through_order(self):
        sup = _make_supervisor()
        order = ["SCROLL_DOWN", "REFRESH_PAGE", "GO_BACK", "CLICK_OFFSET", "escalate"]
        for expected in order:
            assert sup.get_recovery_strategy() == expected

    def test_returns_escalate_when_exhausted(self):
        sup = _make_supervisor()
        for _ in range(10):
            sup.get_recovery_strategy()
        assert sup.get_recovery_strategy() == "escalate"

    def test_reset_restarts_cycle(self):
        sup = _make_supervisor()
        sup.get_recovery_strategy()
        sup.get_recovery_strategy()
        sup.reset()
        assert sup.get_recovery_strategy() == "SCROLL_DOWN"


# ---------------------------------------------------------------------------
# Circuit breaker
# ---------------------------------------------------------------------------


class TestCircuitBreaker:
    def test_not_halted_initially(self):
        sup = _make_supervisor(max_retries=3)
        assert not sup.should_halt()

    def test_halts_after_max_retries_no_change(self):
        sup = _make_supervisor(max_retries=2)
        pre = make_state(b"page")
        post = make_state(b"page")  # same hash -> no change
        critic = CriticResult(did_progress=False)

        for _ in range(2):
            sup.post_action_verify(pre, post, critic)

        assert sup.should_halt()

    def test_reset_clears_halt(self):
        sup = _make_supervisor(max_retries=1)
        pre = make_state(b"same")
        post = make_state(b"same")
        critic = CriticResult(did_progress=False)
        sup.post_action_verify(pre, post, critic)
        assert sup.should_halt()
        sup.reset()
        assert not sup.should_halt()


# ---------------------------------------------------------------------------
# post_action_verify
# ---------------------------------------------------------------------------


class TestPostActionVerify:
    def test_success_when_hash_differs_and_progress(self):
        from core.schemas import StepOutcome  # noqa: PLC0415

        sup = _make_supervisor()
        pre = make_state(b"before")
        post = make_state(b"after")
        critic = CriticResult(did_progress=True)
        outcome = sup.post_action_verify(pre, post, critic)
        assert outcome == StepOutcome.success

    def test_no_change_when_hashes_equal(self):
        from core.schemas import StepOutcome  # noqa: PLC0415

        sup = _make_supervisor()
        state = make_state(b"same-page")
        critic = CriticResult(did_progress=True)
        outcome = sup.post_action_verify(state, state, critic)
        assert outcome == StepOutcome.no_change

    def test_no_change_when_critic_says_no_progress(self):
        from core.schemas import StepOutcome  # noqa: PLC0415

        sup = _make_supervisor()
        pre = make_state(b"before")
        post = make_state(b"after_but_spinner")
        critic = CriticResult(did_progress=False, blocker_type="spinner")
        outcome = sup.post_action_verify(pre, post, critic)
        assert outcome == StepOutcome.no_change


# ---------------------------------------------------------------------------
# confirm_patterns
# ---------------------------------------------------------------------------


class TestConfirmPatterns:
    def test_matching_url_returns_confirm_required(self):
        sup = _make_supervisor(confirm_patterns=["/checkout", "/delete"])
        state = make_state(url="https://shop.example.com/checkout/payment")
        stuck, reason = sup.pre_action_check(state)
        assert not stuck
        assert reason == "confirm_required"

    def test_non_matching_url_passes(self):
        sup = _make_supervisor(confirm_patterns=["/checkout"])
        state = make_state(url="https://shop.example.com/browse")
        stuck, reason = sup.pre_action_check(state)
        assert reason != "confirm_required"
