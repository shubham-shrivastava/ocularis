from __future__ import annotations

import re
from collections import deque
from typing import TYPE_CHECKING

from loguru import logger

from core.schemas import AgentState, ComparisonMode, CriticResult, StepOutcome
from core.settings import SupervisorSettings, SecuritySettings

if TYPE_CHECKING:
    pass


class Supervisor:
    """
    Per-run reliability guard.

    Detects visual stagnation (MD5 hash comparison for speed, SSIM for
    perceptual accuracy), manages recovery strategy cycling, and fires the
    circuit breaker when retries are exhausted.

    One instance per RunSession -- never shared across concurrent runs.
    """

    def __init__(
        self,
        comparison_mode: ComparisonMode,
        supervisor_cfg: SupervisorSettings,
        security_cfg: SecuritySettings,
    ) -> None:
        self._mode = comparison_mode
        self._cfg = supervisor_cfg
        self._security = security_cfg

        self._hash_history: deque[str] = deque(maxlen=supervisor_cfg.history_size)
        # Only populated when mode == ssim
        self._screenshot_history: deque[bytes] = deque(maxlen=2)

        self._retry_count: int = 0
        self._recovery_index: int = 0

    # ------------------------------------------------------------------
    # Public API consumed by RunSession
    # ------------------------------------------------------------------

    def pre_action_check(self, state: AgentState) -> tuple[bool, str | None]:
        """
        Check state before executing an action.

        Returns:
            (is_stuck: bool, reason: str | None)
            reason is None when healthy, "confirm_required" when URL matches a
            sensitive pattern, or "stuck_md5"/"stuck_ssim" when stagnation detected.
        """
        # Sensitive URL check takes priority
        confirm_reason = self._check_confirm_patterns(state.url)
        if confirm_reason:
            return False, confirm_reason

        stuck, reason = self._detect_stuck(state)
        return stuck, reason

    def post_action_verify(
        self,
        pre: AgentState,
        post: AgentState,
        critic_result: CriticResult,
    ) -> StepOutcome:
        """
        Evaluate whether an action made progress.

        Uses the structured CriticResult from model_client.critique() along with
        the hash/bytes comparison to determine outcome.
        """
        hashes_differ = pre.state_hash != post.state_hash

        if hashes_differ and critic_result.did_progress:
            self._retry_count = 0
            self._recovery_index = 0
            logger.debug("Step verified: progress made")
            return StepOutcome.success

        if not hashes_differ:
            logger.info("Post-action state unchanged (no_change)", url=post.url)
            self._retry_count += 1
            return StepOutcome.no_change

        # Hash differs but critic says no progress (e.g. spinner replaced spinner).
        # The page DID change visually, so don't count this toward the circuit breaker.
        # Only truly stuck screens (identical hash) should trigger escalation.
        logger.info(
            "State changed but critic sees no progress",
            blocker=critic_result.blocker_type,
        )
        return StepOutcome.no_change

    def get_recovery_strategy(self) -> str:
        """
        Return the next recovery action to try.

        Cycles through recovery_order; returns "escalate" when exhausted.
        """
        order = self._cfg.recovery_order
        if self._recovery_index >= len(order):
            return "escalate"
        strategy = order[self._recovery_index]
        self._recovery_index += 1
        logger.info("Recovery strategy selected", strategy=strategy, index=self._recovery_index)
        return strategy

    def should_halt(self) -> bool:
        """Return True when the circuit breaker threshold is reached."""
        return self._retry_count >= self._cfg.max_retries

    def reset(self) -> None:
        """Reset retry and recovery state on a successful step."""
        self._retry_count = 0
        self._recovery_index = 0

    # ------------------------------------------------------------------
    # Internal detection
    # ------------------------------------------------------------------

    def _detect_stuck(self, state: AgentState) -> tuple[bool, str | None]:
        if self._mode == ComparisonMode.ssim:
            stuck, score = self._is_stuck_ssim(state.screenshot_bytes)
            return stuck, "stuck_ssim" if stuck else None
        stuck = self._is_stuck_md5(state.state_hash)
        return stuck, "stuck_md5" if stuck else None

    def _is_stuck_md5(self, current_hash: str) -> bool:
        """
        True when the same hash appears >= stuck_threshold times in history.
        Always appends hash for next call.
        """
        count = self._hash_history.count(current_hash)
        self._hash_history.append(current_hash)
        stuck = count >= self._cfg.stuck_threshold
        if stuck:
            logger.warning("Stuck detected (MD5)", hash=current_hash, count=count + 1)
        return stuck

    def _is_stuck_ssim(self, screenshot_bytes: bytes) -> tuple[bool, float]:
        """
        Compare current screenshot to last via SSIM.

        Returns (is_stuck, ssim_score). The caller gets the boolean decision
        and the raw score for logging. Score is 0.0 when there's nothing to
        compare yet (first call).
        """
        import io

        import numpy as np
        from PIL import Image
        from skimage.metrics import structural_similarity as ssim

        self._screenshot_history.append(screenshot_bytes)

        if len(self._screenshot_history) < 2:
            return False, 0.0

        def _to_gray(b: bytes) -> np.ndarray:
            img = Image.open(io.BytesIO(b)).convert("L")
            return np.array(img)

        prev_gray = _to_gray(self._screenshot_history[-2])
        curr_gray = _to_gray(self._screenshot_history[-1])

        if prev_gray.shape != curr_gray.shape:
            curr_gray = np.array(
                Image.fromarray(curr_gray).resize(
                    (prev_gray.shape[1], prev_gray.shape[0])
                )
            )

        score: float = float(ssim(prev_gray, curr_gray, data_range=255))
        stuck = score >= self._cfg.ssim_similarity_floor

        if stuck:
            logger.warning("Stuck detected (SSIM)", score=score, floor=self._cfg.ssim_similarity_floor)
        else:
            logger.debug("SSIM check passed", score=score)

        return stuck, score

    def _check_confirm_patterns(self, url: str) -> str | None:
        """Return 'confirm_required' if URL matches any sensitive pattern."""
        for pattern in self._security.confirm_patterns:
            if re.search(pattern, url):
                logger.info("Confirm pattern matched", url=url, pattern=pattern)
                return "confirm_required"
        return None
