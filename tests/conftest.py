from __future__ import annotations

import base64
import hashlib
from unittest.mock import AsyncMock, MagicMock

import pytest

from core.schemas import (
    ActionRequest,
    ActionType,
    AgentState,
    ComparisonMode,
)
from core.settings import (
    GoalEvaluatorSettings,
    SecuritySettings,
    SupervisorSettings,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _make_screenshot(content: bytes = b"fake-png-data") -> tuple[bytes, str]:
    """Return (raw_bytes, base64_string) for a fake screenshot."""
    b64 = base64.b64encode(content).decode()
    return content, b64


def make_state(content: bytes = b"fake-png-data", url: str = "https://example.com") -> AgentState:
    raw, b64 = _make_screenshot(content)
    return AgentState(
        screenshot_b64=b64,
        screenshot_bytes=raw,
        url=url,
        state_hash=hashlib.md5(raw).hexdigest(),
    )


def default_supervisor_cfg(**overrides) -> SupervisorSettings:
    data = dict(
        stuck_threshold=3,
        history_size=5,
        ssim_similarity_floor=0.98,
        max_retries=3,
        recovery_order=["SCROLL_DOWN", "REFRESH_PAGE", "GO_BACK", "CLICK_OFFSET", "escalate"],
    )
    data.update(overrides)
    return SupervisorSettings.model_validate(data)


def default_security_cfg(**overrides) -> SecuritySettings:
    data = dict(
        allowed_domains=[],
        confirm_patterns=[],
        block_password_fields=True,
        cdp_host="127.0.0.1",
        cdp_port=9222,
    )
    data.update(overrides)
    return SecuritySettings.model_validate(data)


@pytest.fixture
def supervisor_cfg():
    return default_supervisor_cfg()


@pytest.fixture
def security_cfg():
    return default_security_cfg()


@pytest.fixture
def goal_evaluator_cfg():
    return GoalEvaluatorSettings(confidence_threshold=0.8)


@pytest.fixture
def click_action():
    return ActionRequest(action_type=ActionType.click, params={"x": 100, "y": 200})
