from __future__ import annotations

import base64
from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_serializer, model_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ActionType(str, Enum):
    click = "click"
    type = "type"
    login = "login"
    scroll = "scroll"
    wait = "wait"
    key_press = "key_press"


class ComparisonMode(str, Enum):
    md5 = "md5"
    ssim = "ssim"


class BrowserMode(str, Enum):
    launch = "launch"
    connect = "connect"


class RunStatus(str, Enum):
    queued = "queued"
    running = "running"
    paused = "paused"
    waiting_for_human = "waiting_for_human"
    completed = "completed"
    max_steps_exceeded = "max_steps_exceeded"
    failed = "failed"


class StepOutcome(str, Enum):
    success = "success"
    no_change = "no_change"
    error = "error"


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------


class ActionRequest(BaseModel):
    action_type: ActionType
    # Flexible params: x/y for click, text for type, direction/amount for scroll,
    # key for key_press, duration_ms for wait.
    params: dict = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Agent State
# ---------------------------------------------------------------------------


class AgentState(BaseModel):
    """Internal representation of browser state. Never crosses the API boundary."""

    screenshot_b64: str
    screenshot_bytes: bytes = Field(exclude=True)  # used by SSIM; excluded from JSON
    url: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    state_hash: str  # MD5 of raw screenshot bytes

    model_config = {"arbitrary_types_allowed": True}

    @field_serializer("screenshot_bytes")
    def _serialize_bytes(self, v: bytes) -> str:  # noqa: PLR6301
        return base64.b64encode(v).decode()


class AgentStateWire(BaseModel):
    """Wire-safe representation of AgentState for API/WebSocket responses."""

    url: str
    timestamp: datetime
    state_hash: str
    # None when the run is ephemeral (no screenshots stored on disk).
    # Dashboard should show a placeholder instead of making a 404 request.
    screenshot_url: str | None = None


# ---------------------------------------------------------------------------
# Step Trace
# ---------------------------------------------------------------------------


class CriticResult(BaseModel):
    did_progress: bool
    blocker_type: str | None = None
    suggested_recovery: str | None = None


class StepTrace(BaseModel):
    """Internal per-step record. Contains raw AgentState objects."""

    step_number: int
    pre_state: AgentState
    post_state: AgentState
    action: ActionRequest
    outcome: StepOutcome
    recovery_used: str | None = None
    duration_ms: int
    critic_analysis: str | None = None


class StepTraceWire(BaseModel):
    """Wire-safe StepTrace for API responses and WebSocket frames (~1KB per step)."""

    step_number: int
    pre_state: AgentStateWire
    post_state: AgentStateWire
    action: ActionRequest
    outcome: StepOutcome
    recovery_used: str | None = None
    duration_ms: int
    critic_analysis: str | None = None


# ---------------------------------------------------------------------------
# Goal Evaluation
# ---------------------------------------------------------------------------


class GoalVerdict(BaseModel):
    status: Literal["achieved", "not_achieved", "uncertain"]
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str


# ---------------------------------------------------------------------------
# Episodic Memory
# ---------------------------------------------------------------------------


class Episode(BaseModel):
    past_goal: str
    past_url: str
    past_action: ActionRequest
    past_outcome: StepOutcome
    similarity_score: float


# ---------------------------------------------------------------------------
# Run Request / Response
# ---------------------------------------------------------------------------


class RunRequest(BaseModel):
    goal: str
    start_url: str
    max_steps: int = Field(default=30, ge=1, le=100)
    comparison_mode: ComparisonMode = ComparisonMode.md5
    use_memory: bool = True  # Per-run override: False opts out of memory for this run
    browser_mode: BrowserMode = BrowserMode.launch
    cdp_url: str | None = None  # required when browser_mode=connect
    ephemeral: bool = False  # True: no screenshot/embedding persistence

    @model_validator(mode="after")
    def _validate_connect_mode(self) -> RunRequest:
        if self.browser_mode == BrowserMode.connect and not self.cdp_url:
            raise ValueError("cdp_url is required when browser_mode is 'connect'")
        return self


class RunAcceptedResponse(BaseModel):
    """Returned immediately by POST /run. Run hasn't started yet."""

    run_id: str
    status: RunStatus  # will be 'queued'


class ResultCandidate(BaseModel):
    title: str
    url: str
    price: str | None = None
    rating: str | None = None
    snippet: str = ""
    fields: dict[str, str] = Field(default_factory=dict)


class RunAnswer(BaseModel):
    result_type: Literal["link", "text", "list", "count"]
    link: str | None = None
    text: str | None = None
    items: list[ResultCandidate] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class ComparisonState(BaseModel):
    target_count: int = 1
    collected_count: int = 0
    collected_items: list[ResultCandidate] = Field(default_factory=list)
    compared_items: list[ResultCandidate] = Field(default_factory=list)
    selected_item: ResultCandidate | None = None
    status: Literal["collecting", "ready", "answered"] = "collecting"


class RunResult(BaseModel):
    summary: str
    final_url: str | None = None
    requested_count: int | None = None
    collected_count: int = 0
    candidates: list[ResultCandidate] = Field(default_factory=list)
    answer: RunAnswer | None = None


class RunDetailResponse(BaseModel):
    """Returned by GET /runs/{id} and GET /runs/{id}/replay."""

    run_id: str
    goal: str
    start_url: str
    status: RunStatus
    comparison_mode: ComparisonMode
    ephemeral: bool
    steps: list[StepTraceWire] = Field(default_factory=list)
    sub_steps: list[str] = Field(default_factory=list)
    current_sub_step: int = 0
    waiting_reason: str | None = None
    goal_verdict: GoalVerdict | None = None
    result: RunResult | None = None
    comparison_state: ComparisonState | None = None
    created_at: datetime | None = None
    completed_at: datetime | None = None
