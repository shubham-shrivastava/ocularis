from __future__ import annotations

from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class RunRecord(Base):
    """Persisted record of an agent run."""

    __tablename__ = "runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    goal: Mapped[str] = mapped_column(Text, nullable=False)
    start_url: Mapped[str] = mapped_column(Text, nullable=False)
    # RunStatus enum value stored as string
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="queued")
    comparison_mode: Mapped[str] = mapped_column(String(8), nullable=False, default="md5")
    browser_mode: Mapped[str] = mapped_column(String(8), nullable=False, default="launch")
    ephemeral: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    goal_verdict_status: Mapped[str | None] = mapped_column(String(16), nullable=True)
    goal_verdict_confidence: Mapped[float | None] = mapped_column(nullable=True)
    goal_verdict_reasoning: Mapped[str | None] = mapped_column(Text, nullable=True)
    result_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    steps: Mapped[list[StepRecord]] = relationship(
        "StepRecord", back_populates="run", cascade="all, delete-orphan", order_by="StepRecord.step_number"
    )


class StepRecord(Base):
    """Persisted record of a single agent step within a run."""

    __tablename__ = "steps"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(36), ForeignKey("runs.id", ondelete="CASCADE"), nullable=False)
    step_number: Mapped[int] = mapped_column(Integer, nullable=False)

    pre_state_hash: Mapped[str] = mapped_column(String(32), nullable=False)
    post_state_hash: Mapped[str] = mapped_column(String(32), nullable=False)
    pre_state_url: Mapped[str] = mapped_column(Text, nullable=False)
    post_state_url: Mapped[str] = mapped_column(Text, nullable=False)

    action_type: Mapped[str] = mapped_column(String(16), nullable=False)
    action_params: Mapped[str] = mapped_column(Text, nullable=False)  # JSON string

    outcome: Mapped[str] = mapped_column(String(16), nullable=False)
    recovery_used: Mapped[str | None] = mapped_column(String(32), nullable=True)
    duration_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    critic_analysis: Mapped[str | None] = mapped_column(Text, nullable=True)

    # File paths; null when ephemeral=True (no screenshots persisted)
    pre_screenshot_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    post_screenshot_path: Mapped[str | None] = mapped_column(Text, nullable=True)

    # pgvector embedding (1536 dims = text-embedding-3-small); null when ephemeral
    embedding: Mapped[list[float] | None] = mapped_column(Vector(1536), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    run: Mapped[RunRecord] = relationship("RunRecord", back_populates="steps")
