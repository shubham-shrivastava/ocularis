from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.schemas import (
    AgentStateWire,
    GoalVerdict,
    RunAcceptedResponse,
    RunDetailResponse,
    RunResult,
    RunStatus,
    StepOutcome,
    StepTrace,
    StepTraceWire,
)
from db.models import RunRecord, StepRecord


class Repository:
    """
    Async CRUD layer for runs and steps.

    All methods accept an AsyncSession so they participate in the caller's
    transaction. The session is not committed here; callers commit.
    """

    def __init__(self, session: AsyncSession, traces_dir: str = "traces") -> None:
        self._session = session
        self._traces_dir = Path(traces_dir)

    # ------------------------------------------------------------------
    # Run CRUD
    # ------------------------------------------------------------------

    async def create_run(
        self,
        *,
        run_id: str,
        goal: str,
        start_url: str,
        comparison_mode: str,
        browser_mode: str,
        ephemeral: bool,
    ) -> RunRecord:
        record = RunRecord(
            id=run_id,
            goal=goal,
            start_url=start_url,
            status=RunStatus.queued.value,
            comparison_mode=comparison_mode,
            browser_mode=browser_mode,
            ephemeral=ephemeral,
        )
        self._session.add(record)
        await self._session.flush()
        logger.info("Run created", run_id=run_id)
        return record

    async def update_run_status(self, run_id: str, status: RunStatus) -> None:
        record = await self._get_run_or_raise(run_id)
        record.status = status.value
        if status in (
            RunStatus.completed,
            RunStatus.max_steps_exceeded,
            RunStatus.failed,
        ):
            record.completed_at = datetime.utcnow()
        await self._session.flush()

    async def complete_run(
        self,
        run_id: str,
        status: RunStatus,
        verdict: GoalVerdict | None,
        result: RunResult | None = None,
    ) -> None:
        record = await self._get_run_or_raise(run_id)
        record.status = status.value
        record.completed_at = datetime.utcnow()
        if verdict:
            record.goal_verdict_status = verdict.status
            record.goal_verdict_confidence = verdict.confidence
            record.goal_verdict_reasoning = verdict.reasoning
        if result is not None:
            record.result_json = result.model_dump_json()
        await self._session.flush()

    async def list_runs(self, status_filter: RunStatus | None = None) -> list[RunAcceptedResponse]:
        stmt = select(RunRecord)
        if status_filter:
            stmt = stmt.where(RunRecord.status == status_filter.value)
        stmt = stmt.order_by(RunRecord.created_at.desc())
        result = await self._session.execute(stmt)
        records = result.scalars().all()
        return [
            RunAcceptedResponse(run_id=r.id, status=RunStatus(r.status))
            for r in records
        ]

    async def get_run_detail(self, run_id: str, base_url: str = "") -> RunDetailResponse:
        record = await self._get_run_or_raise(run_id)
        step_records = await self._get_steps(run_id)
        steps_wire = [
            self._step_record_to_wire(s, record.ephemeral, base_url)
            for s in step_records
        ]
        verdict = None
        if record.goal_verdict_status:
            verdict = GoalVerdict(
                status=record.goal_verdict_status,
                confidence=record.goal_verdict_confidence or 0.0,
                reasoning=record.goal_verdict_reasoning or "",
            )
        stored_result = self._load_result(record.result_json)
        return RunDetailResponse(
            run_id=record.id,
            goal=record.goal,
            start_url=record.start_url,
            status=RunStatus(record.status),
            comparison_mode=record.comparison_mode,
            ephemeral=record.ephemeral,
            steps=steps_wire,
            goal_verdict=verdict,
            result=stored_result or self._build_basic_result(record.start_url, steps_wire, verdict),
            created_at=record.created_at,
            completed_at=record.completed_at,
        )

    # ------------------------------------------------------------------
    # Step CRUD
    # ------------------------------------------------------------------

    async def log_step(
        self,
        *,
        run_id: str,
        step: StepTrace,
        ephemeral: bool,
    ) -> StepRecord:
        """Persist a step trace. Saves screenshots to disk unless ephemeral."""
        pre_path = post_path = None

        if not ephemeral:
            pre_path, post_path = await self._save_screenshots(
                run_id=run_id,
                step_number=step.step_number,
                pre_bytes=step.pre_state.screenshot_bytes,
                post_bytes=step.post_state.screenshot_bytes,
            )

        record = StepRecord(
            run_id=run_id,
            step_number=step.step_number,
            pre_state_hash=step.pre_state.state_hash,
            post_state_hash=step.post_state.state_hash,
            pre_state_url=step.pre_state.url,
            post_state_url=step.post_state.url,
            action_type=step.action.action_type.value,
            action_params=json.dumps(step.action.params),
            outcome=step.outcome.value,
            recovery_used=step.recovery_used,
            duration_ms=step.duration_ms,
            critic_analysis=step.critic_analysis,
            pre_screenshot_path=str(pre_path) if pre_path else None,
            post_screenshot_path=str(post_path) if post_path else None,
        )
        self._session.add(record)
        await self._session.flush()
        return record

    async def get_step_screenshot_path(
        self, run_id: str, step_number: int, phase: str
    ) -> Path | None:
        """Return file path for pre/post screenshot, or None if not stored."""
        stmt = select(StepRecord).where(
            StepRecord.run_id == run_id,
            StepRecord.step_number == step_number,
        )
        result = await self._session.execute(stmt)
        record = result.scalar_one_or_none()
        if not record:
            return None
        if phase == "pre":
            return Path(record.pre_screenshot_path) if record.pre_screenshot_path else None
        return Path(record.post_screenshot_path) if record.post_screenshot_path else None

    async def find_similar_steps(
        self,
        embedding: list[float],
        top_k: int = 5,
        threshold: float = 0.75,
    ) -> list[tuple[StepRecord, float]]:
        """pgvector cosine similarity search with threshold filtering.

        Returns list of (StepRecord, similarity_score) tuples.
        Cosine distance is converted to similarity: 1 - distance.
        Only rows with similarity >= threshold are returned.
        """
        distance_expr = StepRecord.embedding.cosine_distance(embedding)
        similarity_expr = (1 - distance_expr).label("similarity")
        stmt = (
            select(StepRecord, similarity_expr)
            .where(StepRecord.embedding.isnot(None))
            .where(similarity_expr >= threshold)
            .order_by(distance_expr)
            .limit(top_k)
        )
        result = await self._session.execute(stmt)
        rows = result.all()
        return [(row[0], float(row[1])) for row in rows]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _get_run_or_raise(self, run_id: str) -> RunRecord:
        result = await self._session.execute(select(RunRecord).where(RunRecord.id == run_id))
        record = result.scalar_one_or_none()
        if not record:
            raise ValueError(f"Run not found: {run_id}")
        return record

    async def _get_steps(self, run_id: str) -> list[StepRecord]:
        result = await self._session.execute(
            select(StepRecord)
            .where(StepRecord.run_id == run_id)
            .order_by(StepRecord.step_number)
        )
        return list(result.scalars().all())

    async def _save_screenshots(
        self,
        *,
        run_id: str,
        step_number: int,
        pre_bytes: bytes,
        post_bytes: bytes,
    ) -> tuple[Path, Path]:
        run_dir = self._traces_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        pre_path = run_dir / f"step_{step_number:04d}_pre.jpg"
        post_path = run_dir / f"step_{step_number:04d}_post.jpg"
        pre_path.write_bytes(pre_bytes)
        post_path.write_bytes(post_bytes)
        return pre_path, post_path

    def _step_record_to_wire(
        self, record: StepRecord, ephemeral: bool, base_url: str
    ) -> StepTraceWire:
        from core.schemas import ActionRequest, ActionType  # noqa: PLC0415

        def _screenshot_url(phase: str) -> str | None:
            if ephemeral:
                return None
            return f"{base_url}/runs/{record.run_id}/steps/{record.step_number}/screenshot?phase={phase}"

        return StepTraceWire(
            step_number=record.step_number,
            pre_state=AgentStateWire(
                url=record.pre_state_url,
                timestamp=record.created_at,
                state_hash=record.pre_state_hash,
                screenshot_url=_screenshot_url("pre"),
            ),
            post_state=AgentStateWire(
                url=record.post_state_url,
                timestamp=record.created_at,
                state_hash=record.post_state_hash,
                screenshot_url=_screenshot_url("post"),
            ),
            action=ActionRequest(
                action_type=ActionType(record.action_type),
                params=json.loads(record.action_params),
            ),
            outcome=StepOutcome(record.outcome),
            recovery_used=record.recovery_used,
            duration_ms=record.duration_ms,
            critic_analysis=record.critic_analysis,
        )

    @staticmethod
    def _load_result(payload: str | None) -> RunResult | None:
        if not payload:
            return None
        try:
            return RunResult.model_validate_json(payload)
        except Exception:
            logger.warning("Failed to parse stored run result")
            return None

    @staticmethod
    def _build_basic_result(
        start_url: str,
        steps: list[StepTraceWire],
        verdict: GoalVerdict | None,
    ) -> RunResult | None:
        if not steps and not verdict:
            return None
        final_url = steps[-1].post_state.url if steps else start_url
        summary = verdict.reasoning if verdict else "Run finished without a final verdict."
        return RunResult(
            summary=summary,
            final_url=final_url,
            collected_count=0,
        )
