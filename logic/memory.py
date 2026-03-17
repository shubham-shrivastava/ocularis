from __future__ import annotations

from typing import Protocol, runtime_checkable

from core.schemas import Episode, StepTrace


@runtime_checkable
class MemoryProtocol(Protocol):
    """
    Uniform interface for episodic memory backends.

    Slice 1 wires in NullMemory (no-op). Slice 2 swaps in EpisodicMemory
    (pgvector + OpenAI embeddings). The orchestration loop never checks
    which implementation is active; it always calls recall() and store().
    """

    async def recall(self, state_url: str, goal: str) -> list[Episode]:
        """Return relevant past episodes for the current situation."""
        ...

    async def store(
        self,
        step_trace: StepTrace,
        goal: str,
        run_id: str = "",
        step_number: int = 0,
    ) -> None:
        """Persist a completed step trace for future recall."""
        ...


class NullMemory:
    """
    No-op memory implementation for Slice 1.

    recall() always returns an empty list.
    store() does nothing.
    """

    async def recall(self, state_url: str, goal: str) -> list[Episode]:  # noqa: ARG002
        return []

    async def store(  # noqa: ARG002
        self,
        step_trace: StepTrace,
        goal: str,
        run_id: str = "",
        step_number: int = 0,
    ) -> None:
        return


# ---------------------------------------------------------------------------
# Slice 2: EpisodicMemory (skeleton -- fully implemented in s2-memory)
# ---------------------------------------------------------------------------


class EpisodicMemory:
    """
    Episodic memory backed by pgvector + OpenAI text embeddings.

    Only text-based summaries are embedded (no raw screenshot bytes). This
    keeps embeddings semantically meaningful and avoids the ephemeral
    contradiction (we never store sensitive bytes in the vector DB).

    Ephemeral runs: store() is a no-op. recall() still works from past
    non-ephemeral data (read-only).
    """

    def __init__(self, repository, openai_client, cfg, ephemeral: bool = False) -> None:
        self._repo = repository
        self._openai = openai_client
        self._cfg = cfg
        self._ephemeral = ephemeral

    async def recall(self, state_url: str, goal: str) -> list[Episode]:
        """Embed the current situation and find similar past steps via cosine search."""
        try:
            summary = f"Goal: {goal}\nURL: {state_url}"
            embedding = await self._embed(summary)
            records = await self._repo.find_similar_steps(
                embedding=embedding,
                top_k=self._cfg.top_k,
                threshold=self._cfg.similarity_threshold,
            )
            episodes = []
            for r, sim_score in records:
                import json as _json  # noqa: PLC0415

                from core.schemas import ActionRequest, ActionType, StepOutcome  # noqa: PLC0415

                try:
                    action = ActionRequest(
                        action_type=ActionType(r.action_type),
                        params=_json.loads(r.action_params),
                    )
                    episodes.append(
                        Episode(
                            past_goal=goal,
                            past_url=r.pre_state_url,
                            past_action=action,
                            past_outcome=StepOutcome(r.outcome),
                            similarity_score=sim_score,
                        )
                    )
                except Exception:
                    continue
            return episodes
        except Exception:
            from loguru import logger  # noqa: PLC0415

            logger.exception("EpisodicMemory.recall failed; returning empty")
            return []

    async def store(
        self,
        step_trace: StepTrace,
        goal: str,
        run_id: str = "",
        step_number: int = 0,
    ) -> None:
        """
        Generate a text embedding for this step and persist it.

        No-op when the run is ephemeral. Text summary only -- no screenshot bytes.
        Requires run_id and step_number to update the correct StepRecord.
        """
        if self._ephemeral:
            return
        if not run_id:
            from loguru import logger  # noqa: PLC0415

            logger.warning("EpisodicMemory.store called without run_id; skipping")
            return
        try:
            summary = self._build_summary(step_trace, goal)
            embedding = await self._embed(summary)

            from sqlalchemy import update  # noqa: PLC0415

            from db.models import StepRecord  # noqa: PLC0415

            stmt = (
                update(StepRecord)
                .where(
                    StepRecord.run_id == run_id,
                    StepRecord.step_number == step_number,
                )
                .values(embedding=embedding)
            )
            await self._repo._session.execute(stmt)
        except Exception:
            from loguru import logger  # noqa: PLC0415

            logger.exception("EpisodicMemory.store failed", run_id=run_id, step=step_number)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _build_summary(step_trace: StepTrace, goal: str) -> str:
        return (
            f"Goal: {goal}\n"
            f"URL before: {step_trace.pre_state.url}\n"
            f"URL after: {step_trace.post_state.url}\n"
            f"Action: {step_trace.action.action_type.value} {step_trace.action.params}\n"
            f"Outcome: {step_trace.outcome.value}\n"
            f"Critic: {step_trace.critic_analysis or 'n/a'}"
        )

    async def _embed(self, text: str) -> list[float]:
        response = await self._openai.embeddings.create(
            model=self._cfg.embedding_model,
            input=text,
        )
        return response.data[0].embedding
