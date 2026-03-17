from __future__ import annotations

from loguru import logger

from core.schemas import AgentState, GoalVerdict
from core.settings import GoalEvaluatorSettings


class GoalEvaluator:
    """
    Evaluates whether the agent has achieved the stated goal.

    Delegates to model_client.evaluate_goal() and applies the confidence
    threshold from settings before declaring a task complete.

    One instance per RunSession; takes a ModelClientProtocol reference
    (not a concrete model caller) to remain testable with MockModelClient.
    """

    def __init__(self, model_client, cfg: GoalEvaluatorSettings) -> None:
        self._model = model_client
        self._cfg = cfg

    async def evaluate(
        self,
        state: AgentState,
        goal: str,
        steps_taken: int,
    ) -> GoalVerdict:
        """
        Ask the model whether the goal is complete.

        Returns GoalVerdict. The orchestration loop should terminate only when
        verdict.status == "achieved" and verdict.confidence >= threshold.
        """
        try:
            verdict = await self._model.evaluate_goal(state, goal, steps_taken)
            logger.info(
                "Goal evaluation",
                status=verdict.status,
                confidence=verdict.confidence,
                steps=steps_taken,
            )
            return verdict
        except Exception:
            logger.exception("Goal evaluation failed; returning uncertain")
            return GoalVerdict(
                status="uncertain",
                confidence=0.0,
                reasoning="Evaluation failed due to an error.",
            )

    def is_achieved(self, verdict: GoalVerdict) -> bool:
        """True when the verdict clears the confidence threshold."""
        return (
            verdict.status == "achieved"
            and verdict.confidence >= self._cfg.confidence_threshold
        )
