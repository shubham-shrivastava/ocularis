from __future__ import annotations

import asyncio
import time
import uuid
from copy import deepcopy
import re
from typing import TYPE_CHECKING

from loguru import logger

from core.browser_manager import BrowserManager
from core.model_client import ModelClientProtocol
from core.schemas import (
    ActionRequest,
    ActionType,
    AgentState,
    AgentStateWire,
    BrowserMode,
    ComparisonState,
    CriticResult,
    GoalVerdict,
    ResultCandidate,
    RunAnswer,
    RunRequest,
    RunResult,
    RunStatus,
    StepOutcome,
    StepTrace,
    StepTraceWire,
)
from core.settings import Settings
from logic.goal_evaluator import GoalEvaluator
from logic.content_extractor import ContentExtractor, ExtractedContent, ExtractedItem
from logic.intent_resolver import IntentResolver, TaskIntent
from logic.memory import MemoryProtocol
from logic.page_classifier import PageClassifier, PageContext
from logic.reasoning_executor import ReasoningExecutor
from logic.strategy_router import StrategyRouter
from logic.supervisor import Supervisor
from logic.text_reasoner import TextReasonerProtocol

if TYPE_CHECKING:
    from db.repository import Repository
    from logic.goal_planner import GoalPlanner


class RunSession:
    """
    Owns all mutable state for a single agent run.

    Never shared across concurrent runs -- each POST /run creates a fresh
    RunSession with its own BrowserManager, Supervisor, and step history.
    """

    def __init__(
        self,
        *,
        run_id: str,
        request: RunRequest,
        settings: Settings,
        model_client: ModelClientProtocol,
        memory: MemoryProtocol,
        planner: GoalPlanner | None = None,
        text_reasoner: TextReasonerProtocol | None = None,
    ) -> None:
        self.run_id = run_id
        self.request = request
        self._settings = settings

        self.browser = BrowserManager(settings)
        self.supervisor = Supervisor(
            comparison_mode=request.comparison_mode,
            supervisor_cfg=settings.supervisor,
            security_cfg=settings.security,
        )
        self.goal_evaluator = GoalEvaluator(model_client, settings.goal_evaluator)
        self._model = model_client
        self._memory = memory
        self._planner = planner
        self._text_reasoner = text_reasoner
        self._intent_resolver = IntentResolver(text_reasoner)
        self._page_classifier = PageClassifier()
        self._content_extractor = ContentExtractor(text_reasoner)
        self._reasoning_executor = ReasoningExecutor(text_reasoner)
        self._strategy_router = StrategyRouter()

        self.status: RunStatus = RunStatus.queued
        self.steps: list[StepTrace] = []
        self.goal_verdict: GoalVerdict | None = None
        self.sub_steps: list[str] = []
        self._current_sub_step: int = 0
        self._login_prompted: bool = False
        self._credentials_provided: bool = False
        self._soft_stuck_waits: int = 0
        self._deterministic_option_attempts: dict[str, int] = {}
        self._recovery_attempts: dict[int, int] = {}
        self._pending_recovery: dict | None = None
        self._target_product_count: int = self._extract_requested_product_count(request.goal)
        self._collected_product_signatures: set[str] = set()
        self._collected_product_titles: list[str] = []
        self._collected_products: list[dict[str, str]] = []
        self._current_product_signature: str | None = None
        self._task_intent: TaskIntent | None = None
        self._page_context: PageContext | None = None
        self._final_answer: RunAnswer | None = None
        self._stored_result: RunResult | None = None
        self._comparison_seen_signatures: set[str] = set()
        self._comparison_collected_items: list[ExtractedItem] = []
        self._comparison_compared_items: list[ExtractedItem] = []
        self._comparison_selected_item: ExtractedItem | None = None
        self._comparison_status: str = "collecting"
        self.waiting_reason: str | None = None

        # HITL coordination
        self._pause_event: asyncio.Event = asyncio.Event()
        self._pause_event.set()  # not paused initially
        self._intervention_queue: asyncio.Queue[ActionRequest] = asyncio.Queue()

        # WebSocket broadcast callbacks registered by API layer
        self._ws_callbacks: list = []

    # ------------------------------------------------------------------
    # Orchestration loop
    # ------------------------------------------------------------------

    async def run_loop(self, repository: Repository) -> None:
        """
        Core agent orchestration loop.

        Runs until goal achieved, max_steps exceeded, circuit breaker fires,
        or an unrecoverable error occurs.
        """
        self.status = RunStatus.running
        await repository.update_run_status(self.run_id, RunStatus.running)
        await repository._session.commit()

        try:
            if self.request.browser_mode == BrowserMode.connect:
                await self.browser.connect(
                    self.request.cdp_url or "",
                    self.request.start_url,
                )
            else:
                await self.browser.launch(self.request.start_url)

            # --- Goal decomposition ---
            if self._planner:
                self.sub_steps = await self._planner.decompose(
                    self.request.goal, self.request.start_url
                )
                self._current_sub_step = 0
                logger.info(
                    "Running with sub-steps",
                    run_id=self.run_id,
                    total=len(self.sub_steps),
                )
            else:
                self.sub_steps = [self.request.goal]
                self._current_sub_step = 0

            if self._settings.context_aware.enabled:
                self._task_intent = await self._intent_resolver.resolve(self.request.goal)
                logger.info(
                    "Resolved task intent",
                    run_id=self.run_id,
                    task_type=self._task_intent.task_type,
                    target_count=self._task_intent.target_count,
                    expected_result=self._task_intent.expected_result,
                )

            for step_num in range(1, self.request.max_steps + 1):
                # Respect pause
                await self._pause_event.wait()

                if self.status not in (RunStatus.running, RunStatus.waiting_for_human):
                    break

                logger.info("Starting step", run_id=self.run_id, step=step_num)
                step_start = time.monotonic()

                # --- Pre-action state ---
                pre_state: AgentState = await self.browser.get_state()

                # --- Current sub-step ---
                active_goal = self.sub_steps[self._current_sub_step]
                model_goal = self._goal_with_collection_context(active_goal)
                forced_action: ActionRequest | None = None
                used_recovery_model = False

                # --- Memory recall ---
                episodes = await self._memory.recall(pre_state.url, active_goal)

                # If credentials were already provided, don't waste steps on
                # planner-generated "wait for user input" sub-steps.
                if self._credentials_provided and self._is_human_wait_step(active_goal):
                    logger.info(
                        "Skipping human-wait sub-step after credential handoff",
                        run_id=self.run_id,
                        skipped_step=active_goal,
                    )
                    if self._current_sub_step < len(self.sub_steps) - 1:
                        self._current_sub_step += 1
                        continue

                # Deterministic assist for dropdowns:
                # 1) "Click dropdown labeled X" -> click trigger once to OPEN
                # 2) "Select X from dropdown" -> click option to SELECT
                trigger_label = self._extract_dropdown_trigger(active_goal)
                dropdown_option = self._extract_dropdown_option(active_goal)

                if trigger_label and not dropdown_option:
                    # Open step only: click the trigger to open dropdown (once per sub-step)
                    trigger_key = f"trigger:{self._current_sub_step}"
                    if self._deterministic_option_attempts.get(trigger_key, 0) == 0:
                        self._deterministic_option_attempts[trigger_key] = 1
                        logger.info(
                            "Deterministic dropdown open: clicking trigger",
                            run_id=self.run_id,
                            trigger=trigger_label,
                        )
                        forced_action = ActionRequest(
                            action_type=ActionType.click,
                            params={
                                "selector": f"text={trigger_label}",
                                "_deterministic": "dropdown_trigger_open",
                            },
                        )
                elif dropdown_option:
                    # Select step: try_select_option_text already clicks the option.
                    # Use a no-op (wait) to record the step without double-clicking.
                    attempts = self._deterministic_option_attempts.get(dropdown_option, 0)
                    if attempts < 3:
                        selected = await self.browser.try_select_option_text(dropdown_option)
                        if selected:
                            self._deterministic_option_attempts[dropdown_option] = attempts + 1
                            logger.info(
                                "Deterministic dropdown selection succeeded",
                                run_id=self.run_id,
                                option=dropdown_option,
                            )
                            forced_action = ActionRequest(
                                action_type=ActionType.wait,
                                params={
                                    "duration_ms": 100,
                                    "_deterministic": "dropdown_option_select",
                                },
                            )

                # Deterministic type assist: "Type 'X' in the search box" -> type text (and Enter)
                if forced_action is None:
                    type_text, press_enter = self._extract_type_text(active_goal)
                    if type_text:
                        type_key = f"type:{self._current_sub_step}"
                        if self._deterministic_option_attempts.get(type_key, 0) == 0:
                            self._deterministic_option_attempts[type_key] = 1
                            logger.info(
                                "Deterministic type: typing into search/focused field",
                                run_id=self.run_id,
                                text_preview=type_text[:30] + ("..." if len(type_text) > 30 else ""),
                            )
                            params: dict = {"text": type_text}
                            search_selector = await self.browser.get_search_box_selector()
                            if search_selector:
                                params["selector"] = search_selector
                            if press_enter:
                                params["press_enter"] = True
                            forced_action = ActionRequest(
                                action_type=ActionType.type,
                                params=params,
                            )

                if forced_action is None and self._is_search_box_focus_step(active_goal.lower()):
                    focus_key = f"focus:{self._current_sub_step}"
                    if self._deterministic_option_attempts.get(focus_key, 0) == 0:
                        search_selector = await self.browser.get_search_box_selector()
                        if search_selector:
                            self._deterministic_option_attempts[focus_key] = 1
                            logger.info(
                                "Deterministic search focus: clicking search box",
                                run_id=self.run_id,
                                selector=search_selector,
                            )
                            forced_action = ActionRequest(
                                action_type=ActionType.click,
                                params={
                                    "selector": search_selector,
                                    "_deterministic": "search_box_focus",
                                },
                            )

                # If model keeps emitting wait/no-progress, ask human guidance.
                if self._soft_stuck_waits >= 3:
                    logger.info(
                        "Repeated wait/no-progress detected; requesting guidance",
                        run_id=self.run_id,
                        count=self._soft_stuck_waits,
                    )
                    self.status = RunStatus.waiting_for_human
                    self.waiting_reason = "stuck_needs_guidance"
                    await repository.update_run_status(self.run_id, RunStatus.waiting_for_human)
                    await repository._session.commit()
                    action = await self._wait_for_intervention()
                    if action is None:
                        break
                    forced_action = action
                    self.status = RunStatus.running
                    self.waiting_reason = None
                    self._soft_stuck_waits = 0
                    await repository.update_run_status(self.run_id, RunStatus.running)
                    await repository._session.commit()

                # --- Login handoff ---
                if await self.browser.has_login_form() and not self._login_prompted:
                    logger.info("Login form detected; waiting for user credentials", run_id=self.run_id)
                    self.status = RunStatus.waiting_for_human
                    self.waiting_reason = "login_required"
                    await repository.update_run_status(self.run_id, RunStatus.waiting_for_human)
                    await repository._session.commit()
                    action = await self._wait_for_intervention()
                    if action is None:
                        break
                    forced_action = action
                    self.status = RunStatus.running
                    self.waiting_reason = None
                    self._soft_stuck_waits = 0
                    self._clear_recovery_state_for_current_substep()
                    if action.action_type == ActionType.login:
                        self._credentials_provided = True
                    await repository.update_run_status(self.run_id, RunStatus.running)
                    await repository._session.commit()
                    self._login_prompted = True
                else:
                    self._login_prompted = False

                if forced_action is not None:
                    action = forced_action
                    is_stuck, reason = False, None
                elif await self._should_use_search_submit_fallback():
                    self._recovery_attempts[self._current_sub_step] = 1
                    submitted = await self.browser.try_submit_search()
                    if submitted:
                        logger.info("Using deterministic search submit fallback", run_id=self.run_id)
                        action = ActionRequest(
                            action_type=ActionType.wait,
                            params={"duration_ms": 150, "_deterministic": "search_submit_click"},
                        )
                    else:
                        logger.info("Deterministic search submit fallback unavailable", run_id=self.run_id)
                        action = ActionRequest(action_type=ActionType.wait, params={"duration_ms": 300})
                    is_stuck, reason = False, None
                elif await self._should_retry_search_box_focus(active_goal):
                    search_selector = await self.browser.get_search_box_selector()
                    self._recovery_attempts[self._current_sub_step] = (
                        self._recovery_attempts.get(self._current_sub_step, 0) + 1
                    )
                    if search_selector:
                        logger.info(
                            "Using deterministic search focus recovery",
                            run_id=self.run_id,
                            selector=search_selector,
                        )
                        action = ActionRequest(
                            action_type=ActionType.click,
                            params={
                                "selector": search_selector,
                                "_deterministic": "search_box_focus",
                            },
                        )
                    else:
                        logger.info("Search focus recovery unavailable; no search box detected", run_id=self.run_id)
                        action = ActionRequest(action_type=ActionType.wait, params={"duration_ms": 300})
                    is_stuck, reason = False, None
                elif await self._should_return_to_results(pre_state.url, active_goal):
                    returned = await self.browser.leave_current_page()
                    if returned:
                        logger.info("Using deterministic return-to-results action", run_id=self.run_id)
                        action = ActionRequest(
                            action_type=ActionType.wait,
                            params={"duration_ms": 150, "_deterministic": "return_to_results"},
                        )
                    else:
                        action = ActionRequest(action_type=ActionType.wait, params={"duration_ms": 300})
                    is_stuck, reason = False, None
                elif await self._should_leave_bad_product_page():
                    self._recovery_attempts[self._current_sub_step] = (
                        self._recovery_attempts.get(self._current_sub_step, 0) + 1
                    )
                    left_page = await self.browser.leave_current_page()
                    if left_page:
                        logger.info("Leaving duplicate/irrelevant product page", run_id=self.run_id)
                        action = ActionRequest(
                            action_type=ActionType.wait,
                            params={"duration_ms": 150, "_deterministic": "leave_bad_product_page"},
                        )
                    else:
                        action = ActionRequest(action_type=ActionType.wait, params={"duration_ms": 300})
                    is_stuck, reason = False, None
                elif self._should_use_recovery_model():
                    recovery_context = self._pending_recovery or {}
                    logger.info(
                        "Calling model for recovery action",
                        run_id=self.run_id,
                        sub_step=active_goal,
                        reason=recovery_context.get("reason"),
                    )
                    try:
                        action = await self._model.predict_recovery_action(
                            pre_state,
                            self.request.goal,
                            model_goal,
                            self.steps,
                            episodes,
                            str(recovery_context.get("reason", "sub_step_stuck")),
                            recovery_context.get(
                                "last_action",
                                ActionRequest(action_type=ActionType.wait, params={"duration_ms": 500}),
                            ),
                        )
                        self._recovery_attempts[self._current_sub_step] = (
                            self._recovery_attempts.get(self._current_sub_step, 0) + 1
                        )
                        used_recovery_model = True
                    except Exception:
                        logger.warning("Model recovery prediction failed; using wait", run_id=self.run_id)
                        action = ActionRequest(action_type=ActionType.wait, params={"duration_ms": 2000})
                    is_stuck, reason = False, None
                else:
                    # --- Stuck / confirm check ---
                    is_stuck, reason = self.supervisor.pre_action_check(pre_state)

                    if reason == "confirm_required":
                        logger.info("Confirm pattern triggered", url=pre_state.url)
                        self.status = RunStatus.waiting_for_human
                        self.waiting_reason = "confirm_required"
                        await repository.update_run_status(self.run_id, RunStatus.waiting_for_human)
                        await repository._session.commit()

                        action = await self._wait_for_intervention()
                        if action is None:
                            break
                        self.status = RunStatus.running
                        self.waiting_reason = None
                        await repository.update_run_status(self.run_id, RunStatus.running)
                        await repository._session.commit()

                    elif is_stuck:
                        strategy = self.supervisor.get_recovery_strategy()
                        logger.info("Recovery triggered", strategy=strategy, run_id=self.run_id)

                        if strategy == "escalate" or self.supervisor.should_halt():
                            self.status = RunStatus.waiting_for_human
                            self.waiting_reason = "supervisor_escalation"
                            await repository.update_run_status(self.run_id, RunStatus.waiting_for_human)
                            await repository._session.commit()
                            action = await self._wait_for_intervention()
                            if action is None:
                                break
                            self.status = RunStatus.running
                            self.waiting_reason = None
                            await repository.update_run_status(self.run_id, RunStatus.running)
                            await repository._session.commit()
                        else:
                            action = self._recovery_to_action(strategy)
                    else:
                        # --- Model predicts action ---
                        logger.bind(category="thought", run_id=self.run_id).debug(
                            "Calling model for next action", step=step_num
                        )
                        try:
                            action = await self._model.predict(
                                pre_state, model_goal, self.steps, episodes
                            )
                        except Exception:
                            logger.warning("Model predict failed (timeout?); using wait", run_id=self.run_id)
                            action = ActionRequest(action_type=ActionType.wait, params={"duration_ms": 3000})

                # --- Execute action ---
                safe_action = self._safe_action_for_trace(action)
                logger.bind(category="action", run_id=self.run_id).info(
                    "Executing action",
                    action_type=safe_action.action_type,
                    params=safe_action.params,
                )
                action_ok = await self.browser.execute_action(action)

                if not action_ok:
                    post_state = pre_state
                    outcome = StepOutcome.error
                    critic_result = None
                    postcondition_met = False
                    postcondition_reason = None
                    recovery_reason = "action_execution_failed"
                else:
                    # --- Post-action state ---
                    post_state = await self.browser.get_state()

                    # --- Critic ---
                    try:
                        critic_result = await self._model.critique(
                            pre_state, post_state, action, active_goal
                        )
                    except Exception:
                        logger.warning("Model critique failed (timeout?); assuming progress", run_id=self.run_id)
                        critic_result = CriticResult(did_progress=True)

                    # --- Supervisor verify ---
                    outcome = self.supervisor.post_action_verify(pre_state, post_state, critic_result)
                    postcondition_met, postcondition_reason, recovery_reason = (
                        await self._check_sub_step_postcondition(
                            active_goal=active_goal,
                            action=action,
                            pre_state=pre_state,
                            post_state=post_state,
                            outcome=outcome,
                        )
                    )
                    if postcondition_met and outcome != StepOutcome.error:
                        outcome = StepOutcome.success

                duration_ms = int((time.monotonic() - step_start) * 1000)

                # --- Build trace ---
                critic_analysis = (
                    f"did_progress={critic_result.did_progress} blocker={critic_result.blocker_type}"
                    if critic_result
                    else "action_failed"
                )
                if postcondition_reason:
                    critic_analysis = f"{critic_analysis} postcondition={postcondition_reason}"
                trace = StepTrace(
                    step_number=step_num,
                    pre_state=pre_state,
                    post_state=post_state,
                    action=safe_action,
                    outcome=outcome,
                    recovery_used=reason if is_stuck else None,
                    duration_ms=duration_ms,
                    critic_analysis=critic_analysis,
                )
                self.steps.append(trace)

                # --- Persist step ---
                await repository.log_step(
                    run_id=self.run_id,
                    step=trace,
                    ephemeral=self.request.ephemeral,
                )
                await repository._session.commit()

                # --- Memory store ---
                if not self.request.ephemeral:
                    await self._memory.store(
                        trace,
                        self.request.goal,
                        run_id=self.run_id,
                        step_number=step_num,
                    )

                # --- Broadcast to WebSocket subscribers ---
                await self._broadcast(trace, post_state)

                if postcondition_met:
                    self._clear_recovery_state_for_current_substep()

                    if await self._maybe_complete_from_context(post_state):
                        self.status = RunStatus.completed
                        final_result = self.build_result()
                        await repository.complete_run(
                            self.run_id,
                            RunStatus.completed,
                            self.goal_verdict,
                            final_result,
                        )
                        await repository._session.commit()
                        logger.info(
                            "Run completed via context-aware answer at postcondition",
                            run_id=self.run_id,
                        )
                        return

                    if self._current_sub_step < len(self.sub_steps) - 1:
                        self._current_sub_step += 1
                        logger.info(
                            "Deterministic postcondition met; advancing sub-step",
                            run_id=self.run_id,
                            completed=active_goal,
                            next_step=self.sub_steps[self._current_sub_step],
                            reason=postcondition_reason,
                        )
                        self.supervisor.reset()
                        self._soft_stuck_waits = 0
                        self._deterministic_option_attempts.clear()
                        continue

                    self.goal_verdict = GoalVerdict(
                        status="achieved",
                        confidence=1.0,
                        reasoning=f"Deterministic postcondition met: {postcondition_reason or 'sub-step complete'}",
                    )
                    self.status = RunStatus.completed
                    await repository.complete_run(
                        self.run_id,
                        RunStatus.completed,
                        self.goal_verdict,
                        self.build_result(),
                    )
                    await repository._session.commit()
                    logger.info("All sub-steps completed via deterministic verification", run_id=self.run_id)
                    return

                if recovery_reason:
                    self._set_pending_recovery(recovery_reason, safe_action)
                elif outcome == StepOutcome.success or used_recovery_model:
                    self._clear_recovery_state_for_current_substep()

                # --- Goal evaluation (skip when action failed) ---
                if outcome == StepOutcome.error:
                    continue
                strict_valid, strict_reason = await self._strict_sub_step_validation(active_goal, post_state)
                if strict_valid is False:
                    logger.info(
                        "Strict sub-step validation blocked advancement",
                        run_id=self.run_id,
                        sub_step=active_goal,
                        reason=strict_reason,
                    )
                    self._set_pending_recovery(strict_reason or "strict_sub_step_validation_failed", safe_action)
                    continue

                if await self._maybe_complete_from_context(post_state):
                    self.status = RunStatus.completed
                    final_result = self.build_result()
                    await repository.complete_run(
                        self.run_id,
                        RunStatus.completed,
                        self.goal_verdict,
                        final_result,
                    )
                    await repository._session.commit()
                    logger.info("Run completed via context-aware answer", run_id=self.run_id)
                    return

                try:
                    verdict = await self.goal_evaluator.evaluate(post_state, model_goal, step_num)
                except Exception:
                    logger.warning("Goal evaluation failed (timeout?); continuing", run_id=self.run_id)
                    continue
                if self.goal_evaluator.is_achieved(verdict):
                    if self._current_sub_step < len(self.sub_steps) - 1:
                        self._clear_recovery_state_for_current_substep()
                        self._current_sub_step += 1
                        logger.info(
                            "Sub-step completed; advancing",
                            run_id=self.run_id,
                            completed=active_goal,
                            next_step=self.sub_steps[self._current_sub_step],
                            progress=f"{self._current_sub_step}/{len(self.sub_steps)}",
                        )
                        self.supervisor.reset()
                        self._soft_stuck_waits = 0
                        self._deterministic_option_attempts.clear()
                        continue

                    self.goal_verdict = verdict
                    if self._target_product_count > 1 and len(self._collected_product_signatures) < self._target_product_count:
                        logger.info(
                            "Completion blocked: not enough distinct products collected",
                            run_id=self.run_id,
                            collected=len(self._collected_product_signatures),
                            target=self._target_product_count,
                        )
                        self._set_pending_recovery("need_more_distinct_products", safe_action)
                        continue
                    self.status = RunStatus.completed
                    await repository.complete_run(
                        self.run_id,
                        RunStatus.completed,
                        verdict,
                        self.build_result(),
                    )
                    await repository._session.commit()
                    logger.info("All sub-steps completed", run_id=self.run_id, step=step_num)
                    return

                # --- Circuit breaker ---
                if self.supervisor.should_halt():
                    self.status = RunStatus.waiting_for_human
                    self.waiting_reason = "supervisor_escalation"
                    await repository.update_run_status(self.run_id, RunStatus.waiting_for_human)
                    await repository._session.commit()
                    action = await self._wait_for_intervention()
                    if action is None:
                        break
                    self.status = RunStatus.running
                    self.waiting_reason = None
                    await repository.update_run_status(self.run_id, RunStatus.running)
                    await repository._session.commit()
                    self.supervisor.reset()

                # Soft stuck counter tracks repeated wait + no_change loops
                if outcome == StepOutcome.no_change and action.action_type == ActionType.wait:
                    self._soft_stuck_waits += 1
                else:
                    self._soft_stuck_waits = 0

            # Loop exhausted without goal achieved
            if self.status == RunStatus.running:
                self.status = RunStatus.max_steps_exceeded
                await repository.complete_run(
                    self.run_id,
                    RunStatus.max_steps_exceeded,
                    None,
                    self.build_result(),
                )
                await repository._session.commit()
                logger.warning("Max steps exceeded", run_id=self.run_id)

        except Exception:
            logger.exception("Run failed with unrecoverable error", run_id=self.run_id)
            self.status = RunStatus.failed
            try:
                await repository.complete_run(self.run_id, RunStatus.failed, None, self.build_result())
                await repository._session.commit()
            except Exception:
                logger.exception("Failed to persist run failure state")
        finally:
            await self.browser.close()

    # ------------------------------------------------------------------
    # HITL controls
    # ------------------------------------------------------------------

    async def pause(self) -> None:
        self._pause_event.clear()
        self.status = RunStatus.paused
        logger.info("Run paused", run_id=self.run_id)

    async def resume(self) -> None:
        self.status = RunStatus.running
        self._pause_event.set()
        logger.info("Run resumed", run_id=self.run_id)

    async def intervene(self, action: ActionRequest) -> None:
        """Push a human-supplied action into the intervention queue."""
        await self._intervention_queue.put(action)
        logger.info("Intervention received", run_id=self.run_id, action=action.action_type)

    async def _wait_for_intervention(self, timeout: float = 300.0) -> ActionRequest | None:
        """Block until a human provides an action or timeout expires."""
        try:
            action = await asyncio.wait_for(self._intervention_queue.get(), timeout=timeout)
            return action
        except asyncio.TimeoutError:
            logger.warning("Intervention timeout; marking run failed", run_id=self.run_id)
            self.status = RunStatus.failed
            return None

    # ------------------------------------------------------------------
    # WebSocket broadcast
    # ------------------------------------------------------------------

    def register_ws_callback(self, cb) -> None:
        self._ws_callbacks.append(cb)

    def unregister_ws_callback(self, cb) -> None:
        self._ws_callbacks.discard(cb) if hasattr(self._ws_callbacks, "discard") else None
        if cb in self._ws_callbacks:
            self._ws_callbacks.remove(cb)

    async def _broadcast(self, trace: StepTrace, post_state: AgentState) -> None:
        if not self._ws_callbacks:
            return

        def _screenshot_url(phase: str) -> str | None:
            if self.request.ephemeral:
                return None
            return f"/runs/{self.run_id}/steps/{trace.step_number}/screenshot?phase={phase}"

        wire = StepTraceWire(
            step_number=trace.step_number,
            pre_state=AgentStateWire(
                url=trace.pre_state.url,
                timestamp=trace.pre_state.timestamp,
                state_hash=trace.pre_state.state_hash,
                screenshot_url=_screenshot_url("pre"),
            ),
            post_state=AgentStateWire(
                url=post_state.url,
                timestamp=post_state.timestamp,
                state_hash=post_state.state_hash,
                screenshot_url=_screenshot_url("post"),
            ),
            action=trace.action,
            outcome=trace.outcome,
            recovery_used=trace.recovery_used,
            duration_ms=trace.duration_ms,
            critic_analysis=trace.critic_analysis,
        )
        payload = wire.model_dump_json()
        dead = []
        for cb in self._ws_callbacks:
            try:
                await cb(payload)
            except Exception:
                dead.append(cb)
        for cb in dead:
            if cb in self._ws_callbacks:
                self._ws_callbacks.remove(cb)

    # ------------------------------------------------------------------
    # Recovery helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _recovery_to_action(strategy: str) -> ActionRequest:
        from core.schemas import ActionType  # noqa: PLC0415

        mapping = {
            "SCROLL_DOWN": ActionRequest(action_type=ActionType.scroll, params={"direction": "down", "amount": 400}),
            "REFRESH_PAGE": ActionRequest(action_type=ActionType.key_press, params={"key": "F5"}),
            "GO_BACK": ActionRequest(action_type=ActionType.key_press, params={"key": "Alt+ArrowLeft"}),
            "CLICK_OFFSET": ActionRequest(action_type=ActionType.click, params={"x": 640, "y": 400}),
        }
        return mapping.get(strategy, ActionRequest(action_type=ActionType.wait, params={"duration_ms": 2000}))

    @staticmethod
    def _safe_action_for_trace(action: ActionRequest) -> ActionRequest:
        """Redact sensitive action params before logs/persistence/websocket payloads."""
        params = deepcopy(action.params)
        if action.action_type == ActionType.login and "password" in params:
            params["password"] = "***REDACTED***"
        if params.get("secret") and "text" in params:
            params["text"] = "***REDACTED***"
        return ActionRequest(action_type=action.action_type, params=params)

    @staticmethod
    def _is_human_wait_step(step_text: str) -> bool:
        text = step_text.lower()
        wait_markers = [
            "wait for the user",
            "wait for user",
            "wait for me",
            "wait for credentials",
            "fill in the user id and password",
            "fill user id and password",
            "enter credentials",
        ]
        return any(marker in text for marker in wait_markers)

    @staticmethod
    def _extract_dropdown_trigger(step_text: str) -> str | None:
        """Extract trigger label from 'click/open dropdown labeled X' steps."""
        text = step_text.lower()
        if "select" in text and ("from" in text or "option" in text):
            return None  # This is a select step, not an open step
        if not any(w in text for w in ["dropdown", "menu", "filter"]):
            return None
        if not any(w in text for w in ["click", "open"]):
            return None
        # Prefer quoted: "Click the dropdown menu labeled 'Owned by me'"
        quoted = re.search(r"['\"]([^'\"]{2,120})['\"]", step_text)
        if quoted:
            return quoted.group(1).strip()
        # Unquoted: "Click the dropdown labeled Owned by me"
        m = re.search(r"labeled\s+(.{2,80}?)(?:\s|$|\.)", step_text, flags=re.IGNORECASE)
        if m:
            return m.group(1).strip()
        return None

    @staticmethod
    def _extract_dropdown_option(step_text: str) -> str | None:
        """Extract option text from 'select X from dropdown' steps."""
        text = step_text.lower()
        if not any(w in text for w in ["select", "choose", "change"]):
            return None
        if not any(w in text for w in ["dropdown", "menu", "filter", "from", "option"]):
            return None
        quoted = re.search(r"['\"]([^'\"]{2,120})['\"]", step_text)
        if quoted:
            return quoted.group(1).strip()
        m = re.search(
            r"select\s+(?:the\s+option\s+)?(.{2,120}?)\s+from\s+(?:the\s+)?dropdown",
            step_text,
            flags=re.IGNORECASE,
        )
        if m:
            candidate = m.group(1).strip(" .,:;")
            if candidate and "dropdown" not in candidate.lower():
                return candidate
        return None

    @staticmethod
    def _extract_type_text(step_text: str) -> tuple[str | None, bool]:
        """Extract text to type from 'Type X in the search box' steps. Returns (text, press_enter)."""
        text = step_text.lower()
        if "type" not in text:
            return None, False
        # Quoted: "Type 'demo dashboard 2' in the search box"
        quoted = re.search(r"type\s+['\"]([^'\"]{1,200})['\"]", step_text, flags=re.IGNORECASE)
        if quoted:
            to_type = quoted.group(1).strip()
            press_enter = "enter" in text or "press enter" in text
            return to_type, press_enter
        # Unquoted: "Type demo dashboard 2 in the search box"
        m = re.search(r"type\s+(.+?)\s+in\s+(?:the\s+)?(?:search\s+)?(?:box|field)", step_text, flags=re.IGNORECASE | re.DOTALL)
        if m:
            to_type = m.group(1).strip(" .,;")
            press_enter = "enter" in text or "press enter" in text
            return to_type, press_enter
        return None, False

    def _recent_deterministic_repeat(self, option: str) -> bool:
        if len(self.steps) < 2:
            return False
        recent = self.steps[-2:]
        marker = f"text={option}"
        for step in recent:
            if step.action.action_type != ActionType.click:
                return False
            selector = str(step.action.params.get("selector", ""))
            det = step.action.params.get("_deterministic")
            if det != "dropdown_option_select" or selector != marker:
                return False
        return True

    def _should_use_recovery_model(self) -> bool:
        if not self._pending_recovery:
            return False
        if self._pending_recovery.get("sub_step_index") != self._current_sub_step:
            return False
        return self._recovery_attempts.get(self._current_sub_step, 0) < 2

    async def _should_use_search_submit_fallback(self) -> bool:
        if not self._pending_recovery:
            return False
        if self._pending_recovery.get("sub_step_index") != self._current_sub_step:
            return False
        if self._pending_recovery.get("reason") != "search_submit_needs_assist":
            return False
        return self._recovery_attempts.get(self._current_sub_step, 0) == 0

    async def _should_retry_search_box_focus(self, active_goal: str) -> bool:
        if not self._pending_recovery:
            return False
        if self._pending_recovery.get("sub_step_index") != self._current_sub_step:
            return False
        if self._pending_recovery.get("reason") != "search_box_not_focused":
            return False
        if not self._is_search_box_focus_step(active_goal.lower()):
            return False
        return self._recovery_attempts.get(self._current_sub_step, 0) == 0

    async def _should_return_to_results(self, pre_url: str, active_goal: str) -> bool:
        goal_text = active_goal.lower()
        if "return to the search results" not in goal_text:
            return False
        return self._looks_like_product_page(pre_url)

    async def _should_leave_bad_product_page(self) -> bool:
        if not self._pending_recovery:
            return False
        if self._pending_recovery.get("sub_step_index") != self._current_sub_step:
            return False
        if self._pending_recovery.get("reason") not in {
            "duplicate_product_page",
            "irrelevant_product_page",
            "need_more_distinct_products",
        }:
            return False
        return self._recovery_attempts.get(self._current_sub_step, 0) == 0

    def _set_pending_recovery(self, reason: str, last_action: ActionRequest) -> None:
        self._pending_recovery = {
            "sub_step_index": self._current_sub_step,
            "reason": reason,
            "last_action": last_action,
        }
        if reason == "search_submit_needs_assist":
            self._recovery_attempts.setdefault(self._current_sub_step, 0)

    def _clear_recovery_state_for_current_substep(self) -> None:
        self._recovery_attempts.pop(self._current_sub_step, None)
        if self._pending_recovery and self._pending_recovery.get("sub_step_index") == self._current_sub_step:
            self._pending_recovery = None

    async def _check_sub_step_postcondition(
        self,
        *,
        active_goal: str,
        action: ActionRequest,
        pre_state: AgentState,
        post_state: AgentState,
        outcome: StepOutcome,
    ) -> tuple[bool, str | None, str | None]:
        marker = str(action.params.get("_deterministic", ""))

        if marker == "dropdown_trigger_open":
            if await self.browser.is_dropdown_open() or pre_state.state_hash != post_state.state_hash:
                return True, "dropdown_opened", None
            return False, None, "dropdown_open_not_verified"

        if marker == "dropdown_option_select":
            if pre_state.state_hash != post_state.state_hash or pre_state.url != post_state.url:
                return True, "dropdown_option_applied", None
            return False, None, "dropdown_option_not_applied"

        if marker == "search_submit_click":
            search_text = self._extract_search_text_from_context(active_goal, action)
            if await self._is_expected_search_results_page(post_state.url, search_text):
                return True, "search_submitted", None
            return False, None, "search_submit_needs_assist"

        if marker == "search_box_focus":
            if await self.browser.is_search_box_focused():
                return True, "search_box_focused", None
            return False, None, "search_box_not_focused"

        if marker == "leave_bad_product_page":
            if not self._looks_like_product_page(post_state.url) or pre_state.url != post_state.url:
                return True, "left_product_page", None
            return False, None, "still_on_bad_product_page"

        if marker == "return_to_results":
            if self._looks_like_search_results_page(post_state.url, self._current_search_text()):
                return True, "returned_to_results", None
            return False, None, "search_results_not_visible"

        if action.action_type == ActionType.type and action.params.get("press_enter") is True:
            search_text = self._extract_search_text_from_context(active_goal, action)
            if (
                self._looks_like_search_submitted(pre_state.url, post_state.url, search_text)
                and await self._is_expected_search_results_page(post_state.url, search_text)
            ):
                return True, "search_submitted", None
            return False, None, "search_submit_needs_assist"

        if action.params.get("_finished") and outcome != StepOutcome.success:
            return False, None, "model_finished_without_progress"

        current_goal = active_goal.lower()
        if any(token in current_goal for token in ["product page", "details page"]) and pre_state.url != post_state.url:
            return True, "navigated_to_new_page", None

        return False, None, None

    @staticmethod
    def _looks_like_search_submitted(pre_url: str, post_url: str, search_text: str) -> bool:
        if post_url != pre_url:
            return True
        normalized = search_text.strip().lower().replace(" ", "+")
        if not normalized:
            return False
        return normalized in post_url.lower()

    async def _strict_sub_step_validation(
        self,
        active_goal: str,
        post_state: AgentState,
    ) -> tuple[bool | None, str | None]:
        goal_text = active_goal.lower()
        if self._is_search_box_focus_step(goal_text):
            if await self.browser.is_search_box_focused():
                return True, None
            return False, "search_box_not_focused"

        if self._is_search_submission_step(goal_text):
            search_text = self._extract_search_text_from_goal(active_goal)
            if await self._is_expected_search_results_page(post_state.url, search_text):
                return True, None
            return False, "search_results_not_visible"

        if self._is_results_navigation_step(goal_text):
            if await self._is_expected_search_results_page(post_state.url, self._current_search_text()):
                return True, None
            return False, "search_results_not_visible"

        if self._is_relevant_results_step(goal_text):
            if not await self._is_expected_search_results_page(post_state.url, self._current_search_text()):
                return False, "search_results_not_visible"
            visible_text = (await self.browser.get_visible_text()).lower()
            if self._page_contains_required_terms(visible_text):
                return True, None
            return False, "relevant_results_not_visible"

        if self._is_product_selection_step(goal_text):
            title = (await self.browser.get_page_title()).lower()
            text = (await self.browser.get_visible_text()).lower()
            if not self._looks_like_product_page(post_state.url):
                self._current_product_signature = None
                return False, "product_page_not_reached"
            if self._is_relevant_product_page(post_state.url, title, text):
                candidate = self._build_product_candidate(post_state.url, title, text)
                signature = candidate["signature"]
                if signature == self._current_product_signature:
                    return True, None
                if signature in self._collected_product_signatures:
                    return False, "duplicate_product_page"
                self._current_product_signature = signature
                self._collected_product_signatures.add(signature)
                self._collected_product_titles.append(candidate["title"])
                self._collected_products.append(candidate)
                logger.info(
                    "Collected distinct product candidate",
                    run_id=self.run_id,
                    title=candidate["title"],
                    total=len(self._collected_product_signatures),
                )
                return True, None
            return False, "irrelevant_product_page"

        return None, None

    @staticmethod
    def _is_search_box_focus_step(goal_text: str) -> bool:
        return "click the search box" in goal_text or "search box at the top" in goal_text

    @staticmethod
    def _is_search_submission_step(goal_text: str) -> bool:
        return "type" in goal_text and "enter" in goal_text

    @staticmethod
    def _is_results_navigation_step(goal_text: str) -> bool:
        return "return to the search results" in goal_text

    @staticmethod
    def _is_relevant_results_step(goal_text: str) -> bool:
        return "relevant product title matches the requested attributes" in goal_text

    @staticmethod
    def _is_product_selection_step(goal_text: str) -> bool:
        return any(
            phrase in goal_text
            for phrase in [
                "click on the first product",
                "click a relevant product title",
                "click a different relevant product title",
                "different relevant product",
                "view details",
                "product details page",
                "product page",
            ]
        )

    def _extract_search_text_from_context(self, active_goal: str, action: ActionRequest) -> str:
        action_text = str(action.params.get("text", "")).strip()
        if action_text:
            return action_text
        return self._extract_search_text_from_goal(active_goal)

    @classmethod
    def _extract_search_text_from_goal(cls, active_goal: str) -> str:
        text, _ = cls._extract_type_text(active_goal)
        return text or ""

    @staticmethod
    def _looks_like_search_results_page(url: str, search_text: str) -> bool:
        url_l = url.lower()
        if "/s?" in url_l or "search" in url_l:
            return True
        normalized = search_text.strip().lower().replace(" ", "+")
        if normalized and normalized in url_l:
            return True
        return False

    async def _is_expected_search_results_page(self, url: str, search_text: str) -> bool:
        if not self._looks_like_search_results_page(url, search_text):
            return False
        if not search_text.strip():
            return True
        url_query = self._extract_search_query_from_url(url)
        if url_query:
            return self._normalized_text(url_query) == self._normalized_text(search_text)
        search_value = await self.browser.get_search_box_value()
        if search_value:
            return self._normalized_text(search_value) == self._normalized_text(search_text)
        return False

    @staticmethod
    def _extract_search_query_from_url(url: str) -> str:
        from urllib.parse import parse_qs, urlparse

        parsed = urlparse(url)
        qs = parse_qs(parsed.query)
        for key in ("k", "q", "query"):
            values = qs.get(key)
            if values:
                return values[0]
        return ""

    @staticmethod
    def _normalized_text(text: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", text.lower())

    def _current_search_text(self) -> str:
        for step in reversed(self.steps):
            if step.action.action_type == ActionType.type:
                value = str(step.action.params.get("text", "")).strip()
                if value:
                    return value
        return ""

    def _page_contains_required_terms(self, text: str) -> bool:
        required = self._required_goal_terms()
        if not required:
            return True
        matches = sum(1 for term in required if term in text)
        return matches >= min(2, len(required))

    def _required_goal_terms(self) -> list[str]:
        goal = self.request.goal.lower()
        words = re.findall(r"[a-z0-9]+", goal)
        stop = {
            "search", "for", "and", "give", "me", "best", "top", "products", "product",
            "the", "a", "an", "of", "to", "find", "show", "with", "that", "this",
        }
        return [w for w in words if w not in stop and len(w) >= 3][:5]

    def _is_relevant_product_page(self, url: str, title: str, text: str) -> bool:
        if not self._looks_like_product_page(url):
            return False
        haystack = f"{title}\n{text}"
        goal = self.request.goal.lower()
        if "ssd" in goal and "ssd" not in haystack:
            return False
        if "ssd" in goal and "hdd" in haystack:
            return False
        if "1tb" in goal and "1tb" not in haystack:
            return False
        return True

    @staticmethod
    def _build_product_candidate(url: str, title: str, text: str) -> dict[str, str]:
        clean_title = " ".join(title.split())
        if not clean_title:
            clean_title = " ".join(text.split())[:160]
        signature = f"{url.lower()}::{clean_title.lower()}"
        return {"url": url, "title": clean_title, "signature": signature}

    @staticmethod
    def _looks_like_product_page(url: str) -> bool:
        url_l = url.lower()
        return "/dp/" in url_l or "/gp/" in url_l or "/aw/d/" in url_l

    @staticmethod
    def _extract_requested_product_count(goal: str) -> int:
        text = goal.lower()
        patterns = [
            r"\btop\s+(\d+)\b",
            r"\bbest\s+(\d+)\b",
            r"\b(\d+)\s+(?:products|options|items|recommendations)\b",
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    return max(1, int(match.group(1)))
                except ValueError:
                    continue
        return 1

    def _goal_with_collection_context(self, active_goal: str) -> str:
        if self._target_product_count <= 1:
            return active_goal
        if not self._collected_product_titles:
            return f"{active_goal}\nNeed {self._target_product_count} distinct relevant products in total."
        collected = "; ".join(self._collected_product_titles[-3:])
        return (
            f"{active_goal}\n"
            f"Need {self._target_product_count} distinct relevant products in total.\n"
            f"Already inspected products: {collected}\n"
            "Choose a different relevant product than the ones already inspected."
        )

    async def _maybe_complete_from_context(self, post_state: AgentState) -> bool:
        if not self._settings.context_aware.enabled or not self._task_intent:
            return False
        if self._pending_recovery:
            return False
        if not self._text_reasoner or not self._text_reasoner.available:
            return False

        self._page_context = await self._page_classifier.classify(post_state, self.browser)
        logger.info(
            "Context-aware page classified",
            run_id=self.run_id,
            page_type=self._page_context.page_type,
            structure_hint=self._page_context.structure_hint,
        )

        if not self._strategy_router.should_extract(
            intent=self._task_intent,
            page_context=self._page_context,
            has_pending_recovery=self._pending_recovery is not None,
        ):
            return False

        if self._page_context.page_type == "product_listing" and self._task_intent.search_query:
            if not self._looks_like_search_results_page(
                post_state.url, self._task_intent.search_query
            ):
                logger.debug(
                    "Skipping extraction from generic listing; search not performed yet",
                    run_id=self.run_id,
                    url=post_state.url,
                )
                return False

        content = await self._content_extractor.extract(
            self.browser,
            self._page_context,
            self._task_intent,
            max_candidates=self._settings.context_aware.max_candidates,
            max_main_text_chars=self._settings.context_aware.max_main_text_chars,
        )
        logger.info(
            "Context-aware extraction complete",
            run_id=self.run_id,
            page_type=self._page_context.page_type,
            item_count=len(content.items),
            text_chars=len(content.raw_text),
        )

        if self._page_context.page_type in {"search_results", "product_listing"} and content.items:
            self._merge_comparison_items(content.items)
            target_count = self._comparison_target_count()
            self._comparison_status = (
                "ready" if len(self._comparison_collected_items) >= target_count else "collecting"
            )
            if len(self._comparison_collected_items) < target_count:
                logger.info(
                    "Context-aware comparison waiting for more candidates",
                    run_id=self.run_id,
                    collected=len(self._comparison_collected_items),
                    target=target_count,
                )
                return False
            content = ExtractedContent(items=list(self._comparison_collected_items), raw_text=content.raw_text)

        reasoning = await self._reasoning_executor.reason(
            content=content,
            intent=self._task_intent,
            user_goal=self.request.goal,
        )
        if not reasoning or not reasoning.answer:
            return False

        self._final_answer = reasoning.answer
        if reasoning.compared_items:
            self._comparison_compared_items = reasoning.compared_items
        if reasoning.selected_item:
            self._comparison_selected_item = reasoning.selected_item
        self._comparison_status = "answered"

        self.goal_verdict = GoalVerdict(
            status="achieved",
            confidence=reasoning.answer.confidence,
            reasoning=self._context_answer_summary(reasoning.answer),
        )
        self._stored_result = self._build_context_result(post_state.url)
        logger.info(
            "Context-aware reasoning completed",
            run_id=self.run_id,
            result_type=reasoning.answer.result_type,
            compared=len(self._comparison_compared_items),
        )
        return True

    def _merge_comparison_items(self, items: list[ExtractedItem]) -> None:
        for item in items:
            signature = self._comparison_signature(item)
            if signature in self._comparison_seen_signatures:
                continue
            self._comparison_seen_signatures.add(signature)
            self._comparison_collected_items.append(item)
            if len(self._comparison_collected_items) >= self._settings.context_aware.max_candidates:
                break

    def _comparison_signature(self, item: ExtractedItem) -> str:
        title = " ".join(item.title.lower().split())
        url = item.url.lower().strip()
        return f"{url}::{title}"

    def _comparison_target_count(self) -> int:
        if self._task_intent and self._task_intent.target_count:
            return max(1, self._task_intent.target_count)
        return 1

    def _context_answer_summary(self, answer: RunAnswer) -> str:
        if answer.result_type == "link" and answer.link:
            return f"Selected link: {answer.link}"
        if answer.text:
            return answer.text
        if answer.items:
            return f"Compared {len(answer.items)} candidates."
        return "Context-aware answer produced."

    def _build_context_result(self, final_url: str) -> RunResult:
        candidates = [self._to_result_candidate(item) for item in self._comparison_collected_items]
        return RunResult(
            summary=self._context_answer_summary(self._final_answer) if self._final_answer else "Context-aware answer produced.",
            final_url=final_url,
            requested_count=self._comparison_target_count() if self._comparison_target_count() > 1 else None,
            collected_count=len(candidates),
            candidates=candidates,
            answer=self._final_answer,
        )

    @staticmethod
    def _to_result_candidate(item: ExtractedItem) -> ResultCandidate:
        return ResultCandidate(
            title=item.title,
            url=item.url,
            price=item.price,
            rating=item.rating,
            snippet=item.snippet,
            fields=item.fields,
        )

    def build_comparison_state(self) -> ComparisonState | None:
        if not self._comparison_collected_items and not self._comparison_compared_items and not self._comparison_selected_item:
            return None
        return ComparisonState(
            target_count=self._comparison_target_count(),
            collected_count=len(self._comparison_collected_items),
            collected_items=[self._to_result_candidate(item) for item in self._comparison_collected_items],
            compared_items=[self._to_result_candidate(item) for item in self._comparison_compared_items],
            selected_item=self._to_result_candidate(self._comparison_selected_item) if self._comparison_selected_item else None,
            status=self._comparison_status,  # type: ignore[arg-type]
        )

    def build_result(self) -> RunResult | None:
        if self._stored_result is not None:
            return self._stored_result
        final_url = self.steps[-1].post_state.url if self.steps else self.request.start_url
        summary = (
            self.goal_verdict.reasoning
            if self.goal_verdict
            else ("Run still in progress." if self.status == RunStatus.running else f"Run status: {self.status.value}")
        )
        candidates = [
            ResultCandidate(
                title=item["title"],
                url=item["url"],
                price=item.get("price"),
                rating=item.get("rating"),
            )
            for item in self._collected_products
        ]
        if candidates and self._target_product_count > 1:
            summary = (
                f"Collected {len(candidates)} of {self._target_product_count} distinct relevant products."
                if not self.goal_verdict
                else self.goal_verdict.reasoning
            )
        if not self.steps and not candidates and not self.goal_verdict:
            return None
        return RunResult(
            summary=summary,
            final_url=final_url,
            requested_count=self._target_product_count if self._target_product_count > 1 else None,
            collected_count=len(candidates),
            candidates=candidates,
            answer=self._final_answer,
        )


# ---------------------------------------------------------------------------
# Run Registry (singleton per process)
# ---------------------------------------------------------------------------


class RunRegistry:
    """
    Global in-memory registry of active RunSessions.

    Used by API endpoints to look up sessions for pause/resume/intervene.
    Completed sessions are retained for `retention_seconds` (default 60s) to
    allow WebSocket subscribers to drain and receive final state, then GC'd.
    """

    def __init__(self, retention_seconds: float = 60.0) -> None:
        self._sessions: dict[str, RunSession] = {}
        self._lock = asyncio.Lock()
        self._retention_seconds = retention_seconds

    async def register(self, session: RunSession) -> None:
        async with self._lock:
            self._sessions[session.run_id] = session

    async def get(self, run_id: str) -> RunSession | None:
        return self._sessions.get(run_id)

    async def schedule_removal(self, run_id: str) -> None:
        """Remove session after the retention period elapses."""
        await asyncio.sleep(self._retention_seconds)
        async with self._lock:
            self._sessions.pop(run_id, None)

    async def remove(self, run_id: str) -> None:
        """Immediate removal (for abnormal cleanup)."""
        async with self._lock:
            self._sessions.pop(run_id, None)

    def active_count(self) -> int:
        return len(self._sessions)

    @staticmethod
    def new_run_id() -> str:
        return str(uuid.uuid4())


# Module-level singleton
registry = RunRegistry()
