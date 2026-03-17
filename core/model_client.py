from __future__ import annotations

import json
import re
from typing import Protocol, runtime_checkable

import httpx
from loguru import logger

from core.schemas import (
    ActionRequest,
    ActionType,
    AgentState,
    CriticResult,
    Episode,
    GoalVerdict,
    StepTrace,
)
from core.settings import ModelSettings
from logic import prompts


@runtime_checkable
class ModelClientProtocol(Protocol):
    """
    Abstract interface for the vision model client.

    All three methods must be implemented. Use MockModelClient for tests
    and Slice 1 development without a running model server.
    """

    async def predict(
        self,
        state: AgentState,
        goal: str,
        history: list[StepTrace],
        episodes: list[Episode],
    ) -> ActionRequest:
        """Actor: decide the next browser action from current state."""
        ...

    async def critique(
        self,
        pre_state: AgentState,
        post_state: AgentState,
        action: ActionRequest,
        goal: str,
    ) -> CriticResult:
        """Critic: did the action make progress toward the goal?"""
        ...

    async def evaluate_goal(
        self,
        state: AgentState,
        goal: str,
        steps_taken: int,
    ) -> GoalVerdict:
        """Goal check: is the task complete?"""
        ...

    async def predict_recovery_action(
        self,
        state: AgentState,
        goal: str,
        sub_step: str,
        history: list[StepTrace],
        episodes: list[Episode],
        failure_reason: str,
        last_action: ActionRequest,
    ) -> ActionRequest:
        """Recovery actor: choose the next best action for a stuck sub-step."""
        ...


# ---------------------------------------------------------------------------
# Real implementation (calls UI-TARS / OpenAI-compatible vLLM endpoint)
# ---------------------------------------------------------------------------


class APIModelClient:
    """
    Calls a UI-TARS (or any OpenAI-compatible vision) model endpoint.

    For predict(): uses UI-TARS native prompt format and parses
    ``Thought: ... Action: click(start_box='(x,y)')`` output.

    For critique()/evaluate_goal(): uses JSON-format prompts with
    robust extraction that handles trailing text from the model.
    """

    def __init__(self, settings: ModelSettings) -> None:
        self._cfg = settings
        self._http = httpx.AsyncClient(
            base_url=settings.base_url,
            headers={"Authorization": f"Bearer {settings.api_key}"},
            timeout=settings.timeout_s,
        )
        self._viewport_w = 1280
        self._viewport_h = 800

    def set_viewport(self, width: int, height: int) -> None:
        self._viewport_w = width
        self._viewport_h = height

    async def aclose(self) -> None:
        """Close the HTTP client. Call during app shutdown."""
        await self._http.aclose()

    async def predict(
        self,
        state: AgentState,
        goal: str,
        history: list[StepTrace],
        episodes: list[Episode],
    ) -> ActionRequest:
        history_text = "\n".join(
            f"Step {t.step_number}: {t.action.action_type} {t.action.params}"
            for t in history
        )
        memory_context = prompts.format_memory_context(
            [ep.model_dump() for ep in episodes]
        )
        prompt_text = prompts.format_actor(
            goal=goal,
            url=state.url,
            step_number=len(history) + 1,
            max_steps=50,
            history=history_text,
            memory_context=memory_context,
        )
        raw = await self._call_model(prompt_text, state.screenshot_b64)
        return self._parse_uitars_action(raw)

    async def critique(
        self,
        pre_state: AgentState,
        post_state: AgentState,
        action: ActionRequest,
        goal: str,
    ) -> CriticResult:
        prompt_text = prompts.format_critic(
            goal=goal,
            action_type=action.action_type.value,
            params=action.params,
            pre_url=pre_state.url,
            post_url=post_state.url,
        )
        raw = await self._call_model_multi(
            prompt_text,
            [pre_state.screenshot_b64, post_state.screenshot_b64],
        )
        return self._parse_critic(raw)

    async def evaluate_goal(
        self,
        state: AgentState,
        goal: str,
        steps_taken: int,
    ) -> GoalVerdict:
        prompt_text = prompts.format_goal_check(
            goal=goal,
            url=state.url,
            steps_taken=steps_taken,
        )
        raw = await self._call_model(prompt_text, state.screenshot_b64)
        return self._parse_verdict(raw)

    async def predict_recovery_action(
        self,
        state: AgentState,
        goal: str,
        sub_step: str,
        history: list[StepTrace],
        episodes: list[Episode],
        failure_reason: str,
        last_action: ActionRequest,
    ) -> ActionRequest:
        history_text = "\n".join(
            f"Step {t.step_number}: {t.action.action_type} {t.action.params}"
            for t in history[-3:]
        )
        memory_context = prompts.format_memory_context(
            [ep.model_dump() for ep in episodes]
        )
        prompt_text = prompts.format_recovery_actor(
            goal=goal,
            sub_step=sub_step,
            url=state.url,
            history=history_text,
            failure_reason=failure_reason,
            last_action=f"{last_action.action_type.value} {last_action.params}",
            memory_context=memory_context,
        )
        raw = await self._call_model(prompt_text, state.screenshot_b64)
        return self._parse_uitars_action(raw)

    # ------------------------------------------------------------------
    # HTTP
    # ------------------------------------------------------------------

    async def _call_model(self, prompt: str, screenshot_b64: str) -> str:
        payload = {
            "model": self._cfg.model_name,
            "max_tokens": self._cfg.max_tokens,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{screenshot_b64}"},
                        },
                    ],
                }
            ],
        }
        resp = await self._http.post("/chat/completions", json=payload)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    async def _call_model_multi(self, prompt: str, screenshots: list[str]) -> str:
        content: list[dict] = [{"type": "text", "text": prompt}]
        for b64 in screenshots:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                }
            )
        payload = {
            "model": self._cfg.model_name,
            "max_tokens": self._cfg.max_tokens,
            "messages": [{"role": "user", "content": content}],
        }
        resp = await self._http.post("/chat/completions", json=payload)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    # ------------------------------------------------------------------
    # UI-TARS native action parser
    # ------------------------------------------------------------------

    def _parse_uitars_action(self, raw: str) -> ActionRequest:
        """Parse UI-TARS native ``Thought: ... Action: ...`` output format."""
        try:
            from ui_tars.action_parser import parse_action_to_structure_output  # noqa: PLC0415

            parsed_list = parse_action_to_structure_output(
                raw,
                factor=1000,
                origin_resized_height=self._viewport_h,
                origin_resized_width=self._viewport_w,
                model_type="qwen25vl",
            )
            if not parsed_list:
                raise ValueError("Empty parse result")

            parsed = parsed_list[0]
            action_type = parsed["action_type"]
            inputs = parsed.get("action_inputs", {})
            thought = parsed.get("thought", "")

            if thought:
                logger.bind(category="thought").debug("Model thought: {}", thought)

            return self._map_uitars_to_action(action_type, inputs)

        except Exception:
            action = self._fallback_parse_action(raw)
            if action:
                return action
            logger.warning(
                "Failed to parse UI-TARS action; using wait fallback",
                raw=raw[:300],
            )
            return ActionRequest(action_type=ActionType.wait, params={"duration_ms": 2000})

    def _map_uitars_to_action(self, action_type: str, inputs: dict) -> ActionRequest:
        """Convert parsed UI-TARS action to our ActionRequest."""
        if action_type in ("click", "left_double", "right_single"):
            x, y = self._box_to_pixels(inputs.get("start_box", ""))
            return ActionRequest(action_type=ActionType.click, params={"x": x, "y": y})

        if action_type == "type":
            text = inputs.get("content", "")
            if text.endswith("\\n"):
                text = text[:-2]
            return ActionRequest(action_type=ActionType.type, params={"text": text})

        if action_type == "scroll":
            direction = inputs.get("direction", "down")
            return ActionRequest(
                action_type=ActionType.scroll,
                params={"direction": direction, "amount": 300},
            )

        if action_type == "hotkey":
            key = inputs.get("key", "")
            key_map = {
                "enter": "Enter", "tab": "Tab", "escape": "Escape",
                "backspace": "Backspace", "space": " ",
                "ctrl c": "Control+c", "ctrl v": "Control+v",
                "ctrl a": "Control+a", "ctrl z": "Control+z",
            }
            mapped_key = key_map.get(key.lower(), key.capitalize())
            return ActionRequest(action_type=ActionType.key_press, params={"key": mapped_key})

        if action_type == "wait":
            return ActionRequest(action_type=ActionType.wait, params={"duration_ms": 3000})

        if action_type == "finished":
            return ActionRequest(action_type=ActionType.wait, params={"duration_ms": 500, "_finished": True})

        logger.warning("Unknown UI-TARS action type: {}", action_type)
        return ActionRequest(action_type=ActionType.wait, params={"duration_ms": 2000})

    def _box_to_pixels(self, start_box: str) -> tuple[int, int]:
        """Convert start_box ratio string like '[0.38, 0.12, 0.38, 0.12]' to pixel coords."""
        if not start_box:
            return self._viewport_w // 2, self._viewport_h // 2
        cleaned = start_box.strip("[]")
        parts = [float(p.strip()) for p in cleaned.split(",")]
        x = int(parts[0] * self._viewport_w)
        y = int(parts[1] * self._viewport_h)
        return x, y

    @staticmethod
    def _fallback_parse_action(raw: str) -> ActionRequest | None:
        """Try regex extraction when ui-tars parser fails."""
        click_m = re.search(r"click\(start_box='?\((\d+),\s*(\d+)\)'?\)", raw)
        if click_m:
            x, y = int(click_m.group(1)), int(click_m.group(2))
            return ActionRequest(action_type=ActionType.click, params={"x": x, "y": y})

        type_m = re.search(r"type\(content='([^']*)'\)", raw)
        if type_m:
            return ActionRequest(action_type=ActionType.type, params={"text": type_m.group(1)})

        scroll_m = re.search(r"scroll\(.*direction='(down|up|left|right)'", raw)
        if scroll_m:
            return ActionRequest(
                action_type=ActionType.scroll,
                params={"direction": scroll_m.group(1), "amount": 300},
            )

        if re.search(r"hotkey\(key='([^']*)'\)", raw):
            key = re.search(r"hotkey\(key='([^']*)'\)", raw).group(1)  # type: ignore[union-attr]
            return ActionRequest(action_type=ActionType.key_press, params={"key": key.capitalize()})

        if "wait()" in raw:
            return ActionRequest(action_type=ActionType.wait, params={"duration_ms": 3000})

        if "finished(" in raw:
            return ActionRequest(action_type=ActionType.wait, params={"duration_ms": 500, "_finished": True})

        return None

    # ------------------------------------------------------------------
    # JSON parsers for critic / goal
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_critic(raw: str) -> CriticResult:
        try:
            data = _extract_json(raw)
            return CriticResult.model_validate(data)
        except Exception:
            logger.warning("Failed to parse critic result; assuming no progress", raw=raw[:300])
            return CriticResult(did_progress=False, blocker_type="parse_error")

    @staticmethod
    def _parse_verdict(raw: str) -> GoalVerdict:
        try:
            data = _extract_json(raw)
            return GoalVerdict.model_validate(data)
        except Exception:
            logger.warning("Failed to parse goal verdict; returning uncertain", raw=raw[:300])
            return GoalVerdict(status="uncertain", confidence=0.0, reasoning="parse error")


# ---------------------------------------------------------------------------
# Mock implementation for Slice 1 tests and local development
# ---------------------------------------------------------------------------


class MockModelClient:
    """
    Scripted model client for development and tests. No GPU or API key required.

    predict()        -- cycles through provided action_sequence, then waits.
    critique()       -- returns did_progress=True by default.
    evaluate_goal()  -- returns not_achieved until the final step, then achieved.
    """

    def __init__(
        self,
        action_sequence: list[ActionRequest] | None = None,
        total_steps: int = 3,
    ) -> None:
        self._actions = action_sequence or []
        self._call_count = 0
        self._total_steps = total_steps

    async def predict(
        self,
        state: AgentState,
        goal: str,
        history: list[StepTrace],
        episodes: list[Episode],
    ) -> ActionRequest:
        if self._call_count < len(self._actions):
            action = self._actions[self._call_count]
        else:
            action = ActionRequest(action_type=ActionType.scroll, params={"direction": "down", "amount": 300})
        self._call_count += 1
        logger.debug("MockModelClient.predict", call=self._call_count, action=action.action_type)
        return action

    async def critique(
        self,
        pre_state: AgentState,
        post_state: AgentState,
        action: ActionRequest,
        goal: str,
    ) -> CriticResult:
        return CriticResult(did_progress=True)

    async def evaluate_goal(
        self,
        state: AgentState,
        goal: str,
        steps_taken: int,
    ) -> GoalVerdict:
        if steps_taken >= self._total_steps:
            return GoalVerdict(
                status="achieved",
                confidence=0.95,
                reasoning="Mock: goal achieved after all steps.",
            )
        return GoalVerdict(
            status="not_achieved",
            confidence=0.1,
            reasoning=f"Mock: {steps_taken}/{self._total_steps} steps completed.",
        )

    async def predict_recovery_action(
        self,
        state: AgentState,
        goal: str,
        sub_step: str,
        history: list[StepTrace],
        episodes: list[Episode],
        failure_reason: str,
        last_action: ActionRequest,
    ) -> ActionRequest:
        return await self.predict(state, sub_step or goal, history, episodes)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def _extract_json(text: str) -> dict:
    """Extract the first complete JSON object from a model response.

    Uses brace-counting to find the matching closing brace for the first
    opening brace, so trailing text (Chinese explanations, extra braces)
    from the model doesn't break parsing.
    """
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON found in model response")

    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : i + 1])

    raise ValueError("No complete JSON object found in model response")
