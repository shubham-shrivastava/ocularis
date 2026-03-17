"""
Prompt templates for all model interactions.

All templates use {placeholder} format. Callers are responsible for
supplying every variable; missing keys raise KeyError at format time,
which is the desired fail-fast behavior.
"""
from __future__ import annotations

ACTOR_PROMPT = """\
You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task.

## Output Format
```
Thought: ...
Action: ...
```

## Action Space

click(start_box='(x1, y1)')
left_double(start_box='(x1, y1)')
right_single(start_box='(x1, y1)')
drag(start_box='(x1, y1)', end_box='(x2, y2)')
hotkey(key='ctrl c')
type(content='xxx')
scroll(start_box='(x1, y1)', direction='down or up or right or left')
wait()
finished(content='xxx')

## Note
- Use English in `Thought` part.
- Write a small plan and finally summarize your next action (with its target element) in one sentence in `Thought` part.
- Do NOT use `wait()` repeatedly when actionable UI elements are visible.
- If the current sub-task says to interact with a control (dropdown/search/result link), choose that interaction instead of waiting.
- Use `wait()` only for short loading transitions when no actionable target is available yet.
- For dropdown tasks, click the option text explicitly (the exact value asked in the sub-task), not just the dropdown trigger.
- If typing into a search field did not navigate yet, prefer clicking the visible search/submit button next.
- For shopping/product tasks, prefer results whose visible title matches the requested attributes.
- Avoid choosing products whose visible title contradicts the requested attributes.
- Example: if the goal says `SSD`, avoid a result that clearly says `HDD`; if the goal says `1TB`, prefer a result that clearly says `1TB`.
- Avoid sponsored or obviously irrelevant products when a relevant matching product is visible.
- If the task asks for multiple recommendations or a top N list, choose a different relevant product than the ones already inspected.

## User Instruction
{sub_step}

Overall goal: {goal}
Step {step_number} of {max_steps}. Current URL: {url}

{memory_context}

Previous actions:
{history}
"""

RECOVERY_ACTOR_PROMPT = """\
You are helping a GUI agent recover a stuck browser sub-task.

## Output Format
```
Thought: ...
Action: ...
```

## Action Space

click(start_box='(x1, y1)')
left_double(start_box='(x1, y1)')
right_single(start_box='(x1, y1)')
drag(start_box='(x1, y1)', end_box='(x2, y2)')
hotkey(key='ctrl c')
type(content='xxx')
scroll(start_box='(x1, y1)', direction='down or up or right or left')
wait()
finished(content='xxx')

## Instructions
- Focus only on the current sub-task and the immediately visible UI.
- The previous attempt did not complete the sub-task.
- Return the single next browser action most likely to unblock the current sub-task.
- Prefer prerequisite interactions such as clicking a submit button, selecting an open dropdown option, dismissing an overlay, or focusing the correct field.
- Do NOT invent a new plan.
- Do NOT return `finished(...)` unless the screenshot clearly shows the current sub-task is already complete.
- Avoid repeated `wait()` when a visible target can be acted on now.
- If a typed search did not submit, click the visible search button instead of choosing a random page element.
- If the current page or result title contradicts the requested attributes, choose an action that gets back to relevant results or clicks a relevant matching item.
- Example: if the goal requires `SSD`, do not settle on a product page that clearly says `HDD`.
- If the task asks for multiple products, do not reopen a product that has already been inspected.

## Current sub-task
{sub_step}

Overall goal: {goal}
Current URL: {url}
Failure signal: {failure_reason}
Previous attempted action: {last_action}

{memory_context}

Recent actions:
{history}
"""

CRITIC_PROMPT = """\
You are a critic evaluating whether a web agent action made progress toward a goal.

Goal: {goal}
Action taken: {action_type} with params {params}
Page before action URL: {pre_url}
Page after action URL: {post_url}

Two screenshots are attached: [PRE-ACTION] and [POST-ACTION].

Respond with a JSON object:
{{
  "did_progress": <true|false>,
  "blocker_type": "<spinner|popup|wrong_element|redirect|captcha|none|null>",
  "suggested_recovery": "<REFRESH_PAGE|GO_BACK|SCROLL_DOWN|CLICK_OFFSET|null>"
}}

"did_progress" is true only if the page clearly moved toward the goal.
If the page looks identical, or a blocker appeared, set did_progress to false.
"""

GOAL_CHECK_PROMPT = """\
You are evaluating whether a web agent has successfully completed its goal.

Goal: {goal}
Steps taken: {steps_taken}
Current page URL: {url}

The current page screenshot is attached.

Respond with a JSON object:
{{
  "status": "<achieved|not_achieved|uncertain>",
  "confidence": <0.0 to 1.0>,
  "reasoning": "<one or two sentences explaining your verdict>"
}}

"achieved" means the goal is clearly complete.
"not_achieved" means the agent has not reached the goal yet.
"uncertain" means you cannot tell from the screenshot alone.
"""

MEMORY_CONTEXT_PROMPT = """\
Relevant past experience on similar pages:
{episodes}

Use these past observations to inform your next action, but adapt to the current page state.
"""


def format_actor(
    *,
    goal: str,
    url: str,
    step_number: int,
    max_steps: int,
    history: str,
    memory_context: str = "",
    sub_step: str = "",
) -> str:
    return ACTOR_PROMPT.format(
        goal=goal,
        sub_step=sub_step or goal,
        url=url,
        step_number=step_number,
        max_steps=max_steps,
        history=history or "(no previous actions)",
        memory_context=memory_context,
    )


def format_recovery_actor(
    *,
    goal: str,
    sub_step: str,
    url: str,
    history: str,
    failure_reason: str,
    last_action: str,
    memory_context: str = "",
) -> str:
    return RECOVERY_ACTOR_PROMPT.format(
        goal=goal,
        sub_step=sub_step,
        url=url,
        history=history or "(no previous actions)",
        failure_reason=failure_reason,
        last_action=last_action,
        memory_context=memory_context,
    )


def format_critic(
    *,
    goal: str,
    action_type: str,
    params: dict,
    pre_url: str,
    post_url: str,
) -> str:
    return CRITIC_PROMPT.format(
        goal=goal,
        action_type=action_type,
        params=params,
        pre_url=pre_url,
        post_url=post_url,
    )


def format_goal_check(*, goal: str, url: str, steps_taken: int) -> str:
    return GOAL_CHECK_PROMPT.format(
        goal=goal,
        url=url,
        steps_taken=steps_taken,
    )


def format_memory_context(episodes: list[dict]) -> str:
    if not episodes:
        return ""
    lines = []
    for i, ep in enumerate(episodes, 1):
        lines.append(
            f"{i}. URL: {ep.get('past_url', '?')} | "
            f"Action: {ep.get('past_action', '?')} | "
            f"Outcome: {ep.get('past_outcome', '?')} | "
            f"Similarity: {ep.get('similarity_score', 0):.2f}"
        )
    return MEMORY_CONTEXT_PROMPT.format(episodes="\n".join(lines))
