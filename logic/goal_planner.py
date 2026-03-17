from __future__ import annotations

import json

import httpx
from loguru import logger

from core.settings import PlannerSettings

PLANNER_SYSTEM_PROMPT = """\
You are a web task planner. Given a user's goal and a starting URL, break it \
into 3-8 concrete, sequential browser sub-steps. Each sub-step should be a \
specific, unambiguous action a GUI agent can perform (click, type, scroll, \
select from dropdown, press a key).

Rules:
- Do NOT include vague steps like "find" or "check" or "look for".
- Each step must describe a single, observable browser action.
- Include the exact text to type or element to click when possible.
- The last step should describe the expected end state.
- If the task is shopping/comparison oriented:
  - do NOT say "click the first product" or "open any result"
  - keep the product constraints explicit in the steps (for example: required product type, size/capacity, brand, and contradictory terms to avoid)
  - include a step to inspect relevant results before opening a product page
  - only open a product whose visible title matches the requested attributes
  - include a final step that confirms the chosen product is relevant to the original goal

Good shopping example:
["Click the search box at the top of the page",
"Type '1TB SSD' and press Enter",
"Open the sort menu labeled 'Sort by' if price ordering is needed",
"Select the price low to high option from the sort menu",
"Scroll until a product title clearly shows the required attributes (for example both 'SSD' and '1TB')",
"Click a relevant product title that matches the requested attributes and avoids contradictory terms (for example choose 'SSD' and avoid 'HDD')",
"The product details page now shows a relevant branded 1TB SSD"]

Good multi-product comparison example:
["Click the search box at the top of the page",
"Type '1TB Samsung SSD' and press Enter",
"Scroll until a relevant product title matches the requested attributes",
"Click the first relevant product title and inspect its details",
"Return to the search results",
"Click a different relevant product title that has not been opened yet",
"Return to the search results",
"Click a third different relevant product title that has not been opened yet",
"The agent has now inspected three distinct relevant products and can compare them"]

Output a JSON array of strings, nothing else. Example:
["Click the search box at the top of the page", \
"Type '1TB SSD' and press Enter", \
"Click the 'Sort by: Price Low to High' dropdown option", \
"The search results page is now sorted by price ascending"]
"""


class GoalPlanner:
    """Decomposes a vague user goal into concrete sub-steps using a fast LLM."""

    def __init__(self, settings: PlannerSettings) -> None:
        self._cfg = settings
        self._http = httpx.AsyncClient(
            base_url=settings.base_url,
            headers={"Authorization": f"Bearer {settings.api_key}"},
            timeout=30.0,
        )

    async def aclose(self) -> None:
        """Close the HTTP client. Call during app shutdown."""
        await self._http.aclose()

    async def decompose(self, goal: str, start_url: str) -> list[str]:
        """Break a high-level goal into ordered sub-steps.

        Returns the original goal as a single-element list on any failure.
        """
        try:
            user_msg = f"Goal: {goal}\nStarting URL: {start_url}"
            payload = {
                "model": self._cfg.model_name,
                "max_tokens": 1024,
                "temperature": 0.2,
                "messages": [
                    {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
            }
            resp = await self._http.post("/chat/completions", json=payload)
            resp.raise_for_status()
            raw = resp.json()["choices"][0]["message"]["content"]
            steps = self._parse_steps(raw)

            if not steps:
                logger.warning("Planner returned empty steps; using original goal")
                return [goal]

            if len(steps) > self._cfg.max_sub_steps:
                steps = steps[: self._cfg.max_sub_steps]

            logger.info(
                "Goal decomposed into sub-steps",
                goal=goal,
                num_steps=len(steps),
                steps=steps,
            )
            return steps

        except Exception:
            logger.exception("GoalPlanner.decompose failed; using original goal")
            return [goal]

    @staticmethod
    def _parse_steps(raw: str) -> list[str]:
        """Extract a JSON array of strings from the model response."""
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
            raw = raw.rsplit("```", 1)[0]
            raw = raw.strip()

        start = raw.find("[")
        end = raw.rfind("]") + 1
        if start == -1 or end == 0:
            raise ValueError("No JSON array found in planner response")
        data = json.loads(raw[start:end])
        if not isinstance(data, list):
            raise ValueError("Planner response is not a list")
        return [str(s) for s in data if s]
