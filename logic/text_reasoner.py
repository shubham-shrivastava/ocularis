from __future__ import annotations

import json
from typing import Protocol, runtime_checkable

import httpx
from loguru import logger

from core.settings import ModelSettings, TextModelSettings


@runtime_checkable
class TextReasonerProtocol(Protocol):
    available: bool

    async def generate_json(self, *, system_prompt: str, user_prompt: str) -> dict | None:
        ...

    async def generate_text(self, *, system_prompt: str, user_prompt: str) -> str | None:
        ...


class DisabledTextReasoner:
    available = False

    async def generate_json(self, *, system_prompt: str, user_prompt: str) -> dict | None:  # noqa: ARG002
        return None

    async def generate_text(self, *, system_prompt: str, user_prompt: str) -> str | None:  # noqa: ARG002
        return None


class APITextReasoner:
    available = True

    def __init__(self, settings: ModelSettings | TextModelSettings) -> None:
        self._cfg = settings
        self._http = httpx.AsyncClient(
            base_url=settings.base_url,
            headers={"Authorization": f"Bearer {settings.api_key}"},
            timeout=settings.timeout_s,
        )

    async def aclose(self) -> None:
        await self._http.aclose()

    async def generate_json(self, *, system_prompt: str, user_prompt: str) -> dict | None:
        raw = await self._complete(system_prompt=system_prompt, user_prompt=user_prompt)
        if not raw:
            return None
        try:
            return _extract_json(raw)
        except Exception:
            logger.warning("TextReasoner JSON parse failed")
            return None

    async def generate_text(self, *, system_prompt: str, user_prompt: str) -> str | None:
        raw = await self._complete(system_prompt=system_prompt, user_prompt=user_prompt)
        return raw.strip() if raw else None

    async def _complete(self, *, system_prompt: str, user_prompt: str) -> str | None:
        try:
            payload = {
                "model": self._cfg.model_name,
                "max_tokens": min(self._cfg.max_tokens, 1200),
                "temperature": 0.2,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            }
            resp = await self._http.post("/chat/completions", json=payload)
            resp.raise_for_status()
            return str(resp.json()["choices"][0]["message"]["content"])
        except Exception:
            logger.exception("TextReasoner request failed")
            return None


def _extract_json(text: str) -> dict:
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON found in response")

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
    raise ValueError("No complete JSON object found")
