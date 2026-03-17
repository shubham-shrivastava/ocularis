from __future__ import annotations

from collections.abc import AsyncGenerator

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from core.settings import Settings, load_settings
from db.engine import get_session
from db.repository import Repository


def get_settings() -> Settings:
    return load_settings()


async def get_repository(
    session: AsyncSession = Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> AsyncGenerator[Repository, None]:
    yield Repository(session, traces_dir=settings.traces.dir)
