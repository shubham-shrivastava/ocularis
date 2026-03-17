from __future__ import annotations

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from core.settings import Settings

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def init_engine(settings: Settings) -> AsyncEngine:
    global _engine, _session_factory
    _engine = create_async_engine(
        settings.db.url,
        pool_size=settings.db.pool_size,
        max_overflow=settings.db.max_overflow,
        echo=False,
    )
    _session_factory = async_sessionmaker(_engine, expire_on_commit=False)
    return _engine


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency that yields a database session."""
    if _session_factory is None:
        raise RuntimeError("Database engine not initialized. Call init_engine() first.")
    async with _session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def create_tables(engine: AsyncEngine) -> None:
    """Create all tables (dev/test convenience; use Alembic in production)."""
    from db.models import Base  # noqa: PLC0415

    async with engine.begin() as conn:
        # Ensure pgvector extension exists
        await conn.execute(
            __import__("sqlalchemy", fromlist=["text"]).text("CREATE EXTENSION IF NOT EXISTS vector")
        )
        await conn.run_sync(Base.metadata.create_all)
        await conn.execute(
            __import__("sqlalchemy", fromlist=["text"]).text(
                "ALTER TABLE runs ADD COLUMN IF NOT EXISTS result_json TEXT"
            )
        )
