from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging_setup import setup_logging
from core.model_client import APIModelClient, MockModelClient
from core.run_session import RunRegistry, RunSession, registry
from core.schemas import (
    ActionRequest,
    RunAcceptedResponse,
    RunDetailResponse,
    RunRequest,
    RunStatus,
)
from core.settings import Settings, load_settings
from db.engine import create_tables, get_session, init_engine
from db.repository import Repository
from logic.goal_planner import GoalPlanner
from logic.memory import EpisodicMemory, NullMemory
from logic.text_reasoner import APITextReasoner, DisabledTextReasoner


# ---------------------------------------------------------------------------
# App lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = load_settings()
    setup_logging(settings.logging)

    engine = init_engine(settings)
    await create_tables(engine)

    # Store shared app state
    app.state.settings = settings

    # Model client: use APIModelClient when model.api_key is set, else MockModelClient
    if settings.model.api_key:
        client = APIModelClient(settings.model)
        client.set_viewport(settings.browser.viewport_width, settings.browser.viewport_height)
        app.state.model_client = client
        logger.info("Using APIModelClient", base_url=settings.model.base_url)
    else:
        app.state.model_client = MockModelClient()
        logger.info("No model API key configured; using MockModelClient (development mode)")

    if settings.memory.enabled:
        import openai  # noqa: PLC0415

        app.state.openai_client = openai.AsyncOpenAI()
        app.state.memory_enabled = True
        logger.info("EpisodicMemory enabled", model=settings.memory.embedding_model)
    else:
        app.state.openai_client = None
        app.state.memory_enabled = False
        logger.info("Memory disabled; using NullMemory")

    if settings.planner.enabled and settings.planner.api_key:
        app.state.planner = GoalPlanner(settings.planner)
        logger.info("GoalPlanner enabled", model=settings.planner.model_name)
    else:
        app.state.planner = None
        logger.info("GoalPlanner disabled")

    text_cfg = settings.text_model
    if settings.context_aware.enabled and text_cfg.api_key:
        app.state.text_reasoner = APITextReasoner(text_cfg)
        logger.info(
            "Context-aware text reasoner enabled",
            model=text_cfg.model_name,
            base_url=text_cfg.base_url,
        )
    else:
        app.state.text_reasoner = DisabledTextReasoner()
        logger.info("Context-aware text reasoner disabled")

    logger.info("Ocularis API started")
    yield
    logger.info("Ocularis API shutting down")
    # Close HTTP clients before engine dispose
    client = getattr(app.state, "model_client", None)
    if hasattr(client, "aclose"):
        await client.aclose()
    planner = getattr(app.state, "planner", None)
    if planner is not None and hasattr(planner, "aclose"):
        await planner.aclose()
    text_reasoner = getattr(app.state, "text_reasoner", None)
    if text_reasoner is not None and hasattr(text_reasoner, "aclose"):
        await text_reasoner.aclose()
    await engine.dispose()


app = FastAPI(
    title="Ocularis",
    description="Production-grade orchestration layer for autonomous web agents",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------


def _get_settings() -> Settings:
    return app.state.settings


def _get_model_client():
    return app.state.model_client



async def _get_repo(session: AsyncSession = Depends(get_session)) -> Repository:
    settings: Settings = app.state.settings
    return Repository(session, traces_dir=settings.traces.dir)


# ---------------------------------------------------------------------------
# Run endpoints
# ---------------------------------------------------------------------------


@app.post("/run", response_model=RunAcceptedResponse, status_code=202)
async def create_run(
    request: RunRequest,
    repo: Repository = Depends(_get_repo),
):
    """Start a new agent run. Returns immediately with run_id and queued status."""
    run_id = RunRegistry.new_run_id()
    settings: Settings = app.state.settings

    # Persist run record
    await repo.create_run(
        run_id=run_id,
        goal=request.goal,
        start_url=request.start_url,
        comparison_mode=request.comparison_mode.value,
        browser_mode=request.browser_mode.value,
        ephemeral=request.ephemeral,
    )
    await repo._session.commit()

    session = RunSession(
        run_id=run_id,
        request=request,
        settings=settings,
        model_client=_get_model_client(),
        memory=NullMemory(),
        planner=app.state.planner,
        text_reasoner=app.state.text_reasoner,
    )
    await registry.register(session)

    asyncio.create_task(
        _run_with_new_session(session, settings),
        name=f"run-{run_id}",
    )

    logger.info("Run accepted", run_id=run_id, goal=request.goal)
    return RunAcceptedResponse(run_id=run_id, status=RunStatus.queued)


async def _run_with_new_session(session: RunSession, settings: Settings) -> None:
    """Background task: create a fresh DB session for the run loop."""
    from db.engine import _session_factory  # noqa: PLC0415

    if _session_factory is None:
        logger.error("Session factory not initialized")
        return

    async with _session_factory() as db_session:
        try:
            repo = Repository(db_session, traces_dir=settings.traces.dir)

            if (
                app.state.memory_enabled
                and session.request.use_memory
                and not session.request.ephemeral
            ):
                session._memory = EpisodicMemory(
                    repository=repo,
                    openai_client=app.state.openai_client,
                    cfg=settings.memory,
                    ephemeral=session.request.ephemeral,
                )

            await session.run_loop(repo)
        except Exception:
            logger.exception("Unhandled error in run background task", run_id=session.run_id)
        finally:
            asyncio.create_task(registry.schedule_removal(session.run_id))


@app.get("/runs", response_model=list[RunAcceptedResponse])
async def list_runs(
    status: RunStatus | None = Query(default=None, description="Filter by run status"),
    repo: Repository = Depends(_get_repo),
):
    """List all runs, optionally filtered by status."""
    return await repo.list_runs(status_filter=status)


@app.get("/runs/{run_id}", response_model=RunDetailResponse)
async def get_run(
    run_id: str,
    repo: Repository = Depends(_get_repo),
):
    """Get full details for a single run including all steps so far."""
    try:
        detail = await repo.get_run_detail(run_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Run not found")

    session = await registry.get(run_id)
    if session:
        if session.sub_steps:
            detail.sub_steps = session.sub_steps
            detail.current_sub_step = session._current_sub_step
        detail.waiting_reason = session.waiting_reason
        detail.result = session.build_result() or detail.result
        detail.comparison_state = session.build_comparison_state()
    return detail


@app.get("/runs/{run_id}/replay", response_model=RunDetailResponse)
async def replay_run(
    run_id: str,
    repo: Repository = Depends(_get_repo),
):
    """Return complete step history for a finished run (alias for GET /runs/{id})."""
    try:
        detail = await repo.get_run_detail(run_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Run not found")
    session = await registry.get(run_id)
    if session:
        detail.result = session.build_result() or detail.result
        detail.comparison_state = session.build_comparison_state()
    return detail


# ---------------------------------------------------------------------------
# HITL control endpoints
# ---------------------------------------------------------------------------


@app.post("/runs/{run_id}/pause", status_code=204)
async def pause_run(run_id: str, repo: Repository = Depends(_get_repo)):
    session = await registry.get(run_id)
    if not session:
        raise HTTPException(status_code=404, detail="Active run not found")
    await session.pause()
    await repo.update_run_status(run_id, RunStatus.paused)
    await repo._session.commit()


@app.post("/runs/{run_id}/resume", status_code=204)
async def resume_run(run_id: str, repo: Repository = Depends(_get_repo)):
    session = await registry.get(run_id)
    if not session:
        raise HTTPException(status_code=404, detail="Active run not found")
    await session.resume()
    await repo.update_run_status(run_id, RunStatus.running)
    await repo._session.commit()


@app.post("/runs/{run_id}/intervene", status_code=204)
async def intervene_run(run_id: str, action: ActionRequest):
    """Push a human-supplied action to an agent waiting for intervention."""
    session = await registry.get(run_id)
    if not session:
        raise HTTPException(status_code=404, detail="Active run not found")
    await session.intervene(action)


# ---------------------------------------------------------------------------
# Screenshot endpoint
# ---------------------------------------------------------------------------


@app.get("/runs/{run_id}/steps/{step_number}/screenshot")
async def get_screenshot(
    run_id: str,
    step_number: int,
    phase: str = Query(default="post", pattern="^(pre|post)$"),
    repo: Repository = Depends(_get_repo),
):
    """Serve a step screenshot. Returns 404 for ephemeral runs."""
    # Check if run is ephemeral by looking up a step record
    path = await repo.get_step_screenshot_path(run_id, step_number, phase)

    if path is None:
        # Could be ephemeral or step not found
        try:
            detail = await repo.get_run_detail(run_id)
            if detail.ephemeral:
                return JSONResponse(
                    status_code=404,
                    content={"error": "ephemeral_run"},
                )
        except ValueError:
            pass
        raise HTTPException(status_code=404, detail="Screenshot not found")

    if not path.exists():
        raise HTTPException(status_code=404, detail="Screenshot file missing")

    media_type = "image/jpeg" if str(path).lower().endswith(".jpg") else "image/png"
    return FileResponse(str(path), media_type=media_type)


# ---------------------------------------------------------------------------
# WebSocket live stream
# ---------------------------------------------------------------------------


@app.websocket("/ws/runs/{run_id}")
async def websocket_run(websocket: WebSocket, run_id: str):
    """
    Stream StepTraceWire JSON objects as they complete.

    The client receives lightweight wire objects with screenshot_url
    (not embedded base64). Fetch images separately via the screenshot endpoint.
    """
    await websocket.accept()

    session = await registry.get(run_id)
    if not session:
        await websocket.send_text('{"error":"run_not_found"}')
        await websocket.close()
        return

    async def _send(payload: str) -> None:
        await websocket.send_text(payload)

    session.register_ws_callback(_send)
    logger.info("WebSocket connected", run_id=run_id)

    try:
        while True:
            # Keep connection alive; actual data is pushed via callback
            try:
                await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
            except asyncio.TimeoutError:
                # Send ping-style keepalive
                await websocket.send_text('{"type":"ping"}')
            except WebSocketDisconnect:
                break
    finally:
        session.unregister_ws_callback(_send)
        logger.info("WebSocket disconnected", run_id=run_id)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


@app.get("/health")
async def health():
    return {"status": "ok", "active_runs": registry.active_count()}
