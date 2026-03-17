"""
Loguru structured logging configuration.

Every log entry is tagged with run_id, step_number, and component when
available. JSON output goes to stdout and an optional rotating file.

Categories:
  - "thought": model reasoning / internal decisions
  - "action": browser actions dispatched

Usage:
    from core.logging_setup import setup_logging
    setup_logging(settings)
"""
from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

from core.settings import LoggingSettings


def setup_logging(cfg: LoggingSettings) -> None:
    logger.remove()

    level = cfg.level.upper()

    if cfg.json_format:
        fmt = (
            "{{"
            '"time":"{time:YYYY-MM-DDTHH:mm:ss.SSSZ}",'
            '"level":"{level}",'
            '"message":"{message}",'
            '"extra":{extra}'
            "}}"
        )
    else:
        fmt = (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level> | {extra}"
        )

    logger.add(sys.stdout, level=level, format=fmt, enqueue=True)

    if cfg.file:
        log_path = Path(cfg.file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            str(log_path),
            level=level,
            format=fmt,
            rotation="50 MB",
            retention="7 days",
            enqueue=True,
            serialize=cfg.json_format,
        )

    logger.info("Logging initialized", level=level, json=cfg.json_format, file=cfg.file or "none")
