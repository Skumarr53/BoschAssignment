"""Structured logging setup for the biometric package.

Uses structlog for JSON-structured, machine-parseable logs (ELK/Loki compatible).
Log level is configurable via LOG_LEVEL env var (default: INFO).
Terminal/stdout only — no debug.log files; log aggregation handles persistence.

Configuration is lazy — runs on first get_logger() call, not on import.
"""

import logging
import os
import threading
from typing import cast

import structlog

_configured = False
_lock = threading.Lock()


def _configure_structlog() -> None:
    """Configure structlog with shared processors and LOG_LEVEL from env.

    Thread-safe, idempotent — safe to call multiple times.
    """
    global _configured
    if _configured:
        return
    with _lock:
        if _configured:
            return
        level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
        logging.basicConfig(
            format="%(message)s",
            level=getattr(logging, level_name, logging.INFO),
        )

        structlog.configure(
            processors=[
                structlog.contextvars.merge_contextvars,
                structlog.processors.add_log_level,
                structlog.processors.StackInfoRenderer(),
                structlog.dev.set_exc_info,
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.JSONRenderer(),
            ],
            wrapper_class=structlog.stdlib.BoundLogger,
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )
        _configured = True


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a structlog logger bound to the given module name.

    Lazily configures structlog on first call.

    Args:
        name: Typically __name__ of the calling module.

    Returns:
        Bound logger for structured logging.
    """
    _configure_structlog()
    return cast(structlog.stdlib.BoundLogger, structlog.get_logger(name))
