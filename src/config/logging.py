"""Centralized logging configuration for PhilosophAI."""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Log levels
LOG_LEVEL = os.environ.get("PHILOSOPH_LOG_LEVEL", "INFO").upper()
SCOPE_LOG_LEVEL = os.environ.get("PHILOSOPH_SCOPE_LOG_LEVEL", "DEBUG").upper()
TRACE_LOG_LEVEL = os.environ.get("PHILOSOPH_TRACE_LOG_LEVEL", "INFO").upper()

# Trace verbosity toggles
TRACE_VERBOSE = os.environ.get("PHILOSOPH_TRACE_VERBOSE", "0").lower() in {
    "1",
    "true",
    "yes",
}
TRACE_TRAVERSAL = os.environ.get("PHILOSOPH_TRACE_TRAVERSAL", "0").lower() in {
    "1",
    "true",
    "yes",
}
TRACE_MAX_ITEMS = int(os.environ.get("PHILOSOPH_TRACE_MAX_ITEMS", "10"))
TRACE_MAX_STEPS = int(os.environ.get("PHILOSOPH_TRACE_MAX_STEPS", "200"))

# Log directory
LOG_DIR = Path(os.environ.get("PHILOSOPH_LOG_DIR", "logs"))


class ScopeFormatter(logging.Formatter):
    """Custom formatter for scope enforcement logs with color support."""

    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
        "RESET": "\033[0m",
    }

    SCOPE_ICONS = {
        "SCOPE": "🔒",
        "CHECK": "✓",
        "PASS": "✅",
        "FAIL": "❌",
        "VIOLATION": "🚨",
        "AGENT": "🤖",
        "TOOL": "🧰",
        "RESULT": "📊",
        "DECISION": "🧭",
        "TRAVERSAL": "🧵",
        "SCORE": "🎯",
    }

    def __init__(self, use_color: bool = True):
        super().__init__()
        self.use_color = use_color and sys.stderr.isatty()

    def format(self, record: logging.LogRecord) -> str:
        # Timestamp
        timestamp = datetime.fromtimestamp(record.created).strftime("%H:%M:%S.%f")[:-3]

        # Level with optional color
        level = record.levelname
        if self.use_color:
            color = self.COLORS.get(level, "")
            reset = self.COLORS["RESET"]
            level_str = f"{color}{level:8}{reset}"
        else:
            level_str = f"{level:8}"

        # Message with scope icons
        msg = record.getMessage()
        for tag, icon in self.SCOPE_ICONS.items():
            msg = msg.replace(f"[{tag}]", icon)

        return f"{timestamp} | {level_str} | {msg}"


class ScopeLogger:
    """
    Dedicated logger for scope enforcement with structured output.

    Usage:
        from src.config.logging import scope_logger

        scope_logger.scope_init("Aristotle", chunks=500, edges=1200, entities=800)
        scope_logger.check_pass("vector", total=15, in_scope=15)
        scope_logger.check_fail("seed", total=20, out_of_scope=3, examples=["entity1", "entity2"])
    """

    def __init__(self, name: str = "philosoph.scope"):
        self.logger = logging.getLogger(name)
        self._setup_done = False

    def setup(self, level: str = SCOPE_LOG_LEVEL, log_to_file: bool = True):
        """Configure the scope logger."""
        if self._setup_done:
            return

        self.logger.setLevel(getattr(logging, level))
        self.logger.propagate = False  # Don't bubble up to root

        # Console handler
        console = logging.StreamHandler(sys.stderr)
        console.setFormatter(ScopeFormatter(use_color=True))
        self.logger.addHandler(console)

        # File handler (no color)
        if log_to_file:
            LOG_DIR.mkdir(exist_ok=True)
            file_handler = logging.FileHandler(
                LOG_DIR / "scope.log",
                encoding="utf-8",
            )
            file_handler.setFormatter(ScopeFormatter(use_color=False))
            self.logger.addHandler(file_handler)

        self._setup_done = True

    def scope_init(
        self,
        description: str,
        *,
        strict: bool,
        texts: int,
        chunks: int,
        edges: int | None = None,
        entities: int | None = None,
    ):
        """Log scope initialization."""
        mode = "STRICT" if strict else "NON-STRICT"
        msg = f"[SCOPE] {mode} mode | {description} | texts={texts} chunks={chunks}"
        if edges is not None:
            msg += f" edges={edges}"
        if entities is not None:
            msg += f" entities={entities}"
        self.logger.info(msg)

    def check_pass(
        self,
        stage: str,
        *,
        total: int,
        in_scope: int,
        extra: dict | None = None,
    ):
        """Log a passing scope check."""
        msg = f"[CHECK] [PASS] {stage:12} | total={total} in_scope={in_scope} out_of_scope=0"
        if extra:
            extras = " ".join(f"{k}={v}" for k, v in extra.items())
            msg += f" | {extras}"
        self.logger.info(msg)

    def check_fail(
        self,
        stage: str,
        *,
        total: int,
        out_of_scope: int,
        examples: list[str] | None = None,
    ):
        """Log a failing scope check."""
        in_scope = total - out_of_scope
        msg = f"[CHECK] [FAIL] {stage:12} | total={total} in_scope={in_scope} out_of_scope={out_of_scope}"
        self.logger.error(msg)
        if examples:
            self.logger.error(f"[VIOLATION] Leaked items: {examples[:5]}")

    def traversal_summary(
        self,
        *,
        nodes_visited: int,
        chunks_collected: int,
        edges_filtered: int,
    ):
        """Log traversal summary with scope filtering stats."""
        self.logger.info(
            f"[SCOPE] Traversal complete | nodes={nodes_visited} chunks={chunks_collected} edges_filtered={edges_filtered}"
        )

    def debug(self, msg: str, *args):
        """Pass-through debug logging."""
        self.logger.debug(msg, *args)

    def info(self, msg: str, *args):
        """Pass-through info logging."""
        self.logger.info(msg, *args)

    def warning(self, msg: str, *args):
        """Pass-through warning logging."""
        self.logger.warning(msg, *args)

    def error(self, msg: str, *args):
        """Pass-through error logging."""
        self.logger.error(msg, *args)


class TraceLogger:
    """
    Logger for detailed agent/tool/traversal decisions.
    Controlled via PHILOSOPH_TRACE_* env vars.
    """

    def __init__(self, name: str = "philosoph.trace"):
        self.logger = logging.getLogger(name)
        self._setup_done = False

    def setup(self, level: str = TRACE_LOG_LEVEL, log_to_file: bool = True):
        if self._setup_done:
            return
        self.logger.setLevel(getattr(logging, level))
        self.logger.propagate = False

        console = logging.StreamHandler(sys.stderr)
        console.setFormatter(ScopeFormatter(use_color=True))
        self.logger.addHandler(console)

        if log_to_file:
            LOG_DIR.mkdir(exist_ok=True)
            file_handler = logging.FileHandler(
                LOG_DIR / "trace.log",
                encoding="utf-8",
            )
            file_handler.setFormatter(ScopeFormatter(use_color=False))
            self.logger.addHandler(file_handler)

        self._setup_done = True

    def decision(self, msg: str):
        self.logger.info(f"[DECISION] {msg}")

    def tool_call(self, name: str, **kwargs):
        details = " ".join(f"{k}={v}" for k, v in kwargs.items())
        self.logger.info(f"[TOOL] CALL {name} {details}".strip())

    def tool_result(self, name: str, **kwargs):
        details = " ".join(f"{k}={v}" for k, v in kwargs.items())
        self.logger.info(f"[RESULT] {name} {details}".strip())

    def traversal(self, msg: str):
        self.logger.info(f"[TRAVERSAL] {msg}")

    def score(self, msg: str):
        self.logger.debug(f"[SCORE] {msg}")

    def debug(self, msg: str):
        self.logger.debug(msg)

    def info(self, msg: str):
        self.logger.info(msg)


# Singleton instances
trace_logger = TraceLogger()


# Singleton instance
scope_logger = ScopeLogger()


def setup_logging(level: str = LOG_LEVEL):
    """
    Configure root logging for the application.

    Call this once at application startup (e.g., in main.py).
    """
    # Root logger
    root = logging.getLogger()
    root.setLevel(getattr(logging, level))

    # Console handler with simple format
    if not root.handlers:
        console = logging.StreamHandler(sys.stderr)
        console.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        root.addHandler(console)

    # Setup scope & trace loggers
    scope_logger.setup()
    trace_logger.setup()

    # Quiet noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
