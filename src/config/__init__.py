"""Configuration module for PhilosophAI."""
from .logging import setup_logging, scope_logger, ScopeLogger, trace_logger, TraceLogger

__all__ = ["setup_logging", "scope_logger", "ScopeLogger", "trace_logger", "TraceLogger"]
