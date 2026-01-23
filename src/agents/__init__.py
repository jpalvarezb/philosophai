from .multi_hopper import MultiHopAgent
from .tools import AgentTools
from .trace import TraceRecorder
from .scope import Scope, extract_scope, ScopeFilter, ScopeViolationError
from .philosopher_agent import PhilosopherAgent
from .phases import Phase, PHASE_TOOLS

__all__ = [
    "MultiHopAgent",
    "AgentTools",
    "TraceRecorder",
    "Scope",
    "extract_scope",
    "ScopeFilter",
    "ScopeViolationError",
    "PhilosopherAgent",
    "Phase",
    "PHASE_TOOLS",
]
