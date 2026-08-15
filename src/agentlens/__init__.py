"""AgentLens — find out why your coding agent is expensive.

Reads the session history your coding agents already write to disk, normalizes
it into one schema, and tells you which of your own habits are costing money.

Everything runs locally. Nothing is uploaded.
"""

from __future__ import annotations

__version__ = "0.4.0"

from .schema import Event, Provider, Role, Session, ToolKind, Usage

__all__ = ["__version__", "Event", "Provider", "Role", "Session", "ToolKind", "Usage"]
