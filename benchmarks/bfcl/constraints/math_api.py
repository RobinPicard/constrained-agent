"""Parameter constraints for MathAPI tools.

MathAPI tools are pure functions with no sequential dependencies,
so no parameter-narrowing constraints apply.
"""

from __future__ import annotations

from constrained_agent import Session, ToolRegistry


def math_api_constraints(session: Session, registry: ToolRegistry) -> None:
    pass
