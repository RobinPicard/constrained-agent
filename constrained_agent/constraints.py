from __future__ import annotations
from typing import Callable
from .session import Session
from .tools import ToolRegistry


class Constraint:
    def __init__(self, fn: Callable[[Session, ToolRegistry], None]):
        self._fn = fn

    def apply(self, session: Session, tools: ToolRegistry) -> None:
        self._fn(session, tools)


def constraint(fn: Callable[[Session, ToolRegistry], None]) -> Constraint:
    """Decorator to define a constraint."""
    return Constraint(fn)


class ConstraintEvaluator:
    def __init__(self, constraints: list[Constraint]):
        self.constraints = constraints

    def evaluate(self, session: Session, tools: ToolRegistry) -> None:
        """Reset all tool schemas to base, then apply constraints in order."""
        tools.reset_all()
        for c in self.constraints:
            c.apply(session, tools)
