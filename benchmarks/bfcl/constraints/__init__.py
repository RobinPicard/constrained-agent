"""BFCL class-level parameter constraints.

Each module exports a single function with signature
``(session: Session, registry: ToolRegistry) -> None``
that narrows tool parameters based on previous tool results.
"""

from __future__ import annotations

from typing import Callable

from .trading_bot import trading_bot_constraints
from .ticket_api import ticket_api_constraints
from .twitter_api import twitter_api_constraints
from .message_api import message_api_constraints
from .gorilla_fs import gorilla_fs_constraints
from .vehicle_control import vehicle_control_constraints
from .travel_api import travel_api_constraints
from .math_api import math_api_constraints

CONSTRAINTS: dict[str, Callable] = {
    "TradingBot": trading_bot_constraints,
    "TicketAPI": ticket_api_constraints,
    "TwitterAPI": twitter_api_constraints,
    "MessageAPI": message_api_constraints,
    "GorillaFileSystem": gorilla_fs_constraints,
    "VehicleControlAPI": vehicle_control_constraints,
    "TravelAPI": travel_api_constraints,
    "MathAPI": math_api_constraints,
}


def get_constraints(involved_classes: list[str]) -> list[Callable]:
    """Return constraint functions for the given BFCL classes."""
    return [CONSTRAINTS[cls] for cls in involved_classes if cls in CONSTRAINTS]
