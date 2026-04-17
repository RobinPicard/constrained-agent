"""Parameter constraints for TravelAPI tools.

Business logic:
- After get_flight_cost, book_flight should use the same route parameters.
- After book_flight returns a booking_id, cancel_booking, retrieve_invoice,
  and purchase_insurance are constrained to that booking_id.
"""

from __future__ import annotations

from constrained_agent import Session, ToolRegistry


def _has_param(registry: ToolRegistry, tool: str, param: str) -> bool:
    try:
        return param in registry[tool].params
    except KeyError:
        return False


def travel_api_constraints(session: Session, registry: ToolRegistry) -> None:
    # After get_flight_cost, constrain book_flight route parameters
    if session.tool("get_flight_cost").has_run:
        last = session.tool("get_flight_cost").runs[-1]
        args = last.args
        for param in ("travel_from", "travel_to", "travel_date", "travel_class"):
            if param in args and _has_param(registry, "book_flight", param):
                registry["book_flight"].params[param].enum = [args[param]]

    # After book_flight, constrain booking_id on downstream tools
    if session.tool("book_flight").has_run:
        result = session.tool("book_flight").last_result
        if isinstance(result, dict) and "booking_id" in result:
            bid = result["booking_id"]
            for tool_name in ("cancel_booking", "retrieve_invoice", "purchase_insurance"):
                if _has_param(registry, tool_name, "booking_id"):
                    registry[tool_name].params["booking_id"].enum = [bid]
