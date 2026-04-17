"""Parameter constraints for TicketAPI tools.

Business logic:
- ticket_login must be called before create_ticket (auth required).
- After create_ticket, get_ticket/edit_ticket/close_ticket/resolve_ticket
  are constrained to the ticket_id that was returned.
"""

from __future__ import annotations

from constrained_agent import Session, ToolRegistry


def _has_param(registry: ToolRegistry, tool: str, param: str) -> bool:
    try:
        return param in registry[tool].params
    except KeyError:
        return False


def ticket_api_constraints(session: Session, registry: ToolRegistry) -> None:
    # ticket_login must precede create_ticket
    if not session.tool("ticket_login").has_run:
        try:
            registry["create_ticket"].available = False
            registry["create_ticket"].unavailable_reason = (
                "Call 'ticket_login' first to authenticate before creating a ticket"
            )
        except KeyError:
            pass

    # After create_ticket, constrain ticket_id on downstream tools
    if session.tool("create_ticket").has_run:
        result = session.tool("create_ticket").last_result
        if isinstance(result, dict) and "id" in result:
            tid = result["id"]
            for tool_name in ("get_ticket", "edit_ticket", "close_ticket", "resolve_ticket"):
                if _has_param(registry, tool_name, "ticket_id"):
                    registry[tool_name].params["ticket_id"].enum = [tid]
