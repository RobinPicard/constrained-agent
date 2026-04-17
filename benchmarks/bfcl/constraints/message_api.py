"""Parameter constraints for MessageAPI tools.

Business logic:
- message_login must be called before send_message, add_contact,
  delete_message, and other write actions.
- After get_user_id, send_message receiver_id is constrained to
  the returned user_id.
"""

from __future__ import annotations

from constrained_agent import Session, ToolRegistry


_WRITE_TOOLS = ("send_message", "add_contact", "delete_message")


def message_api_constraints(session: Session, registry: ToolRegistry) -> None:
    # message_login must precede write actions
    if not session.tool("message_login").has_run:
        for tool_name in _WRITE_TOOLS:
            try:
                registry[tool_name].available = False
                registry[tool_name].unavailable_reason = (
                    "Call 'message_login' first to log in before using this tool"
                )
            except KeyError:
                pass

    # After get_user_id, constrain send_message receiver_id
    if session.tool("get_user_id").has_run:
        result = session.tool("get_user_id").last_result
        if isinstance(result, dict) and "user_id" in result:
            uid = result["user_id"]
            try:
                if "receiver_id" in registry["send_message"].params:
                    registry["send_message"].params["receiver_id"].enum = [uid]
            except KeyError:
                pass
