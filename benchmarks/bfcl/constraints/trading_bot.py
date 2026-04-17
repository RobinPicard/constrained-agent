"""Parameter constraints for TradingBot tools.

Business logic:
- After get_stock_info, place_order must use the actual market price and
  the queried symbol.
- After place_order, get_order_details and cancel_order are constrained
  to the order_id that was returned.
"""

from __future__ import annotations

from constrained_agent import Session, ToolRegistry


def _has_param(registry: ToolRegistry, tool: str, param: str) -> bool:
    try:
        return param in registry[tool].params
    except KeyError:
        return False


def trading_bot_constraints(session: Session, registry: ToolRegistry) -> None:
    # get_stock_info -> constrain place_order price and symbol
    if session.tool("get_stock_info").has_run:
        last = session.tool("get_stock_info").runs[-1]
        result, args = last.result, last.args
        if isinstance(result, dict) and "price" in result:
            if _has_param(registry, "place_order", "price"):
                registry["place_order"].params["price"].enum = [result["price"]]
            if _has_param(registry, "place_order", "symbol") and "symbol" in args:
                registry["place_order"].params["symbol"].enum = [args["symbol"]]

    # place_order -> constrain order_id on get_order_details and cancel_order
    if session.tool("place_order").has_run:
        result = session.tool("place_order").last_result
        if isinstance(result, dict) and "order_id" in result:
            oid = result["order_id"]
            if _has_param(registry, "get_order_details", "order_id"):
                registry["get_order_details"].params["order_id"].enum = [oid]
            if _has_param(registry, "cancel_order", "order_id"):
                registry["cancel_order"].params["order_id"].enum = [oid]
