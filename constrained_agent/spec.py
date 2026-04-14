"""
JSON-based agent constraint rules.

Schema
------

.. code-block:: json

    {
        "system_prompt": "You are a helpful assistant.",
        "max_turns": 10,
        "rules": [
            {
                "name": "block_transfer_until_balance_checked",
                "if": {"tool": "check_balance", "has_run": false},
                "then": [
                    {"tool": "transfer", "available": false}
                ]
            },
            {
                "name": "cap_transfer_to_balance",
                "if": {"tool": "check_balance", "has_run": true},
                "then": [
                    {
                        "tool": "transfer",
                        "params": {
                            "amount": {"maximum": {"$from": "check_balance.result.currentBalance"}}
                        }
                    }
                ]
            }
        ]
    }

Condition fields (``if``)
-------------------------
- ``tool``     — name of the tool being tested
- ``has_run``  — bool: whether the tool has been called at least once
- ``result``   — dict of ``{field: value}`` equality checks against the last result

Then fields (each item in ``then``)
-------------------------------------
- ``tool``      — name of the tool to modify
- ``available`` — bool: set the tool's availability
- ``reason``    — message explaining unavailability; defaults to the rule name
- ``params``    — per-parameter modifications: ``maximum``, ``minimum``, ``enum``, ``required``

``$from`` expressions
---------------------
Any numeric or list value in ``params`` can be a ``$from`` reference:

- ``{"$from": "tool.result.field"}``           — scalar field
- ``{"$from": "tool.result.items[*].field"}``  — project a key over a list

The expression is resolved lazily; if the referenced tool has not yet run the
constraint is silently skipped.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from .session import Session
from .tools import ToolRegistry


class AgentSpec:
    """
    Parsed representation of a JSON agent spec.

    Attributes
    ----------
    system_prompt:
        Optional system prompt for the agent.
    max_turns:
        Maximum number of turns before the agent stops.
    rules:
        Raw rule dicts from the spec, preserved for introspection.
    """

    def __init__(self, spec: dict):
        self.system_prompt: str | None = spec.get("system_prompt")
        self.format: str | None = spec.get("format")
        self.max_turns: int | None = spec.get("max_turns")
        self.stop_after_first: bool = spec.get("stop_after_first", False)
        self.at_least_one: bool = spec.get("at_least_one", False)
        self.inference_kwargs: dict = spec.get("inference_kwargs", {})
        self.tools_spec: dict = spec.get("tools", {})
        self.rules: list[dict] = spec.get("rules", [])

    @classmethod
    def load(cls, path: str | Path) -> AgentSpec:
        """Load a spec from a JSON file."""
        return cls(json.loads(Path(path).read_text()))

    @classmethod
    def from_dict(cls, spec: dict) -> AgentSpec:
        """Load a spec from a plain dict."""
        return cls(spec)

    def to_constraints(self) -> list[Callable]:
        """Compile rules into a list of callables for ``Agent(constraints=...)``."""
        return [_compile_rule(rule) for rule in self.rules]


# ---------------------------------------------------------------------------
# Compilation
# ---------------------------------------------------------------------------

def _compile_rule(rule: dict) -> Callable:
    condition = rule["if"]
    effects = rule["then"]
    name = rule.get("name")

    def _apply(session: Session, tools: ToolRegistry) -> None:
        if not _evaluate_condition(condition, session):
            return
        for effect in effects:
            _apply_effect(effect, name, session, tools)

    return _apply


def _evaluate_condition(condition: dict, session: Session) -> bool:
    if "allOf" in condition:
        return all(_evaluate_condition(c, session) for c in condition["allOf"])
    if "anyOf" in condition:
        return any(_evaluate_condition(c, session) for c in condition["anyOf"])

    tool_state = session.tool(condition["tool"])

    if "has_run" in condition:
        return tool_state.has_run == condition["has_run"]

    if "result" in condition:
        if not tool_state.has_run:
            return False
        last_result = tool_state.last_result
        for field_path, expected in condition["result"].items():
            if _get_field(last_result, field_path) != expected:
                return False
        return True

    return True


def _apply_effect(
    effect: dict,
    rule_name: str | None,
    session: Session,
    tools: ToolRegistry,
) -> None:
    all_names = {t.name for t in tools.available_tools() + tools.unavailable_tools()}
    if effect["tool"] not in all_names:
        return

    schema = tools[effect["tool"]]

    if "available" in effect:
        schema.available = effect["available"]
        if not effect["available"]:
            schema.unavailable_reason = effect.get("reason") or rule_name

    for param_name, pdef in effect.get("params", {}).items():
        if param_name not in schema.params:
            continue
        param = schema.params[param_name]

        if "maximum" in pdef:
            val = _resolve(pdef["maximum"], session)
            if val is not None:
                param.maximum = val

        if "minimum" in pdef:
            val = _resolve(pdef["minimum"], session)
            if val is not None:
                param.minimum = val

        if "enum" in pdef:
            val = _resolve(pdef["enum"], session)
            if val is not None:
                param.enum = val if isinstance(val, list) else [val]

        if "required" in pdef:
            param.required = pdef["required"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_field(obj: Any, path: str) -> Any:
    """Dot-separated field access for result conditions."""
    for part in path.split("."):
        if isinstance(obj, dict):
            obj = obj.get(part)
        else:
            return None
    return obj


def _resolve(value: Any, session: Session) -> Any:
    """
    Return *value* as-is, or evaluate a ``{"$from": "..."}`` expression.

    Path syntax: ``tool.result.field`` or ``tool.args.field``.
    - ``tool``    — tool name
    - ``result``  — the tool's last return value
    - ``args``    — the arguments of the tool's last call
    - remaining segments navigate into the selected object

    Examples:
    - ``check_balance.result.currentBalance``
    - ``get_cart.result.items[*].product_id``  — project over a list

    Returns ``None`` if the referenced tool has not yet run or any path
    segment is missing.
    """
    if not isinstance(value, dict) or "$from" not in value:
        return value

    parts = value["$from"].split(".")
    if len(parts) < 2:
        return None

    tool_name = parts[0]
    if not session.tool(tool_name).has_run:
        return None

    last_run = session.tool(tool_name).runs[-1]
    accessor = parts[1]
    if accessor == "result":
        current: Any = last_run.result
    elif accessor == "args":
        current = last_run.args
    else:
        return None

    for part in parts[2:]:
        if current is None:
            return None
        if part.endswith("[*]"):
            key = part[:-3]
            if key:
                current = current.get(key) if isinstance(current, dict) else None
        elif isinstance(current, list):
            current = [
                item.get(part) if isinstance(item, dict) else None
                for item in current
            ]
        elif isinstance(current, dict):
            current = current.get(part)
        else:
            return None

    return current
