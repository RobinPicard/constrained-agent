"""Run a single BFCL multi_turn_base test case with constrained_agent."""

from __future__ import annotations

import copy
import importlib
import inspect
import json
from typing import Any, Callable

from bfcl_eval.constants.executable_backend_config import CLASS_FILE_PATH_MAPPING

from constrained_agent import Agent, Session, ToolRegistry
from constrained_agent.backends.base import Backend
from constrained_agent.tools import Tool, ToolRegistry as TR


def _bfcl_params_to_json_schema(params_spec: dict) -> dict:
    """Convert BFCL parameter spec to standard JSON Schema."""
    properties = {}
    for pname, pdef in params_spec.get("properties", {}).items():
        prop = dict(pdef)
        if prop.get("type") == "dict":
            prop["type"] = "object"
        if prop.get("type") == "float":
            prop["type"] = "number"
        if prop.get("type") == "array" and "items" in prop:
            items = prop["items"]
            if items.get("type") == "dict":
                items["type"] = "object"
            if items.get("type") == "float":
                items["type"] = "number"
        if "default" in prop and prop["default"] == "None":
            del prop["default"]
        properties[pname] = prop
    return {
        "properties": properties,
        "required": params_spec.get("required", []),
    }


def _setup_backend_simulators(
    involved_classes: list[str],
    initial_config: dict,
) -> dict[str, Any]:
    """Instantiate BFCL backend simulators and return a method_name -> method map."""
    method_map = {}
    for class_name in involved_classes:
        module_name = CLASS_FILE_PATH_MAPPING[class_name]
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
        instance = cls()
        config = initial_config.get(class_name, {})
        if config:
            instance._load_scenario(copy.deepcopy(config))
        for mname, method in inspect.getmembers(instance, predicate=inspect.ismethod):
            if not mname.startswith("_"):
                method_map[mname] = method
    return method_map


def _build_tools(
    function_specs: list[dict],
    excluded_functions: list[str],
    method_map: dict[str, Any],
    call_log: list[tuple[str, dict]],
) -> list[Tool]:
    """Build Tool objects from BFCL function specs with logging implementations."""
    excluded = set(excluded_functions)
    tools = []

    for func_spec in function_specs:
        fname = func_spec["name"]
        if fname in excluded or fname not in method_map:
            continue

        raw_params = _bfcl_params_to_json_schema(func_spec["parameters"])

        def _make_impl(func_name: str):
            def impl(**kwargs):
                call_log.append((func_name, kwargs))
                method = method_map[func_name]
                result = method(**kwargs)
                if isinstance(result, dict):
                    try:
                        return json.loads(json.dumps(result))
                    except (TypeError, ValueError):
                        return {"result": str(result)}
                elif isinstance(result, str):
                    return {"result": result}
                elif isinstance(result, list):
                    return {"result": result}
                elif result is None:
                    return {"result": "success"}
                else:
                    return {"result": str(result)}
            return impl

        tools.append(Tool(
            name=fname,
            description=func_spec["description"],
            params=raw_params,
            function=_make_impl(fname),
        ))

    return tools


def run_test_case(
    backend: Backend,
    entry: dict,
    ground_truth: list[list[str]],
    constraint_fns: list[Callable] | None = None,
    system_prompt: str | None = None,
    max_turns: int = 15,
    inference_kwargs: dict | None = None,
    verbose: bool = False,
) -> dict:
    """Run a single BFCL multi_turn_base test case.

    Parameters
    ----------
    backend:
        The generation backend to use.
    entry:
        A BFCL test entry dict (from ``load_dataset_entry``).
    ground_truth:
        List of ground truth call lists per turn.
    constraint_fns:
        Constraint functions to apply. Pass ``None`` or ``[]`` for unconstrained.
    system_prompt:
        Override the default system prompt.
    max_turns:
        Maximum internal agent turns per user message.
    inference_kwargs:
        Extra kwargs passed to the backend's generate method.
    verbose:
        Print turn-by-turn debug output.

    Returns
    -------
    dict with keys:
        - ``all_model_responses``: list[list[list[str]]] for the BFCL checker
        - ``turn_details``: per-turn breakdown of calls made vs expected
    """
    call_log: list[tuple[str, dict]] = []

    method_map = _setup_backend_simulators(
        entry["involved_classes"],
        entry.get("initial_config", {}),
    )
    tools = _build_tools(
        entry["function"],
        entry.get("excluded_function", []),
        method_map,
        call_log,
    )

    if system_prompt is None:
        system_prompt = (
            "You are a helpful assistant. "
            "Execute the requested operations step by step using the available tools. "
            "Call tools one at a time in the correct order."
        )

    agent = Agent(
        backend,
        format="qwen3",
        rules=constraint_fns or [],
        system_prompt=system_prompt,
        max_turns=max_turns,
        inference_kwargs=inference_kwargs or {},
        stop_after_first=False,
        at_least_one=False,
        verbose=verbose,
    )
    agent.registry = TR(tools)

    all_model_responses = []
    turn_details = []

    for turn_idx, turn_messages in enumerate(entry["question"]):
        user_msg = turn_messages[0]["content"]

        if verbose:
            print(f"\n{'='*60}")
            print(f"TURN {turn_idx}: {user_msg[:120]}")
            print(f"{'='*60}")
            print(f"Expected: {ground_truth[turn_idx]}")

        log_start = len(call_log)
        response = agent.chat(user_msg)
        turn_calls = call_log[log_start:]

        turn_call_strings = []
        for fname, kwargs in turn_calls:
            arg_parts = [f"{k}={v!r}" for k, v in kwargs.items()]
            turn_call_strings.append(f"{fname}({', '.join(arg_parts)})")

        turn_decoded = [[s] for s in turn_call_strings]
        all_model_responses.append(turn_decoded)

        if verbose:
            print(f"Agent response: {response[:200]}")
            print(f"Calls made: {turn_call_strings}")
            print(f"Expected:   {ground_truth[turn_idx]}")

        turn_details.append({
            "turn_idx": turn_idx,
            "user_message": user_msg,
            "calls": turn_call_strings,
            "expected": ground_truth[turn_idx],
            "response": response[:500],
        })

    return {
        "all_model_responses": all_model_responses,
        "turn_details": turn_details,
    }
