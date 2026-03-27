from __future__ import annotations
from typing import Any
from .session import Session
from .constraints import ConstraintEvaluator
from .tools import ToolRegistry
from .models.base import ModelAdapter


def _format_tool_summary(registry: ToolRegistry) -> str:
    lines = []
    for tool in registry.available_tools():
        schema = tool.schema
        param_parts = []
        for name, param in schema.params.items():
            desc = name
            constraints = []
            if param.minimum is not None:
                constraints.append(f">={param.minimum}")
            if param.maximum is not None:
                constraints.append(f"<={param.maximum}")
            if param.enum is not None:
                constraints.append(f"enum={param.enum}")
            if not param.required:
                constraints.append("optional")
            if constraints:
                desc += f" ({', '.join(constraints)})"
            param_parts.append(desc)
        lines.append(f"  - {schema.name}({', '.join(param_parts)})")
    return "\n".join(lines) if lines else "  (none)"


_CONSTRAINT_NOTICE = (
    "The set of available tools and their parameter constraints (allowed values, ranges, "
    "required fields) reflect the current session state and may change after each tool call. "
    "Always work within the tools and parameter bounds currently provided."
)


class Agent:
    def __init__(
        self,
        model: ModelAdapter,
        registry: ToolRegistry,
        evaluator: ConstraintEvaluator,
        system_prompt: str | None = None,
        max_turns: int = 10,
        verbose: bool = False,
    ):
        self.model = model
        self.registry = registry
        self.evaluator = evaluator
        self.system_prompt = system_prompt
        self.max_turns = max_turns
        self.verbose = verbose
        self.session = Session()

    def reset(self) -> None:
        self.session = Session()

    def run(self, user_message: str) -> str:
        """Single-shot: resets session, runs to completion, returns response."""
        self.reset()
        return self._run_turn(user_message)

    def chat(self, user_message: str) -> str:
        """Multi-turn: preserves session and tool history across calls."""
        return self._run_turn(user_message)

    def _run_turn(self, user_message: str) -> str:
        self.session.add_message({"role": "user", "content": user_message})

        for turn in range(self.max_turns):
            # Evaluate constraints → reset + mutate tool schemas
            self.evaluator.evaluate(self.session, self.registry)

            if self.verbose:
                print(f"[turn {turn + 1}] available tools:\n{_format_tool_summary(self.registry)}")

            # Serialize constrained schemas to provider format
            tools = self.model.format_tools(self.registry)

            # Call the model
            system_parts = []
            if self.system_prompt:
                system_parts.append(self.system_prompt)
            system_parts.append(_CONSTRAINT_NOTICE)
            unavailable = self.registry.unavailable_tools()
            if unavailable:
                lines = ["Currently unavailable tools (exist but cannot be called yet):"]
                for t in unavailable:
                    line = f"- {t.name}: {t.schema.description}"
                    if t.schema.unavailable_reason:
                        line += f" Unavailable: {t.schema.unavailable_reason}"
                    lines.append(line)
                system_parts.append("\n".join(lines))
            system = "\n\n".join(system_parts)
            text, tool_calls, raw = self.model.complete(
                self.session.messages,
                tools,
                system=system,
            )

            if self.verbose and text:
                print(f"[turn {turn + 1}] model: {text}")

            # Record assistant turn in history
            self.session.add_message(
                self.model.build_assistant_message(text, tool_calls, raw)
            )

            # No tool calls → the model is done
            if not tool_calls:
                return text or ""

            # Execute tools and collect results
            results: list[tuple[str, Any]] = []
            for tc in tool_calls:
                if self.verbose:
                    print(f"[turn {turn + 1}] call: {tc['name']}({tc['args']})")
                try:
                    result = self.registry.execute(tc["name"], **tc["args"])
                    self.session.record_tool_call(tc["name"], tc["args"], result)
                except Exception as e:
                    result = {"error": type(e).__name__, "message": str(e)}
                if self.verbose:
                    print(f"[turn {turn + 1}] result: {result}")
                results.append((tc["id"], result))

            # Add tool results to history
            self.session.add_messages(
                self.model.build_tool_result_messages(results)
            )

        return "Max turns reached."
