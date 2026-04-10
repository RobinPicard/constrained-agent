from __future__ import annotations
import json
import re
import outlines
import xgrammar as xgr
from outlines.backends.xgrammar import XGrammarLogitsProcessor
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Literal, Type
from pydantic import BaseModel
from .session import Session
from .tools import Tool, ToolRegistry
from .format import ModelFormat
from .spec import AgentSpec


def _class_to_tool_name(cls: type) -> str:
    """Resolve the tool name for a Pydantic class.

    Checks for an explicit ``name`` class attribute first, then falls back to
    stripping common suffixes (Params, Parameters, Args, Input) and converting
    CamelCase to snake_case: ``CheckBalanceParams`` → ``check_balance``.
    """
    explicit = cls.__dict__.get("name")
    if isinstance(explicit, str):
        return explicit
    name = cls.__name__
    for suffix in ("Parameters", "Params", "Args", "Input"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


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
        model,
        implementations: dict[str, Callable] | None = None,
        format: ModelFormat | str | None = None,
        spec: str | Path | dict | AgentSpec | None = None,
        tools: list[Type[BaseModel]] | None = None,
        tools_mode: Literal["replace", "merge"] = "replace",
        rules: list[Callable] | None = None,
        rules_mode: Literal["replace", "merge"] = "replace",
        system_prompt: str | None = None,
        max_turns: int | None = None,
        inference_kwargs: dict | None = None,
        max_concurrent_tool_calls: int | None = None,
        verbose: bool = False,
    ):
        """Create an Agent.

        Parameters
        ----------
        model:
            The language model to use.
        implementations:
            Mapping from tool name to callable implementation.
        format:
            Model output format — a ``ModelFormat`` object or a format name
            string (e.g. ``"qwen3"``). If omitted, loaded from the spec.
        spec:
            Path to a JSON file, a plain dict, or an ``AgentSpec``. Provides
            defaults for all other arguments; explicit arguments take precedence.
        tools:
            List of Pydantic ``BaseModel`` classes defining tool schemas.
            Set an explicit ``name: ClassVar[str] = "tool_name"`` class attribute
            to control the tool name; otherwise it is derived from the class name.
        tools_mode:
            ``"replace"`` (default) — Python tools replace the spec's tool
            definitions entirely. ``"merge"`` — Python tools are added on top
            of the spec's tools; spec tools with the same name are overridden.
        rules:
            Python constraint callables.
            Use this for patterns not expressible in the JSON DSL.
        rules_mode:
            ``"replace"`` (default) — only the Python rules apply; spec rules
            are discarded. ``"merge"`` — Python rules run after spec rules,
            both apply.
        system_prompt, max_turns, inference_kwargs, max_concurrent_tool_calls:
            Agent configuration. Override spec values when provided.
        verbose:
            Print turn-by-turn debug output.
        """
        # --- load spec ---
        agent_spec: AgentSpec | None = None
        if spec is not None:
            if isinstance(spec, AgentSpec):
                agent_spec = spec
            elif isinstance(spec, dict):
                agent_spec = AgentSpec.from_dict(spec)
            else:
                agent_spec = AgentSpec.load(spec)

        # --- resolve format ---
        if isinstance(format, str):
            format = ModelFormat.load(format)
        if format is None and agent_spec and agent_spec.format:
            format = ModelFormat.load(agent_spec.format)
        if format is None:
            raise ValueError("format must be provided either as an argument or in the spec")

        # --- build tools ---
        impls = implementations or {}

        def _build_python_tools(classes):
            return [
                Tool(
                    _class_to_tool_name(cls),
                    (cls.__doc__ or "").strip(),
                    cls,
                    impls[_class_to_tool_name(cls)],
                )
                for cls in classes
            ]

        def _build_spec_tools():
            if not (agent_spec and agent_spec.tools_spec):
                return []
            return [
                Tool(
                    name,
                    tool_def["description"],
                    {"properties": tool_def.get("params", {}), "required": tool_def.get("required", [])},
                    impls[name],
                )
                for name, tool_def in agent_spec.tools_spec.items()
            ]

        if tools is not None and tools_mode == "replace":
            built_tools = _build_python_tools(tools)
        elif tools is not None and tools_mode == "merge":
            python_tools = {_class_to_tool_name(cls): cls for cls in tools}
            spec_tools = [t for t in _build_spec_tools() if t.name not in python_tools]
            built_tools = spec_tools + _build_python_tools(tools)
        else:
            built_tools = _build_spec_tools()

        # --- build constraints ---
        spec_constraints = agent_spec.to_constraints() if agent_spec else []
        if rules_mode == "merge":
            constraints = spec_constraints + (rules or [])
        else:
            constraints = rules if rules is not None else spec_constraints

        # --- resolve config (explicit args override spec) ---
        spec_inference_kwargs = agent_spec.inference_kwargs if agent_spec else {}

        self.model = model
        self.format = format
        self.registry = ToolRegistry(built_tools)
        self.constraints = constraints
        self.system_prompt = system_prompt if system_prompt is not None else (agent_spec.system_prompt if agent_spec else None)
        self.max_turns = max_turns if max_turns is not None else (agent_spec.max_turns if agent_spec else None) or 10
        self.inference_kwargs = inference_kwargs if inference_kwargs is not None else spec_inference_kwargs
        self.max_concurrent_tool_calls = max_concurrent_tool_calls if max_concurrent_tool_calls is not None else (agent_spec.max_concurrent_tool_calls if agent_spec else None)
        self.verbose = verbose
        self.session = Session()
        tokenizer_info = xgr.TokenizerInfo.from_huggingface(
            model.hf_tokenizer, vocab_size=model.model.config.vocab_size
        )
        self._grammar_compiler = xgr.GrammarCompiler(tokenizer_info)
        self._tensor_library_name = model.tensor_library_name

    def _apply_constraints(self) -> None:
        self.registry.reset_all()
        for c in self.constraints:
            c(self.session, self.registry)

    def _build_system(self) -> str:
        parts = []
        if self.system_prompt:
            parts.append(self.system_prompt)
        tools = self.format.format_tools(self.registry)
        parts.append(self.format.tools_to_text(tools))
        parts.append(self.format.format_instructions())
        parts.append(_CONSTRAINT_NOTICE)
        unavailable = self.registry.unavailable_tools()
        if unavailable:
            lines = ["Currently unavailable tools (exist but cannot be called yet):"]
            for t in unavailable:
                line = f"- {t.name}: {t.schema.description}"
                if t.schema.unavailable_reason:
                    line += f" Unavailable: {t.schema.unavailable_reason}"
                lines.append(line)
            parts.append("\n".join(lines))
        return "\n\n".join(parts)

    def _generate(self, messages: list[dict]) -> str:
        prompt = self.model.tokenizer.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        structural_tag = self.format.structural_tags(
            self.registry,
            stop_after_first=self.stop_after_first,
            at_least_one=self.at_least_one,
        )
        grammar = xgr.Grammar.from_structural_tag(structural_tag)
        compiled = self._grammar_compiler.compile_grammar(grammar)
        processor = XGrammarLogitsProcessor(compiled, self._tensor_library_name)
        generator = outlines.Generator(self.model, processor=processor)
        return generator(prompt, **self.inference_kwargs)

    def reset(self) -> None:
        self.session = Session()

    def run(self, user_message: str) -> str:
        """Single-shot: resets session, runs to completion, returns response."""
        self.reset()
        return self._run_turn(user_message)

    def chat(self, user_message: str) -> str:
        """Multi-turn: preserves session and tool history across calls."""
        return self._run_turn(user_message)

    def _execute_tool(self, tc: dict) -> tuple[str, Any, str, dict]:
        try:
            result = self.registry.execute(tc["name"], **tc["args"])
        except Exception as e:
            result = {"error": type(e).__name__, "message": str(e)}
        return tc["id"], result, tc["name"], tc["args"]

    def _run_turn(self, user_message: str) -> str:
        self.session.add_message({"role": "user", "content": user_message})

        for turn in range(self.max_turns):
            self._apply_constraints()

            if self.verbose:
                print(f"[turn {turn + 1}] available tools:\n{_format_tool_summary(self.registry)}")

            messages = [{"role": "system", "content": self._build_system()}] + self.session.messages
            raw = self._generate(messages)
            parsed = self.format.parse(raw)
            text = parsed.content or None
            tool_calls = parsed.tool_calls

            if self.verbose and text:
                print(f"[turn {turn + 1}] model: {text}")

            self.session.add_message({"role": "assistant", "content": raw})

            if not tool_calls:
                return text or ""

            if self.max_concurrent_tool_calls is not None:
                with ThreadPoolExecutor(max_workers=self.max_concurrent_tool_calls) as executor:
                    raw_results = list(executor.map(self._execute_tool, tool_calls))
            else:
                raw_results = [self._execute_tool(tc) for tc in tool_calls]

            for call_id, result, name, args in raw_results:
                if self.verbose:
                    print(f"[turn {turn + 1}] call: {name}({args})")
                    print(f"[turn {turn + 1}] result: {result}")
                self.session.record_tool_call(name, args, result)

            tool_result_lines = [
                f"{name} ({call_id}): {result if isinstance(result, str) else json.dumps(result)}"
                for call_id, result, name, _ in raw_results
            ]
            self.session.add_message({
                "role": "user",
                "content": "Tool results:\n" + "\n".join(tool_result_lines),
            })

        return "Max turns reached."
