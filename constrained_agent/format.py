"""
Model output format descriptors.

A format is a YAML file in ``constrained_agent/formats/`` that describes how a
specific model family encodes its three output groups — **think**, **content**,
and **tool calls** — and from which both a text parser and an xgrammar
``StructuralTagSchema`` can be derived.

Two function-call styles are supported:

tagged_name
-----------
The tool name is part of the opening delimiter, so each call block has a unique
begin token.  Example (Qwen3)::

    <function=get_balance>
    {"account_id": "42"}
    </function>

Required fields under ``function``:
  prefix  — token immediately before the tool name  (e.g. ``"<function="``)
  sep     — token immediately after the tool name   (e.g. ``">"``)
  close   — token that closes the block             (e.g. ``"</function>"``)

Structural tags: one ``StructuralTagItem`` per available tool.
  begin   = prefix + tool_name + sep
  schema  = the tool's JSON parameter schema
  end     = close
  trigger = prefix

json_body
---------
The tool name is inside the JSON body between fixed open/close delimiters,
so all calls share the same begin token.  Example (Hermes)::

    <tool_call>
    {"name": "get_balance", "arguments": {"account_id": "42"}}
    </tool_call>

With optional wrapper (DeepSeek)::

    <｜tool▁calls▁begin｜>
    <｜tool▁call▁begin｜>
    {"name": "get_balance", "parameters": {"account_id": "42"}}
    <｜tool▁call▁end｜>
    <｜tool▁calls▁end｜>

Required fields under ``function``:
  open        — token that opens each call block    (e.g. ``"<tool_call>"``)
  close       — token that closes each call block   (e.g. ``"</tool_call>"``)
  name_field  — JSON key holding the tool name      (e.g. ``"name"``)
  args_field  — JSON key holding the arguments      (e.g. ``"arguments"``)

Optional fields:
  wrapper_open  — outer token wrapping all calls
  wrapper_close — matching close for the wrapper

Structural tags: one ``StructuralTagItem`` for all tools combined.
  begin   = open
  schema  = anyOf union over all tool schemas, discriminated by name_field
  end     = close
  trigger = wrapper_open if present, else open

Both styles support an optional top-level ``think`` section:
  prefix  — opens the thinking block   (e.g. ``"<think>"``)
  close   — closes the thinking block  (e.g. ``"</think>"``)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_FORMATS_DIR = Path(__file__).parent / "formats"

_TAGGED_NAME_REQUIRED = {"prefix", "sep", "close"}
_JSON_BODY_REQUIRED = {"open", "close", "name_field", "args_field"}


@dataclass
class ParsedOutput:
    """Structured result of parsing raw model output into its three groups."""
    content: str
    tool_calls: list[dict] = field(default_factory=list)
    think: str | None = None


class ModelFormat:
    """
    Loads a YAML format descriptor and exposes parsing and structural-tag
    generation for a specific model family.

    Use :meth:`load` to instantiate from a named descriptor in the
    ``formats/`` directory, or pass a spec dict directly for testing::

        fmt = ModelFormat.load("qwen3")
        fmt = ModelFormat({"function": {"style": "tagged_name", ...}})
    """

    def __init__(self, spec: dict):
        self._spec = spec
        self._think = spec.get("think")
        self._func = spec["function"]
        self._style: str = self._func["style"]
        self._validate()

    @classmethod
    def load(cls, name: str) -> ModelFormat:
        """Load a format descriptor by name from the built-in ``formats/`` directory."""
        path = _FORMATS_DIR / f"{name}.yaml"
        if not path.exists():
            available = [p.stem for p in _FORMATS_DIR.glob("*.yaml")]
            raise FileNotFoundError(
                f"No format descriptor '{name}.yaml' found. "
                f"Available: {available}"
            )
        return cls(yaml.safe_load(path.read_text()))

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate(self) -> None:
        if self._style == "tagged_name":
            missing = _TAGGED_NAME_REQUIRED - self._func.keys()
        elif self._style == "json_body":
            missing = _JSON_BODY_REQUIRED - self._func.keys()
        else:
            raise ValueError(
                f"Unknown function style: {self._style!r}. "
                "Must be 'tagged_name' or 'json_body'."
            )
        if missing:
            raise ValueError(
                f"Style '{self._style}' is missing required fields: {sorted(missing)}"
            )

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def parse(self, text: str) -> ParsedOutput:
        """Parse raw model output into its three groups.

        Extracts the think block first (if the format defines one), then
        dispatches to the style-specific function parser.
        """
        remaining = text
        think: str | None = None

        if self._think:
            tp = re.escape(self._think["prefix"])
            tc = re.escape(self._think["close"])
            m = re.search(f"{tp}(.*?){tc}", remaining, re.DOTALL)
            if m:
                think = m.group(1).strip()
                remaining = remaining[m.end():]

        if self._style == "tagged_name":
            content, tool_calls = self._parse_tagged_name(remaining)
        else:
            content, tool_calls = self._parse_json_body(remaining)

        return ParsedOutput(content=content, tool_calls=tool_calls, think=think)

    def _parse_tagged_name(self, text: str) -> tuple[str, list[dict]]:
        f = self._func
        fp = re.escape(f["prefix"])
        fs = re.escape(f["sep"])
        fc = re.escape(f["close"])

        func_re = re.compile(f"{fp}(\\w+){fs}(.*?){fc}", re.DOTALL)
        first = func_re.search(text)
        content = text[: first.start()].strip() if first else text.strip()

        tool_calls = []
        for i, m in enumerate(func_re.finditer(text)):
            try:
                args = json.loads(m.group(2).strip())
            except json.JSONDecodeError:
                args = {}
            tool_calls.append({"id": f"call_{i}", "name": m.group(1), "args": args})

        return content, tool_calls

    def _parse_json_body(self, text: str) -> tuple[str, list[dict]]:
        f = self._func
        op = re.escape(f["open"])
        cl = re.escape(f["close"])
        name_field = f["name_field"]
        args_field = f["args_field"]

        if "wrapper_open" in f:
            wo = re.escape(f["wrapper_open"])
            wc = re.escape(f["wrapper_close"])
            wrapper_m = re.search(f"{wo}(.*?){wc}", text, re.DOTALL)
            if not wrapper_m:
                return text.strip(), []
            content = text[: wrapper_m.start()].strip()
            inner = wrapper_m.group(1)
        else:
            func_scan = re.compile(f"{op}.*?{cl}", re.DOTALL)
            first = func_scan.search(text)
            content = text[: first.start()].strip() if first else text.strip()
            inner = text

        tool_calls = []
        for i, m in enumerate(re.finditer(f"{op}(.*?){cl}", inner, re.DOTALL)):
            try:
                body = json.loads(m.group(1).strip())
                tool_calls.append({
                    "id": f"call_{i}",
                    "name": body[name_field],
                    "args": body[args_field],
                })
            except (json.JSONDecodeError, KeyError):
                pass

        return content, tool_calls

    # ------------------------------------------------------------------
    # Structural tag schema
    # ------------------------------------------------------------------

    def structural_tags(
        self,
        registry: Any,
        *,
        stop_after_first: bool = False,
        at_least_one: bool = False,
    ) -> dict:
        """Return a structural tag schema dict for the available tools.

        The returned dict follows the xgrammar StructuralTag format and can be
        passed directly to an OpenAI-compatible API as ``response_format``.
        """
        if self._style == "tagged_name":
            return self._structural_tags_tagged_name(
                registry,
                stop_after_first=stop_after_first, at_least_one=at_least_one,
            )
        else:
            return self._structural_tags_json_body(
                registry,
                stop_after_first=stop_after_first, at_least_one=at_least_one,
            )

    def _structural_tags_tagged_name(
        self, registry, *, stop_after_first, at_least_one,
    ) -> dict:
        f = self._func
        tags = [
            {
                "type": "tag",
                "begin": f["prefix"] + tool.name + f["sep"],
                "content": {
                    "type": "json_schema",
                    "json_schema": tool.schema.to_json_schema(),
                    "style": "json",
                },
                "end": f["close"],
            }
            for tool in registry.available_tools()
        ]
        return {
            "type": "structural_tag",
            "format": {
                "type": "triggered_tags",
                "triggers": [f["prefix"]],
                "tags": tags,
                "at_least_one": at_least_one,
                "stop_after_first": stop_after_first,
                "excludes": [],
            },
        }

    def _structural_tags_json_body(
        self, registry, *, stop_after_first, at_least_one,
    ) -> dict:
        f = self._func
        name_field = f["name_field"]
        args_field = f["args_field"]

        union_schema = {
            "anyOf": [
                {
                    "type": "object",
                    "properties": {
                        name_field: {"const": tool.name},
                        args_field: tool.schema.to_json_schema(),
                    },
                    "required": [name_field, args_field],
                }
                for tool in registry.available_tools()
            ]
        }
        trigger = f.get("wrapper_open", f["open"])
        tag = {
            "type": "tag",
            "begin": f["open"],
            "content": {
                "type": "json_schema",
                "json_schema": union_schema,
                "style": "json",
            },
            "end": f["close"],
        }
        return {
            "type": "structural_tag",
            "format": {
                "type": "triggered_tags",
                "triggers": [trigger],
                "tags": [tag],
                "at_least_one": at_least_one,
                "stop_after_first": stop_after_first,
                "excludes": [],
            },
        }

    # ------------------------------------------------------------------
    # System-prompt helpers (used by local adapters)
    # ------------------------------------------------------------------

    def format_instructions(self) -> str:
        """Return a prompt snippet showing the model the expected tool-call format."""
        f = self._func
        if self._style == "tagged_name":
            body = json.dumps({"param1": "value1", "param2": "value2"}, indent=2)
            example = f"{f['prefix']}tool_name{f['sep']}\n{body}\n{f['close']}"
        else:
            body = json.dumps(
                {f["name_field"]: "tool_name", f["args_field"]: {"param1": "value1"}},
                indent=2,
            )
            example = f"{f['open']}\n{body}\n{f['close']}"
            if "wrapper_open" in f:
                example = f"{f['wrapper_open']}\n{example}\n{f['wrapper_close']}"

        return (
            "To call a tool, output it using this exact format "
            "(plain text may appear before the call):\n" + example
        )

    def format_tools(self, registry: Any) -> list[dict]:
        """Serialize available tools to a list of dicts for use with tools_to_text."""
        return [
            {
                "name": tool.name,
                "description": tool.schema.description,
                "parameters": tool.schema.to_json_schema(),
            }
            for tool in registry.available_tools()
        ]

    def tools_to_text(self, tools: list[dict]) -> str:
        """Render tool schemas as human-readable text for the system prompt."""
        lines = ["Available tools:"]
        for tool in tools:
            schema = tool["parameters"]
            props = schema.get("properties", {})
            required = schema.get("required", [])
            params: list[str] = []
            for pname, pdef in props.items():
                req = " (required)" if pname in required else ""
                ptype = pdef.get("type", "any")
                pdesc = f": {pdef['description']}" if "description" in pdef else ""
                params.append(f"    {pname} ({ptype}){req}{pdesc}")
            param_block = ("\n" + "\n".join(params)) if params else ""
            lines.append(f"\n- {tool['name']}: {tool['description']}{param_block}")
        return "\n".join(lines)
