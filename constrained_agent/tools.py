from __future__ import annotations
import inspect
from copy import deepcopy
from typing import Any, Callable, Type
from pydantic import BaseModel, create_model


class ParamSchema:
    def __init__(self, name: str, schema: dict, is_required: bool):
        self.name = name
        self._base_schema = deepcopy(schema)
        self._current: dict = deepcopy(schema)
        self._base_required = is_required
        self.required: bool = is_required

    @property
    def maximum(self) -> float | None:
        return self._current.get("maximum")

    @maximum.setter
    def maximum(self, value: float) -> None:
        self._current["maximum"] = value

    @property
    def minimum(self) -> float | None:
        return self._current.get("minimum")

    @minimum.setter
    def minimum(self, value: float) -> None:
        self._current["minimum"] = value

    @property
    def enum(self) -> list | None:
        return self._current.get("enum")

    @enum.setter
    def enum(self, value: list) -> None:
        self._current["enum"] = value

    def to_schema(self) -> dict:
        return deepcopy(self._current)

    def reset(self) -> None:
        self._current = deepcopy(self._base_schema)
        self.required = self._base_required


class ToolSchema:
    def __init__(self, name: str, description: str, params: dict[str, ParamSchema]):
        self.name = name
        self.description = description
        self.params = params
        self.available: bool = True
        self.unavailable_reason: str | None = None

    def reset(self) -> None:
        for param in self.params.values():
            param.reset()
        self.available = True
        self.unavailable_reason = None

    def to_json_schema(self) -> dict:
        properties = {name: param.to_schema() for name, param in self.params.items()}
        required = [name for name, param in self.params.items() if param.required]
        schema: dict[str, Any] = {"type": "object", "properties": properties}
        if required:
            schema["required"] = required
        return schema


class Tool:
    def __init__(
        self,
        name: str,
        description: str,
        params: dict | Type[BaseModel],
        function: Callable,
    ):
        self.name = name
        self.function = function

        raw = params if isinstance(params, dict) else params.model_json_schema()

        properties = raw.get("properties", {})
        required_fields = set(raw.get("required", []))

        param_schemas = {
            pname: ParamSchema(pname, pdef, pname in required_fields)
            for pname, pdef in properties.items()
        }
        self.schema = ToolSchema(name, description, param_schemas)

    def execute(self, **kwargs) -> Any:
        return self.function(**kwargs)


def _tool_from_callable(func: Callable) -> Tool:
    description = (func.__doc__ or "").strip()
    sig = inspect.signature(func)
    fields: dict[str, Any] = {}
    for name, param in sig.parameters.items():
        annotation = param.annotation if param.annotation is not inspect.Parameter.empty else Any
        if param.default is inspect.Parameter.empty:
            fields[name] = (annotation, ...)
        else:
            fields[name] = (annotation, param.default)
    params_model = create_model(func.__name__ + "_params", **fields)
    return Tool(func.__name__, description, params_model, func)


class ToolRegistry:
    def __init__(self, tools: list[Tool | Callable]):
        normalized = [t if isinstance(t, Tool) else _tool_from_callable(t) for t in tools]
        self._tools: dict[str, Tool] = {t.name: t for t in normalized}

    def __getitem__(self, name: str) -> ToolSchema:
        return self._tools[name].schema

    def reset_all(self) -> None:
        for tool in self._tools.values():
            tool.schema.reset()

    def available_tools(self) -> list[Tool]:
        return [t for t in self._tools.values() if t.schema.available]

    def unavailable_tools(self) -> list[Tool]:
        return [t for t in self._tools.values() if not t.schema.available]

    def execute(self, tool_name: str, **kwargs) -> Any:
        return self._tools[tool_name].execute(**kwargs)
