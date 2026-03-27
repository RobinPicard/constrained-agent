from __future__ import annotations
from dataclasses import dataclass
from typing import Any


@dataclass
class ToolRun:
    args: dict[str, Any]
    result: Any


class ToolHistory:
    def __init__(self):
        self.runs: list[ToolRun] = []

    @property
    def has_run(self) -> bool:
        return bool(self.runs)

    @property
    def run_count(self) -> int:
        return len(self.runs)

    @property
    def last_result(self) -> Any:
        return self.runs[-1].result if self.runs else None


class Session:
    def __init__(self):
        self.messages: list[dict] = []
        self._tool_history: dict[str, ToolHistory] = {}

    def tool(self, name: str) -> ToolHistory:
        if name not in self._tool_history:
            self._tool_history[name] = ToolHistory()
        return self._tool_history[name]

    def record_tool_call(self, name: str, args: dict, result: Any) -> None:
        self.tool(name).runs.append(ToolRun(args=args, result=result))

    def add_message(self, message: dict) -> None:
        self.messages.append(message)

    def add_messages(self, messages: list[dict]) -> None:
        self.messages.extend(messages)
