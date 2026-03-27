from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any
from ..tools import ToolRegistry


class ModelAdapter(ABC):
    @abstractmethod
    def complete(
        self,
        messages: list[dict],
        tools: list[dict],
        system: str | None = None,
    ) -> tuple[str | None, list[dict], Any]:
        """
        Call the model and return (text, tool_calls, raw_response).

        tool_calls: [{"id": str, "name": str, "args": dict}, ...]
        raw_response: provider-specific response object, passed back to
                      build_assistant_message unchanged.
        """

    @abstractmethod
    def format_tools(self, registry: ToolRegistry) -> list[dict]:
        """Serialize available tools to the provider's expected format."""

    @abstractmethod
    def build_assistant_message(
        self, text: str | None, tool_calls: list[dict], raw_response: Any
    ) -> dict:
        """Build the assistant message dict to append to history."""

    @abstractmethod
    def build_tool_result_messages(
        self, results: list[tuple[str, Any]]
    ) -> list[dict]:
        """
        Build message(s) to append for tool results.
        results: [(tool_call_id, result), ...]
        """
