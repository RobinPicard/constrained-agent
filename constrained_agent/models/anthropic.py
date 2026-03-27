from __future__ import annotations
import json
from typing import Any
from anthropic import Anthropic
from .base import ModelAdapter
from ..tools import ToolRegistry


class AnthropicAdapter(ModelAdapter):
    def __init__(self, model: str = "claude-opus-4-6", **client_kwargs):
        self.model = model
        self.client = Anthropic(**client_kwargs)

    def format_tools(self, registry: ToolRegistry) -> list[dict]:
        return [
            {
                "name": tool.name,
                "description": tool.schema.description,
                "input_schema": tool.schema.to_json_schema(),
            }
            for tool in registry.available_tools()
        ]

    def complete(
        self,
        messages: list[dict],
        tools: list[dict],
        system: str | None = None,
    ) -> tuple[str | None, list[dict], Any]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": 4096,
            "messages": messages,
        }
        if tools:
            kwargs["tools"] = tools
        if system:
            kwargs["system"] = system

        response = self.client.messages.create(**kwargs)

        text = None
        tool_calls = []
        for block in response.content:
            if block.type == "text":
                text = block.text
            elif block.type == "tool_use":
                tool_calls.append({
                    "id": block.id,
                    "name": block.name,
                    "args": block.input,
                })

        return text, tool_calls, response

    def build_assistant_message(
        self, text: str | None, tool_calls: list[dict], raw_response: Any
    ) -> dict:
        # raw_response.content is already a list of typed content blocks;
        # pass it through so the SDK objects are preserved for the next request.
        return {"role": "assistant", "content": raw_response.content}

    def build_tool_result_messages(
        self, results: list[tuple[str, Any]]
    ) -> list[dict]:
        # All results from the same turn are combined into one user message.
        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": call_id,
                        "content": result if isinstance(result, str) else json.dumps(result),
                    }
                    for call_id, result in results
                ],
            }
        ]
