from __future__ import annotations
import json
from typing import Any
from openai import OpenAI
from .base import ModelAdapter
from ..tools import ToolRegistry


class OpenAIAdapter(ModelAdapter):
    def __init__(self, model: str = "gpt-4o", **client_kwargs):
        self.model = model
        self.client = OpenAI(**client_kwargs)

    def format_tools(self, registry: ToolRegistry) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.schema.description,
                    "parameters": tool.schema.to_json_schema(),
                },
            }
            for tool in registry.available_tools()
        ]

    def complete(
        self,
        messages: list[dict],
        tools: list[dict],
        system: str | None = None,
    ) -> tuple[str | None, list[dict], Any]:
        all_messages = messages
        if system:
            all_messages = [{"role": "system", "content": system}] + messages

        kwargs: dict[str, Any] = {"model": self.model, "messages": all_messages}
        if tools:
            kwargs["tools"] = tools

        response = self.client.chat.completions.create(**kwargs)
        message = response.choices[0].message

        tool_calls = []
        if message.tool_calls:
            for tc in message.tool_calls:
                tool_calls.append({
                    "id": tc.id,
                    "name": tc.function.name,
                    "args": json.loads(tc.function.arguments),
                })

        return message.content, tool_calls, response

    def build_assistant_message(
        self, text: str | None, tool_calls: list[dict], raw_response: Any
    ) -> dict:
        message = raw_response.choices[0].message
        result: dict[str, Any] = {"role": "assistant", "content": message.content}
        if message.tool_calls:
            result["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in message.tool_calls
            ]
        return result

    def build_tool_result_messages(
        self, results: list[tuple[str, Any]]
    ) -> list[dict]:
        return [
            {
                "role": "tool",
                "tool_call_id": call_id,
                "content": result if isinstance(result, str) else json.dumps(result),
            }
            for call_id, result in results
        ]
