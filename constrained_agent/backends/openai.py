from __future__ import annotations

from typing import Any

from openai import OpenAI

from .base import Backend


class OpenAIBackend(Backend):
    """Generation backend using an OpenAI-compatible API.

    The structural tag is passed as ``response_format`` so the server enforces
    output structure at the token level.

    Parameters
    ----------
    model:
        Model name to pass to the API (e.g. ``"Qwen/Qwen3-5B"``).
    base_url:
        API base URL. Defaults to OpenAI's endpoint.
    api_key:
        API key. Falls back to the ``OPENAI_API_KEY`` environment variable.
    """

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
    ):
        self.model = model
        self._client = OpenAI(base_url=base_url, api_key=api_key)

    def generate(
        self,
        messages: list[dict],
        structural_tag: dict,
        **kwargs,
    ) -> str:
        # Merge extra_body from kwargs with our response_format to avoid
        # duplicate keyword errors when callers also pass extra_body.
        caller_extra = kwargs.pop("extra_body", {})
        extra_body = {**caller_extra, "response_format": structural_tag}
        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            extra_body=extra_body,
            **kwargs,
        )
        return response.choices[0].message.content
