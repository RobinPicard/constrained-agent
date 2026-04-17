from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Backend(ABC):
    """Abstract interface for generation backends.

    A backend takes chat messages and a structural tag dict and returns
    the raw model output string.
    """

    @abstractmethod
    def generate(
        self,
        messages: list[dict],
        structural_tag: dict,
        **kwargs,
    ) -> str:
        """Generate a response.

        Parameters
        ----------
        messages:
            Chat messages in OpenAI format (list of role/content dicts).
        structural_tag:
            The structural tag schema as a dict. Passed to the API as
            ``response_format`` to enforce output structure.
        **kwargs:
            Backend-specific generation parameters (e.g. ``max_tokens``).
        """
        ...
