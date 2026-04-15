"""Token estimation utilities."""

from __future__ import annotations

from math import ceil
from typing import Protocol


class TokenEstimator(Protocol):
    """Interface for model-specific token estimators."""

    def estimate(self, text: str) -> int:
        """Return the estimated token count for text."""


class CharacterTokenEstimator:
    """A cheap character-count approximation for prompt budgeting."""

    def __init__(self, chars_per_token: int = 4) -> None:
        self._chars_per_token = max(1, chars_per_token)

    def estimate(self, text: str) -> int:
        if not text:
            return 0
        return ceil(len(text) / self._chars_per_token)
