"""
エージェントポリシーの抽象基底クラス。

Abstract base class for agent policies.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping

import numpy as np

Observation = Mapping[str, np.ndarray]


class AgentPolicy(ABC):
    """エージェントの行動選択インターフェース / Unified policy API."""

    def __init__(self, name: str | None = None) -> None:
        self.name = name or self.__class__.__name__

    @abstractmethod
    def select_action(self, observation: Observation, agent: str) -> int:
        """行動を選択する / Select an action for the given agent."""

