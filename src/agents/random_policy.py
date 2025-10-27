"""
ランダム方策エージェント。

Random policy agent for baseline evaluation.
"""

from __future__ import annotations

import numpy as np

from .base import AgentPolicy, Observation


class RandomAgent(AgentPolicy):
    """行動可能な手の中から一様に選択するエージェント。"""

    def __init__(
        self,
        rng: np.random.Generator | None = None,
        name: str | None = None,
    ) -> None:
        super().__init__(name)
        self.rng = rng or np.random.default_rng()

    def select_action(self, observation: Observation, agent: str) -> int:
        action_mask_array = observation.get("action_mask")
        if action_mask_array is None:
            raise ValueError(f"action_mask is missing for agent {agent}")

        action_mask = np.asarray(action_mask_array)
        if action_mask.size == 0:
            raise ValueError(f"action_mask is empty for agent {agent}")

        valid_actions = np.flatnonzero(action_mask)
        if valid_actions.size == 0:
            raise ValueError(f"No valid actions available for agent {agent}")

        return int(self.rng.choice(valid_actions))
