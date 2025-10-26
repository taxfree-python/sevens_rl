"""
シンプルなルールベースエージェント。

Rule-based agent that prefers cards near the seven.
"""

from __future__ import annotations

import numpy as np

from .base import AgentPolicy, Observation
from src.sevens_env import Card, NUM_CARDS, SEVEN_RANK


class NearestSevensAgent(AgentPolicy):
    """7に近いカードを優先して出す戦略。"""

    def __init__(
        self,
        prefer_high_rank: bool = False,
        name: str | None = None,
    ) -> None:
        super().__init__(name)
        self.prefer_high_rank = prefer_high_rank

    def select_action(self, observation: Observation, agent: str) -> int:
        action_mask = np.asarray(observation.get("action_mask"))
        if action_mask.size == 0:
            raise ValueError(f"action_mask is missing for agent {agent}")

        playable_cards = np.flatnonzero(action_mask[:-1])
        if playable_cards.size > 0:
            prioritized = min(playable_cards, key=self._priority)
            return int(prioritized)

        if int(action_mask[NUM_CARDS]) == 1:
            return NUM_CARDS

        raise ValueError(f"No legal action (including pass) for agent {agent}")

    def _priority(self, card_id: int) -> tuple[int, int]:
        card = Card.from_id(int(card_id))
        distance = abs(card.rank - SEVEN_RANK)
        if self.prefer_high_rank:
            tie_breaker = -card.rank
        else:
            tie_breaker = card.rank
        return (distance, tie_breaker)
