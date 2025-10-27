"""
シンプルなルールベースエージェント。

Rule-based agent that prefers cards near the seven.
"""

from __future__ import annotations

import numpy as np

from src.sevens_env import NUM_CARDS, SEVEN_RANK, Card

from .base import AgentPolicy, Observation


class NearestSevensAgent(AgentPolicy):
    """7に近いカードを優先して出す戦略。"""

    def __init__(
        self,
        prefer_high_rank: bool = False,
        name: str | None = None,
    ) -> None:
        """ルールの詳細を構成する。

        Args:
            prefer_high_rank: 7からの距離が同じカードが複数ある場合に、高ランク側を優先するか。
                False の場合は低ランク側を優先する。
            name: エージェント名称（省略時はクラス名）。
        """
        super().__init__(name)
        self.prefer_high_rank = prefer_high_rank

    def select_action(self, observation: Observation, agent: str) -> int:
        action_mask_array = observation.get("action_mask")
        if action_mask_array is None:
            raise ValueError(f"action_mask is missing for agent {agent}")

        action_mask = np.asarray(action_mask_array)
        if action_mask.size == 0:
            raise ValueError(f"action_mask is empty for agent {agent}")

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
        tie_breaker = -card.rank if self.prefer_high_rank else card.rank
        return (distance, tie_breaker)
