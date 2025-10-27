"""Rule-based agent that prefers cards near the seven."""

from __future__ import annotations

import numpy as np

from src.sevens_env import NUM_CARDS, SEVEN_RANK, Card

from .base import AgentPolicy, Observation


class NearestSevensAgent(AgentPolicy):
    """
    Strategy that prioritizes playing cards closest to rank 7.

    Parameters
    ----------
    prefer_high_rank : bool, optional
        When multiple cards have the same distance from 7, prefer higher-ranked cards.
        If False, prefer lower-ranked cards. Default is False.
    name : str or None, optional
        Name of the agent. Defaults to the class name if not provided.
    """

    def __init__(
        self,
        prefer_high_rank: bool = False,
        name: str | None = None,
    ) -> None:
        super().__init__(name)
        self.prefer_high_rank = prefer_high_rank

    def select_action(self, observation: Observation, agent: str) -> int:
        """
        Select the card closest to rank 7, or pass if no cards are playable.

        Parameters
        ----------
        observation : Observation
            Current game state observation containing action_mask.
        agent : str
            Agent identifier.

        Returns
        -------
        int
            Action index of the card closest to 7, or pass action if no cards are playable.

        Raises
        ------
        ValueError
            If action_mask is missing, empty, or no legal action exists.
        """
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
        """
        Calculate priority for card selection.

        Parameters
        ----------
        card_id : int
            Card identifier.

        Returns
        -------
        tuple[int, int]
            Priority tuple (distance_from_7, tie_breaker_rank).
            Lower values indicate higher priority.
        """
        card = Card.from_id(int(card_id))
        distance = abs(card.rank - SEVEN_RANK)
        tie_breaker = -card.rank if self.prefer_high_rank else card.rank
        return (distance, tie_breaker)
