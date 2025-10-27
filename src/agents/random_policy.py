"""Random policy agent for baseline evaluation."""

from __future__ import annotations

import numpy as np

from .base import AgentPolicy, Observation


class RandomAgent(AgentPolicy):
    """
    Agent that selects uniformly among valid actions.

    Parameters
    ----------
    rng : np.random.Generator or None, optional
        Random number generator. Defaults to np.random.default_rng() if not provided.
    name : str or None, optional
        Name of the agent. Defaults to the class name if not provided.
    """

    def __init__(
        self,
        rng: np.random.Generator | None = None,
        name: str | None = None,
    ) -> None:
        super().__init__(name)
        self.rng = rng or np.random.default_rng()

    def select_action(self, observation: Observation, agent: str) -> int:
        """
        Select a random valid action for the given agent.

        Parameters
        ----------
        observation : Observation
            Current game state observation containing action_mask.
        agent : str
            Agent identifier.

        Returns
        -------
        int
            Randomly selected valid action index.

        Raises
        ------
        ValueError
            If action_mask is missing, empty, or contains no valid actions.
        """
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
