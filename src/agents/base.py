"""Abstract base class for agent policies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping

import numpy as np

Observation = Mapping[str, np.ndarray]


class AgentPolicy(ABC):
    """
    Unified policy API for agent action selection.

    Parameters
    ----------
    name : str or None, optional
        Name of the policy. Defaults to the class name if not provided.
    """

    def __init__(self, name: str | None = None) -> None:
        self.name = name or self.__class__.__name__

    @abstractmethod
    def select_action(self, observation: Observation, agent: str) -> int:
        """
        Select an action for the given agent.

        Parameters
        ----------
        observation : Observation
            Current game state observation for the agent.
        agent : str
            Agent identifier.

        Returns
        -------
        int
            Selected action index.
        """
