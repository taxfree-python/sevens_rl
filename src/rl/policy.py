"""
Exploration policies for reinforcement learning.

This module provides policies for balancing exploration and exploitation,
including epsilon-greedy strategy.
"""

import numpy as np


class EpsilonGreedyPolicy:
    """
    Epsilon-greedy exploration policy.

    This policy selects random actions with probability epsilon (exploration)
    and greedy actions with probability 1-epsilon (exploitation). Epsilon
    decays over time to gradually shift from exploration to exploitation.

    Parameters
    ----------
    epsilon_start : float, optional
        Initial exploration rate. Default is 1.0.
    epsilon_end : float, optional
        Minimum exploration rate. Default is 0.01.
    epsilon_decay : float, optional
        Decay factor per step. Meaning depends on decay_strategy.
    seed : int or None, optional
        Random seed for reproducibility. Default is None.
    decay_strategy : str, optional
        Decay schedule: 'exponential' (multiplicative) or 'linear'. Default is 'exponential'.

    Attributes
    ----------
    epsilon : float
        Current exploration rate.
    epsilon_start : float
        Initial exploration rate.
    epsilon_end : float
        Minimum exploration rate.
    epsilon_decay : float
        Decay factor.
    rng : np.random.Generator
        Random number generator.
    """

    def __init__(
        self,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        seed: int = None,
        decay_strategy: str = "exponential",
    ):
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.rng = np.random.default_rng(seed)
        self.decay_strategy = decay_strategy.lower()

        if self.decay_strategy not in {"exponential", "linear"}:
            raise ValueError(
                f"Unsupported decay_strategy={decay_strategy}. "
                "Choose 'exponential' or 'linear'."
            )

        if self.decay_strategy == "linear":
            step = (self.epsilon_start - self.epsilon_end) * (1 - self.epsilon_decay)
            self._linear_decay_step = max(step, 0.0)
        else:
            self._linear_decay_step = 0.0

    def select_action(
        self,
        q_values: np.ndarray,
        action_mask: np.ndarray = None,
    ) -> int:
        """
        Select an action using epsilon-greedy policy.

        Parameters
        ----------
        q_values : np.ndarray
            Q-values for each action, shape (num_actions,).
        action_mask : np.ndarray or None, optional
            Binary mask indicating valid actions (1 = valid, 0 = invalid).
            If None, all actions are considered valid. Default is None.

        Returns
        -------
        int
            Selected action index.

        Raises
        ------
        ValueError
            If no valid actions are available.
        """
        # Create action mask if not provided
        if action_mask is None:
            action_mask = np.ones_like(q_values, dtype=bool)
        else:
            action_mask = action_mask.astype(bool)

        # Check if any valid actions exist
        valid_actions = np.where(action_mask)[0]
        if len(valid_actions) == 0:
            raise ValueError("No valid actions available")

        # Epsilon-greedy selection
        if self.rng.random() < self.epsilon:
            # Exploration: random valid action
            action = self.rng.choice(valid_actions)
        else:
            # Exploitation: best valid action
            # Set invalid actions to -inf to exclude them
            masked_q_values = np.where(action_mask, q_values, -np.inf)
            action = int(np.argmax(masked_q_values))

        return action

    def decay(self) -> None:
        """
        Decay epsilon by the decay factor.

        Epsilon is multiplied by decay factor but clamped to epsilon_end.
        """
        if self.decay_strategy == "exponential":
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        else:
            self.epsilon = max(self.epsilon_end, self.epsilon - self._linear_decay_step)

    def reset(self) -> None:
        """Reset epsilon to initial value."""
        self.epsilon = self.epsilon_start

    def get_epsilon(self) -> float:
        """
        Get current epsilon value.

        Returns
        -------
        float
            Current exploration rate.
        """
        return self.epsilon

    def set_epsilon(self, epsilon: float) -> None:
        """
        Set epsilon to a specific value.

        Parameters
        ----------
        epsilon : float
            New exploration rate (clamped between epsilon_end and 1.0).
        """
        self.epsilon = np.clip(epsilon, self.epsilon_end, 1.0)
