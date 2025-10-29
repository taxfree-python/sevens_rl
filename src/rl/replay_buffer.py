"""
Replay buffer for experience replay in DQN.

This module provides a circular buffer for storing and sampling
past experiences to break temporal correlations in training.
"""

from collections import deque
from typing import Any

import numpy as np


class ReplayBuffer:
    """
    Replay buffer for storing and sampling experiences.

    This buffer stores experiences (state, action, reward, next_state, done)
    and provides random sampling for training. Uses a deque for efficient
    circular buffer implementation.

    Parameters
    ----------
    capacity : int
        Maximum number of experiences to store.
    seed : int or None, optional
        Random seed for reproducibility. Default is None.

    Attributes
    ----------
    capacity : int
        Maximum buffer capacity.
    buffer : deque
        Circular buffer storing experiences.
    rng : np.random.Generator
        Random number generator for sampling.
    """

    def __init__(self, capacity: int, seed: int = None):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.rng = np.random.default_rng(seed)

    def push(
        self,
        state: Any,
        action: int,
        reward: float,
        next_state: Any,
        done: bool,
    ) -> None:
        """
        Add an experience to the buffer.

        If the buffer is full, the oldest experience is automatically
        removed due to the deque's maxlen property.

        Parameters
        ----------
        state : Any
            Current state observation.
        action : int
            Action taken.
        reward : float
            Reward received.
        next_state : Any
            Next state observation.
        done : bool
            Whether the episode terminated.
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> tuple:
        """
        Sample a random batch of experiences.

        Parameters
        ----------
        batch_size : int
            Number of experiences to sample.

        Returns
        -------
        tuple
            Tuple of (states, actions, rewards, next_states, dones) where
            each element is a list of sampled experiences.

        Raises
        ------
        ValueError
            If batch_size is larger than buffer size.
        """
        if batch_size > len(self.buffer):
            raise ValueError(
                f"Cannot sample {batch_size} experiences from buffer "
                f"with only {len(self.buffer)} experiences"
            )

        # Convert to list once so index access is O(1)
        experiences = list(self.buffer)

        # Sample indices without replacement
        indices = self.rng.choice(len(experiences), size=batch_size, replace=False)

        # Gather experiences
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for idx in indices:
            state, action, reward, next_state, done = experiences[idx]
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)

        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        """
        Get current number of experiences in buffer.

        Returns
        -------
        int
            Number of stored experiences.
        """
        return len(self.buffer)

    def clear(self) -> None:
        """Clear all experiences from the buffer."""
        self.buffer.clear()

    def is_ready(self, min_size: int) -> bool:
        """
        Check if buffer has enough experiences for training.

        Parameters
        ----------
        min_size : int
            Minimum number of experiences required.

        Returns
        -------
        bool
            True if buffer has at least min_size experiences.
        """
        return len(self.buffer) >= min_size
