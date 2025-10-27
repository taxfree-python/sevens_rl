"""
Deep Q-Network (DQN) agent implementation.

This module provides a DQN agent that can learn to play Sevens through
reinforcement learning with experience replay and target networks.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.agents.base import AgentPolicy
from src.rl.network import QNetwork
from src.rl.policy import EpsilonGreedyPolicy
from src.rl.replay_buffer import ReplayBuffer


class DQNAgent(AgentPolicy):
    """
    Deep Q-Network agent for Sevens game.

    This agent uses deep reinforcement learning with experience replay
    and target networks to learn optimal play strategies. Supports both
    standard DQN and Double DQN algorithms.

    Parameters
    ----------
    state_dim : int
        Dimension of flattened state observation.
    action_dim : int
        Number of possible actions.
    name : str or None, optional
        Human-readable identifier for the agent. Defaults to class name.
    hidden_layers : list of int, optional
        Hidden layer sizes for Q-network. Default is [512, 256, 128].
    learning_rate : float, optional
        Learning rate for optimizer. Default is 0.0001.
    gamma : float, optional
        Discount factor for future rewards. Default is 0.95.
    replay_buffer_size : int, optional
        Maximum size of replay buffer. Default is 100000.
    batch_size : int, optional
        Batch size for training. Default is 128.
    target_update_freq : int, optional
        Frequency (in episodes) to update target network. Default is 20.
    epsilon_start : float, optional
        Initial exploration rate. Default is 1.0.
    epsilon_end : float, optional
        Minimum exploration rate. Default is 0.05.
    epsilon_decay : float, optional
        Epsilon decay factor. Default is 0.999.
    double_dqn : bool, optional
        Whether to use Double DQN algorithm. Default is True.
    dueling : bool, optional
        Whether to use Dueling DQN architecture. Default is False.
    gradient_clip : float or None, optional
        Gradient clipping value. If None, no clipping. Default is 1.0.
    device : str, optional
        Device for training ('cpu' or 'cuda'). Default is 'cpu'.
    seed : int or None, optional
        Random seed for reproducibility. Default is None.

    Attributes
    ----------
    q_network : QNetwork
        Online Q-network for action selection and training.
    target_network : QNetwork
        Target Q-network for stable value estimation.
    optimizer : torch.optim.Optimizer
        Optimizer for training Q-network.
    replay_buffer : ReplayBuffer
        Experience replay buffer.
    policy : EpsilonGreedyPolicy
        Epsilon-greedy exploration policy.
    episode_count : int
        Number of episodes trained.
    total_steps : int
        Total number of steps taken.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        name: str | None = None,
        hidden_layers: list[int] = None,
        learning_rate: float = 0.0001,
        gamma: float = 0.95,
        replay_buffer_size: int = 100000,
        batch_size: int = 128,
        target_update_freq: int = 20,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.999,
        double_dqn: bool = True,
        dueling: bool = False,
        gradient_clip: float = 1.0,
        device: str = "cpu",
        seed: int = None,
    ):
        super().__init__(name=name)
        if hidden_layers is None:
            hidden_layers = [512, 256, 128]

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.double_dqn = double_dqn
        self.gradient_clip = gradient_clip
        self.device = torch.device(device)

        # Set random seed
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # Initialize networks
        self.q_network = QNetwork(
            input_dim=state_dim,
            output_dim=action_dim,
            hidden_layers=hidden_layers,
            dueling=dueling,
        ).to(self.device)

        self.target_network = QNetwork(
            input_dim=state_dim,
            output_dim=action_dim,
            hidden_layers=hidden_layers,
            dueling=dueling,
        ).to(self.device)

        # Copy weights from Q-network to target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()  # Target network is always in eval mode

        # Initialize optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(capacity=replay_buffer_size, seed=seed)

        # Initialize exploration policy
        self.policy = EpsilonGreedyPolicy(
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            epsilon_decay=epsilon_decay,
            seed=seed,
        )

        # Training statistics
        self.episode_count = 0
        self.total_steps = 0

    def select_action(self, observation: dict, agent_id: str) -> int:
        """
        Select an action given the current observation.

        Parameters
        ----------
        observation : dict
            Dictionary containing 'board', 'hand', and 'action_mask'.
        agent_id : str
            Agent identifier (unused, for compatibility with AgentPolicy).

        Returns
        -------
        int
            Selected action index.
        """
        # Flatten observation
        state = self._observation_to_state(observation)

        # Get Q-values from network
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor).cpu().numpy()[0]

        # Select action using epsilon-greedy policy
        action_mask = observation.get("action_mask")
        action = self.policy.select_action(q_values, action_mask)

        return action

    def store_experience(
        self,
        state: dict,
        action: int,
        reward: float,
        next_state: dict,
        done: bool,
    ) -> None:
        """
        Store an experience in the replay buffer.

        Parameters
        ----------
        state : dict
            Current state observation.
        action : int
            Action taken.
        reward : float
            Reward received.
        next_state : dict
            Next state observation.
        done : bool
            Whether episode terminated.
        """
        # Convert observations to flat arrays for storage
        state_array = self._observation_to_state(state)
        next_state_array = self._observation_to_state(next_state)

        self.replay_buffer.push(state_array, action, reward, next_state_array, done)
        self.total_steps += 1

    def train_step(self) -> dict[str, float]:
        """
        Perform one training step using experience replay.

        Returns
        -------
        dict
            Training metrics including 'loss' and 'q_value_mean'.
            Returns empty dict if buffer not ready for training.
        """
        # Check if replay buffer has enough samples
        if not self.replay_buffer.is_ready(self.batch_size):
            return {}

        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )

        # Convert to tensors
        states_t = torch.FloatTensor(np.array(states)).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)

        # Compute current Q-values
        current_q_values = self.q_network(states_t).gather(1, actions_t.unsqueeze(1))

        # Compute target Q-values
        with torch.no_grad():
            if self.double_dqn:
                # Double DQN: use online network to select actions,
                # target network to evaluate them
                next_actions = self.q_network(next_states_t).argmax(dim=1)
                next_q_values = self.target_network(next_states_t).gather(
                    1, next_actions.unsqueeze(1)
                )
            else:
                # Standard DQN: use target network for both selection and evaluation
                next_q_values = (
                    self.target_network(next_states_t).max(dim=1)[0].unsqueeze(1)
                )

            target_q_values = rewards_t.unsqueeze(1) + (
                self.gamma * next_q_values * (1 - dones_t.unsqueeze(1))
            )

        # Compute loss
        loss = nn.functional.mse_loss(current_q_values, target_q_values)

        # Optimize the network
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping if specified
        if self.gradient_clip is not None:
            nn.utils.clip_grad_norm_(self.q_network.parameters(), self.gradient_clip)

        self.optimizer.step()

        # Decay epsilon
        self.policy.decay()

        # Return training metrics
        return {
            "loss": loss.item(),
            "q_value_mean": current_q_values.mean().item(),
            "epsilon": self.policy.get_epsilon(),
        }

    def update_target_network(self) -> None:
        """Update target network by copying weights from Q-network."""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def end_episode(self) -> None:
        """
        Called at the end of each episode.

        Updates episode count and target network if needed.
        """
        self.episode_count += 1

        # Update target network periodically
        if self.episode_count % self.target_update_freq == 0:
            self.update_target_network()

    def save(self, path: str) -> None:
        """
        Save agent state to file.

        Parameters
        ----------
        path : str
            Path to save checkpoint file.
        """
        checkpoint = {
            "q_network_state_dict": self.q_network.state_dict(),
            "target_network_state_dict": self.target_network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "episode_count": self.episode_count,
            "total_steps": self.total_steps,
            "epsilon": self.policy.get_epsilon(),
        }
        torch.save(checkpoint, path)

    def load(self, path: str) -> None:
        """
        Load agent state from file.

        Parameters
        ----------
        path : str
            Path to checkpoint file.
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint["q_network_state_dict"])
        self.target_network.load_state_dict(checkpoint["target_network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.episode_count = checkpoint["episode_count"]
        self.total_steps = checkpoint["total_steps"]
        self.policy.set_epsilon(checkpoint["epsilon"])

    def _observation_to_state(self, observation: dict) -> np.ndarray:
        """
        Convert dictionary observation to flat state array.

        Parameters
        ----------
        observation : dict
            Observation dictionary with 'board', 'hand', 'action_mask'.

        Returns
        -------
        np.ndarray
            Flattened state array.
        """
        board = observation["board"].astype(np.float32)
        hand = observation["hand"].astype(np.float32)
        action_mask = observation["action_mask"].astype(np.float32)

        # Concatenate all components
        state = np.concatenate([board, hand, action_mask])
        return state
