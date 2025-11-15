"""Example training script using Hydra for configuration management.

This script demonstrates how to train a DQN agent using Hydra for
managing hyperparameters and experiment configurations.

Example usage::

    # Train with default configuration bundle
    python examples/train_dqn_with_hydra.py

    # Override algorithm hyperparameters
    python examples/train_dqn_with_hydra.py algorithm.learning_rate=0.001 algorithm.batch_size=128

    # Launch a sweep across multiple values
    python examples/train_dqn_with_hydra.py -m algorithm.learning_rate=0.0001,0.001 algorithm.epsilon_decay=0.995,0.999
"""

from __future__ import annotations

import logging

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from src.rl.dqn_agent import DQNAgent
from src.utils.env_utils import calculate_action_dim, calculate_state_dim

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    """Train a DQN agent using Hydra configuration.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration.
    """
    logger.info("Training configuration:\n%s", OmegaConf.to_yaml(cfg))

    algorithm_cfg = cfg.algorithm
    experiment_cfg = cfg.experiment

    if experiment_cfg.seed is not None:
        np.random.seed(experiment_cfg.seed)

    # Calculate state and action dimensions for Sevens game
    num_players = cfg.env.num_players
    state_dim = calculate_state_dim(num_players)
    action_dim = calculate_action_dim()

    # Create DQN agent
    logger.info("Creating DQN agent...")
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_layers=algorithm_cfg.hidden_layers,
        learning_rate=algorithm_cfg.learning_rate,
        gamma=algorithm_cfg.gamma,
        replay_buffer_size=algorithm_cfg.buffer_size,
        batch_size=algorithm_cfg.batch_size,
        target_update_freq=algorithm_cfg.target_update_freq,
        epsilon_start=algorithm_cfg.epsilon_start,
        epsilon_end=algorithm_cfg.epsilon_end,
        epsilon_decay=algorithm_cfg.epsilon_decay,
        double_dqn=algorithm_cfg.double_dqn,
        dueling=algorithm_cfg.dueling_dqn,
        gradient_clip=algorithm_cfg.gradient_clip_norm,
        activation=algorithm_cfg.activation,
        dropout=algorithm_cfg.dropout,
        tau=algorithm_cfg.tau,
        epsilon_decay_strategy=algorithm_cfg.epsilon_decay_strategy,
        device=experiment_cfg.device or algorithm_cfg.device,
        seed=experiment_cfg.seed
        if experiment_cfg.seed is not None
        else algorithm_cfg.seed,
    )

    logger.info(f"Agent created with epsilon={agent.policy.get_epsilon():.3f}")
    logger.info("Algorithm parameters:")
    logger.info(f"  - Learning rate: {algorithm_cfg.learning_rate}")
    logger.info(f"  - Gamma: {algorithm_cfg.gamma}")
    logger.info(f"  - Batch size: {algorithm_cfg.batch_size}")
    logger.info(f"  - Hidden layers: {algorithm_cfg.hidden_layers}")
    logger.info(f"  - Double DQN: {algorithm_cfg.double_dqn}")
    logger.info(f"  - Dueling DQN: {algorithm_cfg.dueling_dqn}")

    # This is a demonstration script showing how to:
    # 1. Load agent parameters from Hydra config
    # 2. Create a DQN agent with those parameters
    # 3. Override parameters from command line
    #
    # For a full training loop, see other examples in this directory

    logger.info("\nTo run training with different parameters, try:")
    logger.info(
        "  python examples/train_dqn_with_hydra.py algorithm.learning_rate=0.001"
    )
    logger.info(
        "  python examples/train_dqn_with_hydra.py algorithm.batch_size=128 algorithm.gamma=0.99"
    )
    logger.info(
        "  python examples/train_dqn_with_hydra.py -m algorithm.learning_rate=0.0001,0.001"
    )

    logger.info("\nConfiguration successfully loaded and agent created!")


if __name__ == "__main__":
    main()
