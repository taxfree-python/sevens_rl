"""Example training script using Hydra for configuration management.

This script demonstrates how to train a DQN agent using Hydra for
managing hyperparameters and experiment configurations.

Usage:
    # Train with default config
    python examples/train_dqn_with_hydra.py

    # Override specific parameters
    python examples/train_dqn_with_hydra.py agent.learning_rate=0.001 agent.batch_size=128

    # Use different config file
    python examples/train_dqn_with_hydra.py --config-name=train_dqn

    # Run multiple experiments with different hyperparameters
    python examples/train_dqn_with_hydra.py -m agent.learning_rate=0.0001,0.001 agent.gamma=0.95,0.99
"""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from src.rl.dqn_agent import DQNAgent

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs/agent", config_name="dqn")
def main(cfg: DictConfig) -> None:
    """Train a DQN agent using Hydra configuration.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration.
    """
    # Print configuration
    logger.info("Training configuration:")
    logger.info(OmegaConf.to_yaml(cfg))

    # Set random seed
    if cfg.seed is not None:
        np.random.seed(cfg.seed)

    # Calculate state and action dimensions for Sevens game
    # State: board (52) + hand (52) + action_mask (53) = 157
    state_dim = 157
    action_dim = 53

    # Create DQN agent
    logger.info("Creating DQN agent...")
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_layers=cfg.hidden_layers,
        learning_rate=cfg.learning_rate,
        gamma=cfg.gamma,
        replay_buffer_size=cfg.buffer_size,
        batch_size=cfg.batch_size,
        target_update_freq=cfg.target_update_freq,
        epsilon_start=cfg.epsilon_start,
        epsilon_end=cfg.epsilon_end,
        epsilon_decay=cfg.epsilon_decay,
        double_dqn=cfg.double_dqn,
        dueling=cfg.dueling_dqn,
        gradient_clip=cfg.gradient_clip_norm,
        device=cfg.device,
        seed=cfg.seed,
    )

    logger.info(f"Agent created with epsilon={agent.policy.get_epsilon():.3f}")
    logger.info(f"Agent parameters from Hydra config:")
    logger.info(f"  - Learning rate: {cfg.learning_rate}")
    logger.info(f"  - Gamma: {cfg.gamma}")
    logger.info(f"  - Batch size: {cfg.batch_size}")
    logger.info(f"  - Hidden layers: {cfg.hidden_layers}")
    logger.info(f"  - Double DQN: {cfg.double_dqn}")
    logger.info(f"  - Dueling DQN: {cfg.dueling_dqn}")

    # This is a demonstration script showing how to:
    # 1. Load agent parameters from Hydra config
    # 2. Create a DQN agent with those parameters
    # 3. Override parameters from command line
    #
    # For a full training loop, see other examples in this directory

    logger.info("\nTo run training with different parameters, try:")
    logger.info("  python examples/train_dqn_with_hydra.py learning_rate=0.001")
    logger.info("  python examples/train_dqn_with_hydra.py batch_size=128 gamma=0.99")
    logger.info(
        "  python examples/train_dqn_with_hydra.py -m learning_rate=0.0001,0.001"
    )

    logger.info("\nConfiguration successfully loaded and agent created!")


if __name__ == "__main__":
    main()
