"""
Utilities for loading and managing Hydra configurations.

This module provides helper functions for working with Hydra
configuration files and initializing components from config.
"""

from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from src.rl.dqn_agent import DQNAgent


def load_config(config_path: str) -> DictConfig:
    """
    Load a Hydra configuration file.

    Parameters
    ----------
    config_path : str
        Path to the YAML configuration file.

    Returns
    -------
    DictConfig
        Loaded configuration as OmegaConf DictConfig.

    Raises
    ------
    FileNotFoundError
        If config file does not exist.
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    return OmegaConf.load(config_file)


def create_dqn_agent_from_config(
    cfg: DictConfig,
    state_dim: int,
    action_dim: int,
) -> DQNAgent:
    """
    Create a DQN agent from Hydra configuration.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration containing training and network parameters.
    state_dim : int
        Dimension of state observation.
    action_dim : int
        Number of possible actions.

    Returns
    -------
    DQNAgent
        Initialized DQN agent.

    Examples
    --------
    >>> cfg = load_config("configs/train_dqn.yaml")
    >>> agent = create_dqn_agent_from_config(cfg, state_dim=157, action_dim=53)
    """
    # Extract training parameters
    training_cfg = cfg.get("training", {})
    network_cfg = cfg.get("network", {})
    training_opts_cfg = cfg.get("training_opts", {})
    experiment_cfg = cfg.get("experiment", {})

    # Create agent with config parameters
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_layers=network_cfg.get("hidden_layers", [512, 256, 128]),
        learning_rate=training_cfg.get("learning_rate", 0.0001),
        gamma=training_cfg.get("gamma", 0.95),
        replay_buffer_size=training_cfg.get("replay_buffer_size", 100000),
        batch_size=training_cfg.get("batch_size", 128),
        target_update_freq=training_cfg.get("target_update_freq", 20),
        epsilon_start=training_cfg.get("epsilon_start", 1.0),
        epsilon_end=training_cfg.get("epsilon_end", 0.05),
        epsilon_decay=training_cfg.get("epsilon_decay", 0.999),
        double_dqn=training_opts_cfg.get("double_dqn", True),
        dueling=network_cfg.get("dueling", False),
        gradient_clip=training_opts_cfg.get("gradient_clip", 1.0),
        device=experiment_cfg.get("device", "cpu"),
        seed=experiment_cfg.get("seed", None),
    )

    return agent


def get_training_params(cfg: DictConfig) -> dict[str, Any]:
    """
    Extract training parameters from config.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration.

    Returns
    -------
    dict
        Dictionary of training parameters.
    """
    training_cfg = cfg.get("training", {})

    return {
        "num_episodes": training_cfg.get("num_episodes", 10000),
        "batch_size": training_cfg.get("batch_size", 128),
        "min_replay_size": training_cfg.get("min_replay_size", 1000),
    }


def get_experiment_params(cfg: DictConfig) -> dict[str, Any]:
    """
    Extract experiment parameters from config.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration.

    Returns
    -------
    dict
        Dictionary of experiment parameters.
    """
    experiment_cfg = cfg.get("experiment", {})

    return {
        "name": experiment_cfg.get("name", "sevens_baseline"),
        "seed": experiment_cfg.get("seed", 42),
        "device": experiment_cfg.get("device", "cpu"),
    }


def get_env_params(cfg: DictConfig) -> dict[str, Any]:
    """
    Extract environment parameters from config.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration.

    Returns
    -------
    dict
        Dictionary of environment parameters.
    """
    env_cfg = cfg.get("env", {})

    return {
        "num_players": env_cfg.get("num_players", 4),
        "render_mode": env_cfg.get("render_mode", None),
        "max_steps": env_cfg.get("max_steps", 1000),
    }


def merge_configs(base_cfg: DictConfig, override_cfg: DictConfig) -> DictConfig:
    """
    Merge two configurations with override taking precedence.

    Parameters
    ----------
    base_cfg : DictConfig
        Base configuration.
    override_cfg : DictConfig
        Override configuration.

    Returns
    -------
    DictConfig
        Merged configuration.
    """
    return OmegaConf.merge(base_cfg, override_cfg)
