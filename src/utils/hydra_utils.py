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
    algorithm_cfg = cfg.get("algorithm", cfg.get("agent", cfg))
    training_cfg = cfg.get("training", {})
    experiment_cfg = cfg.get("experiment", {})

    def _cfg_get(mapping: DictConfig | dict, key: str, default: Any) -> Any:
        if isinstance(mapping, DictConfig):
            return mapping.get(key, default)
        return mapping.get(key, default)

    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_layers=_cfg_get(algorithm_cfg, "hidden_layers", [512, 256, 128]),
        learning_rate=_cfg_get(
            algorithm_cfg,
            "learning_rate",
            _cfg_get(training_cfg, "learning_rate", 0.0001),
        ),
        gamma=_cfg_get(algorithm_cfg, "gamma", _cfg_get(training_cfg, "gamma", 0.95)),
        replay_buffer_size=_cfg_get(
            algorithm_cfg,
            "replay_buffer_size",
            _cfg_get(
                algorithm_cfg,
                "buffer_size",
                _cfg_get(training_cfg, "replay_buffer_size", 100000),
            ),
        ),
        batch_size=_cfg_get(
            algorithm_cfg, "batch_size", _cfg_get(training_cfg, "batch_size", 128)
        ),
        target_update_freq=_cfg_get(
            algorithm_cfg,
            "target_update_freq",
            _cfg_get(training_cfg, "target_update_freq", 20),
        ),
        epsilon_start=_cfg_get(
            algorithm_cfg, "epsilon_start", _cfg_get(training_cfg, "epsilon_start", 1.0)
        ),
        epsilon_end=_cfg_get(
            algorithm_cfg, "epsilon_end", _cfg_get(training_cfg, "epsilon_end", 0.05)
        ),
        epsilon_decay=_cfg_get(
            algorithm_cfg,
            "epsilon_decay",
            _cfg_get(training_cfg, "epsilon_decay", 0.999),
        ),
        double_dqn=_cfg_get(algorithm_cfg, "double_dqn", True),
        dueling=_cfg_get(algorithm_cfg, "dueling_dqn", False),
        gradient_clip=_cfg_get(
            algorithm_cfg,
            "gradient_clip_norm",
            _cfg_get(training_cfg, "gradient_clip", 1.0),
        ),
        device=_cfg_get(
            algorithm_cfg, "device", _cfg_get(experiment_cfg, "device", "cpu")
        ),
        seed=_cfg_get(algorithm_cfg, "seed", _cfg_get(experiment_cfg, "seed", None)),
        activation=_cfg_get(algorithm_cfg, "activation", "relu"),
        dropout=_cfg_get(algorithm_cfg, "dropout", 0.2),
        tau=_cfg_get(algorithm_cfg, "tau", _cfg_get(training_cfg, "tau", None)),
        epsilon_decay_strategy=_cfg_get(
            algorithm_cfg,
            "epsilon_decay_strategy",
            _cfg_get(training_cfg, "epsilon_decay_strategy", "exponential"),
        ),
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
    algorithm_cfg = cfg.get("algorithm", cfg.get("agent", {}))

    return {
        "num_episodes": training_cfg.get("num_episodes", 10000),
        "batch_size": training_cfg.get(
            "batch_size", algorithm_cfg.get("batch_size", 128)
        ),
        "min_replay_size": training_cfg.get(
            "min_replay_size", algorithm_cfg.get("min_buffer_size", 1000)
        ),
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
