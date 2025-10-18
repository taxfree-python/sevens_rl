"""
Configuration validation utilities

Provides validation functions for Hydra configurations to catch
invalid parameter values early.
"""

from omegaconf import DictConfig


def validate_env_config(cfg: DictConfig) -> None:
    """
    Validate environment configuration.

    Args:
        cfg: Environment configuration from Hydra

    Raises:
        ValueError: If configuration is invalid
    """
    if cfg.env.num_players not in [2, 3, 4]:
        raise ValueError(
            f"env.num_players must be 2, 3, or 4, got {cfg.env.num_players}"
        )

    if cfg.env.max_steps <= 0:
        raise ValueError(f"env.max_steps must be positive, got {cfg.env.max_steps}")

    if cfg.env.render_mode not in [None, "human"]:
        raise ValueError(
            f"env.render_mode must be None or 'human', got {cfg.env.render_mode}"
        )


def validate_training_config(cfg: DictConfig) -> None:
    """
    Validate training configuration.

    Args:
        cfg: Training configuration from Hydra

    Raises:
        ValueError: If configuration is invalid
    """
    if cfg.training.num_episodes <= 0:
        raise ValueError(
            f"training.num_episodes must be positive, got {cfg.training.num_episodes}"
        )

    if cfg.training.batch_size <= 0:
        raise ValueError(
            f"training.batch_size must be positive, got {cfg.training.batch_size}"
        )

    if cfg.training.learning_rate <= 0:
        raise ValueError(
            f"training.learning_rate must be positive, got {cfg.training.learning_rate}"
        )

    if not 0 <= cfg.training.gamma <= 1:
        raise ValueError(
            f"training.gamma must be in [0, 1], got {cfg.training.gamma}"
        )

    if not 0 <= cfg.training.epsilon_start <= 1:
        raise ValueError(
            f"training.epsilon_start must be in [0, 1], got {cfg.training.epsilon_start}"
        )

    if not 0 <= cfg.training.epsilon_end <= 1:
        raise ValueError(
            f"training.epsilon_end must be in [0, 1], got {cfg.training.epsilon_end}"
        )

    if cfg.training.epsilon_end > cfg.training.epsilon_start:
        raise ValueError(
            f"training.epsilon_end ({cfg.training.epsilon_end}) must be <= "
            f"epsilon_start ({cfg.training.epsilon_start})"
        )

    if not 0 < cfg.training.epsilon_decay <= 1:
        raise ValueError(
            f"training.epsilon_decay must be in (0, 1], got {cfg.training.epsilon_decay}"
        )


def validate_network_config(cfg: DictConfig) -> None:
    """
    Validate network configuration.

    Args:
        cfg: Network configuration from Hydra

    Raises:
        ValueError: If configuration is invalid
    """
    if not cfg.network.hidden_layers:
        raise ValueError("network.hidden_layers cannot be empty")

    for i, layer_size in enumerate(cfg.network.hidden_layers):
        if layer_size <= 0:
            raise ValueError(
                f"network.hidden_layers[{i}] must be positive, got {layer_size}"
            )

    if not 0 <= cfg.network.dropout < 1:
        raise ValueError(
            f"network.dropout must be in [0, 1), got {cfg.network.dropout}"
        )

    valid_activations = ["relu", "tanh", "sigmoid", "leaky_relu", "elu"]
    if cfg.network.activation not in valid_activations:
        raise ValueError(
            f"network.activation must be one of {valid_activations}, "
            f"got {cfg.network.activation}"
        )


def validate_experiment_config(cfg: DictConfig) -> None:
    """
    Validate experiment configuration.

    Args:
        cfg: Experiment configuration from Hydra

    Raises:
        ValueError: If configuration is invalid
    """
    if cfg.experiment.seed < 0:
        raise ValueError(f"experiment.seed must be >= 0, got {cfg.experiment.seed}")

    if cfg.experiment.device not in ["cpu", "cuda"]:
        raise ValueError(
            f"experiment.device must be 'cpu' or 'cuda', got {cfg.experiment.device}"
        )

    if cfg.experiment.num_workers <= 0:
        raise ValueError(
            f"experiment.num_workers must be positive, got {cfg.experiment.num_workers}"
        )


def validate_config(cfg: DictConfig) -> None:
    """
    Validate all configuration sections.

    Args:
        cfg: Full Hydra configuration

    Raises:
        ValueError: If any configuration is invalid
    """
    validate_env_config(cfg)
    validate_training_config(cfg)
    validate_network_config(cfg)
    validate_experiment_config(cfg)
