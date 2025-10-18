"""Tests for utility modules"""

import logging

import pytest
from omegaconf import OmegaConf

from src.utils import get_log_level, setup_logger, validate_config


def test_get_log_level_with_int():
    """Test get_log_level with integer input"""
    assert get_log_level(logging.INFO) == logging.INFO
    assert get_log_level(logging.DEBUG) == logging.DEBUG


def test_get_log_level_with_string():
    """Test get_log_level with string input"""
    assert get_log_level("INFO") == logging.INFO
    assert get_log_level("info") == logging.INFO
    assert get_log_level("DEBUG") == logging.DEBUG
    assert get_log_level("WARNING") == logging.WARNING
    assert get_log_level("ERROR") == logging.ERROR
    assert get_log_level("CRITICAL") == logging.CRITICAL


def test_get_log_level_invalid():
    """Test get_log_level with invalid input"""
    with pytest.raises(ValueError, match="Invalid log level"):
        get_log_level("INVALID")


def test_setup_logger_basic():
    """Test basic logger setup"""
    logger = setup_logger("test_logger", level="INFO")
    assert logger.name == "test_logger"
    assert logger.level == logging.INFO
    assert len(logger.handlers) > 0


def test_setup_logger_no_duplicates():
    """Test that setup_logger doesn't add duplicate handlers"""
    logger1 = setup_logger("test_no_dup", level="INFO")
    handler_count1 = len(logger1.handlers)

    logger2 = setup_logger("test_no_dup", level="INFO")
    handler_count2 = len(logger2.handlers)

    assert handler_count1 == handler_count2
    assert logger1 is logger2


def test_validate_config_valid(tmp_path):
    """Test config validation with valid configuration"""
    cfg = OmegaConf.create(
        {
            "env": {
                "num_players": 4,
                "max_steps": 1000,
                "render_mode": None,
            },
            "training": {
                "num_episodes": 1000,
                "batch_size": 64,
                "learning_rate": 0.001,
                "gamma": 0.99,
                "epsilon_start": 1.0,
                "epsilon_end": 0.01,
                "epsilon_decay": 0.995,
            },
            "network": {
                "hidden_layers": [256, 128],
                "activation": "relu",
                "dropout": 0.1,
            },
            "experiment": {
                "seed": 42,
                "device": "cpu",
                "num_workers": 1,
            },
        }
    )

    # Should not raise
    validate_config(cfg)


def test_validate_config_invalid_num_players():
    """Test config validation with invalid num_players"""
    cfg = OmegaConf.create(
        {
            "env": {"num_players": 5, "max_steps": 1000, "render_mode": None},
            "training": {
                "num_episodes": 1000,
                "batch_size": 64,
                "learning_rate": 0.001,
                "gamma": 0.99,
                "epsilon_start": 1.0,
                "epsilon_end": 0.01,
                "epsilon_decay": 0.995,
            },
            "network": {
                "hidden_layers": [256],
                "activation": "relu",
                "dropout": 0.1,
            },
            "experiment": {"seed": 42, "device": "cpu", "num_workers": 1},
        }
    )

    with pytest.raises(ValueError, match="num_players must be 2, 3, or 4"):
        validate_config(cfg)


def test_validate_config_invalid_gamma():
    """Test config validation with invalid gamma"""
    cfg = OmegaConf.create(
        {
            "env": {"num_players": 4, "max_steps": 1000, "render_mode": None},
            "training": {
                "num_episodes": 1000,
                "batch_size": 64,
                "learning_rate": 0.001,
                "gamma": 1.5,  # Invalid: > 1
                "epsilon_start": 1.0,
                "epsilon_end": 0.01,
                "epsilon_decay": 0.995,
            },
            "network": {
                "hidden_layers": [256],
                "activation": "relu",
                "dropout": 0.1,
            },
            "experiment": {"seed": 42, "device": "cpu", "num_workers": 1},
        }
    )

    with pytest.raises(ValueError, match="gamma must be in"):
        validate_config(cfg)


def test_validate_config_invalid_activation():
    """Test config validation with invalid activation function"""
    cfg = OmegaConf.create(
        {
            "env": {"num_players": 4, "max_steps": 1000, "render_mode": None},
            "training": {
                "num_episodes": 1000,
                "batch_size": 64,
                "learning_rate": 0.001,
                "gamma": 0.99,
                "epsilon_start": 1.0,
                "epsilon_end": 0.01,
                "epsilon_decay": 0.995,
            },
            "network": {
                "hidden_layers": [256],
                "activation": "invalid_activation",
                "dropout": 0.1,
            },
            "experiment": {"seed": 42, "device": "cpu", "num_workers": 1},
        }
    )

    with pytest.raises(ValueError, match="activation must be one of"):
        validate_config(cfg)
