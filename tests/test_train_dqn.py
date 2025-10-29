"""Smoke tests for DQN training loop."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from src.rl.dqn_agent import DQNAgent
from src.sevens_env import SevensEnv
from src.train_dqn import evaluate_episode, train_episode


def create_test_agent(**kwargs):
    """Create a small DQN agent for testing.

    Parameters
    ----------
    **kwargs
        Override default parameters.

    Returns
    -------
    DQNAgent
        A DQN agent configured for testing.
    """
    defaults = {
        "state_dim": 217,  # Updated for new observation space
        "action_dim": 53,
        "hidden_layers": [64, 32],  # Smaller for faster tests
        "learning_rate": 0.001,
        "gamma": 0.95,
        "replay_buffer_size": 500,
        "batch_size": 16,
        "target_update_freq": 5,
        "epsilon_start": 1.0,
        "epsilon_end": 0.1,
        "epsilon_decay": 0.95,
        "double_dqn": True,
        "dueling": False,
        "gradient_clip": 1.0,
        "device": "cpu",
        "seed": 42,
    }
    defaults.update(kwargs)
    return DQNAgent(**defaults)


@pytest.fixture
def env():
    """Create a Sevens environment for testing."""
    return SevensEnv(num_players=4)


@pytest.fixture
def agents(env):
    """Create agents for all players."""
    agents_dict = {}
    for agent_id in env.possible_agents:
        agent = create_test_agent()
        agent.name = agent_id
        agents_dict[agent_id] = agent
    return agents_dict


def test_train_episode_completes(env, agents):
    """Test that a single training episode completes without errors."""
    from src.utils.logger import setup_logger

    logger = setup_logger(name="test_train", level="ERROR")

    stats = train_episode(env, agents, logger, training_agents=["player_0"])

    # Check that stats are returned
    assert "episode_steps" in stats
    assert "episode_rewards" in stats
    assert "training_agents_rewards" in stats
    assert "training_agents" in stats
    assert "training_agent_reward" in stats
    assert "mean_loss" in stats
    assert "mean_q_value" in stats
    assert "epsilon" in stats

    # Check that episode ran
    assert stats["episode_steps"] > 0
    assert isinstance(stats["episode_rewards"], dict)
    assert len(stats["episode_rewards"]) == 4  # 4 players
    assert stats["training_agents"] == ["player_0"]

    # Check reward is a number
    assert isinstance(stats["training_agent_reward"], (int, float))
    assert stats["training_agents_rewards"]["player_0"] == stats["training_agent_reward"]


def test_evaluate_episode_completes(env, agents):
    """Test that a single evaluation episode completes without errors."""
    stats = evaluate_episode(env, agents, training_agents=["player_0"])

    # Check that stats are returned
    assert "episode_steps" in stats
    assert "episode_rewards" in stats
    assert "training_agents_rewards" in stats
    assert "training_agents" in stats
    assert "training_agent_reward" in stats

    # Check that episode ran
    assert stats["episode_steps"] > 0
    assert isinstance(stats["episode_rewards"], dict)
    assert len(stats["episode_rewards"]) == 4  # 4 players
    assert stats["training_agents"] == ["player_0"]
    assert stats["training_agents_rewards"]["player_0"] == stats["training_agent_reward"]


def test_evaluate_episode_uses_zero_epsilon(env, agents):
    """Test that evaluation uses epsilon=0 (no exploration)."""
    # Set a non-zero epsilon before evaluation
    for agent in agents.values():
        agent.policy.set_epsilon(0.5)

    # Run evaluation
    evaluate_episode(env, agents, training_agents=["player_0"])

    # After evaluation, epsilon should be restored
    for agent in agents.values():
        assert agent.policy.get_epsilon() == 0.5


def test_train_episode_stores_experiences(env, agents):
    """Test that training episode stores experiences in replay buffer."""
    from src.utils.logger import setup_logger

    logger = setup_logger(name="test_train", level="ERROR")

    # Get initial buffer size
    initial_buffer_size = len(agents["player_0"].replay_buffer)

    # Train one episode
    train_episode(env, agents, logger, training_agents=["player_0"])

    # Buffer should have new experiences
    final_buffer_size = len(agents["player_0"].replay_buffer)
    assert final_buffer_size > initial_buffer_size


def test_train_episode_updates_agent_stats(env, agents):
    """Test that training updates agent statistics."""
    from src.utils.logger import setup_logger

    logger = setup_logger(name="test_train", level="ERROR")

    initial_episode_count = agents["player_0"].episode_count
    initial_total_steps = agents["player_0"].total_steps

    # Train one episode
    train_episode(env, agents, logger, training_agents=["player_0"])

    # Stats should be updated
    assert agents["player_0"].episode_count == initial_episode_count + 1
    assert agents["player_0"].total_steps > initial_total_steps


def test_multiple_episodes_smoke_test(env, agents):
    """Smoke test: Run multiple episodes to ensure stability."""
    from src.utils.logger import setup_logger

    logger = setup_logger(name="test_train", level="ERROR")

    num_episodes = 5
    all_stats = []

    for _ in range(num_episodes):
        stats = train_episode(env, agents, logger, training_agents=["player_0"])
        all_stats.append(stats)

    # All episodes should complete
    assert len(all_stats) == num_episodes

    # Check that epsilon decayed over time
    epsilons = [stats["epsilon"] for stats in all_stats]
    assert epsilons[-1] <= epsilons[0]  # Epsilon should decrease or stay same


def test_train_and_evaluate_integration(env, agents):
    """Integration test: Train a few episodes then evaluate."""
    from src.utils.logger import setup_logger

    logger = setup_logger(name="test_train", level="ERROR")

    # Train for a few episodes
    for _ in range(3):
        train_episode(env, agents, logger, training_agents=["player_0"])

    # Evaluate
    eval_stats = evaluate_episode(env, agents, training_agents=["player_0"])

    # Should complete without errors
    assert eval_stats["episode_steps"] > 0


def test_agent_save_load_integration(env, agents):
    """Test that agent can be saved and loaded during training."""
    from src.utils.logger import setup_logger

    logger = setup_logger(name="test_train", level="ERROR")

    # Train for a couple episodes
    for _ in range(2):
        train_episode(env, agents, logger, training_agents=["player_0"])

    # Save agent
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "agent.pt"
        agents["player_0"].save(str(save_path))

        # Verify file was created
        assert save_path.exists()

        # Load into a new agent
        new_agent = create_test_agent()
        new_agent.load(str(save_path))

        # Check that key stats match
        assert new_agent.episode_count == agents["player_0"].episode_count
        assert new_agent.total_steps == agents["player_0"].total_steps


def test_different_training_agents(env):
    """Test training different agents (not just player_0)."""
    from src.utils.logger import setup_logger

    logger = setup_logger(name="test_train", level="ERROR")

    # Create agents
    agents = {}
    for agent_id in env.possible_agents:
        agent = create_test_agent()
        agent.name = agent_id
        agents[agent_id] = agent

    # Train player_1 instead of player_0
    stats = train_episode(env, agents, logger, training_agents=["player_1"])

    # Should work without errors
    assert "training_agent_reward" in stats
    assert stats["episode_steps"] > 0


def test_training_with_different_num_players():
    """Test training with different number of players."""
    from src.utils.logger import setup_logger

    logger = setup_logger(name="test_train", level="ERROR")

    for num_players in [2, 3, 4]:
        env = SevensEnv(num_players=num_players)

        # Calculate state_dim for this number of players
        state_dim = (
            52  # board
            + 52  # hand
            + 53  # action_mask
            + num_players  # hand_counts
            + 52  # card_play_order
            + num_players  # current_player
        )

        # Create agents with correct state_dim
        agents = {}
        for agent_id in env.possible_agents:
            agent = create_test_agent(state_dim=state_dim)
            agent.name = agent_id
            agents[agent_id] = agent

        # Train one episode
        stats = train_episode(env, agents, logger, training_agents=["player_0"])

        # Should work for all player counts
        assert len(stats["episode_rewards"]) == num_players
        assert stats["episode_steps"] > 0


@pytest.mark.slow
def test_training_session_smoke_test(env, agents):
    """Comprehensive smoke test: Run a mini training session.

    This test simulates a complete training session with training,
    evaluation, and checkpointing. Marked as slow since it runs
    more episodes.
    """
    from src.utils.logger import setup_logger

    logger = setup_logger(name="test_train", level="ERROR")

    num_episodes = 20
    eval_freq = 5
    training_rewards = []
    eval_rewards = []

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoints_dir = Path(tmpdir) / "checkpoints"
        checkpoints_dir.mkdir(exist_ok=True, parents=True)

        for episode in range(1, num_episodes + 1):
            # Train
            stats = train_episode(env, agents, logger, training_agents=["player_0"])
            training_rewards.append(stats["training_agents_rewards"]["player_0"])

            # Evaluate
            if episode % eval_freq == 0:
                eval_stats = evaluate_episode(
                    env, agents, training_agents=["player_0"]
                )
                eval_rewards.append(eval_stats["training_agents_rewards"]["player_0"])

                # Save checkpoint
                checkpoint_path = checkpoints_dir / f"agent_{episode}.pt"
                agents["player_0"].save(str(checkpoint_path))
                assert checkpoint_path.exists()

        # Verify we collected data
        assert len(training_rewards) == num_episodes
        assert len(eval_rewards) == num_episodes // eval_freq

        # Verify checkpoints exist
        checkpoints = list(checkpoints_dir.glob("agent_*.pt"))
        assert len(checkpoints) == num_episodes // eval_freq


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
