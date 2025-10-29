"""Tests for DQN agent implementation."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from src.rl.dqn_agent import DQNAgent


def create_test_agent(**kwargs):
    """Create a DQN agent with default test parameters.

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
        "state_dim": 157,
        "action_dim": 53,
        "hidden_layers": [128, 64],
        "learning_rate": 0.001,
        "gamma": 0.95,
        "replay_buffer_size": 1000,
        "batch_size": 32,
        "target_update_freq": 10,
        "epsilon_start": 1.0,
        "epsilon_end": 0.05,
        "epsilon_decay": 0.99,
        "double_dqn": True,
        "dueling": False,
        "gradient_clip": 1.0,
        "device": "cpu",
    }
    defaults.update(kwargs)
    return DQNAgent(**defaults)


@pytest.fixture
def dqn_agent():
    """Create a DQN agent for testing."""
    return create_test_agent(seed=42)


def test_dqn_agent_initialization(dqn_agent):
    """Test DQN agent initialization."""
    assert dqn_agent.state_dim == 157
    assert dqn_agent.action_dim == 53
    assert dqn_agent.gamma == 0.95
    assert dqn_agent.batch_size == 32
    assert dqn_agent.episode_count == 0
    assert dqn_agent.total_steps == 0


def test_dqn_agent_networks(dqn_agent):
    """Test that Q-network and target network exist."""
    assert dqn_agent.q_network is not None
    assert dqn_agent.target_network is not None
    assert dqn_agent.optimizer is not None
    assert dqn_agent.replay_buffer is not None
    assert dqn_agent.policy is not None


def test_dqn_agent_select_action(dqn_agent):
    """Test action selection."""
    observation = {
        "board": np.zeros(52, dtype=np.int8),
        "hand": np.zeros(52, dtype=np.int8),
        "action_mask": np.ones(53, dtype=np.int8),
    }

    action = dqn_agent.select_action(observation, agent_id="player_0")

    assert isinstance(action, (int, np.integer))
    assert 0 <= action < 53


def test_dqn_agent_select_action_respects_mask(dqn_agent):
    """Test that action selection respects action mask."""
    # Set epsilon to 0 for deterministic behavior (might still explore, so we test many times)
    dqn_agent.policy.set_epsilon(0.0)

    observation = {
        "board": np.zeros(52, dtype=np.int8),
        "hand": np.zeros(52, dtype=np.int8),
        "action_mask": np.zeros(53, dtype=np.int8),
    }
    # Only allow actions 0 and 1
    observation["action_mask"][0] = 1
    observation["action_mask"][1] = 1

    actions = [
        dqn_agent.select_action(observation, agent_id="player_0") for _ in range(100)
    ]

    # All actions should be 0 or 1
    assert all(a in [0, 1] for a in actions)


def test_dqn_agent_store_experience(dqn_agent):
    """Test storing experience in replay buffer."""
    state = {
        "board": np.zeros(52, dtype=np.int8),
        "hand": np.zeros(52, dtype=np.int8),
        "action_mask": np.ones(53, dtype=np.int8),
    }
    next_state = state.copy()

    initial_buffer_size = len(dqn_agent.replay_buffer)

    dqn_agent.store_experience(
        state, action=0, reward=1.0, next_state=next_state, done=False
    )

    assert len(dqn_agent.replay_buffer) == initial_buffer_size + 1
    assert dqn_agent.total_steps == 1


def test_dqn_agent_train_step_not_ready(dqn_agent):
    """Test training step when buffer not ready."""
    # Buffer is empty, should return empty dict
    metrics = dqn_agent.train_step()
    assert metrics == {}


def test_dqn_agent_train_step_ready(dqn_agent):
    """Test training step when buffer is ready."""
    # Fill buffer with minimum required experiences
    state = {
        "board": np.zeros(52, dtype=np.int8),
        "hand": np.zeros(52, dtype=np.int8),
        "action_mask": np.ones(53, dtype=np.int8),
    }

    for i in range(dqn_agent.batch_size + 10):
        dqn_agent.store_experience(
            state=state,
            action=i % 53,
            reward=float(i),
            next_state=state,
            done=False,
        )

    # Now training should work
    metrics = dqn_agent.train_step()

    assert "loss" in metrics
    assert "q_value_mean" in metrics
    assert "epsilon" in metrics
    assert isinstance(metrics["loss"], float)


def test_dqn_agent_target_network_update(dqn_agent):
    """Test target network update."""
    # Get initial target network weights
    initial_params = [p.clone() for p in dqn_agent.target_network.parameters()]

    # Modify Q-network (simulate training)
    for p in dqn_agent.q_network.parameters():
        p.data.add_(torch.randn_like(p) * 0.1)

    # Update target network
    dqn_agent.update_target_network()

    # Check that target network weights changed
    updated_params = list(dqn_agent.target_network.parameters())

    # At least one parameter should be different
    params_changed = any(
        not torch.equal(initial, updated)
        for initial, updated in zip(initial_params, updated_params, strict=True)
    )
    assert params_changed


def test_dqn_agent_end_episode(dqn_agent):
    """Test end episode behavior."""
    initial_episode_count = dqn_agent.episode_count

    # End episode (not at update frequency)
    dqn_agent.end_episode()
    assert dqn_agent.episode_count == initial_episode_count + 1

    # End multiple episodes to trigger target update
    for _ in range(dqn_agent.target_update_freq):
        dqn_agent.end_episode()

    # Episode count should have increased
    assert dqn_agent.episode_count > initial_episode_count


def test_dqn_agent_save_load(dqn_agent):
    """Test saving and loading agent."""
    # Train a bit to change weights
    state = {
        "board": np.zeros(52, dtype=np.int8),
        "hand": np.zeros(52, dtype=np.int8),
        "action_mask": np.ones(53, dtype=np.int8),
    }

    for i in range(50):
        dqn_agent.store_experience(state, i % 53, float(i), state, False)

    dqn_agent.train_step()
    dqn_agent.end_episode()

    # Save to temporary file
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "agent.pt"
        dqn_agent.save(str(save_path))

        # Create new agent and load
        new_agent = create_test_agent()
        new_agent.load(str(save_path))

        # Check that states match
        assert new_agent.episode_count == dqn_agent.episode_count
        assert new_agent.total_steps == dqn_agent.total_steps
        assert (
            abs(new_agent.policy.get_epsilon() - dqn_agent.policy.get_epsilon()) < 1e-6
        )


def test_dqn_agent_double_dqn():
    """Test Double DQN configuration."""
    agent_double = create_test_agent(double_dqn=True)
    agent_standard = create_test_agent(double_dqn=False)

    assert agent_double.double_dqn is True
    assert agent_standard.double_dqn is False


def test_dqn_agent_dueling_dqn():
    """Test Dueling DQN architecture."""
    agent = create_test_agent(dueling=True)

    # Check that networks use dueling architecture
    assert agent.q_network.dueling is True
    assert agent.target_network.dueling is True


def test_dqn_agent_gradient_clip():
    """Test gradient clipping."""
    agent_with_clip = create_test_agent(gradient_clip=1.0)
    agent_without_clip = create_test_agent(gradient_clip=0.0)  # 0 means no clipping

    assert agent_with_clip.gradient_clip == 1.0
    assert agent_without_clip.gradient_clip == 0.0

    # Ensure training still updates parameters when clipping disabled
    state = {
        "board": np.zeros(52, dtype=np.int8),
        "hand": np.zeros(52, dtype=np.int8),
        "action_mask": np.ones(53, dtype=np.int8),
    }

    for i in range(agent_without_clip.batch_size + 5):
        agent_without_clip.store_experience(
            state=state,
            action=i % 53,
            reward=float(i),
            next_state=state,
            done=False,
        )

    params_before = [p.clone() for p in agent_without_clip.q_network.parameters()]
    agent_without_clip.train_step()
    params_after = list(agent_without_clip.q_network.parameters())

    assert any(
        not torch.equal(before, after)
        for before, after in zip(params_before, params_after, strict=True)
    )


def test_dqn_agent_observation_to_state(dqn_agent):
    """Test observation to state conversion."""
    observation = {
        "board": np.ones(52, dtype=np.int8),
        "hand": np.zeros(52, dtype=np.int8),
        "action_mask": np.ones(53, dtype=np.int8),
    }

    state = dqn_agent._observation_to_state(observation)

    assert isinstance(state, np.ndarray)
    assert state.shape == (157,)  # 52 + 52 + 53
    assert state.dtype == np.float32


def test_dqn_agent_device():
    """Test device configuration."""
    agent_cpu = create_test_agent(device="cpu")
    assert agent_cpu.device.type == "cpu"

    # Only test cuda if available
    if torch.cuda.is_available():
        agent_cuda = create_test_agent(device="cuda")
        assert agent_cuda.device.type == "cuda"


def test_dqn_agent_seed_reproducibility():
    """Test that agents with same seed behave consistently."""
    seed = 42

    agent1 = create_test_agent(seed=seed)
    agent2 = create_test_agent(seed=seed)

    observation = {
        "board": np.zeros(52, dtype=np.int8),
        "hand": np.zeros(52, dtype=np.int8),
        "action_mask": np.ones(53, dtype=np.int8),
    }

    # Generate multiple actions
    actions1 = [agent1.select_action(observation, "player_0") for _ in range(50)]
    actions2 = [agent2.select_action(observation, "player_0") for _ in range(50)]

    # Should produce same sequence (with same seed)
    assert actions1 == actions2


def test_dqn_agent_epsilon_decay_during_training(dqn_agent):
    """Test that epsilon decays during training."""
    initial_epsilon = dqn_agent.policy.get_epsilon()

    # Fill buffer and train
    state = {
        "board": np.zeros(52, dtype=np.int8),
        "hand": np.zeros(52, dtype=np.int8),
        "action_mask": np.ones(53, dtype=np.int8),
    }

    for i in range(100):
        dqn_agent.store_experience(state, i % 53, float(i), state, False)

    # Train multiple steps
    for _ in range(10):
        dqn_agent.train_step()

    final_epsilon = dqn_agent.policy.get_epsilon()

    # Epsilon should have decayed
    assert final_epsilon < initial_epsilon


def test_dqn_agent_soft_update_tau():
    """Test that tau performs soft updates instead of hard copy."""
    agent = create_test_agent(tau=0.5)

    # Modify Q-network
    for param in agent.q_network.parameters():
        param.data.add_(torch.randn_like(param) * 0.1)

    initial_target_params = [p.clone() for p in agent.target_network.parameters()]
    agent.update_target_network()
    updated_target_params = list(agent.target_network.parameters())

    # Soft update should move parameters but not copy exactly
    assert any(
        not torch.equal(initial, updated)
        for initial, updated in zip(
            initial_target_params, updated_target_params, strict=True
        )
    )
    assert any(
        not torch.equal(updated, current)
        for updated, current in zip(
            updated_target_params, agent.q_network.parameters(), strict=True
        )
    )


def test_dqn_agent_linear_epsilon_decay_strategy():
    """Test DQN agent with linear epsilon decay strategy."""
    agent = create_test_agent(
        epsilon_decay_strategy="linear",
        epsilon_decay=0.99,
    )

    before = agent.policy.get_epsilon()
    agent.policy.decay()
    after = agent.policy.get_epsilon()

    assert after < before
