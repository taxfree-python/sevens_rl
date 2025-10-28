"""Tests for epsilon-greedy policy implementation."""

import numpy as np
import pytest

from src.rl.policy import EpsilonGreedyPolicy


def test_epsilon_greedy_initialization():
    """Test epsilon-greedy policy initialization."""
    policy = EpsilonGreedyPolicy(
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
    )

    assert policy.epsilon == 1.0
    assert policy.epsilon_start == 1.0
    assert policy.epsilon_end == 0.01
    assert policy.epsilon_decay == 0.995


def test_epsilon_greedy_select_action_shape():
    """Test action selection output."""
    policy = EpsilonGreedyPolicy(seed=42)

    q_values = np.array([0.1, 0.5, 0.3, 0.8, 0.2])
    action = policy.select_action(q_values)

    assert isinstance(action, (int, np.integer))
    assert 0 <= action < len(q_values)


def test_epsilon_greedy_with_action_mask():
    """Test action selection with action mask."""
    policy = EpsilonGreedyPolicy(epsilon_start=0.0, seed=42)  # Pure greedy

    q_values = np.array([0.1, 0.9, 0.5, 0.3])
    action_mask = np.array([1, 0, 1, 1])  # Block action 1

    action = policy.select_action(q_values, action_mask)

    # Should not select masked action (action 1)
    assert action != 1
    assert action in [0, 2, 3]


def test_epsilon_greedy_exploitation():
    """Test exploitation behavior (epsilon=0)."""
    policy = EpsilonGreedyPolicy(epsilon_start=0.0)

    q_values = np.array([0.1, 0.9, 0.5, 0.3])

    # With epsilon=0, should always select best action
    actions = [policy.select_action(q_values) for _ in range(100)]

    # All actions should be 1 (highest Q-value)
    assert all(a == 1 for a in actions)


def test_epsilon_greedy_exploration():
    """Test exploration behavior (epsilon=1)."""
    policy = EpsilonGreedyPolicy(epsilon_start=1.0, seed=42)

    q_values = np.array([0.1, 0.9, 0.5, 0.3])

    # With epsilon=1, should explore randomly
    actions = [policy.select_action(q_values) for _ in range(100)]

    # Should have variety of actions (not all optimal)
    unique_actions = set(actions)
    assert len(unique_actions) > 1


def test_epsilon_greedy_decay():
    """Test epsilon decay."""
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay = 0.95

    policy = EpsilonGreedyPolicy(
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay=epsilon_decay,
    )

    assert policy.epsilon == epsilon_start

    # Decay once
    policy.decay()
    assert policy.epsilon == pytest.approx(epsilon_start * epsilon_decay)

    # Decay many times
    for _ in range(1000):
        policy.decay()

    # Should reach minimum
    assert policy.epsilon == epsilon_end


def test_epsilon_greedy_reset():
    """Test resetting epsilon to initial value."""
    policy = EpsilonGreedyPolicy(epsilon_start=1.0)

    # Decay epsilon
    for _ in range(10):
        policy.decay()

    assert policy.epsilon < 1.0

    # Reset
    policy.reset()
    assert policy.epsilon == 1.0


def test_epsilon_greedy_get_epsilon():
    """Test getting current epsilon value."""
    epsilon_start = 0.5
    policy = EpsilonGreedyPolicy(epsilon_start=epsilon_start)

    assert policy.get_epsilon() == epsilon_start

    policy.decay()
    assert policy.get_epsilon() < epsilon_start


def test_epsilon_greedy_set_epsilon():
    """Test setting epsilon to specific value."""
    policy = EpsilonGreedyPolicy()

    policy.set_epsilon(0.5)
    assert policy.epsilon == 0.5

    # Test clamping to epsilon_end
    policy.set_epsilon(0.001)
    assert policy.epsilon == policy.epsilon_end

    # Test clamping to 1.0
    policy.set_epsilon(2.0)
    assert policy.epsilon == 1.0


def test_epsilon_greedy_no_valid_actions():
    """Test error when no valid actions available."""
    policy = EpsilonGreedyPolicy()

    q_values = np.array([0.1, 0.5, 0.3])
    action_mask = np.array([0, 0, 0])  # All actions invalid

    with pytest.raises(ValueError, match="No valid actions"):
        policy.select_action(q_values, action_mask)


def test_epsilon_greedy_single_valid_action():
    """Test behavior with only one valid action."""
    policy = EpsilonGreedyPolicy(seed=42)

    q_values = np.array([0.1, 0.9, 0.5, 0.3])
    action_mask = np.array([0, 0, 1, 0])  # Only action 2 valid

    # Should always select the only valid action
    actions = [policy.select_action(q_values, action_mask) for _ in range(100)]
    assert all(a == 2 for a in actions)


def test_epsilon_greedy_reproducibility():
    """Test that policy with same seed produces same results."""
    seed = 42

    policy1 = EpsilonGreedyPolicy(epsilon_start=0.5, seed=seed)
    policy2 = EpsilonGreedyPolicy(epsilon_start=0.5, seed=seed)

    q_values = np.array([0.1, 0.5, 0.3, 0.8, 0.2])

    actions1 = [policy1.select_action(q_values) for _ in range(50)]
    actions2 = [policy2.select_action(q_values) for _ in range(50)]

    assert actions1 == actions2


def test_epsilon_greedy_greedy_with_mask():
    """Test that greedy selection respects action mask."""
    policy = EpsilonGreedyPolicy(epsilon_start=0.0)  # Pure greedy

    q_values = np.array([0.1, 0.9, 0.5, 0.8])
    action_mask = np.array([1, 0, 1, 1])  # Block highest Q-value (action 1)

    action = policy.select_action(q_values, action_mask)

    # Should select action 3 (highest valid Q-value)
    assert action == 3


def test_epsilon_greedy_mixed_strategy():
    """Test epsilon-greedy mixed strategy."""
    epsilon = 0.5
    policy = EpsilonGreedyPolicy(epsilon_start=epsilon, seed=42)

    q_values = np.array([0.1, 0.9, 0.1, 0.1])  # Action 1 is clearly best

    # Collect many actions
    num_trials = 10000
    actions = [policy.select_action(q_values) for _ in range(num_trials)]

    # Count greedy actions (action 1)
    greedy_count = sum(1 for a in actions if a == 1)
    greedy_ratio = greedy_count / num_trials

    # Should be roughly (1-epsilon) + epsilon/4 = 0.5 + 0.125 = 0.625
    # Allow for some statistical variation
    expected_ratio = (1 - epsilon) + (epsilon / len(q_values))
    assert abs(greedy_ratio - expected_ratio) < 0.05


def test_epsilon_greedy_linear_decay():
    """Test linear epsilon decay strategy."""
    policy = EpsilonGreedyPolicy(
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay=0.99,
        decay_strategy="linear",
    )

    policy.decay()
    expected_step = (1.0 - 0.1) * (1 - 0.99)
    assert policy.get_epsilon() == pytest.approx(1.0 - expected_step)


def test_epsilon_greedy_invalid_decay_strategy():
    """Invalid decay strategy should raise ValueError."""
    with pytest.raises(ValueError):
        EpsilonGreedyPolicy(decay_strategy="unsupported")
