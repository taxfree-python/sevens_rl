"""Tests for replay buffer implementation."""

import numpy as np
import pytest

from src.rl.replay_buffer import ReplayBuffer


def test_replay_buffer_initialization():
    """Test replay buffer initialization."""
    capacity = 1000
    buffer = ReplayBuffer(capacity=capacity)

    assert buffer.capacity == capacity
    assert len(buffer) == 0


def test_replay_buffer_push():
    """Test adding experiences to buffer."""
    buffer = ReplayBuffer(capacity=100)

    # Add single experience
    state = np.array([1, 2, 3])
    action = 0
    reward = 1.0
    next_state = np.array([4, 5, 6])
    done = False

    buffer.push(state, action, reward, next_state, done)
    assert len(buffer) == 1

    # Add multiple experiences
    for i in range(10):
        buffer.push(state, i, float(i), next_state, False)
    assert len(buffer) == 11


def test_replay_buffer_circular():
    """Test circular buffer behavior when capacity exceeded."""
    capacity = 5
    buffer = ReplayBuffer(capacity=capacity)

    state = np.array([1, 2, 3])
    next_state = np.array([4, 5, 6])

    # Fill buffer beyond capacity
    for i in range(10):
        buffer.push(state, i, float(i), next_state, False)

    # Should only keep last 'capacity' experiences
    assert len(buffer) == capacity


def test_replay_buffer_sample():
    """Test sampling from buffer."""
    buffer = ReplayBuffer(capacity=100, seed=42)

    # Add experiences
    for i in range(50):
        state = np.array([i])
        action = i % 5
        reward = float(i)
        next_state = np.array([i + 1])
        done = i == 49

        buffer.push(state, action, reward, next_state, done)

    # Sample batch
    batch_size = 10
    states, actions, rewards, next_states, dones = buffer.sample(batch_size)

    assert len(states) == batch_size
    assert len(actions) == batch_size
    assert len(rewards) == batch_size
    assert len(next_states) == batch_size
    assert len(dones) == batch_size


def test_replay_buffer_sample_insufficient():
    """Test sampling with insufficient buffer size."""
    buffer = ReplayBuffer(capacity=100)

    # Add only 5 experiences
    for i in range(5):
        buffer.push(np.array([i]), i, float(i), np.array([i + 1]), False)

    # Try to sample more than available
    with pytest.raises(ValueError, match="Cannot sample"):
        buffer.sample(batch_size=10)


def test_replay_buffer_sample_reproducibility():
    """Test that sampling with same seed is reproducible."""
    seed = 42

    # Create two buffers with same seed
    buffer1 = ReplayBuffer(capacity=100, seed=seed)
    buffer2 = ReplayBuffer(capacity=100, seed=seed)

    # Add same experiences to both
    for i in range(50):
        state = np.array([i])
        buffer1.push(state, i, float(i), state, False)
        buffer2.push(state, i, float(i), state, False)

    # Sample from both
    batch1 = buffer1.sample(10)
    batch2 = buffer2.sample(10)

    # Check rewards are same (as proxy for same sampling)
    assert batch1[2] == batch2[2]


def test_replay_buffer_clear():
    """Test clearing buffer."""
    buffer = ReplayBuffer(capacity=100)

    # Add experiences
    for i in range(20):
        buffer.push(np.array([i]), i, float(i), np.array([i + 1]), False)

    assert len(buffer) == 20

    # Clear buffer
    buffer.clear()
    assert len(buffer) == 0


def test_replay_buffer_is_ready():
    """Test is_ready method."""
    buffer = ReplayBuffer(capacity=100)

    min_size = 10
    assert not buffer.is_ready(min_size)

    # Add experiences up to min_size
    for i in range(min_size):
        buffer.push(np.array([i]), i, float(i), np.array([i + 1]), False)

    assert buffer.is_ready(min_size)
    assert buffer.is_ready(min_size - 1)
    assert not buffer.is_ready(min_size + 1)


def test_replay_buffer_diverse_types():
    """Test buffer with diverse state/action types."""
    buffer = ReplayBuffer(capacity=100)

    # Dictionary state (like Sevens environment)
    state = {
        "board": np.array([1, 0, 1]),
        "hand": np.array([0, 1, 0]),
        "action_mask": np.array([1, 1, 0, 1]),
    }
    next_state = {
        "board": np.array([1, 1, 1]),
        "hand": np.array([0, 0, 0]),
        "action_mask": np.array([0, 0, 0, 1]),
    }

    buffer.push(state, 1, 1.0, next_state, True)
    assert len(buffer) == 1

    states, actions, rewards, next_states, dones = buffer.sample(1)
    assert len(states) == 1
    assert actions[0] == 1
    assert rewards[0] == 1.0
    assert dones[0] is True


def test_replay_buffer_large_capacity():
    """Test buffer with large capacity."""
    capacity = 100000
    buffer = ReplayBuffer(capacity=capacity)

    # Add many experiences
    num_experiences = 1000
    for i in range(num_experiences):
        buffer.push(np.array([i]), i % 10, float(i), np.array([i + 1]), False)

    assert len(buffer) == num_experiences

    # Sample large batch
    batch_size = 500
    states, actions, rewards, next_states, dones = buffer.sample(batch_size)
    assert len(states) == batch_size


def test_replay_buffer_sample_without_replacement():
    """Test that sampling is without replacement within a batch."""
    buffer = ReplayBuffer(capacity=100, seed=42)

    # Add unique experiences
    for i in range(50):
        buffer.push(np.array([i]), i, float(i), np.array([i + 1]), False)

    # Sample batch
    batch_size = 20
    states, actions, rewards, next_states, dones = buffer.sample(batch_size)

    # Check rewards are unique (no duplicates in batch)
    assert len(set(rewards)) == batch_size
