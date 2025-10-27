"""
ベースラインエージェントの単体テスト。

Unit tests for baseline agents.
"""

import numpy as np
import pytest

from src.agents import NearestSevensAgent, RandomAgent
from src.agents.base import Observation
from src.sevens_env import NUM_CARDS, SEVEN_RANK, Card


def _fake_observation(mask_indices: list[int]) -> Observation:
    action_mask = np.zeros(NUM_CARDS + 1, dtype=np.int8)
    for idx in mask_indices:
        action_mask[idx] = 1
    return {
        "board": np.zeros(NUM_CARDS, dtype=np.int8),
        "hand": np.zeros(NUM_CARDS, dtype=np.int8),
        "action_mask": action_mask,
    }


def test_random_agent_respects_action_mask():
    rng = np.random.default_rng(123)
    agent = RandomAgent(rng)
    card_a = Card(0, SEVEN_RANK - 2).to_id()
    card_b = Card(1, SEVEN_RANK + 3).to_id()
    obs = _fake_observation([card_a, card_b, NUM_CARDS])

    for _ in range(20):
        action = agent.select_action(obs, "player_0")
        assert action in {card_a, card_b, NUM_CARDS}


def test_random_agent_raises_without_actions():
    agent = RandomAgent(np.random.default_rng(0))
    obs = _fake_observation([])

    with pytest.raises(ValueError):
        agent.select_action(obs, "player_0")


def test_random_agent_requires_action_mask():
    agent = RandomAgent(np.random.default_rng(0))
    obs: Observation = {
        "board": np.zeros(NUM_CARDS, dtype=np.int8),
        "hand": np.zeros(NUM_CARDS, dtype=np.int8),
    }

    with pytest.raises(ValueError, match="missing"):
        agent.select_action(obs, "player_0")


def test_random_agent_rejects_empty_mask():
    agent = RandomAgent(np.random.default_rng(0))
    obs: Observation = {
        "board": np.zeros(NUM_CARDS, dtype=np.int8),
        "hand": np.zeros(NUM_CARDS, dtype=np.int8),
        "action_mask": np.zeros(0, dtype=np.int8),
    }

    with pytest.raises(ValueError, match="empty"):
        agent.select_action(obs, "player_0")


def test_nearest_sevens_agent_prefers_closest_card():
    agent = NearestSevensAgent()
    close_card = Card(2, SEVEN_RANK - 1).to_id()
    far_card = Card(3, SEVEN_RANK + 4).to_id()
    obs = _fake_observation([close_card, far_card, NUM_CARDS])

    action = agent.select_action(obs, "player_1")
    assert action == close_card


def test_nearest_sevens_agent_prefers_high_rank_when_requested():
    agent = NearestSevensAgent(prefer_high_rank=True)
    low_card = Card(0, SEVEN_RANK - 1).to_id()
    high_card = Card(1, SEVEN_RANK + 1).to_id()
    obs = _fake_observation([low_card, high_card])

    action = agent.select_action(obs, "player_2")
    assert action == high_card


def test_nearest_sevens_agent_passes_when_only_pass_available():
    agent = NearestSevensAgent()
    obs = _fake_observation([NUM_CARDS])

    action = agent.select_action(obs, "player_3")
    assert action == NUM_CARDS


def test_nearest_sevens_agent_raises_without_any_action():
    agent = NearestSevensAgent()
    obs = {
        "board": np.zeros(NUM_CARDS, dtype=np.int8),
        "hand": np.zeros(NUM_CARDS, dtype=np.int8),
        "action_mask": np.zeros(NUM_CARDS + 1, dtype=np.int8),
    }

    with pytest.raises(ValueError):
        agent.select_action(obs, "player_4")


def test_nearest_sevens_agent_requires_action_mask():
    agent = NearestSevensAgent()
    obs: Observation = {
        "board": np.zeros(NUM_CARDS, dtype=np.int8),
        "hand": np.zeros(NUM_CARDS, dtype=np.int8),
    }

    with pytest.raises(ValueError, match="missing"):
        agent.select_action(obs, "player_5")


def test_nearest_sevens_agent_rejects_empty_mask():
    agent = NearestSevensAgent()
    obs: Observation = {
        "board": np.zeros(NUM_CARDS, dtype=np.int8),
        "hand": np.zeros(NUM_CARDS, dtype=np.int8),
        "action_mask": np.zeros(0, dtype=np.int8),
    }

    with pytest.raises(ValueError, match="empty"):
        agent.select_action(obs, "player_6")
