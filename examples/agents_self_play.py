"""
ベースラインエージェント同士の自己対局を対話的に実行するスクリプト。

Interactive script for running self-play matches between baseline agents.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents import AgentPolicy, NearestSevensAgent, RandomAgent
from src.sevens_env import NUM_CARDS, Card, SevensEnv

AgentFactory = Callable[[int], AgentPolicy]


def _prompt_int(prompt: str, *, default: int, minimum: int, maximum: int) -> int:
    while True:
        raw = input(f"{prompt} [{default}]: ").strip()
        if not raw:
            return default
        if raw.isdecimal():
            value = int(raw)
            if minimum <= value <= maximum:
                return value
        print(f"無効な入力です。{minimum}-{maximum} の整数を入力してください。")


def _prompt_choice(prompt: str, options: dict[str, str], *, default: str) -> str:
    option_text = ", ".join(f"{key}({label})" for key, label in options.items())
    while True:
        raw = input(f"{prompt} [{default}] {{{option_text}}}: ").strip().lower()
        if not raw:
            return default
        if raw in options:
            return raw
        print(f"無効な入力です。{list(options)} から選択してください。")


def _describe_action(action: int) -> str:
    if action == NUM_CARDS:
        return "pass"
    card = Card.from_id(action)
    return repr(card)


def _build_factories(seed: int) -> dict[str, AgentFactory]:
    return {
        "random": lambda idx: RandomAgent(np.random.default_rng(seed + idx)),
        "nearest": lambda idx: NearestSevensAgent(prefer_high_rank=False),
        "nearest_high": lambda idx: NearestSevensAgent(prefer_high_rank=True),
    }


def main() -> None:
    print("=" * 60)
    print("七並べ 自己対局デモ (Baseline Agents Self-Play)")
    print("=" * 60)

    num_players = _prompt_int(
        "プレイヤー数を入力してください", default=4, minimum=2, maximum=4
    )
    seed = _prompt_int(
        "乱数シードを入力してください", default=0, minimum=0, maximum=10_000
    )

    factories = _build_factories(seed)
    labels = {
        "random": "ランダム",
        "nearest": "7に近いカードを優先（低ランク優先）",
        "nearest_high": "7に近いカードを優先（高ランク優先）",
    }

    policies: dict[str, AgentPolicy] = {}
    for idx in range(num_players):
        agent_name = f"player_{idx}"
        choice = _prompt_choice(
            f"{agent_name} のエージェント種別を選択してください",
            options=labels,
            default="random",
        )
        policies[agent_name] = factories[choice](idx)

    env = SevensEnv(num_players=num_players)
    env.reset(seed=seed)

    print("\n--- 対局開始 ---\n")
    step_count = 0

    while env.agents:
        agent = env.agent_selection
        policy = policies[agent]
        observation = env.observe(agent)
        action = policy.select_action(observation, agent)
        env.step(action)

        print(
            f"[{step_count:03d}] {agent} ({policy.name}) -> {_describe_action(action)}"
        )
        step_count += 1

    print("\n--- 対局終了 ---\n")
    print(f"総ターン数: {step_count}")
    print("順位:")
    for rank, agent in enumerate(env.finished_order, start=1):
        reward = env._cumulative_rewards.get(agent, 0.0)
        print(f"  {rank}位: {agent} (報酬: {reward:+.1f})")

    remaining = set(env.possible_agents) - set(env.finished_order)
    if remaining:
        print("未完了:")
        for agent in remaining:
            reward = env._cumulative_rewards.get(agent, 0.0)
            print(f"  - {agent} (報酬: {reward:+.1f})")


if __name__ == "__main__":
    main()
