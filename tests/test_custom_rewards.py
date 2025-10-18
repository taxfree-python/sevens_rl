"""カスタム報酬設定のテスト"""

from src.sevens_env import SevensEnv
import numpy as np


def test_custom_rewards():
    """カスタム報酬設定でゲームを実行"""
    # カスタム報酬: 1位のみ+10、それ以外0
    custom_rewards = {1: 10.0, 2: 0.0, 3: 0.0, 4: 0.0}

    env = SevensEnv(num_players=4, reward_config=custom_rewards)
    env.reset(seed=42)

    print("カスタム報酬設定テスト")
    print(f"報酬設定: {custom_rewards}")
    print()

    # ランダムエージェント
    for step in range(1000):
        agent = env.agent_selection

        if env.terminations[agent] or env.truncations[agent]:
            break

        observation = env.observe(agent)
        action_mask = observation['action_mask']
        valid_actions = np.where(action_mask == 1)[0]

        if len(valid_actions) > 0:
            action = np.random.choice(valid_actions)
            env.step(action)

    print("\n" + "="*50)
    print("ゲーム終了!")
    print("="*50)
    print(f"総ステップ数: {step + 1}")
    print(f"上がり順: {env.finished_order}")
    print(f"累積報酬: {env._cumulative_rewards}")
    print()


def test_default_rewards():
    """デフォルト報酬設定でゲームを実行"""
    env = SevensEnv(num_players=4)  # reward_config=None
    env.reset(seed=42)

    print("デフォルト報酬設定テスト")
    print(f"報酬設定: {env.reward_config}")
    print()

    # ランダムエージェント
    for step in range(1000):
        agent = env.agent_selection

        if env.terminations[agent] or env.truncations[agent]:
            break

        observation = env.observe(agent)
        action_mask = observation['action_mask']
        valid_actions = np.where(action_mask == 1)[0]

        if len(valid_actions) > 0:
            action = np.random.choice(valid_actions)
            env.step(action)

    print("\n" + "="*50)
    print("ゲーム終了!")
    print("="*50)
    print(f"総ステップ数: {step + 1}")
    print(f"上がり順: {env.finished_order}")
    print(f"累積報酬: {env._cumulative_rewards}")
    print()


if __name__ == "__main__":
    test_default_rewards()
    print("\n" + "="*70 + "\n")
    test_custom_rewards()
