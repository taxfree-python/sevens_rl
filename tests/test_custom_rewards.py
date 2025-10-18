"""カスタム報酬設定のテスト"""

import pytest
import numpy as np
from src.sevens_env import SevensEnv
from configs.config import DEFAULT_REWARDS


def run_game_with_rewards(reward_config, seed=42):
    """指定された報酬設定でゲームを実行"""
    env = SevensEnv(num_players=4, reward_config=reward_config)
    env.reset(seed=seed)

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

    return env, step + 1


def test_custom_winner_takes_all_rewards():
    """カスタム報酬設定（Winner-takes-all）でゲームを実行"""
    # カスタム報酬: 1位のみ+10、それ以外0
    custom_rewards = {1: 10.0, 2: 0.0, 3: 0.0, 4: 0.0}

    env, step_count = run_game_with_rewards(custom_rewards)

    # 報酬設定が正しく適用されていることを確認
    assert env.reward_config == custom_rewards
    # ゲームが終了していることを確認
    assert len(env.finished_order) > 0
    # 1位のプレイヤーだけが報酬を得ていることを確認
    first_place = env.finished_order[0]
    assert env._cumulative_rewards[first_place] == 10.0
    # 他のプレイヤーの報酬が0であることを確認
    for player, reward in env._cumulative_rewards.items():
        if player != first_place:
            assert reward == 0.0


def test_default_rewards():
    """デフォルト報酬設定でゲームを実行"""
    env, step_count = run_game_with_rewards(None)

    # デフォルト報酬設定が適用されていることを確認（4人用）
    assert env.reward_config == DEFAULT_REWARDS[4]
    # ゲームが終了していることを確認
    assert len(env.finished_order) > 0
    # 累積報酬が計算されていることを確認
    assert len(env._cumulative_rewards) == 4


def test_sparse_rewards():
    """スパース報酬設定でゲームを実行"""
    # スパース報酬: 1位と最下位のみ報酬/ペナルティ
    sparse_rewards = {1: 10.0, 2: 0.0, 3: 0.0, 4: -10.0}

    env, step_count = run_game_with_rewards(sparse_rewards)

    # 報酬設定が正しく適用されていることを確認
    assert env.reward_config == sparse_rewards
    # ゲームが終了していることを確認
    assert len(env.finished_order) == 4
    # 1位と最下位の報酬を確認
    first_place = env.finished_order[0]
    last_place = env.finished_order[-1]
    assert env._cumulative_rewards[first_place] == 10.0
    assert env._cumulative_rewards[last_place] == -10.0


def test_negative_rewards():
    """全員にペナルティを与える報酬設定でゲームを実行"""
    negative_rewards = {1: -1.0, 2: -2.0, 3: -3.0, 4: -4.0}

    env, step_count = run_game_with_rewards(negative_rewards)

    # 全プレイヤーが負の報酬を得ていることを確認
    for player, reward in env._cumulative_rewards.items():
        assert reward < 0


def run_interactive_comparison():
    """対話的な報酬設定比較（pytest実行時はスキップ）"""
    print("\n" + "="*70)
    print("デフォルト報酬設定テスト")
    print("="*70)
    env, step_count = run_game_with_rewards(None)
    print(f"報酬設定: {env.reward_config}")
    print(f"総ステップ数: {step_count}")
    print(f"上がり順: {env.finished_order}")
    print(f"累積報酬: {env._cumulative_rewards}")

    print("\n" + "="*70)
    print("カスタム報酬設定テスト (Winner-takes-all)")
    print("="*70)
    custom_rewards = {1: 10.0, 2: 0.0, 3: 0.0, 4: 0.0}
    env, step_count = run_game_with_rewards(custom_rewards)
    print(f"報酬設定: {custom_rewards}")
    print(f"総ステップ数: {step_count}")
    print(f"上がり順: {env.finished_order}")
    print(f"累積報酬: {env._cumulative_rewards}")


if __name__ == "__main__":
    run_interactive_comparison()
