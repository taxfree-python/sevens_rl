"""
config.pyの各種設定をテストするスクリプト
"""

import numpy as np

from configs.config import (
    DEFAULT_CONFIG,
    EXPERIMENT_SPARSE_CONFIG,
    EXPERIMENT_WTA_CONFIG,
    TRAINING_CONFIG,
    SevensConfig,
)
from src.sevens_env import SevensEnv


def run_game_with_config(env_config, seed=42):
    """設定を使ってゲームを実行"""
    env = SevensEnv(
        num_players=env_config.num_players,
        render_mode=env_config.render_mode,
        reward_config=env_config.reward_config,
    )
    env.reset(seed=seed)

    # ランダムエージェントでゲームを実行
    step_count = 0
    for _step in range(1000):
        agent = env.agent_selection

        if env.terminations[agent] or env.truncations[agent]:
            break

        observation = env.observe(agent)
        action_mask = observation['action_mask']
        valid_actions = np.where(action_mask == 1)[0]

        if len(valid_actions) > 0:
            action = np.random.choice(valid_actions)
            env.step(action)

        step_count += 1

    return env, step_count


def test_default_config():
    """デフォルト設定でゲームを実行"""
    env, step_count = run_game_with_config(DEFAULT_CONFIG)

    # 設定が正しく適用されていることを確認
    assert env.num_players == DEFAULT_CONFIG.num_players
    assert env.reward_config == DEFAULT_CONFIG.reward_config
    # ゲームが終了していることを確認
    assert len(env.finished_order) > 0
    assert len(env._cumulative_rewards) == DEFAULT_CONFIG.num_players


def test_training_config():
    """トレーニング設定でゲームを実行"""
    env, step_count = run_game_with_config(TRAINING_CONFIG)

    # 設定が正しく適用されていることを確認
    assert env.num_players == TRAINING_CONFIG.num_players
    assert env.reward_config == TRAINING_CONFIG.reward_config
    # ゲームが終了していることを確認
    assert len(env.finished_order) > 0


def test_winner_takes_all_config():
    """Winner-takes-all設定でゲームを実行"""
    env, step_count = run_game_with_config(EXPERIMENT_WTA_CONFIG)

    # 設定が正しく適用されていることを確認
    assert env.num_players == EXPERIMENT_WTA_CONFIG.num_players
    assert env.reward_config == EXPERIMENT_WTA_CONFIG.reward_config
    # ゲームが終了していることを確認
    assert len(env.finished_order) > 0
    # 1位のみが報酬を得ていることを確認
    first_place = env.finished_order[0]
    assert env._cumulative_rewards[first_place] > 0
    for player, reward in env._cumulative_rewards.items():
        if player != first_place:
            assert reward == 0.0


def test_sparse_rewards_config():
    """スパース報酬設定でゲームを実行"""
    env, step_count = run_game_with_config(EXPERIMENT_SPARSE_CONFIG)

    # 設定が正しく適用されていることを確認
    assert env.num_players == EXPERIMENT_SPARSE_CONFIG.num_players
    assert env.reward_config == EXPERIMENT_SPARSE_CONFIG.reward_config
    # ゲームが終了していることを確認
    assert len(env.finished_order) == 4


def test_custom_config_three_players():
    """カスタム設定（3人プレイ）でゲームを実行"""
    custom_config = SevensConfig(
        num_players=3,
        reward_config={1: 5.0, 2: 1.0, 3: -2.0},
    )
    env, step_count = run_game_with_config(custom_config)

    # 設定が正しく適用されていることを確認
    assert env.num_players == 3
    assert env.reward_config == {1: 5.0, 2: 1.0, 3: -2.0}
    # ゲームが終了していることを確認
    assert len(env.finished_order) == 3
    assert len(env._cumulative_rewards) == 3


def test_config_dataclass_defaults():
    """SevensConfigのデフォルト値をテスト"""
    config = SevensConfig()

    assert config.num_players == 4
    assert config.render_mode is None
    # デフォルトでは4人用の報酬設定が適用される
    from configs.config import DEFAULT_REWARDS
    assert config.reward_config == DEFAULT_REWARDS[4]


def run_interactive_config_tests():
    """対話的な設定テスト（pytest実行時はスキップ）"""
    configs = [
        ("デフォルト設定", DEFAULT_CONFIG),
        ("トレーニング設定", TRAINING_CONFIG),
        ("Winner-takes-all設定", EXPERIMENT_WTA_CONFIG),
        ("スパース報酬設定", EXPERIMENT_SPARSE_CONFIG),
    ]

    print("\n" + "="*70)
    print("Sevens Environment - Configuration Tests")
    print("="*70)

    for config_name, env_config in configs:
        print(f"\n{'='*70}")
        print(f"テスト: {config_name}")
        print(f"{'='*70}")
        env, step_count = run_game_with_config(env_config)
        print(f"プレイヤー数: {env_config.num_players}")
        print(f"報酬設定: {env_config.reward_config}")
        print(f"描画モード: {env_config.render_mode}")
        print(f"総ステップ数: {step_count}")
        print(f"上がり順: {env.finished_order}")
        print(f"累積報酬: {env._cumulative_rewards}")

    # カスタム設定
    print(f"\n{'='*70}")
    print("テスト: カスタム設定 (3人プレイ)")
    print(f"{'='*70}")
    custom_config = SevensConfig(
        num_players=3,
        reward_config={1: 5.0, 2: 1.0, 3: -2.0},
    )
    env, step_count = run_game_with_config(custom_config)
    print(f"プレイヤー数: {custom_config.num_players}")
    print(f"報酬設定: {custom_config.reward_config}")
    print(f"総ステップ数: {step_count}")
    print(f"上がり順: {env.finished_order}")
    print(f"累積報酬: {env._cumulative_rewards}")

    print("\n" + "="*70)
    print("全テスト完了!")
    print("="*70 + "\n")


if __name__ == "__main__":
    run_interactive_config_tests()
