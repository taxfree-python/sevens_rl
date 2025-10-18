"""
config.pyの各種設定をテストするスクリプト
"""

import numpy as np
from src.sevens_env import SevensEnv
from configs.config import (
    DEFAULT_CONFIG,
    TRAINING_CONFIG,
    EXPERIMENT_WTA_CONFIG,
    EXPERIMENT_SPARSE_CONFIG,
    WINNER_TAKES_ALL_REWARDS,
    SPARSE_REWARDS,
)


def run_test_game(config_name, env_config, seed=42):
    """設定を使ってゲームを実行"""
    print(f"\n{'='*70}")
    print(f"テスト: {config_name}")
    print(f"{'='*70}")

    env = SevensEnv(
        num_players=env_config.num_players,
        render_mode=env_config.render_mode,
        reward_config=env_config.reward_config,
    )
    env.reset(seed=seed)

    print(f"プレイヤー数: {env_config.num_players}")
    print(f"報酬設定: {env_config.reward_config}")
    print(f"描画モード: {env_config.render_mode}")
    print()

    # ランダムエージェントでゲームを実行
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

    print("ゲーム終了!")
    print(f"総ステップ数: {step + 1}")
    print(f"上がり順: {env.finished_order}")
    print(f"累積報酬: {env._cumulative_rewards}")


def main():
    """各種設定でテストを実行"""
    print("\n" + "="*70)
    print("Sevens Environment - Configuration Tests")
    print("="*70)

    # 1. デフォルト設定
    run_test_game("デフォルト設定", DEFAULT_CONFIG)

    # 2. トレーニング設定
    run_test_game("トレーニング設定", TRAINING_CONFIG)

    # 3. Winner-takes-all設定
    run_test_game("Winner-takes-all設定", EXPERIMENT_WTA_CONFIG)

    # 4. スパース報酬設定
    run_test_game("スパース報酬設定", EXPERIMENT_SPARSE_CONFIG)

    # 5. カスタム設定の例
    from configs.config import SevensConfig
    custom_config = SevensConfig(
        num_players=3,
        reward_config={1: 5.0, 2: 1.0, 3: -2.0},
    )
    run_test_game("カスタム設定 (3人プレイ)", custom_config)

    print("\n" + "="*70)
    print("全テスト完了!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
