"""
七並べ環境のテストスクリプト
ランダムエージェントで動作確認
"""

import numpy as np
from src.sevens_env import SevensEnv
from configs.config import EVAL_CONFIG


def random_agent(observation, agent):
    """ランダムに有効なアクションを選択"""
    action_mask = observation['action_mask']
    valid_actions = np.where(action_mask == 1)[0]
    return np.random.choice(valid_actions)


def test_game(num_players=4, render=True, reward_config=None):
    """ゲームをテスト"""
    env = SevensEnv(
        num_players=num_players,
        render_mode='human' if render else None,
        reward_config=reward_config
    )
    env.reset()

    if render:
        env.render()

    step_count = 0
    max_steps = 1000

    while env.agents and step_count < max_steps:
        agent = env.agent_selection
        observation = env.observe(agent)

        # ランダムエージェント
        action = random_agent(observation, agent)

        # ステップ実行
        env.step(action)

        if render:
            if action < 52:
                from src.sevens_env import Card
                card = Card.from_id(action)
                print(f"{agent} が {card} を出しました")
            else:
                print(f"{agent} がパスしました")
            env.render()

        step_count += 1

    # 結果表示
    print("\n" + "="*50)
    print("ゲーム終了!")
    print("="*50)
    print(f"総ステップ数: {step_count}")
    print(f"上がり順: {env.finished_order}")
    print(f"累積報酬: {env._cumulative_rewards}")


if __name__ == "__main__":
    print("七並べ環境テスト - ランダムエージェント")
    print()
    test_game(num_players=4, render=True)
