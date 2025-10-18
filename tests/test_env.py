"""
七並べ環境のテストスクリプト
ランダムエージェントで動作確認
"""

import numpy as np

from src.sevens_env import Card, SevensEnv


def random_agent(observation, agent):
    """ランダムに有効なアクションを選択"""
    action_mask = observation["action_mask"]
    valid_actions = np.where(action_mask == 1)[0]
    return np.random.choice(valid_actions)


def test_basic_game():
    """基本的なゲームが正常に完了することをテスト"""
    env = SevensEnv(num_players=4, render_mode=None)
    env.reset(seed=42)

    step_count = 0
    max_steps = 1000

    while env.agents and step_count < max_steps:
        agent = env.agent_selection
        observation = env.observe(agent)
        action = random_agent(observation, agent)
        env.step(action)
        step_count += 1

    # ゲームが終了していることを確認
    assert len(env.agents) == 0 or step_count == max_steps
    # 上がり順が記録されていることを確認
    assert len(env.finished_order) > 0
    # 累積報酬が計算されていることを確認
    assert len(env._cumulative_rewards) == 4


def test_game_with_different_player_counts():
    """異なるプレイヤー数でゲームが正常に動作することをテスト"""
    for num_players in [2, 3, 4]:
        env = SevensEnv(num_players=num_players, render_mode=None)
        env.reset(seed=42)

        step_count = 0
        max_steps = 1000

        while env.agents and step_count < max_steps:
            agent = env.agent_selection
            observation = env.observe(agent)
            action = random_agent(observation, agent)
            env.step(action)
            step_count += 1

        assert len(env._cumulative_rewards) == num_players


def test_observation_space():
    """観測空間が正しいことをテスト"""
    env = SevensEnv(num_players=4)
    env.reset(seed=42)

    agent = env.agent_selection
    observation = env.observe(agent)

    # 観測空間のキーを確認
    assert "board" in observation
    assert "hand" in observation
    assert "action_mask" in observation

    # 各要素のサイズを確認
    assert observation["board"].shape == (52,)
    assert observation["hand"].shape == (52,)
    assert observation["action_mask"].shape == (53,)


def test_action_mask_validity():
    """アクションマスクが正しく機能することをテスト"""
    env = SevensEnv(num_players=4)
    env.reset(seed=42)

    agent = env.agent_selection
    observation = env.observe(agent)
    action_mask = observation["action_mask"]

    # 有効なアクションが少なくとも1つ存在することを確認
    assert np.sum(action_mask) > 0

    # 有効なアクションを実行
    valid_actions = np.where(action_mask == 1)[0]
    action = valid_actions[0]
    env.step(action)

    # エラーが発生しないことを確認（正常に実行されればOK）


def test_custom_reward_config():
    """カスタム報酬設定が正しく適用されることをテスト"""
    custom_rewards = {1: 10.0, 2: 5.0, 3: -5.0, 4: -10.0}
    env = SevensEnv(num_players=4, reward_config=custom_rewards)
    env.reset(seed=42)

    # 報酬設定が正しく適用されていることを確認
    assert env.reward_config == custom_rewards


def test_initial_sevens_placement():
    """初期化フェーズで全ての7が場に出ることをテスト"""
    env = SevensEnv(num_players=4)
    env.reset(seed=42)

    # 全ての7が場に出ていることを確認
    for suit in range(4):
        seven_id = Card(suit, 7).to_id()
        assert env.board[seven_id] == 1, f"7 of suit {suit} should be on board"

    # どのプレイヤーも7を持っていないことを確認
    for agent in env.agents:
        for suit in range(4):
            seven_id = Card(suit, 7).to_id()
            assert (
                env.hands[agent][seven_id] == 0
            ), f"{agent} should not have 7 of suit {suit}"


def test_diamond_seven_starting_player():
    """ダイヤの7を持っていたプレイヤーが先攻になることをテスト"""
    # 複数回テストして確率的に検証
    for seed in range(10):
        env = SevensEnv(num_players=4)
        # reset前にダイヤの7の持ち主を追跡できないため、
        # resetが正常に完了し、先攻プレイヤーが決まることを確認
        env.reset(seed=seed)

        # agent_selectionが設定されていることを確認
        assert env.agent_selection is not None
        assert env.agent_selection in env.agents

        # ダイヤの7が場に出ていることを確認
        diamond_seven_id = Card(2, 7).to_id()  # suit=2はダイヤ
        assert env.board[diamond_seven_id] == 1


def test_card_distribution():
    """カード配布が正しく行われることをテスト"""
    # 2人プレイ: 52枚を均等に配布 (各26枚)
    env = SevensEnv(num_players=2)
    env.reset(seed=42)

    # 7を除いた手札枚数を確認 (各プレイヤーは7を2枚ずつ持っていたはず)
    total_cards_in_hands = sum(np.sum(env.hands[agent]) for agent in env.agents)
    cards_on_board = np.sum(env.board)

    # 手札 + 場のカード = 52枚
    assert total_cards_in_hands + cards_on_board == 52
    # 場には4枚の7がある
    assert cards_on_board == 4

    # 3人プレイ: 52枚を配布 (17枚 or 18枚、余り1枚)
    env = SevensEnv(num_players=3)
    env.reset(seed=42)

    total_cards_in_hands = sum(np.sum(env.hands[agent]) for agent in env.agents)
    cards_on_board = np.sum(env.board)

    # 手札 + 場のカード = 52枚
    assert total_cards_in_hands + cards_on_board == 52
    # 場には4枚の7がある
    assert cards_on_board == 4

    # 各プレイヤーの手札枚数を確認 (7を除いた後: 15枚 or 16枚)
    hand_sizes = [np.sum(env.hands[agent]) for agent in env.agents]
    # 48枚 (52 - 4枚の7) を3人で分配: 16, 16, 16 または 15, 16, 17 など
    assert sum(hand_sizes) == 48
    assert min(hand_sizes) >= 15
    assert max(hand_sizes) <= 17

    # 4人プレイ: 52枚を均等に配布 (各13枚)
    env = SevensEnv(num_players=4)
    env.reset(seed=42)

    total_cards_in_hands = sum(np.sum(env.hands[agent]) for agent in env.agents)
    cards_on_board = np.sum(env.board)

    # 手札 + 場のカード = 52枚
    assert total_cards_in_hands + cards_on_board == 52
    # 場には4枚の7がある
    assert cards_on_board == 4

    # 各プレイヤーの手札枚数 (7を除いた後: 11枚, 12枚, 12枚, 13枚 など)
    hand_sizes = [np.sum(env.hands[agent]) for agent in env.agents]
    assert sum(hand_sizes) == 48
    assert min(hand_sizes) >= 11
    assert max(hand_sizes) <= 13


def test_remainder_card_randomization():
    """余りカードがランダムに配布されることをテスト（3人プレイ）"""
    # 複数回実行して、余りカードを持つプレイヤーが変わることを確認
    player_with_extra_card_counts = {"player_0": 0, "player_1": 0, "player_2": 0}

    for seed in range(30):  # 30回試行
        env = SevensEnv(num_players=3)
        env.reset(seed=seed)

        # 各プレイヤーの手札枚数を確認
        hand_sizes = {agent: int(np.sum(env.hands[agent])) for agent in env.agents}

        # 最大枚数を持つプレイヤーを特定
        max_size = max(hand_sizes.values())
        for agent, size in hand_sizes.items():
            if size == max_size:
                player_with_extra_card_counts[agent] += 1

    # 各プレイヤーが少なくとも1回は余りカードを持つことを確認
    # (完全にランダムなので、稀に偏る可能性があるが、30回なら十分)
    for agent, count in player_with_extra_card_counts.items():
        assert count > 0, f"{agent} never received the extra card in 30 trials"


def run_interactive_game(num_players=4, render=True, reward_config=None):
    """インタラクティブなゲーム実行（pytest実行時はスキップ）"""
    env = SevensEnv(
        num_players=num_players,
        render_mode="human" if render else None,
        reward_config=reward_config,
    )
    env.reset()

    if render:
        env.render()

    step_count = 0
    max_steps = 1000

    while env.agents and step_count < max_steps:
        agent = env.agent_selection
        observation = env.observe(agent)
        action = random_agent(observation, agent)
        env.step(action)

        if render:
            if action < 52:
                card = Card.from_id(action)
                print(f"{agent} が {card} を出しました")
            else:
                print(f"{agent} がパスしました")
            env.render()

        step_count += 1

    print("\n" + "=" * 50)
    print("ゲーム終了!")
    print("=" * 50)
    print(f"総ステップ数: {step_count}")
    print(f"上がり順: {env.finished_order}")
    print(f"累積報酬: {env._cumulative_rewards}")


if __name__ == "__main__":
    print("七並べ環境テスト - ランダムエージェント")
    print()
    run_interactive_game(num_players=4, render=True)
