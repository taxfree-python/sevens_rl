"""
七並べ (Sevens) environment using PettingZoo AEC API.

ゲームルール:
- 52枚のカード (4スート × 13ランク、ジョーカーなし)
- 各プレイヤーに均等配布
- 最初に7を出す
- その後、場に出ているカードに隣接するカードのみ出せる
- 出せるカードがない、または戦略的に出さない場合はパス
- 手札を最初になくしたプレイヤーの勝ち
"""

import numpy as np
from gymnasium import spaces
from pettingzoo import AECEnv
from pettingzoo.utils.agent_selector import AgentSelector

from configs.config import DEFAULT_REWARDS

# カードの定義
SUITS = ["♠", "♥", "♦", "♣"]  # スペード、ハート、ダイヤ、クラブ
RANKS = list(range(1, 14))  # 1(A) から 13(K) まで
NUM_CARDS = 52
SEVEN_RANK = 7


class Card:
    """カードクラス"""

    def __init__(self, suit: int, rank: int):
        self.suit = suit  # 0-3
        self.rank = rank  # 1-13

    def __repr__(self):
        return f"{SUITS[self.suit]}{self.rank}"

    def __eq__(self, other):
        return self.suit == other.suit and self.rank == other.rank

    def __hash__(self):
        return hash((self.suit, self.rank))

    def to_id(self) -> int:
        """カードをユニークなID (0-51) に変換"""
        return self.suit * 13 + (self.rank - 1)

    @staticmethod
    def from_id(card_id: int) -> "Card":
        """IDからカードを生成"""
        suit = card_id // 13
        rank = (card_id % 13) + 1
        return Card(suit, rank)


class SevensEnv(AECEnv):
    """七並べ環境 (PettingZoo AEC API)"""

    metadata = {
        "render_modes": ["human"],
        "name": "sevens_v0",
        "is_parallelizable": False,
    }

    def __init__(
        self,
        num_players: int = 4,
        render_mode: str | None = None,
        reward_config: dict[int, float] | None = None,
    ):
        super().__init__()

        assert 2 <= num_players <= 4, "プレイヤー数は2-4人"

        self.num_players = num_players
        self.render_mode = render_mode

        # 報酬設定（カスタム設定がなければデフォルトを使用）
        if reward_config is None:
            self.reward_config = DEFAULT_REWARDS[num_players]
        else:
            self.reward_config = reward_config

        # エージェント定義
        self.possible_agents = [f"player_{i}" for i in range(num_players)]

        # 観測空間: [場の状態(52), 自分の手札(52), 有効なアクション(53)]
        # 場の状態: 各カードが場に出ているか (0 or 1)
        # 手札: 各カードを持っているか (0 or 1)
        # 有効なアクション: 各アクション(52カード+1パス)が有効か (0 or 1)
        self.observation_spaces = {
            agent: spaces.Dict(
                {
                    "board": spaces.MultiBinary(NUM_CARDS),
                    "hand": spaces.MultiBinary(NUM_CARDS),
                    "action_mask": spaces.MultiBinary(NUM_CARDS + 1),
                }
            )
            for agent in self.possible_agents
        }

        # 行動空間: 52枚のカード + パス (ID=52)
        self.action_spaces = {
            agent: spaces.Discrete(NUM_CARDS + 1) for agent in self.possible_agents
        }

        self.reset()

    def reset(self, seed: int | None = None, options: dict | None = None):
        """環境をリセット"""
        if seed is not None:
            np.random.seed(seed)

        # 状態初期化
        self.agents = self.possible_agents[:]
        self.rewards = dict.fromkeys(self.agents, 0)
        self._cumulative_rewards = dict.fromkeys(self.agents, 0)
        self.terminations = dict.fromkeys(self.agents, False)
        self.truncations = dict.fromkeys(self.agents, False)
        self.infos = {agent: {} for agent in self.agents}

        # ゲーム状態
        self.board = np.zeros(NUM_CARDS, dtype=np.int8)  # 場に出ているカード
        self.hands = {
            agent: np.zeros(NUM_CARDS, dtype=np.int8) for agent in self.agents
        }
        self.finished_order = []  # 上がった順番

        # カード配布
        self._deal_cards()

        # 初期化フェーズ: 全ての7を場に出す
        starting_player = self._place_all_sevens()
        self.starting_player = starting_player

        # エージェント選択の初期化
        # ダイヤの7を出したプレイヤーが先攻
        self._agent_selector = AgentSelector(self.agents)
        # starting_playerまで進める
        while self._agent_selector.next() != starting_player:
            pass
        self.agent_selection = starting_player

        return self.observe(self.agent_selection), self.infos[self.agent_selection]

    def _deal_cards(self):
        """カードをシャッフルして配る"""
        deck = np.arange(NUM_CARDS)
        np.random.shuffle(deck)

        # ラウンドロビン方式で1枚ずつ配る（余りカードを公平に分配）
        for i, card_id in enumerate(deck):
            agent_idx = i % self.num_players
            agent = self.agents[agent_idx]
            self.hands[agent][card_id] = 1

    def _place_all_sevens(self) -> str:
        """
        初期化フェーズ: 全プレイヤーが持っている7を場に出す

        Returns:
            str: ダイヤの7を持っていたプレイヤー（先攻プレイヤー）
        """
        starting_player = None
        diamond_seven_id = Card(2, SEVEN_RANK).to_id()  # ダイヤの7 (suit=2)

        # 全プレイヤーの手札から7を探して場に出す
        for agent in self.agents:
            for suit in range(4):  # 4スート
                seven_id = Card(suit, SEVEN_RANK).to_id()
                if self.hands[agent][seven_id] == 1:
                    # 手札から削除して場に出す
                    self.hands[agent][seven_id] = 0
                    self.board[seven_id] = 1

                    # ダイヤの7を持っていたプレイヤーを記録
                    if seven_id == diamond_seven_id:
                        starting_player = agent

        # ダイヤの7を持っているプレイヤーが必ずいるはず
        assert starting_player is not None, "ダイヤの7が見つかりません"

        return starting_player

    def observe(self, agent: str) -> dict:
        """エージェントの観測を取得"""
        observation = {
            "board": self.board.copy(),
            "hand": self.hands[agent].copy(),
            "action_mask": self._get_action_mask(agent),
        }
        return observation

    def _get_action_mask(self, agent: str) -> np.ndarray:
        """有効なアクションのマスクを取得"""
        mask = np.zeros(NUM_CARDS + 1, dtype=np.int8)

        # パスは常に可能
        mask[NUM_CARDS] = 1

        # 手札の各カードについて、出せるかチェック
        for card_id in range(NUM_CARDS):
            if self.hands[agent][card_id] == 1 and self._is_valid_play(card_id):
                mask[card_id] = 1

        return mask

    def _is_valid_play(self, card_id: int) -> bool:
        """カードが出せるかチェック"""
        card = Card.from_id(card_id)

        # 7は常に出せる (場に出ていなければ)
        if card.rank == SEVEN_RANK:
            return self.board[card_id] == 0

        # 7より大きい数字: 1つ小さい数字が場にあるか
        if card.rank > SEVEN_RANK:
            prev_card = Card(card.suit, card.rank - 1)
            return self.board[prev_card.to_id()] == 1

        # 7より小さい数字: 1つ大きい数字が場にあるか
        if card.rank < SEVEN_RANK:
            next_card = Card(card.suit, card.rank + 1)
            return self.board[next_card.to_id()] == 1

        return False

    def _get_reward_for_rank(self, rank: int) -> float:
        """順位に応じた報酬を計算"""
        return self.reward_config.get(rank, 0.0)

    def step(self, action: int):
        """アクションを実行"""
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            return self._was_dead_step(action)

        agent = self.agent_selection

        # アクション検証
        action_mask = self._get_action_mask(agent)
        if action_mask[action] == 0:
            raise ValueError(f"Invalid action {action} for {agent}")

        # アクション実行
        if action < NUM_CARDS:  # カードを出す
            self.hands[agent][action] = 0
            self.board[action] = 1

            # 手札がなくなったら上がり
            if np.sum(self.hands[agent]) == 0:
                self.finished_order.append(agent)
                self.terminations[agent] = True

                # 順位に応じた報酬を設定
                rank = len(self.finished_order)
                self.rewards[agent] = self._get_reward_for_rank(rank)

                # 全員終了したらゲーム終了
                if len(self.finished_order) == self.num_players - 1:
                    # 残り1人を自動的に最下位にする
                    for a in self.agents:
                        if a not in self.finished_order:
                            self.finished_order.append(a)
                            self.terminations[a] = True
                            self.rewards[a] = self._get_reward_for_rank(
                                self.num_players
                            )
        else:  # パス
            pass

        # 累積報酬更新
        self._accumulate_rewards()

        if len(self.finished_order) == self.num_players:
            self.agents = []
            return

        # 次のエージェントを選択 (終了していないエージェント)
        self.agent_selection = self._agent_selector.next()

        # 全員終了していないか、次の有効なエージェントを探す
        max_iter = self.num_players
        iter_count = 0
        while self.terminations[self.agent_selection] and iter_count < max_iter:
            self.agent_selection = self._agent_selector.next()
            iter_count += 1

    def _accumulate_rewards(self):
        """報酬を累積"""
        for agent in self.agents:
            self._cumulative_rewards[agent] += self.rewards[agent]
            self.rewards[agent] = 0

    def _was_dead_step(self, action):
        """終了したエージェントがアクションを取った場合"""
        # 終了済みエージェントの報酬はリセット
        self.rewards[self.agent_selection] = 0
        self._accumulate_rewards()
        self.agent_selection = self._agent_selector.next()
        return None

    def render(self):
        """環境を描画"""
        if self.render_mode != "human":
            return

        print("\n" + "=" * 50)
        print("七並べ (Sevens)")
        print("=" * 50)

        # 場の状態を表示
        print("\n場のカード:")
        for suit_idx, suit in enumerate(SUITS):
            cards_on_board = []
            for rank in RANKS:
                card_id = Card(suit_idx, rank).to_id()
                if self.board[card_id] == 1:
                    cards_on_board.append(str(rank))
                else:
                    cards_on_board.append("--")
            print(f"{suit}: {' '.join(cards_on_board)}")

        # 各プレイヤーの手札枚数
        print("\nプレイヤーの手札:")
        for agent in self.agents:
            num_cards = int(np.sum(self.hands[agent]))
            status = "上がり" if self.terminations[agent] else f"{num_cards}枚"
            marker = " <--" if agent == self.agent_selection else ""
            print(f"{agent}: {status}{marker}")

        print()
