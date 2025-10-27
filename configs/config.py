"""Sevens RL environment configuration and hyperparameters."""


# =============================================================================
# 報酬設定 (Reward Configuration)
# =============================================================================

# デフォルト報酬設定: プレイヤー数ごとの順位別報酬
DEFAULT_REWARDS = {
    2: {1: 1.0, 2: -1.0},  # 2人プレイ
    3: {1: 1.0, 2: 0.0, 3: -1.0},  # 3人プレイ
    4: {1: 1.0, 2: 0.3, 3: -0.3, 4: -1.0},  # 4人プレイ
}

# 実験用の代替報酬設定例
# Winner-takes-all: 1位のみ報酬
WINNER_TAKES_ALL_REWARDS = {
    2: {1: 1.0, 2: 0.0},
    3: {1: 1.0, 2: 0.0, 3: 0.0},
    4: {1: 1.0, 2: 0.0, 3: 0.0, 4: 0.0},
}

# 均等分配: 勝ち負けを明確に
BINARY_REWARDS = {
    2: {1: 1.0, 2: -1.0},
    3: {1: 1.0, 2: 0.0, 3: -1.0},
    4: {1: 1.0, 2: 0.0, 3: 0.0, 4: -1.0},
}

# スパース報酬: 極端な勝敗のみ
SPARSE_REWARDS = {
    2: {1: 10.0, 2: -10.0},
    3: {1: 10.0, 2: 0.0, 3: -10.0},
    4: {1: 10.0, 2: 0.0, 3: 0.0, 4: -10.0},
}


# =============================================================================
# 環境設定 (Environment Configuration)
# =============================================================================

class SevensConfig:
    """
    Configuration class for Sevens environment.

    Parameters
    ----------
    num_players : int, optional
        Number of players (2-4). Default is 4.
    reward_config : dict[int, float] or None, optional
        Reward configuration mapping rank to reward value.
        If None, uses default rewards. Default is None.
    render_mode : str or None, optional
        Rendering mode ('human' or None). Default is None.
    """

    def __init__(
        self,
        num_players: int = 4,
        reward_config: dict[int, float] = None,
        render_mode: str = None,
    ):
        self.num_players = num_players
        self.render_mode = render_mode

        # 報酬設定
        if reward_config is None:
            self.reward_config = DEFAULT_REWARDS[num_players]
        else:
            self.reward_config = reward_config

    def to_dict(self) -> dict:
        """
        Get configuration as dictionary.

        Returns
        -------
        dict
            Configuration dictionary.
        """
        return {
            'num_players': self.num_players,
            'reward_config': self.reward_config,
            'render_mode': self.render_mode,
        }


# =============================================================================
# プリセット設定 (Preset Configurations)
# =============================================================================

# 標準設定: 4人プレイ、デフォルト報酬
DEFAULT_CONFIG = SevensConfig(
    num_players=4,
    reward_config=None,  # DEFAULT_REWARDSを使用
)

# トレーニング用設定: 報酬なし描画
TRAINING_CONFIG = SevensConfig(
    num_players=4,
    reward_config=None,
    render_mode=None,
)

# 評価用設定: 描画あり
EVAL_CONFIG = SevensConfig(
    num_players=4,
    reward_config=None,
    render_mode='human',
)

# 実験用: Winner-takes-all
EXPERIMENT_WTA_CONFIG = SevensConfig(
    num_players=4,
    reward_config=WINNER_TAKES_ALL_REWARDS[4],
)

# 実験用: スパース報酬
EXPERIMENT_SPARSE_CONFIG = SevensConfig(
    num_players=4,
    reward_config=SPARSE_REWARDS[4],
)
