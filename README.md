# Sevens RL

七並べ (Sevens) を対象としたマルチエージェント強化学習プロジェクトです。PettingZoo の AEC API を用いて環境を実装し、PyTorch ベースのエージェント実装・評価に向けた基盤を整えています。

## 主な特徴 / Features
- PettingZoo 互換の `SevensEnv` を `src/sevens_env.py` に実装済み
- 観測空間 (board / hand / action mask) と 53 次元の離散行動空間をサポート
- 順位に基づく柔軟な報酬テーブルを `configs/config.py` で管理
- ベースラインのランダム／ルールベースエージェントを `src/agents/` に提供
- pytest + ruff によるテスト・Lint ワークフローを整備
- Docker / VSCode Dev Container による再現性の高い開発環境

## クイックスタート
### 1. Dev Container / Docker
1. VSCode で本リポジトリを開き、`Reopen in Container` を選択
2. コンテナ内で依存パッケージが自動インストールされます (`postCreateCommand`)
3. そのまま Python REPL や pytest を実行可能です

Docker Compose を直接利用する場合:
```bash
docker-compose up -d
docker-compose exec sevens-rl bash
```

### 2. ローカル (Docker なし)
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 使用例
環境の初期化とランダムプレイの例:
```python
from sevens_env import SevensEnv

env = SevensEnv(num_players=4)
obs, info = env.reset(seed=0)
agent = env.agent_selection

while env.agents:
    mask = obs["action_mask"]
    action = mask.argmax()  # ここでは最初に可能な行動を選択
    obs, reward, termination, truncation, info = env.step(action)
```
インタラクティブなスモークテストは `python tests/test_env.py`、より詳細な報酬テストは `python -m tests.test_custom_rewards` を参照してください。

## ベースラインエージェント / Baseline Agents
`src/agents/` には環境確認やベースライン評価に使えるシンプルな方策が含まれています。

```python
from src.agents import RandomAgent, NearestSevensAgent
from src.sevens_env import SevensEnv

env = SevensEnv(num_players=4)
obs, _ = env.reset(seed=1)
agent_id = env.agent_selection

policy = RandomAgent()  # または NearestSevensAgent()
while env.agents:
    action = policy.select_action(obs, agent_id)
    obs, reward, terminated, truncated, info = env.step(action)
    agent_id = env.agent_selection

print(env.finished_order)
```

`NearestSevensAgent` は 7 に近いカードを優先して出す単純な規則で行動し、`RandomAgent` はマスクされた合法手から一様サンプリングします。

ターミナル上で自己対局を試す場合は次のコマンドを利用してください。

```bash
python -m examples.agents_self_play
```

## テストとLint
```bash
pytest tests/ -v
ruff check .
```

## プロジェクト構成
```
├── configs/          # Hydra 設定・報酬テーブル
├── examples/         # ロギング設定デモ
├── src/              # 環境実装とユーティリティ
│   └── utils/        # ロガー・設定バリデータ
├── tests/            # pytest ベースのユニットテスト
├── Dockerfile        # CPU 版 PyTorch 付き開発環境
├── docker-compose.yml
```

## 今後のロードマップ
- DQN など PyTorch 強化学習エージェントと学習ループの構築
- ベースラインエージェントと学習エージェントの対戦評価スクリプト
- 学習曲線・勝率可視化やハイパーパラメータ探索の自動化

開発タスクの詳細管理は GitHub の Issue / PR で運用しています。
