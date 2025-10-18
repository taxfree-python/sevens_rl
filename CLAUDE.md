# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Reinforcement learning project for the Japanese card game Sevens (七並べ). Uses PyTorch for RL models and PettingZoo/Gymnasium for multi-agent environments. Currently implements a working PettingZoo AEC environment with 52 cards (13 ranks × 4 suits, no jokers).

## Development Environment

VSCode Dev Container with Python 3.11, CPU-only PyTorch, Node.js 22, and uv package manager.

**Installed packages**: torch (CPU), gymnasium, pettingzoo, numpy, pandas, torchvision, torchaudio, pytest, pytest-cov

**Tools**: black (formatting), flake8 and mypy (linting), pytest (testing), basic type checking enabled

**Project structure**:
```
sevens_rl/
├── src/              # Source code
│   └── sevens_env.py # PettingZoo environment implementation
├── tests/            # Test files
│   ├── test_env.py
│   ├── test_custom_rewards.py
│   └── test_configs.py
├── configs/          # Configuration files
│   └── config.py     # Reward configurations
└── notebooks/        # Jupyter notebooks (for experiments)
```

## Development Commands

### Using Docker (recommended)

```bash
# Build and run with docker-compose (for non-VSCode users)
docker-compose up -d
docker-compose exec sevens-rl bash

# Or build directly
docker build -t sevens-rl:latest .
docker run -it -v $(pwd):/workspace -w /workspace sevens-rl:latest bash

# Inside container
pip install --no-cache-dir -r requirements.txt  # if needed
pytest tests/ -v
```

### Using VSCode Dev Container

1. Open folder in VSCode
2. Click "Reopen in Container" when prompted
3. Container auto-installs dependencies via `postCreateCommand`

### Local development (without Docker)

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install --no-cache-dir -r requirements.txt

# Run tests with pytest
pytest tests/ -v                    # Run all tests with verbose output
pytest tests/test_env.py -v        # Run specific test file
pytest tests/ -v --cov=src         # Run with coverage report

# Run tests as standalone scripts (for interactive debugging)
python -m tests.test_env
python -m tests.test_custom_rewards
python -m tests.test_configs
```

## Implemented Architecture

### Sevens Game Environment (`src/sevens_env.py`)

**Core classes**:
- `Card`: Represents individual cards with suit (0-3) and rank (1-13)
  - `to_id()` / `from_id()`: Convert cards to/from unique IDs (0-51)
  - ID mapping: `card_id = suit * 13 + (rank - 1)`

- `SevensEnv(AECEnv)`: PettingZoo AEC API implementation
  - Supports 2-4 players (default: 4)
  - Turn-based gameplay with automatic agent selection
  - Terminates when all but one player finish

**Observation space** (Dict):
- `board`: MultiBinary(52) - cards on board
- `hand`: MultiBinary(52) - current player's hand
- `action_mask`: MultiBinary(53) - valid actions (52 cards + pass)

**Action space**: Discrete(53)
- 0-51: Play card with that ID
- 52: Pass

**Game rules implemented**:
- Sevens (rank 7) can always be played if not on board
- Cards adjacent to existing board cards are valid plays
- Higher ranks require rank-1 to be on board, lower ranks require rank+1
- Players finish when hand is empty
- Rewards: +1.0 for 1st place, -1.0 for last place

### Configuration (`configs/config.py`)

Provides predefined reward configurations:
- `DEFAULT_REWARDS`: {1: 1.0, 2: 0.3, 3: -0.3, 4: -1.0}
- `WINNER_TAKES_ALL_REWARDS`: Only 1st place gets reward
- `SPARSE_REWARDS`: High reward/penalty for 1st/last place
- `SevensConfig`: Dataclass for environment configuration

### Test Scripts (`tests/`)

All tests use pytest framework. Run with `pytest tests/ -v`.

**test_env.py**: Environment functionality tests
- `test_basic_game()`: Verifies game completes successfully
- `test_game_with_different_player_counts()`: Tests 2-4 player games
- `test_observation_space()`: Validates observation structure
- `test_action_mask_validity()`: Checks action masking works correctly
- `test_custom_reward_config()`: Tests custom reward configurations
- `run_interactive_game()`: Manual testing with visualization (run script directly)

**test_custom_rewards.py**: Reward configuration tests
- `test_custom_winner_takes_all_rewards()`: Winner-takes-all reward scheme
- `test_default_rewards()`: Default reward configuration
- `test_sparse_rewards()`: Sparse reward scheme (1st/last only)
- `test_negative_rewards()`: All-negative rewards

**test_configs.py**: Configuration presets tests
- `test_default_config()`: DEFAULT_CONFIG validation
- `test_training_config()`: TRAINING_CONFIG validation
- `test_winner_takes_all_config()`: EXPERIMENT_WTA_CONFIG validation
- `test_sparse_rewards_config()`: EXPERIMENT_SPARSE_CONFIG validation
- `test_custom_config_three_players()`: Custom 3-player game
- `test_config_dataclass_defaults()`: SevensConfig defaults

## Coding Style

- **Bilingual approach**: Japanese docstrings for game logic, English for RL/technical code
- **Type hints**: Use for function signatures (matches PettingZoo API patterns)
- **Naming**: snake_case, descriptive identifiers (e.g., `_get_action_mask`, `_is_valid_play`)
- **Testing**: pytest framework with deterministic seeds (`env.reset(seed=42)`) for reproducible tests

## Planned Extensions

- Baseline agents: rule-based players
- RL agents: DQN or Actor-Critic with PyTorch
- Training infrastructure: experience replay, ε-greedy, self-play

## Notes

- Project planning docs in Japanese (ToDo.md, memo.md)
- Dev container auto-installs dependencies on creation
- Self-play training is the primary learning approach

# Answer rule
日本語で回答してください