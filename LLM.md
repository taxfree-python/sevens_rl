# Sevens RL Project

Reinforcement learning environment for the Japanese card game **Sevens** (七並べ). Built with PettingZoo/Gymnasium for multi-agent environments and PyTorch for RL agents. Supports 2-4 player games with 52 cards (no jokers) and enables self-play training.

## Project Structure

```
sevens_rl/
├── src/
│   ├── sevens_env.py       # PettingZoo AEC environment implementation
│   ├── baseline_agents.py  # Rule-based baseline agents
│   └── utils/
│       └── logger.py       # Logging configuration
├── tests/
│   ├── test_env.py
│   ├── test_custom_rewards.py
│   └── test_configs.py
├── configs/
│   ├── config.py           # Reward configurations
│   ├── default.yaml        # Default hyperparameters
│   └── train_dqn.yaml      # DQN training configuration
├── examples/               # Example scripts and demos
└── notebooks/              # Jupyter notebooks for experiments
```

## Coding Standards

### Docstrings and Comments
- **All docstrings and comments must be in English**
- Use **NumPy style docstrings** for all functions, classes, and modules
- Example:
  ```python
  def calculate_reward(placement: int, num_players: int) -> float:
      """
      Calculate reward based on player placement.

      Parameters
      ----------
      placement : int
          Player's finishing position (1-indexed)
      num_players : int
          Total number of players in the game

      Returns
      -------
      float
          Reward value for the placement
      """
      ...
  ```

### Type Hints
- Use modern Python 3.10+ type hints
- Prefer `dict[str, int]` over `Dict[str, int]`
- Prefer `str | None` over `Optional[str]`
- Type all function signatures

### Naming Conventions
- **Functions and variables**: `snake_case`
- **Classes**: `PascalCase`
- **Private methods**: `_leading_underscore`
- Use descriptive names (e.g., `_get_action_mask`, `_is_valid_play`)

### Code Formatting
- Use **Ruff** for linting and formatting
- Run `ruff check .` to check for issues
- Run `ruff format .` to auto-format code
- 4-space indentation, follow PEP 8

## Development Workflow

### Testing
```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_env.py -v

# Run with coverage
pytest tests/ -v --cov=src

# Run tests as standalone scripts (for debugging)
python -m tests.test_env
```

**Testing guidelines**:
- Use deterministic seeds (`env.reset(seed=42)`) for reproducible tests
- Name test functions as `test_<behavior>()`
- Test edge cases (forced passes, single-player endgame, etc.)

### Linting and Formatting
```bash
ruff check .              # Check for linting issues
ruff check --fix .        # Auto-fix issues
ruff format .             # Format code
```

## Working with AI Agents

**Prefer Edit tools over Python heredoc commands**

AI agents should use dedicated file editing tools (Read/Edit/Write) instead of Python heredoc scripts for file modifications. If an agent suggests commands like:

```bash
python - <<'PY'
  from pathlib import Path
  path = Path('tests/test_agents.py')
  text = path.read_text()
  # ... file modification logic
PY
```

Ask the agent to use proper Edit tools instead. These are safer, more trackable, and easier to review.

## Game Architecture

### SevensEnv (PettingZoo AEC)
- **Observation space**: Dict with `board`, `hand`, `action_mask`
- **Action space**: Discrete(53) - 52 cards + pass action
- **Supports**: 2-4 players (default: 4)
- **Card mapping**: `card_id = suit * 13 + (rank - 1)` for IDs 0-51

### Game Rules
- Sevens (rank 7) can always be played if not on board
- Other cards require adjacent cards to be played first
- Players pass when no valid moves available
- Game ends when all but one player finish

### Reward Configuration
- Configurable via `configs/config.py`
- Presets: `DEFAULT_REWARDS`, `WINNER_TAKES_ALL_REWARDS`, `SPARSE_REWARDS`
- Use `SevensConfig` dataclass for environment configuration

## Commit Guidelines

- Use present-tense, imperative commit subjects (e.g., "Add pass counter", "Fix action mask bug")
- Keep commit messages concise and focused
- Reference issues when available (e.g., "Refs #42")

## Additional Resources

- See `CLAUDE.md` for detailed environment setup, Docker configuration, and implementation details
- See `ToDo.md` and `memo.md` for project planning notes (in Japanese)
