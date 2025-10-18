# Repository Guidelines

## Project Structure & Module Organization
- `sevens_env.py`: Core PettingZoo AEC environment for 七並べ, encapsulating card definitions, environment state, and step logic.
- `test_env.py`: Random-agent simulation used for smoke-testing gameplay and verifying environment resets, masks, and victory handling.
- `requirements.txt`: Minimal runtime stack; install into a virtual environment before running examples or tests.
- `CLAUDE.md`, `ToDo.md`, `memo.md`: Context and planning notes—review before large refactors to stay aligned with documented goals.

## Build, Test, and Development Commands
- `python -m venv .venv && source .venv/bin/activate`: Create and activate a local virtual environment; required for isolating dependencies.
- `pip install -r requirements.txt`: Install PettingZoo, Gymnasium, and data-science utilities needed by the environment and tests.
- `python test_env.py`: Run the bundled random-agent loop; prints per-move traces (when `render=True`) and summarizes winners and rewards.
- Example usage inside a REPL:
  ```python
  from sevens_env import SevensEnv
  env = SevensEnv(num_players=4)
  obs, _ = env.reset(seed=0)
  ```

## Coding Style & Naming Conventions
- Follow PEP 8 with 4-space indentation, descriptive snake_case identifiers, and typed function signatures mirroring the existing modules.
- Keep docstrings bilingual where practical; prefer concise Japanese summaries paired with technical English for clarity.
- Organize helpers near their call sites; favor pure functions for reward/state transforms and keep environment state on the `SevensEnv` instance.

## Testing Guidelines
- Extend `test_env.py` with targeted scenarios (e.g., forced passes, single-player end game) before adding new mechanics.
- Name new tests `test_<behavior>()` and gate stochastic checks behind deterministic seeds (`env.reset(seed=...)`).
- For formal suites, introduce `pytest` gradually but keep the random-agent smoke test runnable via `python test_env.py`.

## Commit & Pull Request Guidelines
- Use present-tense, imperative commit subjects (`Add pass counter`, `Fix action mask bug`) and keep bodies focused on rationale.
- Reference issues or TODO IDs when available (`Refs ToDo#3`).
- PRs should summarize behavioral changes, outline new tests, and include screenshots or trace snippets if gameplay output changes.

# Answer rule
日本語で回答してください