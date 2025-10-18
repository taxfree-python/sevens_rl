"""
Demo script for logging and configuration management

Demonstrates how to use:
- Hydra for configuration management
- Custom logger for structured logging
- Integration with Sevens RL environment
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import hydra
from omegaconf import DictConfig, OmegaConf

from src.sevens_env import SevensEnv
from src.utils import setup_logger


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(cfg: DictConfig) -> None:
    """
    Main function demonstrating logging and configuration.

    Args:
        cfg: Hydra configuration loaded from configs/default.yaml
    """
    # Set up logger with configuration
    logger = setup_logger(
        name="demo",
        level=cfg.logging.level,
        log_file=f"{cfg.logging.log_dir}/{cfg.logging.log_file}",
    )

    logger.info("="*70)
    logger.info("Sevens RL - Logging & Configuration Demo")
    logger.info("="*70)

    # Display configuration
    logger.info("Configuration:")
    logger.info("\n" + OmegaConf.to_yaml(cfg))

    # Create environment with config settings
    logger.info(f"Creating environment with {cfg.env.num_players} players...")
    env = SevensEnv(
        num_players=cfg.env.num_players,
        render_mode=cfg.env.render_mode,
        reward_config=cfg.env.reward_config,
    )

    # Run sample episodes
    num_episodes = 3
    logger.info(f"Running {num_episodes} sample episodes...")

    for episode in range(num_episodes):
        logger.info(f"\n--- Episode {episode + 1}/{num_episodes} ---")
        env.reset(seed=cfg.experiment.seed + episode)

        step_count = 0
        while env.agents and step_count < cfg.env.max_steps:
            agent = env.agent_selection
            observation = env.observe(agent)

            # Random action selection
            import numpy as np
            action_mask = observation['action_mask']
            valid_actions = np.where(action_mask == 1)[0]
            action = np.random.choice(valid_actions)

            env.step(action)
            step_count += 1

        logger.info(f"Episode finished in {step_count} steps")
        logger.info(f"Finish order: {env.finished_order}")
        logger.info(f"Cumulative rewards: {env._cumulative_rewards}")

    logger.info("\n" + "="*70)
    logger.info("Demo completed successfully!")
    logger.info("="*70)


if __name__ == "__main__":
    main()
