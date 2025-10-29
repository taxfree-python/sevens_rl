"""DQN Training Script with Hydra Configuration."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from src.rl.dqn_agent import DQNAgent
from src.sevens_env import SevensEnv
from src.utils.hydra_utils import (
    create_dqn_agent_from_config,
    get_env_params,
    get_experiment_params,
    get_training_params,
)
from src.utils.logger import setup_logger


def flatten_observation(observation: dict[str, np.ndarray]) -> np.ndarray:
    """Flatten observation dict into a single vector.

    Args:
        observation: Dict with 'board', 'hand', 'action_mask' keys

    Returns:
        Flattened numpy array of shape (157,)
    """
    board = observation["board"]
    hand = observation["hand"]
    action_mask = observation["action_mask"]
    return np.concatenate([board, hand, action_mask])


def train_episode(
    env: SevensEnv,
    agents: dict[str, DQNAgent],
    logger: logging.Logger,
    training_agent: str = "player_0",
) -> dict[str, Any]:
    """Train for one episode.

    Args:
        env: Sevens environment
        agents: Dictionary of agent_id -> DQNAgent
        logger: Logger instance
        training_agent: Which agent to train (default: player_0)

    Returns:
        Dictionary with episode statistics
    """
    env.reset()
    episode_rewards = {agent_id: 0.0 for agent_id in env.agents}
    episode_steps = 0
    training_losses = []
    training_q_values = []

    # Store previous observation for experience replay
    prev_observations = {}
    prev_actions = {}

    for agent_id in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        episode_rewards[agent_id] += reward
        done = termination or truncation

        if done:
            # Store final transition for training agent
            if agent_id == training_agent and agent_id in prev_observations:
                agents[training_agent].store_experience(
                    prev_observations[agent_id],
                    prev_actions[agent_id],
                    reward,
                    observation,
                    done=True,
                )
            env.step(None)
            continue

        # Store previous transition if exists
        if agent_id in prev_observations:
            agents[agent_id].store_experience(
                prev_observations[agent_id],
                prev_actions[agent_id],
                reward,
                observation,
                done=False,
            )

        # Select action
        action = agents[agent_id].select_action(observation, agent_id)

        # Store for next transition
        prev_observations[agent_id] = observation
        prev_actions[agent_id] = action

        # Take action
        env.step(action)
        episode_steps += 1

        # Train the training agent
        if agent_id == training_agent:
            train_result = agents[training_agent].train_step()
            if train_result and "loss" in train_result:
                training_losses.append(train_result["loss"])
                training_q_values.append(train_result["q_value_mean"])

    # End episode for all agents
    for agent in agents.values():
        agent.end_episode()

    # Calculate statistics
    stats = {
        "episode_steps": episode_steps,
        "episode_rewards": episode_rewards,
        "training_agent_reward": episode_rewards[training_agent],
        "mean_loss": np.mean(training_losses) if training_losses else 0.0,
        "mean_q_value": np.mean(training_q_values) if training_q_values else 0.0,
        "epsilon": agents[training_agent].policy.get_epsilon(),
    }

    return stats


def evaluate_episode(
    env: SevensEnv,
    agents: dict[str, DQNAgent],
    training_agent: str = "player_0",
) -> dict[str, Any]:
    """Evaluate agents for one episode (no training).

    Args:
        env: Sevens environment
        agents: Dictionary of agent_id -> DQNAgent
        training_agent: Which agent is being trained

    Returns:
        Dictionary with episode statistics
    """
    env.reset()
    episode_rewards = {agent_id: 0.0 for agent_id in env.agents}
    episode_steps = 0

    # Temporarily set epsilon to 0 for evaluation (pure exploitation)
    original_epsilons = {}
    for agent_id, agent in agents.items():
        original_epsilons[agent_id] = agent.policy.get_epsilon()
        agent.policy.set_epsilon(0.0)

    for agent_id in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        episode_rewards[agent_id] += reward
        done = termination or truncation

        if done:
            env.step(None)
            continue

        # Select action (no exploration)
        action = agents[agent_id].select_action(observation, agent_id)
        env.step(action)
        episode_steps += 1

    # Restore epsilon values
    for agent_id, agent in agents.items():
        agent.policy.set_epsilon(original_epsilons[agent_id])

    stats = {
        "episode_steps": episode_steps,
        "episode_rewards": episode_rewards,
        "training_agent_reward": episode_rewards[training_agent],
    }

    return stats


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    """Main training function with Hydra configuration.

    Args:
        cfg: Hydra configuration
    """
    # Setup logging
    logger = setup_logger(name="train_dqn", level=logging.INFO)
    logger.info("=" * 80)
    logger.info("Starting DQN Training")
    logger.info("=" * 80)
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # Extract parameters
    env_params = get_env_params(cfg)
    training_params = get_training_params(cfg)
    experiment_params = get_experiment_params(cfg)

    # Set seed
    if experiment_params.get("seed") is not None:
        torch.manual_seed(experiment_params["seed"])
        np.random.seed(experiment_params["seed"])
        logger.info(f"Random seed set to: {experiment_params['seed']}")

    # Create environment
    env = SevensEnv(
        num_players=env_params["num_players"],
        render_mode=env_params.get("render_mode"),
    )
    logger.info(f"Environment created: {env_params['num_players']} players")

    # Calculate state and action dimensions
    state_dim = 157  # 52 (board) + 52 (hand) + 53 (action_mask)
    action_dim = 53  # 52 cards + 1 pass

    # Create agents for all players
    agents = {}
    training_agent = "player_0"  # Train player_0 by default

    for agent_id in env.possible_agents:
        agent = create_dqn_agent_from_config(
            cfg,
            state_dim=state_dim,
            action_dim=action_dim,
        )
        agent.name = agent_id
        agents[agent_id] = agent
        logger.info(f"Created DQN agent: {agent_id}")

    logger.info(f"Training agent: {training_agent}")
    logger.info(f"Q-Network parameters: {agents[training_agent].q_network.get_num_parameters()}")

    # Setup output directories
    output_dir = Path.cwd()  # Hydra sets CWD to output directory
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True, parents=True)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Checkpoints directory: {checkpoints_dir}")

    # Training loop
    num_episodes = training_params["num_episodes"]
    eval_freq = training_params["eval_freq"]
    save_freq = training_params["save_freq"]
    log_freq = training_params["log_freq"]

    logger.info("=" * 80)
    logger.info("Starting Training Loop")
    logger.info("=" * 80)

    # Track statistics
    episode_rewards_history = []
    episode_wins = []  # Track if training agent won

    for episode in range(1, num_episodes + 1):
        # Train episode
        stats = train_episode(env, agents, logger, training_agent)

        # Track rewards and wins
        training_reward = stats["training_agent_reward"]
        episode_rewards_history.append(training_reward)

        # Determine winner (agent with highest cumulative reward)
        winner = max(stats["episode_rewards"].items(), key=lambda x: x[1])[0]
        won = winner == training_agent
        episode_wins.append(won)

        # Log progress
        if episode % log_freq == 0:
            recent_rewards = episode_rewards_history[-log_freq:]
            recent_wins = episode_wins[-log_freq:]
            avg_reward = np.mean(recent_rewards)
            win_rate = np.mean(recent_wins) * 100

            logger.info(
                f"Episode {episode}/{num_episodes} | "
                f"Steps: {stats['episode_steps']} | "
                f"Reward: {training_reward:.3f} | "
                f"Avg Reward (last {log_freq}): {avg_reward:.3f} | "
                f"Win Rate: {win_rate:.1f}% | "
                f"Loss: {stats['mean_loss']:.4f} | "
                f"Q-value: {stats['mean_q_value']:.3f} | "
                f"Epsilon: {stats['epsilon']:.4f}"
            )

        # Evaluation
        if episode % eval_freq == 0:
            logger.info("-" * 80)
            logger.info(f"Evaluating at episode {episode}...")

            eval_rewards = []
            eval_wins = []
            num_eval_episodes = 10

            for _ in range(num_eval_episodes):
                eval_stats = evaluate_episode(env, agents, training_agent)
                eval_rewards.append(eval_stats["training_agent_reward"])

                # Determine winner
                winner = max(eval_stats["episode_rewards"].items(), key=lambda x: x[1])[0]
                eval_wins.append(winner == training_agent)

            avg_eval_reward = np.mean(eval_rewards)
            eval_win_rate = np.mean(eval_wins) * 100

            logger.info(
                f"Evaluation Results | "
                f"Avg Reward: {avg_eval_reward:.3f} | "
                f"Win Rate: {eval_win_rate:.1f}% "
                f"({sum(eval_wins)}/{num_eval_episodes} wins)"
            )
            logger.info("-" * 80)

        # Save checkpoint
        if episode % save_freq == 0:
            checkpoint_path = checkpoints_dir / f"agent_{episode}.pt"
            agents[training_agent].save(str(checkpoint_path))
            logger.info(f"Saved checkpoint: {checkpoint_path}")

    # Final evaluation
    logger.info("=" * 80)
    logger.info("Training Complete - Running Final Evaluation")
    logger.info("=" * 80)

    final_eval_rewards = []
    final_eval_wins = []
    num_final_eval = 100

    for _ in range(num_final_eval):
        eval_stats = evaluate_episode(env, agents, training_agent)
        final_eval_rewards.append(eval_stats["training_agent_reward"])
        winner = max(eval_stats["episode_rewards"].items(), key=lambda x: x[1])[0]
        final_eval_wins.append(winner == training_agent)

    avg_final_reward = np.mean(final_eval_rewards)
    final_win_rate = np.mean(final_eval_wins) * 100

    logger.info(
        f"Final Evaluation ({num_final_eval} episodes) | "
        f"Avg Reward: {avg_final_reward:.3f} | "
        f"Win Rate: {final_win_rate:.1f}% "
        f"({sum(final_eval_wins)}/{num_final_eval} wins)"
    )

    # Save final model
    final_checkpoint_path = checkpoints_dir / "agent_final.pt"
    agents[training_agent].save(str(final_checkpoint_path))
    logger.info(f"Saved final model: {final_checkpoint_path}")

    logger.info("=" * 80)
    logger.info("Training Session Complete")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
