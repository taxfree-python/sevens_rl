"""DQN Training Script with Hydra Configuration - Simple Self-Play Version."""

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
from src.utils.env_utils import calculate_action_dim, calculate_state_dim
from src.utils.hydra_utils import (
    create_dqn_agent_from_config,
    get_env_params,
    get_experiment_params,
    get_training_params,
)
from src.utils.logger import setup_logger


def _normalize_training_agents(
    training_agents: list[str] | str | None,
    possible_agents: list[str],
) -> list[str]:
    """Normalize training_agents parameter to a list of agent IDs.

    Parameters
    ----------
    training_agents : list[str] | str | None
        Collection of agent IDs to train, or None for all agents.
        Can be a list of agent IDs, a single agent ID string, or None.
    possible_agents : list[str]
        List of all possible agent IDs in the environment

    Returns
    -------
    list[str]
        List of agent IDs to train
    """
    if training_agents is None:
        return list(possible_agents)
    elif isinstance(training_agents, str):
        return [training_agents]
    else:
        return list(training_agents)


def train_episode(
    env: SevensEnv,
    agents: dict[str, DQNAgent],
    logger: logging.Logger,
    training_agents: list[str] | str | None = None,
) -> dict[str, Any]:
    """Train for one episode.

    Parameters
    ----------
    env : SevensEnv
        Sevens environment
    agents : dict[str, DQNAgent]
        Dictionary of agent_id -> DQNAgent
    logger : logging.Logger
        Logger instance
    training_agents : list[str] | str | None, optional
        Collection of agent IDs to train. If None, all agents
        are trained. Can be a list of agent IDs or a single agent ID string.

    Returns
    -------
    dict[str, Any]
        Dictionary with episode statistics
    """
    training_agents = _normalize_training_agents(training_agents, env.possible_agents)

    env.reset()
    episode_rewards = dict.fromkeys(env.agents, 0.0)
    episode_steps = 0
    training_losses = []
    training_q_values = []

    # Store previous observation for experience replay
    prev_observations = {}
    prev_actions = {}

    # Create network representatives to avoid duplicate training on shared networks
    # For shared networks, only the first agent in training_agents will call train_step
    network_representatives = {}
    for agent_id in training_agents:
        agent = agents[agent_id]
        network_id = id(agent)
        if network_id not in network_representatives:
            network_representatives[network_id] = agent_id

    for agent_id in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        episode_rewards[agent_id] += reward
        done = termination or truncation

        if done:
            # Store final transition for training agents
            if agent_id in training_agents and agent_id in prev_observations:
                agent = agents[agent_id]
                agent.store_experience(
                    prev_observations[agent_id],
                    prev_actions[agent_id],
                    reward,
                    observation,
                    done=True,
                )
            env.step(None)
            continue

        # Store previous transition if exists
        if agent_id in training_agents and agent_id in prev_observations:
            agent = agents[agent_id]
            agent.store_experience(
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

        # Train the training agents (only once per unique network)
        if agent_id in training_agents:
            agent = agents[agent_id]
            network_id = id(agent)

            # Only train if this agent is the representative for its network
            # This prevents duplicate training when multiple agents share the same network
            if network_representatives.get(network_id) == agent_id:
                train_result = agent.train_step()
                if train_result and "loss" in train_result:
                    training_losses.append(train_result["loss"])
                    training_q_values.append(train_result["q_value_mean"])

    # End episode for DQN agents
    for agent in agents.values():
        agent.end_episode()

    # Calculate statistics for training agents
    training_agents_rewards = {
        agent_id: episode_rewards[agent_id] for agent_id in training_agents
    }

    # Get epsilon from first DQN training agent
    epsilon = None
    for agent_id in training_agents:
        agent = agents[agent_id]
        epsilon = agent.policy.get_epsilon()
        break

    # Determine winner from finished_order (first to finish wins)
    winner = env.finished_order[0] if env.finished_order else None

    stats: dict[str, Any] = {
        "episode_steps": episode_steps,
        "episode_rewards": episode_rewards,
        "training_agents_rewards": training_agents_rewards,
        "mean_loss": np.mean(training_losses) if training_losses else 0.0,
        "mean_q_value": np.mean(training_q_values) if training_q_values else 0.0,
        "epsilon": epsilon,
        "training_agents": training_agents,
        "winner": winner,
        "finished_order": env.finished_order,
    }

    if len(training_agents) == 1:
        single_agent = training_agents[0]
        stats["training_agent_reward"] = training_agents_rewards[single_agent]
        stats["training_agent"] = single_agent

    return stats


def evaluate_episode(
    env: SevensEnv,
    agents: dict[str, DQNAgent],
    training_agents: list[str] | str | None = None,
) -> dict[str, Any]:
    """Evaluate agents for one episode (no training).

    Parameters
    ----------
    env : SevensEnv
        Sevens environment
    agents : dict[str, DQNAgent]
        Dictionary of agent_id -> DQNAgent
    training_agents : list[str] | str | None, optional
        Collection of agents being tracked.
        Can be a list of agent IDs or a single agent ID string.

    Returns
    -------
    dict[str, Any]
        Dictionary with episode statistics
    """
    training_agents = _normalize_training_agents(training_agents, env.possible_agents)

    env.reset()
    episode_rewards = dict.fromkeys(env.agents, 0.0)
    episode_steps = 0

    # Temporarily set epsilon to 0 for DQN agents evaluation (pure exploitation)
    # Use id() to avoid setting epsilon multiple times for shared networks
    original_epsilons = {}
    processed_agents = set()
    for _agent_id, agent in agents.items():
        if id(agent) not in processed_agents:
            original_epsilons[id(agent)] = agent.policy.get_epsilon()
            agent.policy.set_epsilon(0.0)
            processed_agents.add(id(agent))

    for agent_id in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        episode_rewards[agent_id] += reward
        done = termination or truncation

        if done:
            env.step(None)
            continue

        # Select action (no exploration for DQN agents)
        action = agents[agent_id].select_action(observation, agent_id)
        env.step(action)
        episode_steps += 1

    # Restore epsilon values for DQN agents
    for _agent_id, agent in agents.items():
        agent_id_key = id(agent)
        if agent_id_key in original_epsilons:
            agent.policy.set_epsilon(original_epsilons[agent_id_key])

    # Calculate statistics for training agents
    training_agents_rewards = {
        agent_id: episode_rewards[agent_id] for agent_id in training_agents
    }

    # Determine winner
    winner = env.finished_order[0] if env.finished_order else None

    stats: dict[str, Any] = {
        "episode_steps": episode_steps,
        "episode_rewards": episode_rewards,
        "training_agents_rewards": training_agents_rewards,
        "training_agents": training_agents,
        "winner": winner,
        "finished_order": env.finished_order,
    }

    if len(training_agents) == 1:
        single_agent = training_agents[0]
        stats["training_agent_reward"] = training_agents_rewards[single_agent]
        stats["training_agent"] = single_agent

    return stats


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    """Main training function for simple self-play training.

    This is a simplified version that creates DQN agents directly in main().
    All players use DQN agents (self-play mode only).

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration
    """
    # Setup logging
    logger = setup_logger(name="train_dqn", level=logging.INFO)
    logger.info("=" * 80)
    logger.info("Starting DQN Self-Play Training")
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
    num_players = env_params["num_players"]
    state_dim = calculate_state_dim(num_players)
    action_dim = calculate_action_dim()

    logger.info(f"State dimension: {state_dim} (for {num_players} players)")
    logger.info(f"Action dimension: {action_dim}")

    # Create DQN agents for self-play
    # All agents share the same network for efficient self-play
    logger.info("Setting up self-play agents with shared network")
    shared_agent = create_dqn_agent_from_config(
        cfg, state_dim=state_dim, action_dim=action_dim
    )

    agents = {}
    for agent_id in env.possible_agents:
        agents[agent_id] = shared_agent
        logger.info(f"Assigned shared DQN to {agent_id}")

    # All agents are training agents in self-play
    training_agents = list(env.possible_agents)

    logger.info(
        f"Q-Network parameters: {shared_agent.q_network.get_num_parameters()}"
    )

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
    episode_rewards_history = {agent_id: [] for agent_id in training_agents}
    episode_wins = {agent_id: [] for agent_id in training_agents}

    for episode in range(1, num_episodes + 1):
        # Train episode
        stats = train_episode(env, agents, logger, training_agents)

        # Track rewards and wins for each training agent
        for agent_id in training_agents:
            training_reward = stats["training_agents_rewards"][agent_id]
            episode_rewards_history[agent_id].append(training_reward)

            # Determine if this agent won (first in finished_order)
            winner = stats["winner"]
            won = winner == agent_id if winner else False
            episode_wins[agent_id].append(won)

        # Log progress
        if episode % log_freq == 0:
            # Aggregate statistics across training agents
            all_recent_rewards = []
            all_recent_wins = []
            for agent_id in training_agents:
                recent_rewards = episode_rewards_history[agent_id][-log_freq:]
                recent_wins = episode_wins[agent_id][-log_freq:]
                all_recent_rewards.extend(recent_rewards)
                all_recent_wins.extend(recent_wins)

            avg_reward = np.mean(all_recent_rewards)
            win_rate = np.mean(all_recent_wins) * 100

            logger.info(
                f"Episode {episode}/{num_episodes} | "
                f"Steps: {stats['episode_steps']} | "
                f"Reward: {avg_reward:.3f} | "
                f"Win Rate: {win_rate:.1f}% | "
                f"Loss: {stats['mean_loss']:.4f} | "
                f"Q-value: {stats['mean_q_value']:.2f} | "
                f"Epsilon: {stats['epsilon']:.3f}"
            )

        # Evaluate
        if episode % eval_freq == 0:
            eval_stats = evaluate_episode(env, agents, training_agents)

            # Calculate evaluation statistics
            eval_rewards = []
            eval_wins = []
            for agent_id in training_agents:
                eval_rewards.append(eval_stats["training_agents_rewards"][agent_id])
                winner = eval_stats["winner"]
                eval_wins.append(winner == agent_id if winner else False)

            avg_eval_reward = np.mean(eval_rewards)
            eval_win_rate = np.mean(eval_wins) * 100

            logger.info(
                f"Evaluation Results | "
                f"Avg Reward: {avg_eval_reward:.3f} | "
                f"Win Rate: {eval_win_rate:.1f}%"
            )

        # Save checkpoint
        if episode % save_freq == 0:
            checkpoint_path = checkpoints_dir / f"agent_{episode}.pt"
            shared_agent.save(str(checkpoint_path))
            logger.info(f"Saved checkpoint: {checkpoint_path}")

    # Save final model
    final_model_path = checkpoints_dir / "agent_final.pt"
    shared_agent.save(str(final_model_path))
    logger.info(f"Saved final model: {final_model_path}")

    logger.info("=" * 80)
    logger.info("Training Session Complete")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
