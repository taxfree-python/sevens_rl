"""DQN Training Script with Hydra Configuration."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Union

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from src.agents import NearestSevensAgent, RandomAgent
from src.rl.dqn_agent import DQNAgent
from src.sevens_env import SevensEnv
from src.utils.hydra_utils import (
    create_dqn_agent_from_config,
    get_env_params,
    get_experiment_params,
    get_training_params,
)
from src.utils.logger import setup_logger

# Type alias for agent types
Agent = Union[DQNAgent, RandomAgent, NearestSevensAgent]


def _normalize_training_agents(
    training_agents: list[str] | str | None,
    possible_agents: list[str],
) -> list[str]:
    """Normalize training_agents parameter to a list of agent IDs.

    Args:
        training_agents: Collection of agent IDs to train, or None for all agents.
            Can be a list of agent IDs, a single agent ID string, or None.
        possible_agents: List of all possible agent IDs in the environment

    Returns:
        List of agent IDs to train
    """
    if training_agents is None:
        return list(possible_agents)
    elif isinstance(training_agents, str):
        return [training_agents]
    else:
        return list(training_agents)


def _setup_self_play_agents(
    cfg: DictConfig,
    env: SevensEnv,
    state_dim: int,
    action_dim: int,
    shared_network: bool,
    logger: logging.Logger,
) -> dict[str, Agent]:
    """Setup agents for self-play mode.

    Args:
        cfg: Hydra configuration
        env: Sevens environment
        state_dim: State dimension
        action_dim: Action dimension
        shared_network: Whether to use shared network
        logger: Logger instance

    Returns:
        Dictionary of agent_id -> Agent
    """
    agents = {}

    if shared_network:
        # Create one shared DQN agent
        shared_dqn_agent = create_dqn_agent_from_config(
            cfg, state_dim=state_dim, action_dim=action_dim
        )
        for agent_id in env.possible_agents:
            agents[agent_id] = shared_dqn_agent
            logger.info(f"Assigned shared DQN to {agent_id}")
    else:
        # Create independent DQN agents
        for agent_id in env.possible_agents:
            agent = create_dqn_agent_from_config(
                cfg, state_dim=state_dim, action_dim=action_dim
            )
            agent.name = agent_id
            agents[agent_id] = agent
            logger.info(f"Created independent DQN agent: {agent_id}")

    return agents


def _setup_vs_random_agents(
    cfg: DictConfig,
    env: SevensEnv,
    state_dim: int,
    action_dim: int,
    training_agents: list[str],
    logger: logging.Logger,
) -> dict[str, Agent]:
    """Setup agents for vs_random mode.

    Args:
        cfg: Hydra configuration
        env: Sevens environment
        state_dim: State dimension
        action_dim: Action dimension
        training_agents: List of training agent IDs
        logger: Logger instance

    Returns:
        Dictionary of agent_id -> Agent
    """
    agents = {}

    for agent_id in env.possible_agents:
        if agent_id in training_agents:
            agent = create_dqn_agent_from_config(
                cfg, state_dim=state_dim, action_dim=action_dim
            )
            agent.name = agent_id
            agents[agent_id] = agent
            logger.info(f"Created DQN agent: {agent_id}")
        else:
            agents[agent_id] = RandomAgent()
            logger.info(f"Created Random agent: {agent_id}")

    return agents


def _setup_custom_agents(
    cfg: DictConfig,
    env: SevensEnv,
    state_dim: int,
    action_dim: int,
    players_cfg: DictConfig,
    logger: logging.Logger,
) -> dict[str, Agent]:
    """Setup agents for custom mode.

    Args:
        cfg: Hydra configuration
        env: Sevens environment
        state_dim: State dimension
        action_dim: Action dimension
        players_cfg: Players configuration
        logger: Logger instance

    Returns:
        Dictionary of agent_id -> Agent
    """
    agents = {}
    custom_players = players_cfg.get("players", {})
    networks_cfg = players_cfg.get("networks", {})

    # Create network instances if specified
    network_instances = {}
    if networks_cfg:
        for net_id, net_config in networks_cfg.items():
            if "shared_by" in net_config:
                # Create shared network
                network_instances[net_id] = create_dqn_agent_from_config(
                    cfg, state_dim=state_dim, action_dim=action_dim
                )
                logger.info(f"Created shared network: {net_id}")

    # Create agents based on custom config
    for agent_id in env.possible_agents:
        player_cfg = custom_players.get(agent_id, {})
        agent_type = player_cfg.get("type", "dqn")

        if agent_type == "dqn":
            network_id = player_cfg.get("network")
            if network_id and network_id in network_instances:
                # Use shared network
                agents[agent_id] = network_instances[network_id]
                logger.info(f"Assigned shared network '{network_id}' to {agent_id}")
            else:
                # Create independent network
                agent = create_dqn_agent_from_config(
                    cfg, state_dim=state_dim, action_dim=action_dim
                )
                agent.name = agent_id
                agents[agent_id] = agent
                logger.info(f"Created independent DQN agent: {agent_id}")

        elif agent_type == "random":
            agents[agent_id] = RandomAgent()
            logger.info(f"Created Random agent: {agent_id}")

        elif agent_type == "nearest_sevens":
            prefer_high = player_cfg.get("prefer_high_rank", False)
            agents[agent_id] = NearestSevensAgent(prefer_high_rank=prefer_high)
            logger.info(
                f"Created NearestSevens agent: {agent_id} "
                f"(prefer_high_rank={prefer_high})"
            )

        else:
            msg = f"Unknown agent type: {agent_type}"
            raise ValueError(msg)

    return agents


def setup_agents(
    cfg: DictConfig,
    env: SevensEnv,
    state_dim: int,
    action_dim: int,
    logger: logging.Logger,
) -> tuple[dict[str, Agent], list[str]]:
    """Setup agents based on players configuration.

    Args:
        cfg: Hydra configuration
        env: Sevens environment
        state_dim: State dimension
        action_dim: Action dimension
        logger: Logger instance

    Returns:
        Tuple of (agents dict, training_agents list)
    """
    players_cfg = cfg.get("players", {})
    mode = players_cfg.get("mode", "self_play")
    shared_network = players_cfg.get("shared_network", True)
    training_players = players_cfg.get("training_players", "all")

    # Convert training_players to list
    if training_players == "all":
        training_agents = list(env.possible_agents)
    else:
        training_agents = [f"player_{i}" for i in training_players]

    logger.info(f"Setting up agents with mode: {mode}")
    logger.info(f"Shared network: {shared_network}")
    logger.info(f"Training agents: {training_agents}")

    if mode == "self_play":
        agents = _setup_self_play_agents(
            cfg, env, state_dim, action_dim, shared_network, logger
        )
    elif mode == "vs_random":
        agents = _setup_vs_random_agents(
            cfg, env, state_dim, action_dim, training_agents, logger
        )
    elif mode == "custom":
        agents = _setup_custom_agents(
            cfg, env, state_dim, action_dim, players_cfg, logger
        )
    else:
        msg = f"Unknown mode: {mode}"
        raise ValueError(msg)

    return agents, training_agents


def train_episode(
    env: SevensEnv,
    agents: dict[str, Agent],
    logger: logging.Logger,
    training_agents: list[str] | str | None = None,
) -> dict[str, Any]:
    """Train for one episode.

    Args:
        env: Sevens environment
        agents: Dictionary of agent_id -> Agent (DQN, Random, etc.)
        logger: Logger instance
        training_agents: Collection of agent IDs to train. If None, all agents
            are trained. Can be a list of agent IDs or a single agent ID string.

    Returns:
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

    for agent_id in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        episode_rewards[agent_id] += reward
        done = termination or truncation

        if done:
            # Store final transition for training agents
            if agent_id in training_agents and agent_id in prev_observations:
                # Only DQNAgents can store experiences
                agent = agents[agent_id]
                if hasattr(agent, "store_experience"):
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
            if hasattr(agent, "store_experience"):
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

        # Train the training agents
        if agent_id in training_agents:
            agent = agents[agent_id]
            if hasattr(agent, "train_step"):
                train_result = agent.train_step()
                if train_result and "loss" in train_result:
                    training_losses.append(train_result["loss"])
                    training_q_values.append(train_result["q_value_mean"])

    # End episode for DQN agents
    for agent in agents.values():
        if hasattr(agent, "end_episode"):
            agent.end_episode()

    # Calculate statistics for training agents
    training_agents_rewards = {
        agent_id: episode_rewards[agent_id] for agent_id in training_agents
    }

    # Get epsilon from first DQN training agent
    epsilon = None
    for agent_id in training_agents:
        agent = agents[agent_id]
        if hasattr(agent, "policy"):
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
    agents: dict[str, Agent],
    training_agents: list[str] | str | None = None,
) -> dict[str, Any]:
    """Evaluate agents for one episode (no training).

    Args:
        env: Sevens environment
        agents: Dictionary of agent_id -> Agent (DQN, Random, etc.)
        training_agents: Collection of agents being trained and tracked.
            Can be a list of agent IDs or a single agent ID string, or None for all.

    Returns:
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
        if hasattr(agent, "policy") and id(agent) not in processed_agents:
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
        if agent_id_key in original_epsilons and hasattr(agent, "policy"):
            agent.policy.set_epsilon(original_epsilons[agent_id_key])

    # Calculate statistics for training agents
    training_agents_rewards = {
        agent_id: episode_rewards[agent_id] for agent_id in training_agents
    }

    # Determine winner from finished_order (first to finish wins)
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
    num_players = env_params["num_players"]
    state_dim = (
        52  # board
        + 52  # hand
        + 53  # action_mask
        + num_players  # hand_counts
        + 52  # card_play_order
        + num_players  # current_player
    )
    action_dim = 53  # 52 cards + 1 pass

    logger.info(f"State dimension: {state_dim} (for {num_players} players)")
    logger.info(f"Action dimension: {action_dim}")

    # Setup agents based on players configuration
    agents, training_agents = setup_agents(cfg, env, state_dim, action_dim, logger)

    # Get Q-Network parameters from first DQN agent
    first_dqn_agent = None
    for agent in agents.values():
        if hasattr(agent, "q_network"):
            first_dqn_agent = agent
            break

    if first_dqn_agent:
        logger.info(
            f"Q-Network parameters: {first_dqn_agent.q_network.get_num_parameters()}"
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
                all_recent_rewards.extend(episode_rewards_history[agent_id][-log_freq:])
                all_recent_wins.extend(episode_wins[agent_id][-log_freq:])

            avg_reward = np.mean(all_recent_rewards)
            win_rate = np.mean(all_recent_wins) * 100

            log_msg = (
                f"Episode {episode}/{num_episodes} | "
                f"Steps: {stats['episode_steps']} | "
                f"Avg Reward: {avg_reward:.3f} | "
                f"Win Rate: {win_rate:.1f}% | "
                f"Loss: {stats['mean_loss']:.4f} | "
                f"Q-value: {stats['mean_q_value']:.3f}"
            )

            if stats["epsilon"] is not None:
                log_msg += f" | Epsilon: {stats['epsilon']:.4f}"

            logger.info(log_msg)

            # Log individual agent performance if multiple training agents
            if len(training_agents) > 1:
                for agent_id in training_agents:
                    agent_recent_rewards = episode_rewards_history[agent_id][-log_freq:]
                    agent_recent_wins = episode_wins[agent_id][-log_freq:]
                    agent_avg_reward = np.mean(agent_recent_rewards)
                    agent_win_rate = np.mean(agent_recent_wins) * 100
                    logger.info(
                        f"  {agent_id}: Avg Reward: {agent_avg_reward:.3f} | "
                        f"Win Rate: {agent_win_rate:.1f}%"
                    )

        # Evaluation
        if episode % eval_freq == 0:
            logger.info("-" * 80)
            logger.info(f"Evaluating at episode {episode}...")

            eval_rewards = {agent_id: [] for agent_id in training_agents}
            eval_wins = {agent_id: [] for agent_id in training_agents}
            num_eval_episodes = 10

            for _ in range(num_eval_episodes):
                eval_stats = evaluate_episode(env, agents, training_agents)

                # Track results for each training agent
                for agent_id in training_agents:
                    eval_rewards[agent_id].append(
                        eval_stats["training_agents_rewards"][agent_id]
                    )

                    # Determine if this agent won (first in finished_order)
                    winner = eval_stats["winner"]
                    eval_wins[agent_id].append(
                        winner == agent_id if winner else False
                    )

            # Aggregate evaluation results
            all_eval_rewards = []
            all_eval_wins = []
            for agent_id in training_agents:
                all_eval_rewards.extend(eval_rewards[agent_id])
                all_eval_wins.extend(eval_wins[agent_id])

            avg_eval_reward = np.mean(all_eval_rewards)
            eval_win_rate = np.mean(all_eval_wins) * 100

            logger.info(
                f"Evaluation Results | "
                f"Avg Reward: {avg_eval_reward:.3f} | "
                f"Win Rate: {eval_win_rate:.1f}% "
                f"({sum(all_eval_wins)}/{len(all_eval_wins)} wins)"
            )

            # Log individual agent evaluation results if multiple training agents
            if len(training_agents) > 1:
                for agent_id in training_agents:
                    agent_eval_avg_reward = np.mean(eval_rewards[agent_id])
                    agent_eval_win_rate = np.mean(eval_wins[agent_id]) * 100
                    agent_eval_wins_count = sum(eval_wins[agent_id])
                    logger.info(
                        f"  {agent_id}: Avg Reward: {agent_eval_avg_reward:.3f} | "
                        f"Win Rate: {agent_eval_win_rate:.1f}% "
                        f"({agent_eval_wins_count}/{num_eval_episodes} wins)"
                    )

            logger.info("-" * 80)

        # Save checkpoint (save all unique DQN agents)
        if episode % save_freq == 0:
            saved_agents = set()
            for agent_id in training_agents:
                agent = agents[agent_id]
                if hasattr(agent, "save") and id(agent) not in saved_agents:
                    checkpoint_path = checkpoints_dir / f"{agent_id}_{episode}.pt"
                    agent.save(str(checkpoint_path))
                    logger.info(f"Saved checkpoint: {checkpoint_path}")
                    saved_agents.add(id(agent))

    # Final evaluation
    logger.info("=" * 80)
    logger.info("Training Complete - Running Final Evaluation")
    logger.info("=" * 80)

    final_eval_rewards = {agent_id: [] for agent_id in training_agents}
    final_eval_wins = {agent_id: [] for agent_id in training_agents}
    num_final_eval = 100

    for _ in range(num_final_eval):
        eval_stats = evaluate_episode(env, agents, training_agents)

        for agent_id in training_agents:
            final_eval_rewards[agent_id].append(
                eval_stats["training_agents_rewards"][agent_id]
            )
            winner = eval_stats["winner"]
            final_eval_wins[agent_id].append(winner == agent_id if winner else False)

    # Aggregate final results
    all_final_rewards = []
    all_final_wins = []
    for agent_id in training_agents:
        all_final_rewards.extend(final_eval_rewards[agent_id])
        all_final_wins.extend(final_eval_wins[agent_id])

    avg_final_reward = np.mean(all_final_rewards)
    final_win_rate = np.mean(all_final_wins) * 100

    logger.info(
        f"Final Evaluation ({num_final_eval} episodes) | "
        f"Avg Reward: {avg_final_reward:.3f} | "
        f"Win Rate: {final_win_rate:.1f}% "
        f"({sum(all_final_wins)}/{len(all_final_wins)} wins)"
    )

    # Log individual agent final evaluation results if multiple training agents
    if len(training_agents) > 1:
        for agent_id in training_agents:
            agent_final_avg_reward = np.mean(final_eval_rewards[agent_id])
            agent_final_win_rate = np.mean(final_eval_wins[agent_id]) * 100
            agent_final_wins_count = sum(final_eval_wins[agent_id])
            logger.info(
                f"  {agent_id}: Avg Reward: {agent_final_avg_reward:.3f} | "
                f"Win Rate: {agent_final_win_rate:.1f}% "
                f"({agent_final_wins_count}/{num_final_eval} wins)"
            )

    # Save final model (save all unique DQN agents)
    saved_agents = set()
    for agent_id in training_agents:
        agent = agents[agent_id]
        if hasattr(agent, "save") and id(agent) not in saved_agents:
            final_checkpoint_path = checkpoints_dir / f"{agent_id}_final.pt"
            agent.save(str(final_checkpoint_path))
            logger.info(f"Saved final model: {final_checkpoint_path}")
            saved_agents.add(id(agent))

    logger.info("=" * 80)
    logger.info("Training Session Complete")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
