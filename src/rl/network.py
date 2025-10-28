"""
Q-network implementation for DQN agent.

This module provides neural network architectures for estimating
Q-values in Deep Q-Network reinforcement learning.
"""

import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """
    Q-network for estimating action values.

    This network takes the game state as input and outputs Q-values
    for each possible action. Supports both standard DQN and Dueling DQN
    architectures.

    Parameters
    ----------
    input_dim : int
        Dimension of input observation (flattened).
    output_dim : int
        Number of possible actions.
    hidden_layers : list of int, optional
        List of hidden layer sizes. Default is [512, 256, 128].
    activation : str, optional
        Activation function ('relu', 'tanh', 'elu', 'leaky_relu', 'selu'). Default is 'relu'.
    dropout : float, optional
        Dropout rate for regularization. Default is 0.2.
    dueling : bool, optional
        Whether to use Dueling DQN architecture. Default is False.

    Attributes
    ----------
    features : nn.Sequential
        Shared feature extraction layers.
    value_stream : nn.Sequential or None
        Value stream for Dueling DQN (if enabled).
    advantage_stream : nn.Sequential or None
        Advantage stream for Dueling DQN (if enabled).
    output_layer : nn.Linear or None
        Output layer for standard DQN (if dueling is False).
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_layers: list[int] = None,
        activation: str = "relu",
        dropout: float = 0.2,
        dueling: bool = False,
    ):
        super().__init__()

        if hidden_layers is None:
            hidden_layers = [512, 256, 128]

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dueling = dueling

        # Activation function mapping
        activation_map = {
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "elu": nn.ELU,
            "leaky_relu": nn.LeakyReLU,
            "selu": nn.SELU,
        }
        act_fn = activation_map.get(activation.lower(), nn.ReLU)

        # Build shared feature extraction layers
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_layers:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    act_fn(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim

        self.features = nn.Sequential(*layers)

        if dueling:
            # Dueling DQN: separate value and advantage streams
            last_hidden = hidden_layers[-1]

            # Value stream: outputs single value V(s)
            self.value_stream = nn.Sequential(
                nn.Linear(last_hidden, last_hidden // 2),
                act_fn(),
                nn.Linear(last_hidden // 2, 1),
            )

            # Advantage stream: outputs advantage A(s,a) for each action
            self.advantage_stream = nn.Sequential(
                nn.Linear(last_hidden, last_hidden // 2),
                act_fn(),
                nn.Linear(last_hidden // 2, output_dim),
            )

            self.output_layer = None
        else:
            # Standard DQN: direct Q-value output
            self.output_layer = nn.Linear(hidden_layers[-1], output_dim)
            self.value_stream = None
            self.advantage_stream = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Parameters
        ----------
        x : torch.Tensor
            Input observation tensor of shape (batch_size, input_dim).

        Returns
        -------
        torch.Tensor
            Q-values for each action of shape (batch_size, output_dim).
        """
        # Extract features
        features = self.features(x)

        if self.dueling:
            # Dueling DQN: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
            value = self.value_stream(features)
            advantage = self.advantage_stream(features)

            # Subtract mean advantage for identifiability
            q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        else:
            # Standard DQN
            q_values = self.output_layer(features)

        return q_values

    def get_num_parameters(self) -> int:
        """
        Get total number of trainable parameters.

        Returns
        -------
        int
            Total number of trainable parameters.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
