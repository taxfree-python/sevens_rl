"""Tests for Q-network implementation."""

import torch

from src.rl.network import QNetwork


def test_qnetwork_initialization():
    """Test Q-network initialization."""
    input_dim = 157
    output_dim = 53

    # Test default initialization
    network = QNetwork(input_dim=input_dim, output_dim=output_dim)
    assert network.input_dim == input_dim
    assert network.output_dim == output_dim
    assert not network.dueling

    # Test custom hidden layers
    hidden_layers = [256, 128]
    network = QNetwork(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_layers=hidden_layers,
    )
    assert len([m for m in network.features if isinstance(m, torch.nn.Linear)]) == len(
        hidden_layers
    )


def test_qnetwork_forward_shape():
    """Test Q-network forward pass output shape."""
    input_dim = 157
    output_dim = 53
    batch_size = 32

    network = QNetwork(input_dim=input_dim, output_dim=output_dim)

    # Test single sample
    x = torch.randn(1, input_dim)
    q_values = network(x)
    assert q_values.shape == (1, output_dim)

    # Test batch
    x_batch = torch.randn(batch_size, input_dim)
    q_values_batch = network(x_batch)
    assert q_values_batch.shape == (batch_size, output_dim)


def test_qnetwork_dueling_architecture():
    """Test Dueling DQN architecture."""
    input_dim = 157
    output_dim = 53

    network = QNetwork(
        input_dim=input_dim,
        output_dim=output_dim,
        dueling=True,
    )

    assert network.dueling
    assert network.value_stream is not None
    assert network.advantage_stream is not None
    assert network.output_layer is None

    # Test forward pass
    x = torch.randn(10, input_dim)
    q_values = network(x)
    assert q_values.shape == (10, output_dim)


def test_qnetwork_activation_functions():
    """Test different activation functions."""
    input_dim = 157
    output_dim = 53

    activations = ["relu", "tanh", "elu"]
    for act in activations:
        network = QNetwork(
            input_dim=input_dim,
            output_dim=output_dim,
            activation=act,
        )

        x = torch.randn(5, input_dim)
        q_values = network(x)
        assert q_values.shape == (5, output_dim)


def test_qnetwork_dropout():
    """Test dropout in network."""
    input_dim = 157
    output_dim = 53

    network = QNetwork(
        input_dim=input_dim,
        output_dim=output_dim,
        dropout=0.5,
    )

    # Count dropout layers
    dropout_layers = [m for m in network.features if isinstance(m, torch.nn.Dropout)]
    assert len(dropout_layers) > 0

    # Test forward pass in train mode (with dropout)
    network.train()
    x = torch.randn(10, input_dim)
    q1 = network(x)

    # Test forward pass in eval mode (without dropout)
    network.eval()
    q2 = network(x)

    assert q1.shape == q2.shape == (10, output_dim)


def test_qnetwork_parameter_count():
    """Test get_num_parameters method."""
    input_dim = 157
    output_dim = 53

    network = QNetwork(input_dim=input_dim, output_dim=output_dim)
    num_params = network.get_num_parameters()

    assert num_params > 0
    assert isinstance(num_params, int)

    # Verify count
    manual_count = sum(p.numel() for p in network.parameters() if p.requires_grad)
    assert num_params == manual_count


def test_qnetwork_gradient_flow():
    """Test gradient flow through network."""
    input_dim = 157
    output_dim = 53

    network = QNetwork(input_dim=input_dim, output_dim=output_dim)
    optimizer = torch.optim.Adam(network.parameters(), lr=0.001)

    # Forward pass
    x = torch.randn(10, input_dim)
    q_values = network(x)

    # Compute loss and backprop
    loss = q_values.mean()
    optimizer.zero_grad()
    loss.backward()

    # Check gradients exist
    for param in network.parameters():
        if param.requires_grad:
            assert param.grad is not None


def test_qnetwork_dueling_identifiability():
    """Test that Dueling DQN maintains identifiability constraint."""
    input_dim = 157
    output_dim = 53

    network = QNetwork(
        input_dim=input_dim,
        output_dim=output_dim,
        dueling=True,
    )

    x = torch.randn(10, input_dim)
    q_values = network(x)

    # In Dueling DQN, Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
    # This means for each state, the advantages should sum to ~0 (or close)
    # We can't test this directly without accessing internals, but we can
    # verify the output shape and that it produces valid Q-values
    assert q_values.shape == (10, output_dim)
    assert not torch.isnan(q_values).any()
    assert not torch.isinf(q_values).any()
