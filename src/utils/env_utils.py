"""Environment utility functions for Sevens RL."""

from __future__ import annotations


def calculate_state_dim(num_players: int) -> int:
    """Calculate the state dimension for the Sevens environment.

    The state consists of:
    - board: 52 binary values (1 per card)
    - hand: 52 binary values (1 per card)
    - action_mask: 53 binary values (52 cards + 1 pass action)
    - hand_counts: num_players values (hand size for each player)
    - card_play_order: 52 float values (play order for each card, normalized)
    - current_player: num_players binary values (one-hot encoded)

    Parameters
    ----------
    num_players : int
        Number of players in the game (2-4)

    Returns
    -------
    int
        Total state dimension

    Examples
    --------
    >>> calculate_state_dim(4)
    217
    >>> calculate_state_dim(2)
    213
    """
    state_dim = (
        52  # board
        + 52  # hand
        + 53  # action_mask
        + num_players  # hand_counts
        + 52  # card_play_order
        + num_players  # current_player
    )
    return state_dim


def calculate_action_dim() -> int:
    """Calculate the action dimension for the Sevens environment.

    The action space consists of:
    - 52 card actions (one per card)
    - 1 pass action

    Returns
    -------
    int
        Total action dimension (always 53)

    Examples
    --------
    >>> calculate_action_dim()
    53
    """
    return 53  # 52 cards + 1 pass action
