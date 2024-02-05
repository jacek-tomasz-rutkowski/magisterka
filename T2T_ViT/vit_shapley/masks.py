from typing import Literal

import numpy as np
import torch


def make_masks_from_player_values(
    num_zeroes: torch.Tensor, player_values: torch.Tensor, players_to_mask: Literal["best", "worst", "random"]
) -> torch.Tensor:
    """
    Create masks with a given number of best/words/random players masked (zeroed).

    Args:
    - num_zeroes: shape (B, num_masks_per_image), dtype long.
    - player_values: shape (B, num_players), any dtype.
    - players_to_mask: which players to mask (set to zero); "best" means highest values.

    Returns:
        masks of shape (B, num_masks_per_image, num_players), dtype bool.
    """
    B, num_masks_per_image = num_zeroes.shape
    _B, num_players = player_values.shape
    assert B == _B
    assert num_zeroes.max() <= num_players
    assert num_zeroes.min() >= 0
    device = player_values.device

    # Get indices of best/worst/random players.
    sorted_players: torch.Tensor  # (B, num_players)
    if players_to_mask == "best":
        sorted_players = player_values.argsort(dim=1, descending=True)
    elif players_to_mask == "worst":
        sorted_players = player_values.argsort(dim=1, descending=False)
    elif players_to_mask == "random":
        sorted_players = torch.stack([torch.randperm(num_players) for _ in range(B)], dim=0).to(device)
    else:
        raise ValueError(f"Got unexpected {players_to_mask=}")
    assert sorted_players.shape == (B, num_players)

    # Create masks.
    # masks[b, n, p] := False if p < num_zeroes[b, n] else True
    masks = ~(torch.arange(num_players, device=device).unsqueeze(0).unsqueeze(0) < num_zeroes.unsqueeze(2))
    assert masks.shape == (B, num_masks_per_image, num_players)
    # masks[b, n, sorted_players[b, p]] := masks[b, n, p]
    inverse_sorted_players = torch.argsort(sorted_players, dim=1)  # (B, num_players)
    masks = masks.gather(dim=2, index=inverse_sorted_players.unsqueeze(1).expand(B, num_masks_per_image, num_players))
    assert masks.shape == (B, num_masks_per_image, num_players)
    return masks


def get_distances_from_center(num_players: int) -> torch.Tensor:
    """
    Get the closeness to the center of each player, shape (num_players,).

    Assumes num_players is a square of an integer k,
    distances are L2 distances from the center of the k by k square.

    For example, for 9 players, the distances are:
    [1.41, 1.00, 1.41,
     1.00, 0.00, 1.00,
     1.41, 1.00, 1.41]
    """
    H = int(np.sqrt(num_players))
    W = H
    assert H * W == num_players
    hs = torch.arange(H).float() - (H - 1) / 2
    ws = torch.arange(W).float() - (W - 1) / 2
    h, w = torch.meshgrid(hs, ws, indexing="xy")
    return torch.sqrt(h**2 + w**2).view(num_players)
