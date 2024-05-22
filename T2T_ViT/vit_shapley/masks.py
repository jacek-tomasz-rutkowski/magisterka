from typing import Literal

import numpy as np
import torch
from torch import Tensor


def apply_masks_to_batch(images: Tensor, masks: Tensor, labels: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Return a batch of masked images, labels, masks (without a num_masks_per_image dimension).

    See `apply_masks()` for details on masking.

    Args:
    - images: shape (B, C, H, W).
    - masks: shape (B, n_masks_per_image, n_players) or (B, n_players).
    - labels: shape (B,).

    Returns (images, labels, masks) where:
    - images: masked to shape (B * n_masks_per_image, C, H, W).
    - masks: reshaped to shape (B * n_masks_per_image, n_players).
    - labels: repeated to shape (B * n_masks_per_image,).
    """
    if len(masks.shape) == 2:
        masks = masks.unsqueeze(dim=1)

    B, n_masks_per_image, n_players = masks.shape

    images_masked = apply_masks(images, masks)

    if labels is not None:
        assert labels.shape == (B,)
        labels = labels.repeat_interleave(n_masks_per_image)
    else:
        labels = torch.zeros((B * n_masks_per_image,), dtype=torch.long)

    masks = masks.view(B * n_masks_per_image, n_players)

    return images_masked, masks, labels


def apply_masks(images: Tensor, masks: Tensor) -> Tensor:
    """Zeroes out masked pixels in images.

    Each mask provides `n_players` booleans (or 0/1 values).
    Images are split into `sqrt(n_players) ✕ sqrt(n_players)` patches.
    Each patch is masked if the corresponding boolean is False/0.

    Args:
    - images: shape (B, C, H, W).
    - masks: shape (B, n_masks_per_image, n_players).

    Returns shape (B * n_masks_per_image, C, H, W).
    """
    B, C, H, W = images.shape
    B2, n_masks_per_image, n_players = masks.shape
    assert B == B2, f"Expected same batch dim, got {images.shape=} {masks.shape=}"
    assert C in [1, 3], f"Expected 1 or 3 channels, got {images.shape=}"

    mask_H = int(np.round(np.sqrt(n_players)))
    mask_W = mask_H
    assert mask_H * mask_W == n_players, f"{n_players=}, expected a square number."
    masks = masks.view(B, n_masks_per_image, mask_H, mask_W)

    # Upscale masks to image size.
    h_repeats, w_repeats = int(np.ceil(H / mask_H)), int(np.ceil(W / mask_W))
    masks = masks.repeat_interleave(h_repeats, dim=2).repeat_interleave(w_repeats, dim=3)
    masks = masks[:, :, :H, :W]

    masks = masks.unsqueeze(dim=2)  # Add C dimension
    # Masks now have shape (B, n_masks_per_image, 1, H, W)

    images = images.unsqueeze(dim=1)
    # Images now have shape (B, 1, C, H, W)

    images = images * masks
    # Images now have shape (B, n_masks_per_image, C, H, W)

    return images.view(B * n_masks_per_image, C, H, W)


def generate_masks(
    num_players: int,
    num_mask_samples: int = 1,
    paired_mask_samples: bool = True,
    mode: Literal["uniform", "shapley"] = "uniform",
    random_state: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Args:
    - num_players: the number of players in the coalitional game
    - num_mask_samples: the number of masks to generate
    - paired_mask_samples: if True, the generated masks are pairs of x and 1-x (num_mask_samples must be even).
    - mode: the distribution that the number of masked features follows. ('uniform' or 'shapley')
    - random_state: random generator

    Returns int ndarray of shape (num_mask_samples, num_players).
    """
    random_state = random_state or np.random.default_rng()

    num_samples_ = num_mask_samples or 1

    if paired_mask_samples:
        assert num_samples_ % 2 == 0, "'num_samples' must be a multiple of 2 if 'paired' is True"
        num_samples_ = num_samples_ // 2
    else:
        num_samples_ = num_samples_

    if mode == "uniform":
        thresholds = random_state.random((num_samples_, 1))
        masks = (random_state.random((num_samples_, num_players)) > thresholds).astype(np.bool_)
    elif mode == "shapley":
        probs = 1 / np.arange(1, num_players - 1) * (num_players - np.arange(1, num_players - 1))
        probs = probs / np.sum(probs)
        sizes = random_state.choice(np.arange(1, num_players - 1), p=probs, size=(num_samples_,), replace=True)

        mask_list = []
        for i in range(num_samples_):
            all_ones = np.ones((num_players,), dtype=np.bool_)
            all_ones[np.random.choice(num_players, sizes[i], replace=False)] = 0
            mask_list.append(all_ones)
        masks = np.array(mask_list)
    else:
        raise ValueError("'mode' must be 'uniform' or 'shapley'")

    if paired_mask_samples:
        masks = np.stack([masks, 1 - masks], axis=1).reshape(num_samples_ * 2, num_players)

    return masks


def make_masks_from_player_values(
    num_zeroes: torch.Tensor, player_values: torch.Tensor, players_to_mask: Literal["best", "worst", "random"]
) -> torch.Tensor:
    """
    Create masks with a given number of best/words/random players masked (zeroed).

    Args:
    - num_zeroes: shape (B, num_masks_per_image), dtype long.
    - player_values: shape (B, num_players), any dtype.
    - players_to_mask: which players to mask (set to zero); "best" means highest values.

    Returns masks of shape (B, num_masks_per_image, num_players), dtype bool.
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
    Get the distance from the center of each player, shape (num_players,).

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
