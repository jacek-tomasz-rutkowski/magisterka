import glob
import logging
import math
from pathlib import Path
from typing import Sequence, TypeVar, cast

import torch
import torch.nn.functional as F

_logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent  # Path to the T2T_ViT/ directory.


def resize_pos_embed(posemb: torch.Tensor, posemb_new: torch.Tensor) -> torch.Tensor:
    # Copied from T2T_Vit repo.
    # example: 224:(14x14+1)-> 384: (24x24+1)
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    ntok_new = posemb_new.shape[1]
    if True:
        # posemb_tok is for cls token, posemb_grid for the following tokens
        posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
        ntok_new -= 1
    else:
        posemb_tok, posemb_grid = posemb[:, :0], posemb[0]
    gs_old = int(math.sqrt(len(posemb_grid)))  # 14
    gs_new = int(math.sqrt(ntok_new))  # 24
    _logger.info("Position embedding grid-size from %s to %s", gs_old, gs_new)
    # [1, 196, dim]->[1, 14, 14, dim]->[1, dim, 14, 14]
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    # [1, dim, 14, 14] -> [1, dim, 24, 24]
    posemb_grid = F.interpolate(posemb_grid, size=(gs_new, gs_new), mode="bicubic")
    # [1, dim, 24, 24] -> [1, 24*24, dim]
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new * gs_new, -1)
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)  # [1, 24*24+1, dim]
    return posemb


def find_latest_checkpoint(dir: Path) -> Path:
    """Return latest (by mtime) .ckpt file in a given directory (recursively)."""
    files = glob.glob(str(dir) + "/**/*.ckpt", recursive=True)
    if not files:
        raise FileNotFoundError(f"No checkpoints found in: {dir}")
    files = sorted(files, key=lambda x: Path(x).stat().st_mtime)
    return Path(files[-1])


def find_latest_checkpoints(dir: Path) -> list[Path]:
    """Return a list of all .ckpt files in a given directory (recursively), sorted by mtime."""
    files = glob.glob(str(dir) + "/**/*.ckpt", recursive=True)
    if not files:
        raise FileNotFoundError(f"No checkpoints found in: {dir}")
    return [Path(p) for p in sorted(files, key=lambda x: Path(x).stat().st_mtime, reverse=True)]


T = TypeVar("T")


def random_split_sequence(
    sequence: Sequence[T], lengths: Sequence[float] | Sequence[int], generator: torch.Generator
) -> tuple[list[T], ...]:
    """
    Randomly split a sequence into lists of given lengths or proportions.

    If given fractions that sum up to 1, the lengths will be computed
    initially as floor(frac * len(sequence)) and any remainders will be
    distributed in round-robin fashion to the lengths.

    This works exactly like torch.utils.data.random_split(), except it takes any sequence,
    and outputs a tuple of lists (instead of Subset datasets).
    """
    subsets = torch.utils.data.random_split(cast(torch.utils.data.Dataset[T], sequence), lengths, generator)
    return tuple(list(cast(Sequence[T], subset)) for subset in subsets)
