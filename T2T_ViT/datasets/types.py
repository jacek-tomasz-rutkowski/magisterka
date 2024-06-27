from typing import TypedDict

from torch import Tensor
from torchvision import tv_tensors


class ImageLabelDataitem(TypedDict):
    image: tv_tensors.Image  # Shape CHW (RGB, float32 after default transform, uint8 before).
    label: int


class ImageLabelBatch(TypedDict):
    image: tv_tensors.Image  # Shape BCHW, RGB, dtype float32.
    label: Tensor  # Shape B, dtype int64.


class ImageLabelMaskDataitem(ImageLabelDataitem):
    mask: Tensor  # Shape (num_players,), dtype bool.


class ImageLabelMaskBatch(ImageLabelBatch):
    mask: Tensor  # Shape (B, num_players), dtype bool.
