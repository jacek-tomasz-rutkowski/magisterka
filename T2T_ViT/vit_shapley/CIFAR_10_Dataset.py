from typing import Any, Optional
from pathlib import Path

import torch
import torchvision
import numpy as np
import pytorch_lightning as pl
import PIL.Image
import PIL.ImageDraw
from torch.utils.data import random_split, Dataset, DataLoader

PROJECT_ROOT = Path(__file__).parent.parent  # Path to the T2T_ViT/ directory.


def apply_masks_to_batch(
    images: torch.Tensor, labels: torch.Tensor, masks: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return a batch of masked images, labels, masks (without a num_masks_per_image dimension).

    See `apply_masks()` for details on masking.

    Args:
    - images: shape (B, C, H, W).
    - labels: shape (B,).
    - masks: shape (B, n_masks_per_image, n_players).

    Returns (images, labels, masks) where:
    - images: masked to shape (B * n_masks_per_image, C, H, W).
    - labels: repeated to shape (B * n_masks_per_image,).
    - masks: reshaped to shape (B * n_masks_per_image, n_players).
    """
    if len(masks.shape) == 2:
        masks = masks.unsqueeze(dim=1)
    B, n_masks_per_image, n_players = masks.shape
    assert images.shape[0] == B and images.shape[1] in [1, 3], f"Unexpected {images.shape=}"
    assert labels.shape == (B,)
    images_masked = apply_masks(images, masks)
    labels = labels.repeat_interleave(n_masks_per_image)
    masks = masks.view(B * n_masks_per_image, n_players)
    return images_masked, labels, masks


def apply_masks(images: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    """Zeroes out masked pixels in images.

    Each mask provides `n_players` booleans (or 0/1 values).
    Images are split into `sqrt(n_players) ✕ sqrt(n_players)` patches.
    Each patch is masked if the corresponding boolean is False/0.

    Args:
    - images: shape (B, C, H, W).
    - masks: shape (B, n_masks_per_image, n_players).
    Returns:
    - shape (B * n_masks_per_image, C, H, W).
    """
    B, C, H, W = images.shape
    B2, n_masks_per_image, n_players = masks.shape
    assert B == B2

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


class CIFAR_10_Dataset(Dataset):
    """CIFAR-10 dataset, but returns normalized tensors of shape (224, 224, 3) and masks."""

    normalization = torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    classes = ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

    def __init__(
        self,
        num_players: int,
        num_mask_samples: int,
        paired_mask_samples: bool,
        root_path: Path,
        train: bool = True,
        download: bool = True,
    ):
        """
        Args:
            root_path: Directory with all the images (will be downloaded if not present).
            train: whether to access train or test part of the dataset.
            download: whether to download the dataset if it is not present.
            num_players, num_mask_samples, paired_mask_samples: see generate_mask().
        """
        self.num_players = num_players
        self.num_mask_samples = num_mask_samples
        self.paired_mask_samples = paired_mask_samples
        self.root_path = root_path
        self.shape = (224, 224, 3)
        transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(self.shape[:2], torchvision.transforms.InterpolationMode.BILINEAR),
                torchvision.transforms.ToTensor(),
                self.normalization,
            ]
        )
        self.dataset = torchvision.datasets.CIFAR10(root=root_path, train=train, download=download, transform=transform)

    @staticmethod
    def generate_masks(
        num_players: int,
        num_mask_samples: int = 1,
        paired_mask_samples: bool = True,
        mode: str = "uniform",
        random_state: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """
        Args:
            num_players: the number of players in the coalitional game
            num_mask_samples: the number of masks to generate
            paired_mask_samples: if True, the generated masks are pairs of x and 1-x (num_mask_samples must be even).
            mode: the distribution that the number of masked features follows. ('uniform' or 'shapley')
            random_state: random generator

        Returns:
            int ndarray of shape (num_mask_samples, num_players).

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
            masks = (random_state.random((num_samples_, num_players)) > thresholds).astype("int")
        elif mode == "shapley":
            probs = 1 / (np.arange(1, num_players) * (num_players - np.arange(1, num_players)))
            probs = probs / probs.sum()
            thresholds = random_state.choice(np.arange(num_players - 1), p=probs, size=(num_samples_, 1))
            thresholds /= num_players
            masks = (random_state.random((num_samples_, num_players)) > thresholds).astype("int")
        else:
            raise ValueError("'mode' must be 'uniform' or 'shapley'")

        if paired_mask_samples:
            masks = np.stack([masks, 1 - masks], axis=1).reshape(num_samples_ * 2, num_players)

        return masks

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        image, label = self.dataset[idx]
        masks = self.generate_masks(
            num_players=self.num_players,
            num_mask_samples=self.num_mask_samples,
            paired_mask_samples=self.paired_mask_samples,
        )
        return {"images": image, "labels": label, "masks": masks}

    @classmethod
    def to_image(cls, x: torch.Tensor, scale: float = 1.0, label: Optional[str] = None) -> PIL.Image.Image:
        """Converts a tensor of shape (3, 224, 224) to pillow image, reversing normalization."""
        image = _tensor_to_image(x, scale=scale, mean=cls.normalization.mean, std=cls.normalization.std)
        image = _draw_text(image, label, font_size=int(35 * scale))
        return image

    @classmethod
    def labels_to_strings(cls, labels: list[str] | list[int] | torch.Tensor) -> list[str]:
        """Converts a list of labels to a list of strings."""
        labels = [label.item() if isinstance(label, torch.Tensor) else label for label in labels]  # type: ignore
        return [label if isinstance(label, str) else cls.classes[int(label)] for label in labels]  # type: ignore

    @classmethod
    def to_image_grid(
        cls,
        x: torch.Tensor,
        labels: Optional[list[str] | list[int] | torch.Tensor] = None,
        colors: Optional[list[tuple[int, int, int]]] = None,
        scale: float = 1.0,
        n_columns: int = 8,
    ) -> PIL.Image.Image:
        images = [cls.to_image(x[i], scale=scale) for i in range(len(x))]
        str_labels = cls.labels_to_strings(labels) if labels is not None else None
        return _make_image_grid(
            images, labels=str_labels, colors=colors, font_size=int(35 * scale), n_columns=n_columns
        )


class CIFAR_10_Datamodule(pl.LightningDataModule):
    classes = CIFAR_10_Dataset.classes

    def __init__(
        self,
        num_players: int,
        num_mask_samples: int,
        paired_mask_samples: bool,
        batch_size: int = 32,
        num_workers: int = 2,
    ):
        super().__init__()
        self.num_players = num_players
        self.num_mask_samples = num_mask_samples
        self.paired_mask_samples = paired_mask_samples
        self._dataset_kwargs: dict[str, Any] = dict(
            num_players=num_players,
            num_mask_samples=num_mask_samples,
            paired_mask_samples=paired_mask_samples,
            root_path=PROJECT_ROOT / "CIFAR_10_data",
            download=False,
        )
        self.batch_size = batch_size
        self.num_workers = num_workers
        self._dataloader_kwargs: dict[str, Any] = dict(batch_size=batch_size, num_workers=num_workers)
        self.prepare_data_per_node = True

    def prepare_data(self) -> None:
        # Instantiate datasets to make sure they exists or to download them.
        CIFAR_10_Dataset(train=True, **self._dataset_kwargs)
        CIFAR_10_Dataset(train=False, **self._dataset_kwargs)

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit" or stage is None:
            train_set_full = CIFAR_10_Dataset(train=True, **self._dataset_kwargs)
            train_set_size = int(len(train_set_full) * 0.9)
            valid_set_size = len(train_set_full) - train_set_size
            self.train, self.validate = random_split(train_set_full, [train_set_size, valid_set_size])

        if stage == "test" or stage is None:
            self.test = CIFAR_10_Dataset(train=False, **self._dataset_kwargs)

    def train_dataloader(self):
        return DataLoader(self.train, shuffle=True, **self._dataloader_kwargs)

    def val_dataloader(self):
        return DataLoader(self.validate, **self._dataloader_kwargs)

    def test_dataloader(self):
        return DataLoader(self.test, **self._dataloader_kwargs)


def _tensor_to_image(
    x: torch.Tensor, scale: float = 1.0, mean: Optional[np.ndarray] = None, std: Optional[np.ndarray] = None
) -> PIL.Image.Image:
    """Converts a tensor of shape (C, H, W) to a pillow image, reversing normalization."""
    assert len(x.shape) == 3, f"Expected shape (C, H, W), got {x.shape}"
    C, H, W = x.shape
    assert C in [1, 3], f"Expected 1 or 3 channels, got shape {x.shape}"
    img = x.permute(1, 2, 0).numpy()  # shape (H, W, C)
    if mean is None or std is None:
        vmin, vmax = img.min(), img.max()
        img = (img - vmin) / (vmax - vmin)
    else:
        img = img * std + mean
    img = np.clip(img * 255, 0, 255).astype("uint8")
    pil_image = PIL.Image.fromarray(img, mode="RGB")
    pil_image.thumbnail((int(scale * W), int(scale * H)), PIL.Image.Resampling.NEAREST)
    return pil_image


def _draw_text(
    pil_image: PIL.Image.Image,
    text: Optional[str],
    font_size: int = 20,
    color: tuple[int, int, int] = (255, 255, 255),
    margin: int = 5,
) -> PIL.Image.Image:
    """Draws text on a pillow image."""
    if not text:
        return pil_image
    draw = PIL.ImageDraw.Draw(pil_image)
    draw.text((margin, margin), text, font_size=font_size, fill=color)
    return pil_image


def _make_image_grid(
    images: list[PIL.Image.Image],
    labels: Optional[list[str]] = None,
    colors: Optional[list[tuple[int, int, int]]] = None,
    n_columns: int = 8,
    font_size=19,
) -> PIL.Image.Image:
    """Arrange a list of images into a grid image."""
    B = len(images)
    n_rows = int(np.ceil(B / n_columns))
    # Print labels.
    if labels is None:
        labels = [""] * B
    else:
        assert len(labels) == B
    if colors is None:
        colors = [(255, 255, 255)] * B
    else:
        assert len(colors) == B
    images = [_draw_text(img, label, color=c, font_size=font_size) for img, c, label in zip(images, colors, labels)]

    # Paste onto grid.
    # Note that Pillow swaps H and W.
    scaledH, scaledW = max(img.size[1] for img in images), max(img.size[0] for img in images)
    grid = PIL.Image.new("RGB", (n_columns * scaledW, n_rows * scaledH))
    for i, im in enumerate(images):
        grid.paste(im, ((i % n_columns) * scaledW, (i // n_columns) * scaledH))
    return grid
