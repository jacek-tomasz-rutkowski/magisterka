from typing import Any, Literal, Optional
from pathlib import Path

import torch
import torchvision
import numpy as np
import pytorch_lightning as pl
import PIL.Image
import PIL.ImageDraw
from torch.utils.data import random_split, Dataset, DataLoader

from utils import PROJECT_ROOT
from vit_shapley.masks import generate_masks


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
        mode: Literal["uniform", "shapley"] = "uniform",
    ):
        """
        Args:
            root_path: Directory with all the images (will be downloaded if not present).
            train: whether to access train or test part of the dataset.
            download: whether to download the dataset if it is not present.
            num_players, num_mask_samples, paired_mask_samples, mode: see generate_mask().
        """
        self.num_players = num_players
        self.num_mask_samples = num_mask_samples
        self.paired_mask_samples = paired_mask_samples
        self.root_path = root_path
        self.shape = (224, 224, 3)
        self.mode = mode
        transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(self.shape[:2], torchvision.transforms.InterpolationMode.BILINEAR),
                torchvision.transforms.ToTensor(),
                self.normalization,
            ]
        )
        self.dataset = torchvision.datasets.CIFAR10(root=root_path, train=train, download=download, transform=transform)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        image, label = self.dataset[idx]
        masks = generate_masks(
            num_players=self.num_players,
            num_mask_samples=self.num_mask_samples,
            paired_mask_samples=self.paired_mask_samples,
            mode=self.mode
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
        return [label if isinstance(label, str) else cls.classes[int(label)] for label in labels]

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
        train_mode: Literal["uniform", "shapley"] = "uniform",
        val_mode: Literal["uniform", "shapley"] = "uniform",
        test_mode: Literal["uniform", "shapley"] = "uniform",
    ):
        super().__init__()
        self.num_players = num_players
        self.num_mask_samples = num_mask_samples
        self.paired_mask_samples = paired_mask_samples
        self._dataset_kwargs: dict[str, Any] = dict(
            num_players=num_players,
            num_mask_samples=num_mask_samples,
            paired_mask_samples=paired_mask_samples,
            root_path=PROJECT_ROOT / "data",
            download=False,
        )
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_mode = train_mode
        self.val_mode = val_mode
        self.test_mode = test_mode
        self._dataloader_kwargs: dict[str, Any] = dict(batch_size=batch_size, num_workers=num_workers)
        self.prepare_data_per_node = True

    def prepare_data(self) -> None:
        # Instantiate datasets to make sure they exists or to download them.
        CIFAR_10_Dataset(train=True, mode=self.train_mode, **self._dataset_kwargs)
        CIFAR_10_Dataset(train=False, mode=self.test_mode, **self._dataset_kwargs)

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit" or stage is None:
            train_set_full = CIFAR_10_Dataset(train=True, mode=self.train_mode, **self._dataset_kwargs)
            train_set_size = int(len(train_set_full) * 0.9)
            valid_set_size = len(train_set_full) - train_set_size
            self.train, self.validate = random_split(train_set_full, [train_set_size, valid_set_size])
            # Replace self.validate to be the same subset of indices, but use a Dataset with mode=val_mode.
            val_set_full = CIFAR_10_Dataset(train=True, mode=self.val_mode, **self._dataset_kwargs)
            self.validate.dataset = val_set_full

        if stage == "test" or stage is None:
            self.test = CIFAR_10_Dataset(train=False, mode=self.test_mode, **self._dataset_kwargs)

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
    img = x.permute(1, 2, 0).cpu().numpy()  # shape (H, W, C)
    if mean is None or std is None:
        vmin, vmax = img.min(), img.max()
        img = (img - vmin) / (vmax - vmin + 1e-10)
    else:
        img = img * std + mean
    img = np.clip(img * 255, 0, 255).astype("uint8")
    pil_image: PIL.Image.Image = PIL.Image.fromarray(img, mode="RGB")
    pil_image = pil_image.resize((int(scale * W), int(scale * H)), PIL.Image.Resampling.NEAREST)
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
    font_size: int = 19,
    border: int = 1,
    border_color: tuple[int, int, int] = (0, 0, 0),
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
    H, W = max(img.size[1] for img in images), max(img.size[0] for img in images)
    H, W = H + border, W + border
    grid = PIL.Image.new("RGB", (n_columns * W, n_rows * H), color=border_color)
    for i, im in enumerate(images):
        grid.paste(im, ((i % n_columns) * W, (i // n_columns) * H))
    return grid
