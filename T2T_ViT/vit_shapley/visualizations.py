import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import PIL.ImageDraw
import torch
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch import Tensor


CIFAR10_CLASSES = ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def to_image(
    x: Tensor,
    scale: float = 1.0,
    label: str | None = None,
    mean: tuple[float, float, float] = IMAGENET_DEFAULT_MEAN,
    std: tuple[float, float, float] = IMAGENET_DEFAULT_STD,
) -> PIL.Image.Image:
    """Converts a tensor of shape (C, H, W) to pillow image, reversing normalization."""
    image = _tensor_to_image(x, scale=scale, mean=mean, std=std)
    image = _draw_text(image, label, font_size=int(35 * scale))
    return image


def to_image_grid(
    x: Tensor,
    labels: list[str] | list[int] | Tensor | None = None,
    colors: list[tuple[int, int, int]] | None = None,
    scale: float = 1.0,
    n_columns: int = 8,
    classes: list[str] = CIFAR10_CLASSES,
    mean: tuple[float, float, float] = IMAGENET_DEFAULT_MEAN,
    std: tuple[float, float, float] = IMAGENET_DEFAULT_STD,
) -> PIL.Image.Image:
    """
    Converts a tensor of shape (B, C, H, W) to a grid of images.

    Args:
    - x: a batch of images as a tensor.
    """
    images = [to_image(x[i], scale=scale, mean=mean, std=std) for i in range(len(x))]
    str_labels = _labels_to_strings(labels, classes) if labels is not None else None
    return _make_image_grid(images, labels=str_labels, colors=colors, font_size=int(35 * scale), n_columns=n_columns)


def get_heatmaps_grid(
    images: torch.Tensor,
    labels: torch.Tensor,
    shap_values: torch.Tensor,
    scale: float = 0.5,
    alpha: float = 0.6,
    classes: list[str] = CIFAR10_CLASSES,
) -> PIL.Image.Image:
    heatmap_grid = _make_image_grid(_get_heatmaps(shap_values, images.shape, scale=scale))
    image_grid = to_image_grid(images, labels, scale=scale, classes=classes)
    return PIL.Image.blend(image_grid, heatmap_grid, alpha=alpha)


def _labels_to_strings(labels: list[str] | list[int] | Tensor, classes: list[str] = CIFAR10_CLASSES) -> list[str]:
    """Converts a list of labels to a list of strings."""
    labels = [label.item() if isinstance(label, Tensor) else label for label in labels]  # type: ignore
    return [label if isinstance(label, str) else classes[int(label)] for label in labels]


def _tensor_to_image(
    x: Tensor,
    scale: float = 1.0,
    mean: tuple[float, float, float] | None = None,
    std: tuple[float, float, float] | None = None,
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
        img = img * np.array(std) + np.array(mean)
    img = np.clip(img * 255, 0, 255).astype("uint8")
    pil_image: PIL.Image.Image = PIL.Image.fromarray(img, mode="RGB")
    pil_image = pil_image.resize((int(scale * W), int(scale * H)), PIL.Image.Resampling.NEAREST)
    return pil_image


def _draw_text(
    pil_image: PIL.Image.Image,
    text: str | None,
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
    labels: list[str] | None = None,
    colors: list[tuple[int, int, int]] | None = None,
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


def _get_heatmaps(
    values: Tensor,
    images_shape: tuple[int, ...],
    vmin: float | Tensor | None = None,
    vmax: float | Tensor | None = None,
    scale: float = 0.5,
) -> list[PIL.Image.Image]:
    """
    Args:
    - values: (B, num_players)
    - images_shape: (B, C, H, W)
    - vmin: value normalized to zero (default: min per image in batch).
    - vmax: value normalized to one (default: max per image in batch).
    """
    B, num_players = values.shape
    _B, C, H, W = images_shape
    assert B == _B and C == 3, f"Shape mismatch {values.shape=}, {images_shape=}"
    print(f"min={values.min().item():.3f}, max={values.max().item():.3f}")

    mask_H = int(np.round(np.sqrt(num_players)))
    mask_W = mask_H
    assert mask_H * mask_W == num_players, f"{num_players=}, expected a square number."
    values = values.view(B, mask_H, mask_W)

    # Upscale masks to image size.
    h_repeats, w_repeats = int(np.ceil(H / mask_H)), int(np.ceil(W / mask_W))
    values = values.repeat_interleave(h_repeats, dim=1).repeat_interleave(w_repeats, dim=2)
    values = values[:, :H, :W]

    # Normalize to [0, 1]
    if vmin is None:
        # Min per image in batch.
        vmin = values.min(dim=2, keepdim=True).values.min(dim=1, keepdim=True).values
    if vmax is None:
        # Max per image in batch.
        vmax = values.max(dim=2, keepdim=True).values.max(dim=1, keepdim=True).values
    values = (values - vmin) / (vmax - vmin + 1e-8)

    color_map = plt.get_cmap("viridis")
    values = torch.tensor(color_map(values))  # (B, H, W, 4)
    values = values[:, :, :, :3]  # Remove alpha channel
    values = values.permute(0, 3, 1, 2)  # (B, C, H, W)

    return [_tensor_to_image(v, scale=scale, mean=(0., 0., 0.), std=(1., 1., 1.)) for v in values]
