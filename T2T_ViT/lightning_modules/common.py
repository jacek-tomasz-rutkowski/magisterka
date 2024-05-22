from typing import Any, Callable, Generic, Iterable, Protocol, Sequence, Sized, Type, TypedDict, TypeVar

import lightning as L
import torch.nn
import torch.utils.data
from torch import Tensor
from torch.utils.data._utils.collate import collate, default_collate_fn_map  # Private, but officially documented.
from torchvision import tv_tensors


class ImageLabelDataitem(TypedDict):
    image: tv_tensors.Image  # Shape CHW (RGB, float32 after default transform, uint8 before).
    label: int


class ImageLabelBatch(TypedDict):
    image: tv_tensors.Image  # Shape BCHW, RGB, dtype float32.
    label: torch.Tensor  # Shape B, dtype int64.


class ImageLabelMaskDataitem(ImageLabelDataitem):
    mask: torch.Tensor  # Shape (num_players,), dtype bool.


S_co = TypeVar("S_co", covariant=True)
T_co = TypeVar("T_co", covariant=True)


class SizedDataset(torch.utils.data.Dataset[T_co], Sized):
    pass


class TransformedDataset(SizedDataset[T_co]):
    """Applies a transform to dataitems produced by any dataset."""

    def __init__(self, dataset: SizedDataset[S_co] | Sequence[S_co], transform: Callable[[S_co], T_co]):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index: int) -> T_co:
        return self.transform(self.dataset[index])

    def __len__(self) -> int:
        return self.dataset.__len__()


def get_head_and_backbone_parameters(
    model: torch.nn.Module, head: torch.nn.Module
) -> tuple[Iterable[torch.nn.Parameter], Iterable[torch.nn.Parameter]]:
    """Get the parameters (disjoint) of the head and the backbone of a model."""
    head_parameters = head.parameters()
    head_parameter_ids = set(id(p) for p in head_parameters)
    backbone_parameters = [p for p in model.parameters() if id(p) not in head_parameter_ids]
    return head_parameters, backbone_parameters


DataLoaderKwargs = dict[str, Any]  # TypedDicts are not supported by jsonargparse<=4.28.0
# class DataLoaderKwargs(TypedDict, total=False):
#     """Kwargs for torch DataLoader (all optional)."""

#     batch_size: int
#     pin_memory: bool
#     prefetch_factor: int
#     persistent_workers: bool
#     pin_memory_device: str
#     drop_last: bool


class BaseDataModule(L.LightningDataModule, Generic[T_co]):
    """
    A DataModule base class with the typical train/val/test dataloaders.

    The setup() method needs to be defined to initialize self.train_dataset, val_dataset and test_dataset.

    The default collating function (which turns dataitems into batches) is replaced to handle bounding boxes correctly.

    Args:
    - dataloader_kwargs: passed to torch.utils.data.DataLoader().
        Common/default args are batch_size=1, num_workers=0, pin_memory=False.
    """

    num_classes: int

    def __init__(self, dataloader_kwargs: DataLoaderKwargs = {}):
        super().__init__()
        self.dataloader_kwargs: DataLoaderKwargs = {"collate_fn": better_collate_fn, **dataloader_kwargs}
        # These should be initialized in setup().
        self.train_dataset: SizedDataset[T_co]
        self.val_dataset: SizedDataset[T_co]
        self.test_dataset: SizedDataset[T_co]

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(self.train_dataset, **self.dataloader_kwargs, shuffle=True)

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(self.val_dataset, **self.dataloader_kwargs)

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(self.test_dataset, **self.dataloader_kwargs)


class _TimmModelProtocol(Protocol):
    pretrained_cfg: dict[str, Any]

    def forward(self, x: Tensor) -> Tensor: ...
    def __call__(self, x: Tensor) -> Tensor: ...
    def reset_classifier(self, num_classes: int) -> None: ...
    def get_classifier(self) -> torch.nn.Module: ...


class TimmModel(_TimmModelProtocol, torch.nn.Module):
    """Base type for timm models.

    For example:
    - timm.models.vision_transformer.VisionTransformer
    - timm.models.swin_transformer.SwinTransformer
    """

    pass


def collate_bboxes(batch: list[Tensor], *, collate_fn_map=None) -> list[Tensor]:
    """
    To collate a list of bounding-box tensors into a batch, just keep it as a list of tensors.

    Stacking into one tensor is impossible, as numbers of bounding boxes vary.
    We could use torch.nested in the future, but it's currently experimental.
    """
    return batch


better_collate_fn_map: dict[Type | tuple[Type, ...], Callable] = {  # Map from types to collating functions for them.
    **default_collate_fn_map,
    tv_tensors.BoundingBoxes: collate_bboxes,
}


def better_collate_fn(batch: list[Any]) -> Any:
    return collate(batch, collate_fn_map=better_collate_fn_map)
