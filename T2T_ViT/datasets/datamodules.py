from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Generic, Literal, Sequence, Sized, Type, TypedDict, TypeVar

import lightning as L
import torch
import torch.nn
import torch.utils.data
import torchvision.datasets
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch import Tensor
from torch.utils.data import Subset
from torch.utils.data._utils.collate import collate, default_collate_fn_map  # Private, but officially documented.
from torchvision import tv_tensors
from torchvision.transforms import v2

from datasets.gastro import GastroBatch, GastroDataitem, GastroDataset
from datasets.transforms import default_transform
from datasets.types import ImageLabelDataitem, ImageLabelBatch, ImageLabelMaskDataitem, ImageLabelMaskBatch
from utils import PROJECT_ROOT, random_split_sequence
from vit_shapley.masks import generate_masks


S_co = TypeVar("S_co", covariant=True)
T_co = TypeVar("T_co", covariant=True)
T_batch = TypeVar("T_batch")


class SizedDataset(torch.utils.data.Dataset[T_co], Sized):
    """A dataset that supports `len()`."""

    pass


class TransformedDataset(SizedDataset[T_co]):
    """
    Applies a transform to dataitems produced by any dataset.

    This is useful e.g. for datasets that don't allow transforming the whole dataitems, just images.
    """

    def __init__(self, dataset: SizedDataset[S_co] | Sequence[S_co], transform: Callable[[S_co], T_co]):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index: int) -> T_co:
        return self.transform(self.dataset[index])

    def __len__(self) -> int:
        return self.dataset.__len__()


DataLoaderKwargs = dict[str, Any]  # TypedDicts are not supported by jsonargparse<=4.28.0
# class DataLoaderKwargs(TypedDict, total=False):
#     """Kwargs for torch DataLoader (all optional)."""

#     batch_size: int
#     pin_memory: bool
#     prefetch_factor: int
#     persistent_workers: bool
#     pin_memory_device: str
#     drop_last: bool


class BaseDataModule(L.LightningDataModule, Generic[T_co, T_batch]):
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

    def train_dataloader(self) -> torch.utils.data.DataLoader[T_batch]:
        return torch.utils.data.DataLoader(self.train_dataset, **self.dataloader_kwargs, shuffle=True)

    def val_dataloader(self) -> torch.utils.data.DataLoader[T_batch]:
        return torch.utils.data.DataLoader(self.val_dataset, **self.dataloader_kwargs)

    def test_dataloader(self) -> torch.utils.data.DataLoader[T_batch]:
        return torch.utils.data.DataLoader(self.test_dataset, **self.dataloader_kwargs)


class GastroDataModule(BaseDataModule[GastroDataitem, GastroBatch]):
    """
    Lightning DataModule for training a classifier on the Gastro dataset.

    Args:
    - root: directory containing "gastro-hyper-kvasir/".
    - dataloader_kwargs: passed to DataLoader, like batch_size=1, num_workers=0, pin_memory=False.
    """

    num_classes: int = 2

    def __init__(self, root: Path = PROJECT_ROOT / "data", *, dataloader_kwargs: DataLoaderKwargs = {}):
        super().__init__(dataloader_kwargs)
        self.root = root

    def prepare_data(self) -> None:
        GastroDataset(self.root)

    def setup(self, stage: str) -> None:
        assert stage in ["fit", "validate", "test", "predict"]
        generator = torch.Generator().manual_seed(42)  # Fix a train-val split.
        train_ids, val_ids = random_split_sequence(range(len(GastroDataset(self.root))), [0.7, 0.3], generator)

        train_transform = GastroDataset.default_train_transform()
        val_transform = GastroDataset.default_transform()

        self.train_dataset = Subset(GastroDataset(self.root, train_transform), train_ids)  # type: ignore
        self.val_dataset = Subset(GastroDataset(self.root, val_transform), val_ids)  # type: ignore
        self.test_dataset = self.val_dataset


class CIFAR10DataModule(BaseDataModule[ImageLabelDataitem, ImageLabelBatch]):
    """
    Lightning DataModule for training a classifier on the CIFAR10 dataset.

    Args:
    - root: directory containing "cifar-10-batches-py/".
    - dataloader_kwargs: passed to DataLoader, like batch_size=1, num_workers=0, pin_memory=False.
    """

    num_classes: int = 10

    def __init__(self, root: Path = PROJECT_ROOT / "data", *, dataloader_kwargs: DataLoaderKwargs = {}):
        super().__init__(dataloader_kwargs)
        self.root = root

    def prepare_data(self) -> None:
        download = False
        torchvision.datasets.CIFAR10(root=self.root, train=True, download=download)
        torchvision.datasets.CIFAR10(root=self.root, train=False, download=download)

    def setup(self, stage: str) -> None:
        assert stage in ["fit", "validate", "test", "predict"]

        target_size = 224
        val_transform = default_transform(target_size)
        train_transform = v2.Compose(
            [
                v2.RandomRotation(10),
                v2.RandomResizedCrop(target_size, scale=(0.8, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0), antialias=True),
                v2.RandomHorizontalFlip(),
                v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            ]
        )

        if stage == "fit":
            self.train_dataset = self._dataset(train=True, transform=train_transform)
        if stage == "fit" or stage == "validate":
            self.val_dataset = self._dataset(train=False, transform=val_transform)
        if stage == "test":
            self.test_dataset = self._dataset(train=False, transform=val_transform)

    def _dataset(self, train: bool, transform: v2.Transform) -> SizedDataset[ImageLabelDataitem]:
        """Make a CIFAR10Dataset that yields ImageLabelDataitem-s."""
        return TransformedDataset(
            torchvision.datasets.CIFAR10(root=self.root, train=train, download=False),
            transform=lambda tpl: transform(ImageLabelDataitem(image=v2.functional.to_image(tpl[0]), label=tpl[1])),
        )


@dataclass
class GenerateMasksKwargs:  # TypedDicts are not supported by jsonargparse<=4.28.0
    num_players: int
    num_mask_samples: int
    paired_mask_samples: bool
    mode: Literal["uniform", "shapley"]


class DataModuleWithMasks(BaseDataModule[ImageLabelMaskDataitem, ImageLabelMaskBatch]):
    """
    Lightning DataModule for training a Surrogate model.

    Args:
    - wrapped_datamodule: the classifier CIFAR10DataModule or GastroDataModule.
    - generate_masks_kwargs: passed to vit_shapley.masks.generate_masks(),
        like num_players=16, num_mask_samples=2, paired_mask_samples=True, mode="shapley".
        This is only used for training; for validation and testing these are fixed to make scores comparable.
    - dataloader_kwargs: passed to DataLoader, like batch_size=1, num_workers=0, pin_memory=False.
    """
    def __init__(
        self,
        wrapped_datamodule: CIFAR10DataModule | GastroDataModule,
        generate_masks_kwargs: GenerateMasksKwargs,
        dataloader_kwargs: DataLoaderKwargs,
    ):
        super().__init__(dataloader_kwargs=dataloader_kwargs)
        self.wrapped_datamodule = wrapped_datamodule
        self.num_classes = self.wrapped_datamodule.num_classes
        self.generate_masks_kwargs_train = generate_masks_kwargs
        # Keep mask generation for validation and testing fixed, to make scores comparable.
        self.generate_masks_kwargs_val = GenerateMasksKwargs(
            num_players=generate_masks_kwargs.num_players,
            num_mask_samples=2,
            paired_mask_samples=True,
            mode="uniform",
        )

    def prepare_data(self) -> None:
        self.wrapped_datamodule.prepare_data()

    def setup(self, stage: str) -> None:
        self.wrapped_datamodule.setup(stage)
        if stage == "fit":
            self.train_dataset = TransformedDataset(self.wrapped_datamodule.train_dataset, self.add_random_masks_train)
        if stage == "fit" or stage == "validate":
            self.val_dataset = TransformedDataset(self.wrapped_datamodule.val_dataset, self.add_random_masks_val)
        if stage == "test":
            self.test_dataset = TransformedDataset(self.wrapped_datamodule.test_dataset, self.add_random_masks_val)

    def add_random_masks_train(self, dataitem: ImageLabelDataitem) -> ImageLabelMaskDataitem:
        return {**dataitem, "mask": Tensor(generate_masks(**asdict(self.generate_masks_kwargs_train)))}

    def add_random_masks_val(self, dataitem: ImageLabelDataitem) -> ImageLabelMaskDataitem:
        return {**dataitem, "mask": Tensor(generate_masks(**asdict(self.generate_masks_kwargs_val)))}


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
