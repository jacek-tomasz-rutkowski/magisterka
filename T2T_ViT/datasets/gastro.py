"""
Dataset for polyp detection in endoscopic images (mostly colonoscopy, not sure if gastroscopy too).

See:
- https://datasets.simula.no/kvasir-seg/
- https://datasets.simula.no/hyper-kvasir/ (for unlabeled images, which we used to find non-polyp images).
- https://datasets.simula.no/kvasir/ (for general documentation).
"""

import json
from pathlib import Path
from typing import Any, NamedTuple, Optional

import torch
import torchvision.io
import torchvision.models
from torch.utils.data.dataloader import default_collate
from torch.utils.data import random_split, Dataset, DataLoader

from torchvision import tv_tensors
from torchvision.transforms import v2
from torchvision.io import ImageReadMode
import pytorch_lightning as pl
from datasets.CIFAR_10_Dataset import CIFAR_10_Dataset
generate_masks = CIFAR_10_Dataset.generate_masks

PROJECT_ROOT = Path(__file__).parent.parent  # Path to the T2T_ViT/ directory.


class _PreGastroDataitem(NamedTuple):
    image_path: Path
    label: int
    segmentation_path: Path | None
    bboxes: tv_tensors.BoundingBoxes | None


class GastroDataitem(NamedTuple):
    """Item from the GastroDataset."""

    image: tv_tensors.Image  # shape CHW (RGB, before transform uint8, after float32).
    label: int  # 0 - normal, 1 - polyp.
    segmentation: tv_tensors.Mask  # shape CHW, C=1, uint8 (0 - background, 1 - polyp).
    bboxes: tv_tensors.BoundingBoxes  # shape N4, int64, format XYXY.


class GastroBatch(NamedTuple):
    """Batch of GastroDataitems (as collated by collate_gastro_batch)."""

    image: torch.Tensor  # shape BCHW, RGB, normalized float32.
    label: torch.Tensor  # shape B, int64
    segmentation: torch.Tensor  # shape BCHW, C=1, uint8.
    bboxes: list[torch.Tensor]  # list of tensors of shape N4, int64, format XYXY.


class GastroDataset(Dataset):
    """
    Dataset for polyp detection in endoscopic images.

    Contains a 1000 images with polyps (with segmentation and bounding boxes, sometimes more than one),
    and 1000 images without polyps (full-zero segmentation and zero bounding boxes).

    Args:
    - root: path to a directory containing 'gastro-hyper-kvasir/'
    - transform: a torchvision.transforms.v2.Transform
        (a generic transform defined for images, masks, bboxes, possibly labels).
    """

    label_names = ["normal", "polyp"]
    name_to_label = {name: label for label, name in enumerate(label_names)}

    def __init__(self, root: str | Path, transform: v2.Transform | None = None) -> None:
        self.root = Path(root).expanduser()
        self.transform = transform or self.default_transform()

        # Maps image names (without .jpg) to {height, width, bbox: [{label, xmin, ymin, xmax, ymax}, ...]}.
        with open(self.root / "gastro-hyper-kvasir/segmented-images/bounding-boxes.json") as f:
            bounding_boxes: dict[str, dict[str, Any]] = json.load(f)

        self.dataitems = list[_PreGastroDataitem]()

        # Add images with polyps.
        for p in sorted((self.root / "gastro-hyper-kvasir/segmented-images/images").glob("*.jpg")):
            segmentation_p = self.root / "gastro-hyper-kvasir/segmented-images/masks" / p.name
            assert segmentation_p.exists(), f"Mask not found for {p.name}"
            self.dataitems.append(
                _PreGastroDataitem(
                    image_path=p,
                    label=self.name_to_label["polyp"],
                    segmentation_path=segmentation_p,
                    bboxes=self._dict_to_bboxes_tensor(bounding_boxes[p.stem]),
                )
            )

        # Add images without polyps.
        for p in sorted((self.root / "gastro-hyper-kvasir/images-images/images").glob("*.jpg")):
            self.dataitems.append(
                _PreGastroDataitem(
                    image_path=p, label=self.name_to_label["normal"], segmentation_path=None, bboxes=None
                )
            )

    def __getitem__(self, index: int) -> GastroDataitem:
        d = self.dataitems[index]
        image = tv_tensors.Image(torchvision.io.read_image(str(d.image_path), mode=ImageReadMode.RGB))
        H, W = image.shape[-2], image.shape[-1]
        segmentation = (
            tv_tensors.Mask(torchvision.io.read_image(str(d.segmentation_path), mode=ImageReadMode.GRAY))
            if d.segmentation_path
            else torch.zeros_like(image)
        )
        bboxes = self._get_empty_bboxes_tensor(H, W) if d.bboxes is None else d.bboxes
        dataitem = GastroDataitem(image=image, label=d.label, segmentation=segmentation, bboxes=bboxes)
        dataitem = self.transform(dataitem)
        return dataitem

    def __len__(self) -> int:
        return len(self.dataitems)

    def default_transform(self, target_image_size: int = 224) -> v2.Transform:
        # Essentially the same as torchvision.models.ResNet18_Weights.IMAGENET1K_V1.transforms(antialias=True).
        return v2.Compose(
            [
                v2.Resize([target_image_size], antialias=True),  # Resize so that min dimension is image_size.
                v2.CenterCrop(target_image_size),  # Crop rectangle centrally to square.
                v2.ToDtype(torch.float32, scale=True),  # Convert to float and scale to 0..1.
                # Erase the green position visualization that appears on some images:
                Erase(i=target_image_size - 80, j=0, h=80, w=70, v=0),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def default_train_transform(self, target_image_size: int = 224) -> v2.Transform:
        return v2.Compose(
            [
                v2.Resize([2 * target_image_size], antialias=True),
                v2.CenterCrop(2 * target_image_size),
                # Erase the green position visualization that appears on some images:
                Erase(i=2 * (target_image_size - 80), j=0, h=2 * 80, w=2 * 70, v=0),
                v2.RandomResizedCrop([224], scale=(0.5, 1.0), ratio=(0.9, 1.1), antialias=True),
                # TakeRandomCorner((224, 224), (120, 120), ["top_left", "bottom_right"]),
                v2.RandomHorizontalFlip(0.5),
                v2.RandomRotation(180),
                v2.RandomAutocontrast(p=0.5),
                v2.RandomEqualize(0.5),
                v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
                v2.RandomPosterize(bits=3, p=0.5),
                v2.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    @staticmethod
    def _dict_to_bboxes_tensor(d: dict[str, Any]) -> tv_tensors.BoundingBoxes:
        H = d["height"]
        W = d["width"]
        bboxes = []
        for bbox in d["bbox"]:
            assert bbox["label"] == "polyp"
            bboxes.append((bbox["xmin"], bbox["ymin"], bbox["xmax"], bbox["ymax"]))
        return tv_tensors.BoundingBoxes(bboxes, format=tv_tensors.BoundingBoxFormat.XYXY, canvas_size=(H, W))

    @staticmethod
    def _get_empty_bboxes_tensor(image_height: int, image_width: int) -> tv_tensors.BoundingBoxes:
        return GastroDataset._dict_to_bboxes_tensor({"height": image_height, "width": image_width, "bbox": []})


class Erase(v2.Transform):
    """
    Torchvision transform that erases a rectangle from the image (leaves masks and bboxes unchanged).

    Args:
    - i, j: top-left corner of the rectangle.
    - h, w: height and width of the rectangle.
    - v: value to fill the rectangle with.
    """

    def __init__(self, i: int, j: int, h: int, w: int, v: int):
        super().__init__()
        self.i, self.j, self.h, self.w, self.v = i, j, h, w, v

    def _transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        if not isinstance(inpt, tv_tensors.Image):
            return inpt
        return self._call_kernel(v2.functional.erase, inpt, **params, i=self.i, j=self.j, h=self.h, w=self.w, v=self.v)


class TakeRandomCorner(v2.Transform):
    """
    Torchvision transform that takes a random corner of the image (crops masks and bboxes accordingly).

    Args:
    - image_size: input image height and width (a single int means square images).
    - size: output image height and width (a single int means square crops).
    - corners: list of strings, each being one of "top_left", "top_right", "bottom_left", "bottom_right".
        Each call to the transform will randomly choose one of these corners.
    """

    def __init__(self, image_size: tuple[int, int] | int, size: tuple[int, int] | int, corners: list[str]):
        super().__init__()
        self.image_size = image_size if isinstance(image_size, tuple) else (image_size, image_size)
        self.size = size if isinstance(size, tuple) else (size, size)
        self.croppers = [self._get_corner_croppers(c) for c in corners]

    def _transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        cropper_id = int(torch.randint(len(self.croppers), size=(1,)).item())
        cropper = self.croppers[cropper_id]
        return self._call_kernel(v2.functional.crop, inpt, **params, **cropper)

    def _get_corner_croppers(self, corner: str) -> dict[str, int]:
        if corner == "top_left":
            top, left = 0, 0
        elif corner == "top_right":
            top, left = 0, self.image_size[1] - self.size[1]
        elif corner == "bottom_left":
            top, left = self.image_size[0] - self.size[0], 0
        elif corner == "bottom_right":
            top, left = self.image_size[0] - self.size[0], self.image_size[1] - self.size[1]
        else:
            raise ValueError(f"Invalid corner: {corner}")
        return dict(top=top, left=left, height=self.size[0], width=self.size[1])


def collate_gastro_batch(batch: list[GastroDataitem]) -> GastroBatch:
    """Function for collating a list of dataitems into a batch, for use in dataloaders as collate_fn."""
    return GastroBatch(
        image=default_collate([d.image for d in batch]),
        label=default_collate([d.label for d in batch]),
        segmentation=default_collate([d.segmentation for d in batch]),
        bboxes=[d.bboxes for d in batch],  # We could use torch.nested in the future, but it's experimental.
    )
    # Alternatively, a more generic but less readable implementation is:
    # ```
    # from torch.utils.data._utils.collate import collate, default_collate_fn_map  # Private, but officially documented.
    #
    # def collate_bboxes(batch, *, collate_fn_map=None):
    #     return batch
    #
    # collate_fn = partial(collate, collate_fn_map=default_collate_fn_map | {tv_tensors.BoundingBoxes: collate_bboxes})
    # ```

class GastroDatasetWrapper(Dataset):
    """
    Wrapper dataset for polyp detection in endoscopic images.
    """

    def __init__(self, root: str | Path, 
                num_players: int,
                num_mask_samples: int,
                paired_mask_samples: bool,
                root_path: Path,
                mode: str = "uniform",
                transform: v2.Transform | None = None,) -> None:
        
        self.wrapped_dataset = GastroDataset(root_path, transform)
        self.num_players = num_players
        self.num_mask_samples = num_mask_samples
        self.paired_mask_samples = paired_mask_samples
        self.root_path = root_path
        self.mode = mode
  

    def __getitem__(self, index: int) -> GastroDataitem:
        image, label, segmentation, bboxes = self.wrapped_dataset[index]

        masks = generate_masks(
            num_players=self.num_players,
            num_mask_samples=self.num_mask_samples,
            paired_mask_samples=self.paired_mask_samples,
            mode=self.mode
        )
        
        return {"images": image, "labels": label, "masks": masks,
                "segmentation": segmentation, "bboxes": bboxes}


class Gastro_Datamodule(pl.LightningDataModule):
    classes = GastroDataset.label_names

    def __init__(
        self,
        num_players: int,
        num_mask_samples: int,
        paired_mask_samples: bool,
        batch_size: int = 32,
        num_workers: int = 0,
        train_mode: str = "shapley",
        val_mode: str = "uniform",
        test_mode: str = "uniform",
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
        )
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_mode = train_mode
        self.val_mode = val_mode
        self.test_mode = test_mode
        self._dataloader_kwargs: dict[str, Any] = dict(batch_size=batch_size, 
                                                       num_workers=num_workers, 
                                                       collate_fn=collate_gastro_batch)


    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit" or stage is None:
            train_set_full = GastroDatasetWrapper(mode=self.train_mode, **self._dataset_kwargs)
            train_set_size = int(len(train_set_full) * 0.9)
            valid_set_size = len(train_set_full) - train_set_size
            self.train, self.validate = random_split(train_set_full, [train_set_size, valid_set_size])
            # Replace self.validate to be the same subset of indices, but use a Dataset with mode=val_mode.
            val_set_full = GastroDatasetWrapper(mode=self.val_mode, **self._dataset_kwargs)
            self.validate.dataset = val_set_full

        if stage == "test" or stage is None:
            self.test = GastroDatasetWrapper(mode=self.test_mode, **self._dataset_kwargs)

    def train_dataloader(self):
        return DataLoader(self.train, shuffle=True, **self._dataloader_kwargs)

    def val_dataloader(self):
        return DataLoader(self.validate, **self._dataloader_kwargs)

    def test_dataloader(self):
        return DataLoader(self.test, **self._dataloader_kwargs)
    
