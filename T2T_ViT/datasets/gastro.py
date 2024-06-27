"""
Dataset for polyp detection in endoscopic images (mostly colonoscopy, not sure if gastroscopy too).

See:
- https://datasets.simula.no/kvasir-seg/
- https://datasets.simula.no/hyper-kvasir/ (for unlabeled images, which we used to find non-polyp images).
- https://datasets.simula.no/kvasir/ (for general documentation).
"""

import json
from pathlib import Path
from typing import Any, NamedTuple, cast

import torch
import torchvision.io
import torchvision.models
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
from torchvision import tv_tensors
from torchvision.io import ImageReadMode
from torchvision.transforms import v2

from datasets.transforms import Erase, default_transform
from datasets.types import ImageLabelBatch, ImageLabelDataitem

PROJECT_ROOT = Path(__file__).parent.parent  # Path to the T2T_ViT/ directory.


class _PreGastroDataitem(NamedTuple):
    image_path: Path
    label: int
    segmentation_path: Path | None
    bboxes: tv_tensors.BoundingBoxes | None


class GastroDataitem(ImageLabelDataitem):
    """Item from the GastroDataset."""

    segmentation: tv_tensors.Mask  # shape CHW, C=1, uint8 (0 - background, 1 - polyp).
    bboxes: tv_tensors.BoundingBoxes  # shape N4, int64, format XYXY.


class GastroBatch(ImageLabelBatch):
    """Batch of GastroDataitems (as collated by collate_gastro_batch)."""
    segmentation: torch.Tensor  # shape BCHW, C=1, uint8.
    bboxes: list[torch.Tensor]  # list of tensors of shape N4, int64, format XYXY.


class GastroDataset(Dataset[GastroDataitem]):
    """
    Dataset for polyp detection in endoscopic images.

    Contains a 1000 images with polyps (with segmentation and bounding boxes, sometimes more than one),
    and 1000 images without polyps (full-zero segmentation and zero bounding boxes).

    Label 0 means normal, 1 means polyp.

    Args:
    - root: path to a directory containing 'gastro-hyper-kvasir/'
    - transform: a torchvision.transforms.v2.Transform
        (a generic transform defined for images, masks, bboxes, possibly labels).
    """

    label_names = ["normal", "polyp"]
    name_to_label = {name: label for label, name in enumerate(label_names)}

    def __init__(self, root: str | Path, transform: v2.Transform | None = None, cropped: bool = False) -> None:
        self.root = Path(root).expanduser()
        self.transform = transform or self.default_transform()

        # Maps image names (without .jpg) to {height, width, bbox: [{label, xmin, ymin, xmax, ymax}, ...]}.
        with open(self.root / "gastro-hyper-kvasir/segmented-images/bounding-boxes.json") as f:
            bounding_boxes: dict[str, dict[str, Any]] = json.load(f)

        self.dataitems = list[_PreGastroDataitem]()

        # Add images with polyps.
        images_dir = self.root / "gastro-hyper-kvasir/segmented-images/images"
        for p in sorted((images_dir).glob("*.jpg")):
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
        assert len(self.dataitems) == 1000, f"Expected 1000 images in {images_dir}."

        # Add images without polyps.
        # changed path from images-images/images
        self.cropped = cropped
        unlabeled_dir = self.root / "gastro-hyper-kvasir/unlabeled-images-similar/images"
        if cropped:
            unlabeled_dir = self.root / "gastro-hyper-kvasir/unlabeled-images-similar-cropped/images"
        for p in sorted((unlabeled_dir).glob("*.jpg")):
            self.dataitems.append(
                _PreGastroDataitem(
                    image_path=p, label=self.name_to_label["normal"], segmentation_path=None, bboxes=None
                )
            )
        assert len(self.dataitems) == 2000, f"Expected 1000 images in {unlabeled_dir}."

    def __getitem__(self, index: int) -> GastroDataitem:
        d = self.dataitems[index]
        image = tv_tensors.Image(torchvision.io.read_image(str(d.image_path), mode=ImageReadMode.RGB))
        C, H, W = image.shape
        segmentation = tv_tensors.Mask(
            torchvision.io.read_image(str(d.segmentation_path), mode=ImageReadMode.GRAY)
            if d.segmentation_path
            else torch.zeros((1, H, W), dtype=torch.uint8)
        )
        bboxes = self._get_empty_bboxes_tensor(H, W) if d.bboxes is None else d.bboxes
        dataitem = GastroDataitem(image=image, label=d.label, segmentation=segmentation, bboxes=bboxes)
        return cast(GastroDataitem, self.transform(dataitem))

    def __len__(self) -> int:
        return len(self.dataitems)

    @staticmethod
    def default_transform(target_image_size: int = 224) -> v2.Transform:
        return v2.Compose(
            [
                default_transform(target_image_size),
                # Erase the green position visualization that appears on some images:
                Erase(i=target_image_size - 80, j=0, h=80, w=70, v=0),
            ]
        )

    @staticmethod
    def default_train_transform(target_image_size: int = 224) -> v2.Transform:
        return v2.Compose(
            [
                v2.Resize([2 * target_image_size], antialias=True),
                v2.CenterCrop(2 * target_image_size),
                # Erase the green position visualization that appears on some images:
                Erase(i=2 * (target_image_size - 80), j=0, h=2 * 80, w=2 * 70, v=0),
                v2.RandomResizedCrop([224], scale=(0.5, 1.0), ratio=(0.9, 1.1), antialias=True),
                v2.RandomHorizontalFlip(0.5),
                v2.RandomRotation(180),
                v2.RandomAutocontrast(p=0.5),
                v2.RandomEqualize(0.5),
                v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
                v2.RandomPosterize(bits=3, p=0.5),
                v2.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            ]
        )

    @staticmethod
    def _dict_to_bboxes_tensor(d: dict[str, Any]) -> tv_tensors.BoundingBoxes:
        H = d["height"]
        W = d["width"]
        bbox_list = []
        for bbox in d["bbox"]:
            assert bbox["label"] == "polyp"
            bbox_list.append((bbox["xmin"], bbox["ymin"], bbox["xmax"], bbox["ymax"]))
        bboxes = torch.tensor(bbox_list, dtype=torch.int32) if bbox_list else torch.empty(0, 4, dtype=torch.int32)
        return tv_tensors.BoundingBoxes(bboxes, format=tv_tensors.BoundingBoxFormat.XYXY, canvas_size=(H, W))

    @staticmethod
    def _get_empty_bboxes_tensor(image_height: int, image_width: int) -> tv_tensors.BoundingBoxes:
        return GastroDataset._dict_to_bboxes_tensor({"height": image_height, "width": image_width, "bbox": []})


def collate_gastro_batch(batch: list[GastroDataitem]) -> GastroBatch:
    """Function for collating a list of dataitems into a batch, for use in dataloaders as collate_fn."""
    return GastroBatch(
        image=default_collate([d["image"] for d in batch]),
        label=default_collate([d["label"] for d in batch]),
        segmentation=default_collate([d["segmentation"] for d in batch]),
        bboxes=[d["bboxes"] for d in batch],  # We could use torch.nested in the future, but it's experimental.
    )
