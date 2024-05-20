
from typing import cast, Any, Sized, TypedDict, TypeVar

import torch
import torch.utils.data
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision import tv_tensors
from torchvision.transforms import v2


def default_transform(target_image_size: int = 224) -> v2.Transform:
    """Default transform for image classification (as typically taken for ImageNet).

    Essentially the same as torchvision.models.ResNet18_Weights.IMAGENET1K_V1.transforms(antialias=True).
    or timm.data.create_transform(crop_pct=1).
    """
    return v2.Compose(
        [
            v2.Resize([target_image_size], antialias=True),  # Resize so that min dimension is image_size.
            v2.CenterCrop(target_image_size),  # Crop rectangle centrally to square.
            v2.ToDtype(torch.float32, scale=True),  # Convert to float and scale to 0..1.
            v2.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        ]
    )


class ImageLabelDataitem(TypedDict):
    image: tv_tensors.Image  # Before transform: shape CHW, dtype uint8.
    label: int


T_co = TypeVar('T_co', covariant=True)


class SizedDataset(torch.utils.data.Dataset[T_co], Sized):
    pass


class ImageClassificationDatasetWrapper(SizedDataset[ImageLabelDataitem]):
    """
    Turns a dataset that yields (image, label) tuples into one that yields ImageLabelDataitem dicts.

    Images can be one of the following types:
    - torch.Tensor (shape CHW, dtype uint8),
    - PIL Image,
    - np.ndarray (shape HWC, uint8)
    """
    def __init__(self, dataset: SizedDataset[tuple[Any, int]], transform: v2.Transform):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index: int) -> ImageLabelDataitem:
        image, label = self.dataset[index]
        dataitem = ImageLabelDataitem(image=v2.functional.to_image(image), label=label)
        return cast(ImageLabelDataitem, self.transform(dataitem))

    def __len__(self) -> int:
        return len(self.dataset)


class Erase(v2.Transform):
    """
    Torchvision transform that erases a fixed rectangle from the image (leaves masks and bboxes unchanged).

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
