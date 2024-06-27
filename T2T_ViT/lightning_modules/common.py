from typing import Any, Iterable, Protocol

import torch.nn
import torch.utils.data
from torch import Tensor


def get_head_and_backbone_parameters(
    model: torch.nn.Module, head: torch.nn.Module
) -> tuple[Iterable[torch.nn.Parameter], Iterable[torch.nn.Parameter]]:
    """Get the parameters (disjoint) of the head and the backbone of a model."""
    head_parameters = list(head.parameters())
    head_parameter_ids = set(id(p) for p in head_parameters)
    backbone_parameters = [p for p in model.parameters() if id(p) not in head_parameter_ids]
    return head_parameters, backbone_parameters


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
