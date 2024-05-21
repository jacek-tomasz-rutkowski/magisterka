from pathlib import Path
from typing import Any, Protocol, cast

import lightning as L
import timm
import timm.optim
import timm.scheduler
import timm.scheduler.scheduler
import torch
import torch.nn
import torch.utils.data
import torchvision.datasets
from lightning.pytorch.utilities.types import LRSchedulerConfigType, OptimizerLRSchedulerConfig
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch import Tensor
from torchvision.transforms import v2

from datasets.gastro import GastroBatch, GastroDataset, collate_gastro_batch
from datasets.transforms import ImageClassificationDatasetWrapper, default_transform
from utils import PROJECT_ROOT
from .cli import lightning_main


class _TimmModelProtocol(Protocol):
    pretrained_cfg: dict[str, Any]

    def forward(self, x: Tensor) -> Tensor: ...
    def __call__(self, x: Tensor) -> Tensor: ...
    def reset_classifier(self, num_classes: int) -> None: ...
    def get_classifier(self) -> torch.nn.Module: ...


class TimmModel(_TimmModelProtocol, torch.nn.Module):
    """Base type for timm models."""

    pass
    # For example:
    # - timm.models.vision_transformer.VisionTransformer
    # - timm.models.swin_transformer.SwinTransformer


class Classifier(L.LightningModule):
    def __init__(
        self,
        backbone: str,
        num_classes: int,
        optimizer_kwargs: dict[str, Any] = dict(),
        scheduler_kwargs: dict[str, Any] = dict(),
        scriptable: bool = False,
    ):
        """
        Args:
        - backbone: The name of the timm model to use.
        - num_classes: The number of classes in the dataset.
        - optimizer_kwargs: passed to timm.optim.create_optimizer_v2()
            Common/default args are opt='sgd' (or e.g. 'adamw'), lr=None (?), weight_decay=0, momentum=0.9.
            See: https://huggingface.co/docs/timm/reference/optimizers#timm.optim.create_optimizer_v2
            Note sgd means SGD with Nesterov momentum in timm (use "momentum" to disable Nesterov).
        - scheduler_kwargs: passed to timm.scheduler.create_scheduler_v2()
            Common/default args are sched='cosine (or e.g. ...), num_epochs=300, decay_epochs=90,
                decay_milestones=[90, 180, 270], cooldown_epoch=0, patience_epochs=10, decay_rate=0.1,
                min_lr=0, warmup_lr=1e-05, warmup_epochs=0.
            See: https://huggingface.co/docs/timm/reference/schedulers#timm.scheduler.create_scheduler_v2
        """
        super().__init__()
        self.save_hyperparameters()
        self.model = cast(TimmModel, timm.create_model(backbone, pretrained=True, scriptable=scriptable))
        for k in ["input_size", "interpolation", "mean", "std"]:
            print(f"\t{k}:\t{self.model.pretrained_cfg[k]}")
        self.model.reset_classifier(num_classes=num_classes)

        if scriptable:
            self.model = torch.jit.script(self.model)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def training_step(self, batch: GastroBatch, batch_idx: int) -> Tensor:
        return self._common_step(batch, batch_idx, "train")

    def validation_step(self, batch: GastroBatch, batch_idx: int) -> None:
        self._common_step(batch, batch_idx, "val")

    def test_step(self, batch: GastroBatch, batch_idx: int) -> None:
        self._common_step(batch, batch_idx, "test")

    def _common_step(self, batch: GastroBatch, batch_idx: int, phase: str) -> Tensor:
        images, labels = batch["image"], batch["label"]
        batch_size = len(images)
        logits = self.forward(images)
        loss = cast(Tensor, torch.nn.CrossEntropyLoss()(logits, labels))
        accuracy = (logits.argmax(dim=-1) == labels).float().mean()
        self.log(f"{phase}/accuracy", accuracy, prog_bar=True, batch_size=batch_size)
        self.log(f"{phase}/loss", loss, prog_bar=True, batch_size=batch_size)
        return loss

    def configure_optimizers(self) -> OptimizerLRSchedulerConfig:
        optimizer_kwargs = dict(self.hparams["optimizer_kwargs"])
        lr = optimizer_kwargs.pop("lr")
        lr_head = optimizer_kwargs.pop("lr_head", lr)

        head_parameters = self.model.get_classifier().parameters()
        head_parameter_ids = set(id(p) for p in head_parameters)
        backbone_parameters = [p for p in self.model.parameters() if id(p) not in head_parameter_ids]

        optimizer = timm.optim.create_optimizer_v2(
            [
                {"params": backbone_parameters, "lr": lr},
                {"params": head_parameters, "lr": lr_head},
            ],
            **optimizer_kwargs,
        )

        scheduler, num_epochs = timm.scheduler.create_scheduler_v2(optimizer, **self.hparams["scheduler_kwargs"])

        return OptimizerLRSchedulerConfig(
            optimizer=optimizer, lr_scheduler=LRSchedulerConfigType(scheduler=scheduler, monitor="val/loss")
        )

    def lr_scheduler_step(
        self, scheduler: torch.optim.lr_scheduler.LRScheduler | timm.scheduler.scheduler.Scheduler, metric: float | None
    ) -> None:
        if metric is not None:
            assert isinstance(scheduler, timm.scheduler.scheduler.Scheduler)
            scheduler.step(epoch=self.current_epoch, metric=metric)
        else:
            scheduler.step(epoch=self.current_epoch)


class BaseDataModule(L.LightningDataModule):
    def __init__(self, dataloader_kwargs: dict[str, Any]):
        """
        Args:
        - dataloader_kwargs: passed to torch.utils.data.DataLoader().
            Common/default args are batch_size=1, num_workers=0, pin_memory=False.
        """
        super().__init__()
        self.dataloader_kwargs: dict[str, Any] = dataloader_kwargs
        # These should be initialized in setup().
        self.train_dataset: torch.utils.data.Dataset
        self.val_dataset: torch.utils.data.Dataset
        self.test_dataset: torch.utils.data.Dataset

    def setup(self, stage: str) -> None:
        assert stage in ["fit", "validate", "test", "predict"]
        raise NotImplementedError

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(self.train_dataset, **self.dataloader_kwargs, shuffle=True)

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(self.val_dataset, **self.dataloader_kwargs)

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(self.test_dataset, **self.dataloader_kwargs)


class GastroDataModule(BaseDataModule):
    num_classes: int = 2

    def __init__(self, root: Path = PROJECT_ROOT / "data", *, dataloader_kwargs: dict[str, Any]):
        """
        Args:
        - dataloader_kwargs: passed to torch.utils.data.DataLoader().
            Common/default args are batch_size=1, num_workers=0, pin_memory=False.
        """
        super().__init__(dict(collate_fn=collate_gastro_batch) | dataloader_kwargs)
        self.save_hyperparameters()
        self.root = PROJECT_ROOT / "data"

    def prepare_data(self) -> None:
        assert (self.root / "gastro-hyper-kvasir").exists()

    def setup(self, stage: str) -> None:
        assert stage in ["fit", "validate", "test", "predict"]
        generator = torch.Generator().manual_seed(42)  # Fix a train-val split.
        full_dataset = GastroDataset(self.root)
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            full_dataset, [0.7, 0.3], generator=generator
        )
        # Replace .dataset to change the transform, but keep the Subset self.train_dataset intact.
        self.train_dataset.dataset = GastroDataset(
            PROJECT_ROOT / "data", transform=GastroDataset.default_train_transform()
        )
        self.test_dataset = self.val_dataset


class CIFAR10DataModule(BaseDataModule):
    num_classes: int = 10

    def __init__(self, root: Path = PROJECT_ROOT / "data", *, dataloader_kwargs: dict[str, Any]):
        """
        Args:
        - dataloader_kwargs: passed to torch.utils.data.DataLoader().
            Common/default args are batch_size=1, num_workers=0, pin_memory=False.
        """
        super().__init__(dataloader_kwargs)
        self.save_hyperparameters()
        self.root = root

    def prepare_data(self) -> None:
        download = False
        torchvision.datasets.CIFAR10(root=self.root, train=True, download=download)
        torchvision.datasets.CIFAR10(root=self.root, train=False, download=download)

    def setup(self, stage: str) -> None:
        assert stage in ["fit", "validate", "test", "predict"]
        target_image_size = 224
        if stage == "fit":
            self.train_dataset = ImageClassificationDatasetWrapper(
                torchvision.datasets.CIFAR10(root=self.root, train=True, download=False),
                transform=self.default_train_transform(target_image_size),
            )
        if stage == "fit" or stage == "validate":
            self.val_dataset = ImageClassificationDatasetWrapper(
                torchvision.datasets.CIFAR10(root=self.root, train=False, download=False),
                transform=self.default_transform(target_image_size),
            )
        if stage == "test":
            self.test_dataset = ImageClassificationDatasetWrapper(
                torchvision.datasets.CIFAR10(root=self.root, train=False, download=False),
                transform=self.default_transform(target_image_size),
            )

    @staticmethod
    def default_transform(target_image_size: int) -> v2.Transform:
        return default_transform(target_image_size)

    @staticmethod
    def default_train_transform(target_image_size: int) -> v2.Transform:
        return v2.Compose(
            [
                v2.RandomRotation(10),
                v2.RandomResizedCrop(target_image_size, scale=(0.8, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0), antialias=True),
                v2.RandomHorizontalFlip(),
                v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            ]
        )


if __name__ == "__main__":
    lightning_main(
        Classifier,
        # (CIFAR10DataModule, GastroDataModule),
        experiment_name="classifier",
        monitor="val/loss",
        parser_callback=lambda parser: parser.link_arguments(
            "data.num_classes", "model.num_classes", apply_on="instantiate"
        ),
        main_config_callback=lambda config: {
            "datamodule": config.data.class_path.split(".")[-1],
            "backbone": config.model.backbone.split(".")[0],
            "lr": config.model.optimizer_kwargs["lr"],
            "lr_head": config.model.optimizer_kwargs.get("lr_head"),
            "batch": config.data.init_args.dataloader_kwargs["batch_size"],
            "acc": config.trainer.accumulate_grad_batches,
        },
        checkpoint_filename_pattern="epoch={epoch:0>3}_val-acc={val/accuracy:.3f}",
        model_summary_depth=2,
    )
