from pathlib import Path
from typing import cast, Any, Protocol

import lightning as L
import torch
import torch.nn
import torch.utils.data
import torchvision.datasets
import timm
import timm.optim
import timm.scheduler
import timm.scheduler.scheduler
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig, LRSchedulerConfigType
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch import Tensor
from torchvision.transforms import v2

from datasets.gastro import GastroBatch, GastroDataset, collate_gastro_batch
from datasets.transforms import default_transform, ImageClassificationDatasetWrapper
from utils import PROJECT_ROOT


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


class TimmWrapper(L.LightningModule):
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

        optimizer = timm.optim.create_optimizer_v2([
            {"params": backbone_parameters, "lr": lr},
            {"params": head_parameters, "lr": lr_head},
        ], **optimizer_kwargs)

        scheduler, num_epochs = timm.scheduler.create_scheduler_v2(optimizer, **self.hparams["scheduler_kwargs"])

        # optimizer = torch.optim.AdamW(self.parameters(), lr=0.001)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        # optimizer = torch.optim.SGD(self.parameters(), lr=0.01, momentum=0.9)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer, mode='min', factor=0.05, patience=2, threshold=0.0001,
        #     threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=True
        # )

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
                torchvision.datasets.CIFAR10(
                    root=self.root,
                    train=True,
                    download=False
                ),
                transform=self.default_train_transform(target_image_size)
            )
        if stage == "fit" or stage == "validate":
            self.val_dataset = ImageClassificationDatasetWrapper(
                torchvision.datasets.CIFAR10(
                    root=self.root, train=False, download=False
                ),
                transform=self.default_transform(target_image_size)
            )
        if stage == "test":
            self.test_dataset = ImageClassificationDatasetWrapper(
                torchvision.datasets.CIFAR10(
                    root=self.root, train=False, download=False
                ),
                transform=self.default_transform(target_image_size)
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


def main() -> None:
    # TODO set HF_HOME and torch.hub.set_dir()
    # TODO sometimes mean-std is different.

    # Usage:
    #   CUDA_VISIBLE_DEVICES=7 python -m vit_shapley.modules.timm_wrapper fit --config timm_config.yaml
    # Or:
    #   --config default.yaml --config overwrites.yaml
    #   --trainer trainer.yaml --model model.yaml --data data.yaml
    # python main.py fit --model DemoModel --print_config

    torch.set_float32_matmul_precision("medium")
    LightningCLI(
        TimmWrapper,
        trainer_defaults=dict(
            default_root_dir=PROJECT_ROOT / "checkpoints" / "transfer",
        ),
    )
    #  parser_kwargs={"default_config_files": ["my_cli_defaults.yaml"]})

    # For arguments that appear in many places, use
    # https://lightning.ai/docs/pytorch/stable/cli/lightning_cli_expert.html#cli-link-arguments
    # Linking can also create dependencies like "instantiate datamodule first to get number of classes,
    # or link stuff as functions of multiple other settings.
    # For variable interpolation you would need to enable omegaconf.

    # LightningCLI has before_fit and after_fit hooks, for example. It's argparse can also be customized.
    # You can also customize configs/defaults for the fit subcommand and other separately.

    # Use JSONARGPARSE_DEBUG=true to see stacktraces during argparsing.


if __name__ == "__main__":
    main()
