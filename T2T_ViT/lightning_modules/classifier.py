from pathlib import Path
from typing import Any, cast

import lightning as L
import timm
import timm.optim
import timm.scheduler
import timm.scheduler.scheduler
import torch
import torch.nn
import torch.utils.data
import torchvision.datasets
from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch import Tensor
from torch.utils.data import Subset
from torchvision.transforms import v2


from datasets.gastro import GastroDataitem, GastroDataset
from datasets.transforms import default_transform
from lightning_modules.cli import lightning_main
from lightning_modules.common import (
    BaseDataModule,
    DataLoaderKwargs,
    ImageLabelBatch,
    ImageLabelDataitem,
    SizedDataset,
    TimmModel,
    TransformedDataset,
    get_head_and_backbone_parameters,
)
from utils import PROJECT_ROOT, find_latest_checkpoint, random_split_sequence


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
        """
        Input: image batch (B, C, H, W).
        Output: logits (B, output_dim).
        """
        return self.model(x)

    def training_step(self, batch: ImageLabelBatch, batch_idx: int) -> Tensor:
        return self._common_step(batch, batch_idx, "train")

    def validation_step(self, batch: ImageLabelBatch, batch_idx: int) -> None:
        self._common_step(batch, batch_idx, "val")

    def test_step(self, batch: ImageLabelBatch, batch_idx: int) -> None:
        self._common_step(batch, batch_idx, "test")

    def _common_step(self, batch: ImageLabelBatch, batch_idx: int, phase: str) -> Tensor:
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

        head_p, backbone_p = get_head_and_backbone_parameters(self.model, self.model.get_classifier())
        optimizer = timm.optim.create_optimizer_v2(
            [{"params": backbone_p, "lr": lr}, {"params": head_p, "lr": lr_head}],
            **optimizer_kwargs,
        )
        scheduler, num_epochs = timm.scheduler.create_scheduler_v2(optimizer, **self.hparams["scheduler_kwargs"])
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val/loss"}}

    def lr_scheduler_step(
        self, scheduler: torch.optim.lr_scheduler.LRScheduler | timm.scheduler.scheduler.Scheduler, metric: float | None
    ) -> None:
        if metric is not None:
            assert isinstance(scheduler, timm.scheduler.scheduler.Scheduler)
            scheduler.step(epoch=self.current_epoch, metric=metric)
        else:
            scheduler.step(epoch=self.current_epoch)

    @classmethod
    def load_from_latest_checkpoint(cls, path: Path, map_location: Any = None, strict: bool = True) -> "Classifier":
        return cast(
            Classifier, cls.load_from_checkpoint(find_latest_checkpoint(path), map_location=map_location, strict=strict)
        )


class GastroDataModule(BaseDataModule[GastroDataitem]):
    """
    Args:
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


class CIFAR10DataModule(BaseDataModule[ImageLabelDataitem]):
    num_classes: int = 10

    def __init__(self, root: Path = PROJECT_ROOT / "data", *, dataloader_kwargs: DataLoaderKwargs = {}):
        """
        Args:
        - dataloader_kwargs: passed to torch.utils.data.DataLoader().
            Common/default args are batch_size=1, num_workers=0, pin_memory=False.
        """
        super().__init__(dataloader_kwargs)
        self.root = root
        self.target_size = 224
        self.eval_transform = default_transform(self.target_size)
        self.train_transform = v2.Compose(
            [
                v2.RandomRotation(10),
                v2.RandomResizedCrop(self.target_size, scale=(0.8, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0), antialias=True),
                v2.RandomHorizontalFlip(),
                v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            ]
        )

    def prepare_data(self) -> None:
        download = False
        torchvision.datasets.CIFAR10(root=self.root, train=True, download=download)
        torchvision.datasets.CIFAR10(root=self.root, train=False, download=download)

    def setup(self, stage: str) -> None:
        assert stage in ["fit", "validate", "test", "predict"]
        if stage == "fit":
            self.train_dataset = self._dataset(train=True, transform=self.train_transform)
        if stage == "fit" or stage == "validate":
            self.val_dataset = self._dataset(train=False, transform=self.eval_transform)
        if stage == "test":
            self.test_dataset = self._dataset(train=False, transform=self.eval_transform)

    def _dataset(self, train: bool, transform: v2.Transform) -> SizedDataset[ImageLabelDataitem]:
        return TransformedDataset(
            torchvision.datasets.CIFAR10(root=self.root, train=train, download=False),
            transform=lambda tpl: transform(ImageLabelDataitem(image=v2.functional.to_image(tpl[0]), label=tpl[1])),
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
            "dataset": config.data.class_path.split(".")[-1].removesuffix("DataModule"),
            "backbone": config.model.backbone.split(".")[0],
            "lr": config.model.optimizer_kwargs["lr"],
            "lr_head": config.model.optimizer_kwargs.get("lr_head"),
            "batch": config.data.init_args.dataloader_kwargs["batch_size"],
            "acc": config.trainer.accumulate_grad_batches,
        },
        checkpoint_filename_pattern="epoch={epoch:0>3}_val-acc={val/accuracy:.3f}",
        model_summary_depth=2,
    )
