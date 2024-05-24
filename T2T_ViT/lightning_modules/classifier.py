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
from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig
from torch import Tensor

from datasets.types import ImageLabelBatch
from datasets.datamodules import CIFAR10DataModule, GastroDataModule  # noqa: F401
from lightning_modules.cli import lightning_main
from lightning_modules.common import TimmModel, get_head_and_backbone_parameters
from utils import find_latest_checkpoint


class Classifier(L.LightningModule):
    """
    LightningModule for training a classifier by transfer learning.

    Args:
    - backbone: The name of the timm model to use.
    - num_classes: The number of classes in the dataset.
    - optimizer_kwargs: passed to timm.optim.create_optimizer_v2(), except lr_head is used as lr for the head.
        Common/default args are opt='sgd' (or e.g. 'adamw'), lr=None (?), weight_decay=0, momentum=0.9.
        See: https://huggingface.co/docs/timm/reference/optimizers#timm.optim.create_optimizer_v2
        Note sgd means SGD with Nesterov momentum in timm (use "momentum" to disable Nesterov).
    - scheduler_kwargs: passed to timm.scheduler.create_scheduler_v2()
        Common/default args are sched='cosine (or e.g. ...), num_epochs=300, decay_epochs=90,
            decay_milestones=[90, 180, 270], cooldown_epoch=0, patience_epochs=10, decay_rate=0.1,
            min_lr=0, warmup_lr=1e-05, warmup_epochs=0.
        See: https://huggingface.co/docs/timm/reference/schedulers#timm.scheduler.create_scheduler_v2
    """

    def __init__(
        self,
        backbone: str,
        num_classes: int,
        optimizer_kwargs: dict[str, Any] = dict(),
        scheduler_kwargs: dict[str, Any] = dict(),
        scriptable: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = cast(TimmModel, timm.create_model(backbone, pretrained=True, scriptable=scriptable))
        self.model.reset_classifier(num_classes=num_classes)

        # Print some info about how the model was pretrained,
        # if this differs from what we use it may be better to change our transformations to match.
        for k in ["input_size", "interpolation", "mean", "std"]:
            print(f"\t{k}:\t{self.model.pretrained_cfg[k]}")

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

        self.log(f"{phase}/loss", loss, prog_bar=True, batch_size=batch_size)
        self.log(f"{phase}/accuracy", accuracy, prog_bar=True, batch_size=batch_size)

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
        """Load the latest checkpoint found under a given path (directories are search recursively)."""
        return cast(
            Classifier, cls.load_from_checkpoint(find_latest_checkpoint(path), map_location=map_location, strict=strict)
        )


if __name__ == "__main__":
    lightning_main(
        Classifier,
        experiment_name="classifier",
        monitor="val/loss",
        # Set model.num_classes automatically from data config.
        parser_callback=lambda parser: parser.link_arguments(
            "data.num_classes", "model.num_classes", apply_on="instantiate"
        ),
        main_config_callback=lambda config: {
            "datamodule": config.data.class_path.split(".")[-1].removesuffix("DataModule"),
            "backbone": config.model.backbone.split(".")[0],
            "lr": config.model.optimizer_kwargs["lr"],
            "lr_head": config.model.optimizer_kwargs.get("lr_head"),
            "batch": config.data.init_args.dataloader_kwargs["batch_size"],
            "acc": config.trainer.accumulate_grad_batches,
        },
        checkpoint_filename_pattern="epoch={epoch:0>3}_val-acc={val/accuracy:.3f}",
        model_summary_depth=2,
    )
