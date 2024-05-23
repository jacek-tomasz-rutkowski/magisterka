from dataclasses import asdict, dataclass
from typing import Any, Literal, cast

import lightning as L
import timm
import torch
import torch.nn as nn
from jsonargparse import class_from_function
from lightning.pytorch.cli import LightningArgumentParser
from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig
from torch import Tensor
from torch.nn import functional as F

from lightning_modules.classifier import CIFAR10DataModule, Classifier, GastroDataModule
from lightning_modules.cli import lightning_main
from lightning_modules.common import (
    BaseDataModule,
    DataLoaderKwargs,
    ImageLabelDataitem,
    ImageLabelMaskBatch,
    ImageLabelMaskDataitem,
    TimmModel,
    TransformedDataset,
    get_head_and_backbone_parameters,
)
from vit_shapley.masks import apply_masks_to_batch, generate_masks


class Surrogate(L.LightningModule):
    """
    Surrogate model: same as normal classifier, but trained on masked inputs by imitating a target_model.

    Args:
    - backbone: name of timm model or weights file to load.
    - num_classes: number of classes in the dataset.
    - num_players: number of players this model was trained with (not actually used in the model).
    - target_model: This model will be trained to generate output similar to
        the output generated by 'target_model' for the same input.
    - optimizer_kwargs: passed to timm.optim.create_optimizer_v2(), except lr_head is used as lr for the head.
    - scheduler_kwargs: passed to timm.scheduler.create_scheduler_v2()
    """

    def __init__(
        self,
        backbone: str,
        num_classes: int,
        num_players: int,
        target_model: nn.Module | None = None,
        optimizer_kwargs: dict[str, Any] = dict(),
        scheduler_kwargs: dict[str, Any] = dict(),
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["target_model"])
        self.target_model = target_model
        if self.target_model:
            self.target_model.eval()

        self.model = cast(TimmModel, timm.create_model(backbone, pretrained=True))
        self.model.reset_classifier(num_classes=num_classes)
        # TODO or actually copy target_model? Originally we had load_transferred_model(backbone_name).

    def forward(self, x: Tensor) -> Tensor:
        """On input: masked images (B, C, H, W), output: logits (B, output_dim)."""
        return self.model(x)

    @staticmethod
    def _surrogate_loss(logits: Tensor, logits_target: Tensor) -> Tensor:
        """
        Returns KL-divergence(softmax(logits_target) || softmax(logits)).

        Input shapes: (batch_size, num_classes).
        Output shape: (1,).
        """
        return F.kl_div(
            input=torch.log_softmax(logits, dim=1),
            target=torch.log_softmax(logits_target, dim=1),
            reduction="batchmean",
            log_target=True,
        )

    def _common_step(self, batch: ImageLabelMaskBatch, batch_idx: int, phase: str) -> Tensor:
        assert self.target_model
        images, labels, masks = batch["image"], batch["label"], batch["mask"]
        batch_size = len(images)
        num_masks_per_image = masks.shape[1]

        images_masked, masks, labels = apply_masks_to_batch(images, masks, labels)
        # images_masked shape: (B * num_masks_per_image, C, H, W).
        # masks shape: (B * num_masks_per_image, num_players).
        # labels shape: (B * num_masks_per_image,).

        logits = self.forward(images_masked)  # Shape (B * num_masks_per_image, num_classes).
        if phase != "train":
            with torch.no_grad():
                logits_unmasked = self(images)  # Shape (B, num_classes).
                logits_unmasked = logits_unmasked.repeat_interleave(num_masks_per_image, dim=0)

        self.target_model.eval()
        with torch.no_grad():
            logits_target = self.target_model(images)  # Shape (B, num_classes).
            logits_target = logits_target.repeat_interleave(num_masks_per_image, dim=0)

        assert logits.shape[1] == logits_target.shape[1] == self.hparams["num_classes"], (
            "num_classes mismatch, this probably means the target model was trained for a different dataset!"
            + f" {logits.shape[1]=}, {logits_target.shape[1]=}, {self.hparams['num_classes']=}"
        )

        loss = self._surrogate_loss(logits=logits, logits_target=logits_target)
        accuracy = (logits.argmax(dim=1) == labels).float().mean()
        self.log(f"{phase}/loss", loss, prog_bar=True, batch_size=batch_size)
        self.log(f"{phase}/accuracy", accuracy, prog_bar=True, batch_size=batch_size)
        if phase != "train":
            accuracy_unmasked = (logits_unmasked.argmax(dim=1) == labels).float().mean()
            self.log(f"{phase}/acc-unmasked", accuracy_unmasked, prog_bar=True, batch_size=batch_size)
        return loss

    def training_step(self, batch: ImageLabelMaskBatch, batch_idx: int) -> Tensor:
        return self._common_step(batch, batch_idx, "train")

    def validation_step(self, batch: ImageLabelMaskBatch, batch_idx: int) -> None:
        self._common_step(batch, batch_idx, "val")

    def test_step(self, batch: ImageLabelMaskBatch, batch_idx: int) -> None:
        self._common_step(batch, batch_idx, "test")

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

    def state_dict(self, *args, **kwargs):
        """
        Get state_dict (the stuff saved in checkpoints) with 'target_model' removed.

        (We don't need it, because it's specified by the config and doesn't change.)
        """
        # We need to be compatible with the extended API where a 'destination' dict-like may be given,
        # and we need to update and return it (we can't make a copy).
        destination = super().state_dict(*args, **kwargs)
        for k in list(destination.keys()):
            if k.startswith("target_model."):
                del destination[k]
        return destination

    def on_load_checkpoint(self, checkpoint) -> None:
        # Restore target_model to state_dict, otherwise strict load_state_dict would fail.
        if self.target_model:
            for k, v in self.target_model.state_dict().items():
                checkpoint["state_dict"]["target_model." + k] = v


@dataclass
class GenerateMasksKwargs:  # TypedDicts are not supported by jsonargparse<=4.28.0
    num_players: int
    num_mask_samples: int
    paired_mask_samples: bool
    mode: Literal["uniform", "shapley"]


class DataModuleWithMasks(BaseDataModule[ImageLabelMaskDataitem]):
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


# This allows calling Classifier.load_from_latest_checkpoint() from config files.
ClassifierFromLatestCheckpoint = class_from_function(Classifier.load_from_latest_checkpoint)

if __name__ == "__main__":
    # Set model.num_classes and model.num_players automatically from data config.
    def parser_callback(parser: LightningArgumentParser):
        parser.link_arguments("data.num_classes", "model.num_classes", apply_on="instantiate")
        parser.link_arguments("data.generate_masks_kwargs.num_players", "model.num_players")

    lightning_main(
        Surrogate,
        DataModuleWithMasks,
        experiment_name="surrogate",
        monitor="val/loss",
        parser_callback=parser_callback,
        main_config_callback=lambda config: {
            "datamodule": config.data.wrapped_datamodule.class_path.split(".")[-1].removesuffix("DataModule"),
            "num_players": config.data.generate_masks_kwargs.num_players,
            "backbone": config.model.backbone.split(".")[0],
            "target_model": config.model.target_model.init_args.path,
            "lr": config.model.optimizer_kwargs["lr"],
            "lr_head": config.model.optimizer_kwargs.get("lr_head"),
            "batch": config.data.dataloader_kwargs["batch_size"],
            "acc": config.trainer.accumulate_grad_batches,
        },
        checkpoint_filename_pattern="epoch={epoch:0>3}_val-acc={val/accuracy:.3f}_unmasked={val/acc-unmasked:.3f}",
        model_summary_depth=2,
    )
