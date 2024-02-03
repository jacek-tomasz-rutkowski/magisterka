from typing import Literal, cast

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import OptimizerLRSchedulerConfig
from torch.nn import functional as F
from torchmetrics import MeanMetric
from transformers import get_cosine_schedule_with_warmup
from transformers.optimization import AdamW


def set_schedule(pl_module: pl.LightningModule) -> OptimizerLRSchedulerConfig:
    optimizer = AdamW(
        params=pl_module.parameters(),
        lr=pl_module.hparams["learning_rate"],
        weight_decay=pl_module.hparams["weight_decay"],
    )

    if pl_module.trainer.max_steps is None or pl_module.trainer.max_steps == -1:
        assert hasattr(pl_module.trainer, "datamodule")
        assert pl_module.trainer.max_epochs is not None
        n_batches_per_epoch = len(pl_module.trainer.datamodule.train_dataloader())
        max_steps = n_batches_per_epoch * pl_module.trainer.max_epochs // pl_module.trainer.accumulate_grad_batches
    else:
        max_steps = pl_module.trainer.max_steps

    if pl_module.hparams["decay_power"] == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=pl_module.hparams["warmup_steps"], num_training_steps=max_steps
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}
    elif pl_module.hparams["decay_power"] is None:
        return {"optimizer": optimizer}
    else:
        raise NotImplementedError("Only cosine scheduler is implemented for now")


def set_metrics(pl_module: pl.LightningModule) -> None:
    for phase in ["train", "val", "test"]:
        setattr(pl_module, f"{phase}_value_diff", MeanMetric())
        setattr(pl_module, f"{phase}_efficiency_gap", MeanMetric())
        setattr(pl_module, f"{phase}_efficiency_class_gap", MeanMetric())
        setattr(pl_module, f"{phase}_loss", MeanMetric())


def epoch_wrapup(pl_module: pl.LightningModule, phase: Literal["train", "val", "test"]) -> None:
    value_diff = getattr(pl_module, f"{phase}_value_diff").compute()
    getattr(pl_module, f"{phase}_value_diff").reset()
    pl_module.log(f"{phase}/epoch_value_diff", value_diff)

    efficiency_gap = getattr(pl_module, f"{phase}_efficiency_gap").compute()
    getattr(pl_module, f"{phase}_efficiency_gap").reset()
    pl_module.log(f"{phase}/epoch_efficiency_gap", efficiency_gap)

    efficiency_class_gap = getattr(pl_module, f"{phase}_efficiency_class_gap").compute()
    getattr(pl_module, f"{phase}_efficiency_class_gap").reset()
    pl_module.log(f"{phase}/epoch_efficiency_class_gap", efficiency_class_gap)

    loss = getattr(pl_module, f"{phase}_loss").compute()
    getattr(pl_module, f"{phase}_loss").reset()
    pl_module.log(f"{phase}/epoch_loss", loss)

    if pl_module.hparams.checkpoint_metric == "loss":
        checkpoint_metric = -loss
    elif pl_module.hparams.checkpoint_metric == "value_diff":
        checkpoint_metric = -value_diff
    else:
        raise NotImplementedError("Not supported checkpoint metric")
    pl_module.log(f"{phase}/checkpoint_metric", checkpoint_metric)


def compute_metrics(
    pl_module: pl.LightningModule,
    num_players: int,
    values_pred: torch.Tensor,  # (batch, num_players, num_classes)
    values_target: torch.Tensor,  # (batch, num_players, num_classes)
    surrogate_grand: torch.Tensor,  # (batch, num_classes)
    surrogate_null: torch.Tensor,  # (1, num_classes)
    phase: Literal["train", "val", "test"],
) -> torch.Tensor:
    # values_pred should be close to values_target.
    value_diff = num_players * F.mse_loss(input=values_pred, target=values_target, reduction="mean")

    # values_pred summed over players should be close to the difference between the grand coalition and the null coalition.
    efficiency_gap = num_players * F.mse_loss(
        input=values_pred.sum(dim=1), target=surrogate_grand - surrogate_null, reduction="mean"
    )

    # values_pred summed over classes should be close to zero, maybe.
    efficiency_class_gap = F.mse_loss(
        input=values_pred.sum(dim=2), target=torch.zeros_like(values_pred.sum(dim=2)), reduction="mean"
    )

    loss = (
        value_diff
        + pl_module.hparams["efficiency_lambda"] * efficiency_gap
        + pl_module.hparams["efficiency_class_lambda"] * efficiency_class_gap
    )

    value_diff = getattr(pl_module, f"{phase}_value_diff")(value_diff)
    pl_module.log(f"{phase}/value_diff", value_diff)

    efficiency_gap = getattr(pl_module, f"{phase}_efficiency_gap")(efficiency_gap)
    pl_module.log(f"{phase}/efficiency_gap", efficiency_gap)

    efficiency_class_gap = getattr(pl_module, f"{phase}_efficiency_class_gap")(efficiency_class_gap)
    pl_module.log(f"{phase}/efficiency_class_gap", efficiency_class_gap)

    loss = getattr(pl_module, f"{phase}_loss")(loss)
    pl_module.log(f"{phase}/loss", loss)

    return cast(torch.Tensor, loss)
