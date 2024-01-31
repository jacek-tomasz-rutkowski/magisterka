import argparse
from typing import Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.utilities.types import OptimizerLRSchedulerConfig
from torch.nn import functional as F
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup

# from torchvision import models as cnn_models
import models.t2t_vit
from utils import load_checkpoint
from vit_shapley.CIFAR_10_Dataset import PROJECT_ROOT, CIFAR_10_Datamodule, apply_masks, apply_masks_to_batch


class Surrogate(pl.LightningModule):
    """
    `pytorch_lightning` module for surrogate

    Args:
        output_dim: the dimension of output
        target_model: This model will be trained to generate output similar to
            the output generated by 'target_model' for the same input.
        learning_rate: learning rate of optimizer
        weight_decay: weight decay of optimizer
        decay_power: only `cosine` annealing scheduler is supported currently
        warmup_steps: parameter for the `cosine` annealing scheduler
    """

    def __init__(
        self,
        output_dim: int,
        learning_rate: Optional[float],
        weight_decay: Optional[float],
        decay_power: Optional[str],
        warmup_steps: Optional[int],
        num_players: Optional[int] = None,
        target_model: Optional[nn.Module] = None,
    ):
        super().__init__()
        # Save arguments (output_dim, ..., num_players) into self.hparams.
        self.save_hyperparameters(ignore=["target_model"])
        self.target_model = target_model

        # Backbone initialization
        self.backbone = models.t2t_vit.t2t_vit_14(num_classes=output_dim)
        # backbone_path = PROJECT_ROOT / "saved_models/downloaded/imagenet/81.5_T2T_ViT_14.pth"
        # backbone_path = PROJECT_ROOT / "saved_models/downloaded/cifar10/cifar10_t2t-vit_14_98.3.pth"
        backbone_path = PROJECT_ROOT / "saved_models/transferred/cifar10/ckpt_0.01_0.0005_97.5.pth"
        load_checkpoint(backbone_path, self.backbone, ignore_keys=["head.weight", "head.bias"])

        # Nullify classification head built in the backbone module and rebuild.
        head_in_features = self.backbone.head.in_features
        self.backbone.head = nn.Identity()
        self.head = nn.Linear(head_in_features, self.hparams["output_dim"])

        # Set `num_players` variable.
        self.num_players = num_players or 196  # 14 * 14

    def forward(self, images_masked: torch.Tensor) -> torch.Tensor:
        # assert masks.shape[-1] == self.num_players
        # images_masked = apply_masks(images, masks)
        out: torch.Tensor = self.backbone(images_masked)
        logits: torch.Tensor = self.head(out)
        # [:,0].unsqueeze(1) is not needed

        return logits

    @staticmethod
    def _surrogate_loss(logits: torch.Tensor, logits_target: torch.Tensor) -> torch.Tensor:
        """Returns KL-divergence(softmax(logits_target) || softmax(logits)).

        Input shapes: (batch_size, num_classes).
        Output shape: (1,).
        """
        return F.kl_div(
            input=torch.log_softmax(logits, dim=1),
            target=torch.log_softmax(logits_target, dim=1),
            reduction="batchmean",
            log_target=True,
        )

    def training_step(self, batch, batch_idx):
        assert self.target_model is not None
        images, labels, masks = batch["images"], batch["labels"], batch["masks"]
        
        # logits = self(images, masks)  # ['logits']
        images_masked, masks, labels = apply_masks_to_batch(images, masks, labels)

        logits = self(images_masked)  # ['logits']
        self.target_model.eval()
        with torch.no_grad():
            logits_target = self.target_model(images.to(self.target_model.device)).to(self.device)  # ['logits']
        loss = self._surrogate_loss(logits=logits, logits_target=logits_target)
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/accuracy", (logits.argmax(dim=1) == labels).float().mean(), prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        assert self.target_model is not None
        images, labels, masks = batch["images"], batch["labels"], batch["masks"]
        
        # logits = self(images, masks)  # ['logits']
        images_masked, masks, labels = apply_masks_to_batch(images, masks, labels)

        logits = self(images_masked)  # ['logits']
        logits_target = self.target_model(images.to(self.target_model.device)).to(self.device)  # ['logits']
        self.log("val/loss", self._surrogate_loss(logits=logits, logits_target=logits_target), prog_bar=True)
        self.log("val/accuracy", (logits.argmax(dim=1) == labels).float().mean(), prog_bar=True)

    def test_step(self, batch, batch_idx):
        assert self.target_model is not None
        images, labels, masks = batch["images"], batch["labels"], batch["masks"]
        images_masked, masks, labels = apply_masks_to_batch(images, masks, labels)

        logits = self(images_masked)  # ['logits']
        # logits = self(images, masks)  # ['logits']
        logits_target = self.target_model(images.to(self.target_model.device)).to(self.device)  # ['logits']
        self.log("test/loss", self._surrogate_loss(logits=logits, logits_target=logits_target), prog_bar=True)
        self.log("test/accuracy", (logits.argmax(dim=1) == labels).float().mean(), prog_bar=True)

    def configure_optimizers(self) -> OptimizerLRSchedulerConfig:
        optimizer = AdamW(
            params=self.parameters(), lr=self.hparams["learning_rate"], weight_decay=self.hparams["weight_decay"]
        )

        if self.trainer.max_steps is None or self.trainer.max_steps == -1:
            n_train_batches_per_epoch = len(self.trainer.datamodule.train_dataloader())  # type: ignore
            n_train_batches_total = n_train_batches_per_epoch * self.trainer.max_epochs  # type: ignore
            max_steps = n_train_batches_total // self.trainer.accumulate_grad_batches
        else:
            max_steps = self.trainer.max_steps

        assert self.hparams["decay_power"] == "cosine", "Only cosine annealing scheduler is supported currently"
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=self.hparams["warmup_steps"], num_training_steps=max_steps
        )

        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}


def main() -> None:
    torch.set_float32_matmul_precision("medium")

    parser = argparse.ArgumentParser(description="Surrogate training")
    parser.add_argument("--label", required=False, default="", type=str, help="label for checkpoints")
    parser.add_argument("--num_players", required=True, type=int, help="number of players")
    parser.add_argument("--lr", required=False, default=1e-5, type=float, help="learning rate")
    parser.add_argument("--wd", required=False, default=0.0, type=float, help="weight decay")
    parser.add_argument("--b", required=False, default=128, type=int, help="batch size")
    parser.add_argument("--num_workers", required=False, default=0, type=int, help="number of dataloader workers")
    args = parser.parse_args()

    target_model = models.t2t_vit.t2t_vit_14(num_classes=10)
    # target_model_path = PROJECT_ROOT / "saved_models/downloaded/cifar10/cifar10_t2t-vit_14_98.3.pth"
    target_model_path = PROJECT_ROOT / "saved_models/transferred/cifar10/ckpt_0.01_0.0005_97.5.pth"
    load_checkpoint(target_model_path, target_model)

    # num_classes=10

    surrogate = Surrogate(
        output_dim=10,
        target_model=target_model,
        learning_rate=args.lr,
        weight_decay=args.wd,
        decay_power="cosine",
        warmup_steps=2,
        num_players=args.num_players,
    )

    datamodule = CIFAR_10_Datamodule(
        num_players=args.num_players,
        num_mask_samples=1,
        paired_mask_samples=False,
        batch_size=args.b,
        num_workers=args.num_workers,
    )

    log_and_checkpoint_dir = (
        PROJECT_ROOT
        / "checkpoints"
        / "surrogate"
        / f"{args.label}_player{surrogate.num_players}_lr{args.lr}_wd{args.wd}_b{args.b}"
    )
    log_and_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    trainer = pl.Trainer(
        max_epochs=50,
        default_root_dir=log_and_checkpoint_dir,
        callbacks=RichProgressBar(leave=True)
    )  # logger=False)
    trainer.fit(surrogate, datamodule)


if __name__ == "__main__":
    main()
