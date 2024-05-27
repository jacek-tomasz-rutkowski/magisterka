import copy
import math
from pathlib import Path
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

from datasets.datamodules import CIFAR10DataModule, DataModuleWithMasks, GastroDataModule  # noqa: F401
from datasets.types import ImageLabelMaskBatch
from lightning_modules.cli import lightning_main
from lightning_modules.common import TimmModel
from lightning_modules.surrogate import Surrogate
from utils import find_latest_checkpoint
from vit_shapley.masks import apply_masks
from vit_shapley.modules.explainer_utils import get_masked_accuracy


class Explainer(L.LightningModule):
    """
    TODO

    - optimizer_kwargs: passed to timm.optim.create_optimizer_v2(), except lr_head is used as lr for the head.
    - scheduler_kwargs: passed to timm.scheduler.create_scheduler_v2()
    """

    def __init__(
        self,
        surrogate: L.LightningModule,
        backbone: str,
        freeze_backbone: Literal["all", "except_last_two", "none"],
        num_classes: int,
        num_players: int,
        head_num_convolutions: int,
        head_num_attention_blocks: int,
        head_mlp_layer_ratio: int,
        use_tanh: bool = True,
        use_softmax: bool = True,
        divisor: float = 1.0,
        efficiency_lambda: float = 0.0,
        efficiency_class_lambda: float = 0.0,
        target_class_lambda: float = 0.0,
        optimizer_kwargs: dict[str, Any] = dict(),
        scheduler_kwargs: dict[str, Any] = dict(),
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["surrogate"])
        self.surrogate = surrogate
        self.surrogate.eval()

        self.num_players = num_players
        surrogate_num_players = getattr(surrogate, "num_players", None)
        if surrogate_num_players:
            assert (
                num_players == surrogate_num_players
            ), f"Explainer's {num_players=} != {surrogate_num_players=}. It's technically OK but unexpected."

        self._null: Tensor | None = None

        self.backbone: nn.Module
        if backbone == "copy_surrogate":
            self.backbone = copy.deepcopy(self.surrogate.model)
            self.backbone.reset_classifier(0, "")
        else:
            self.backbone = cast(TimmModel, timm.create_model(backbone, pretrained=True, num_classes=0, global_pool=""))
        self._freeze_backbone()

        # Dimension of feature vector at each position, like 768.
        self.embed_dim = self.backbone.feature_info[-1]["num_chs"]
        # Times the image width and height is reduced by the backbone, usually 16 or 32.
        self.reduction = self.backbone.feature_info[-1]["reduction"]
        self._build_head()

    def _build_head(self):
        """Build head modules: convolutions, attention blocks, and a final MLP."""
        self.convs = nn.Sequential()
        for i in range(self.hparams["head_num_convolutions"]):
            self.convs.add_module(
                f"head_conv{i}",
                nn.Conv2d(
                    in_channels=self.embed_dim,
                    out_channels=self.embed_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
            )
            self.convs.add_module(f"head_relu{i}", nn.ReLU())

        self.attention_blocks = nn.ModuleList(
            [
                nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=4, batch_first=True)
                for _ in range(self.hparams["head_num_attention_blocks"])
            ]
        )
        if len(self.attention_blocks):
            self.attention_blocks[0].norm1 = nn.Identity()

        hidden_dim = int(self.hparams["head_mlp_layer_ratio"] * self.embed_dim)
        self.mlp = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(in_features=self.embed_dim, out_features=hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dim, out_features=self.hparams["num_classes"]),
        )

    def _freeze_backbone(self) -> None:
        if self.hparams["freeze_backbone"] == "all":
            for value in self.backbone.parameters():
                value.requires_grad = False
        elif self.hparams["freeze_backbone"] == "except_last_two":
            for value in self.backbone.parameters():
                value.requires_grad = False
            for m in [self.backbone.blocks[-2], self.backbone.blocks[-1], self.backbone.norm]:
                for value in m.parameters():
                    value.requires_grad = True
        elif self.hparams["freeze_backbone"] == "none":
            pass
        else:
            raise ValueError(f"Unknown freeze_backbone value: {self.hparams['freeze_backbone']}")

    def forward(
        self,
        images: Tensor,
        grand: Tensor | None = None,
        null: Tensor | None = None,
        normalize: bool = True,
    ) -> Tensor:
        """
        Forward pass.

        Args:
        - images: (batch, channel, height, width).
        - grand: optionally, surrogate outputs for the grand coallition,
            (no players masked), shape (batch, num_classes).
        - null: optionally, surrogate outputs for the null coallition,
            (all players masked), shape (1, num_classes).

        Returns logits for class predictions of shape (batch, num_players, num_classes).
        """
        B, C, H, W = images.shape
        assert H == W  # Usually 224.
        r = H // self.reduction  # Expected height and width of the feature map, usually 14 or 7.

        x = self.backbone(images)

        # Adapt to shape (B, r, r, embed_dim).
        if x.shape == (B, r, r, self.embed_dim):
            pass
        elif x.shape == (B, self.embed_dim, r, r):
            x = x.permute(0, 2, 3, 1)
        elif x.shape == (B, r * r, self.embed_dim):
            x = x.view(B, r, r, self.embed_dim)
        elif x.shape == (B, r * r + 1, self.embed_dim):
            x = x[:, 1:, :]  # Remove the classification token.
            x = x.view(B, r, r, self.embed_dim)
        else:
            raise ValueError(f"Unexpected features: {x.shape=}; {images.shape=}, {self.reduction=}, {self.embed_dim=}.")

        s = int(math.sqrt(self.num_players))
        assert s * s == self.num_players, f"Expected {self.num_players=} to be a perfect square."

        # Downscale or upscale to (B, s, s, embed_dim).
        if r != s:
            indices = [int((i + 0.5) * r / s) for i in range(s)]
            x = x[:, indices, :, :][:, :, indices, :]  # (B, s, s, embed_dim)

        if self.hparams["head_num_convolutions"]:
            x = x.permute(0, 3, 1, 2)  # (B, embed_dim, s, s)
            x = self.convs(x)  # (B, embed_dim, s, s)
            x = x.permute(0, 2, 3, 1)  # (B, s, s, embed_dim)

        # Flatten to (B, num_players, embed_dim)
        x = x.flatten(1, 2)

        # Run through head modules.
        for layer_module in self.attention_blocks:
            x, _ = layer_module(x, x, x, need_weights=False)

        pred = self.mlp(x)  # (B, num_players, embed_dim) -> # (B, num_players, num_classes)

        if self.hparams["use_tanh"]:
            pred = pred.tanh()

        pred = pred / self.hparams["divisor"]

        # "Additive efficient normalization",
        # which means we add a constant to ensure that (for each item in the batch and each class)
        # the predicted player contributions sum up to grand - null.
        if normalize:
            if grand is None:
                grand = self.grand(images)
            if null is None:
                null = self.null(images)
            # Pred: (batch, num_players, num_classes), grand: (batch, num_classes), null: (1, num_classes).
            deficit = (grand - null) - pred.sum(dim=1)  # (batch, num_classes)
            pred = pred + deficit.unsqueeze(1) / self.num_players

        return cast(Tensor, pred)

    def null(self, images: Tensor | None = None) -> Tensor:
        """
        Class probabilites for a fully masked input (cached after first call).

        Input: (batch, channel, height, width), only the shape is relevant.
        Output: (1, num_classes).
        """
        if self._null is not None:
            return self._null
        if images is None:
            raise RuntimeError("Call explainer.null(images) at least once to get null value.")
        with torch.no_grad():
            masks = torch.zeros(1, 1, self.num_players, device=self.surrogate.device)
            self._null = self.surrogate_multiple_masks(images[:1], masks).squeeze(dim=1)  # (1, num_classes)
            return self._null

    def grand(self, images: Tensor) -> Tensor:
        """Class probabilities for unmasked inputs.

        Input: (batch, channel, height, width).
        Output: (batch, num_classes).
        """
        with torch.no_grad():
            masks = torch.ones(images.shape[0], 1, self.num_players, device=self.surrogate.device)
            return self.surrogate_multiple_masks(images, masks).squeeze(dim=1)  # (batch, num_classes)

    def surrogate_multiple_masks(self, images: Tensor, masks: Tensor) -> Tensor:
        """Get class probabilites from surrogate model, with multiple masks.

        Args:
            images: (batch, channel, height, width).
            masks: (batch, num_mask_samples, num_players).
        Returns class probabilites (batch, num_mask_samples, num_classes).
        """
        assert self.surrogate is not None
        with torch.no_grad():
            batch_size, num_masks_per_image, num_players = masks.shape
            assert (
                num_players == self.num_players
            ), f"Explainer was inited with {self.num_players=} but got {num_players=} from dataloader."
            images_masked = apply_masks(images, masks)  # (B * num_mask_samples, C, H, W)

            logits = self.surrogate(images_masked)  # (B * num_mask_samples, num_classes)
            surrogate_values: Tensor
            if self.hparams["use_softmax"]:
                surrogate_values = torch.nn.Softmax(dim=1)(logits)  # (B * num_mask_samples, num_classes)
            else:
                surrogate_values = logits
        return surrogate_values.reshape(batch_size, num_masks_per_image, -1)

    def _common_step(
        self, batch: ImageLabelMaskBatch, batch_idx: int, phase: Literal["train", "val", "test"]
    ) -> Tensor:
        # Images have shape (batch, channel, height, weight).
        # Masks have shape: (batch, num_masks_per_image, num_players).
        # Labels have shape: (batch,).
        images, masks, labels = batch["image"], batch["mask"], batch["label"]

        # Run surrogate on masked, unmasked and fully masked inputs.
        surrogate_values = self.surrogate_multiple_masks(images, masks)  # (batch, num_masks_per_image, num_classes)
        surrogate_grand = self.grand(images)  # (batch, num_classes)
        surrogate_null = self.null(images)  # (1, num_classes)

        # Run explainer to get shap_values of shape (batch, num_players, num_classes).
        shap_values = self(images, grand=surrogate_grand, null=surrogate_null)

        # Get approximation of surrogate_values from shap_values.
        # Recall surrogate_values[b,n,c] is the value (the class probability outputted by the surrogate)
        # of classifying images[b] masked with masks[b,n] as class c.
        # We want it to be well-approximated by the sum of shap values for class c of players in masks[b,n]
        # plus the null value for c.
        # That is: surrogate_null[.,c] + sum_p masks[b,n,p] * shap_values[b, p, c]
        values_pred = surrogate_null.unsqueeze(0) + masks.float() @ shap_values
        # Shapes: (1, 1, num_classes) + (batch, num_masks_per_image, num_players) @ (batch, num_players, num_classes)
        # = (batch, num_masks_per_image, num_classes).

        # This is the main loss: values_pred should be close to values_target.
        value_loss = self.num_players * F.mse_loss(values_pred, surrogate_values)

        # Contrast loss is similar, but looks only at the contrast between a mask and its complement.
        # This helps reduce constant unavoidable loss due to surrogate_values[m] + surrogate_values[~m]
        # being much larger than grand - null (when the surrogate model handles masked inputs pretty well).
        values_pred_contrast = values_pred[:, 0, :] - values_pred[:, 1, :]
        values_target_contrast = surrogate_values[:, 0, :] - surrogate_values[:, 1, :]
        contrast_loss = F.mse_loss(values_pred_contrast, values_target_contrast)
        # Correlation loss is similar to contrast loss but looks at correlation rather than MSE.
        corr_loss = -(values_pred_contrast * values_target_contrast).mean()

        # A baseline to compare to: constant shap_values (that sum to grand - null)
        # shape (B, num_players, num_classes)
        baseline_shap_values = (surrogate_grand - surrogate_null).div(self.num_players)  # (B, num_classes)
        baseline_shap_values = baseline_shap_values.unsqueeze(dim=1).expand(-1, self.num_players, -1)

        values_baseline = surrogate_null.unsqueeze(0) + masks.float() @ baseline_shap_values
        values_baseline_contrast = values_baseline[:, 0, :] - values_baseline[:, 1, :]
        contrast_loss_baseline = F.mse_loss(values_baseline_contrast, values_target_contrast)

        # Values restricted to the target class only.
        # target_values_pred[b, p] := values_pred[b, p, targets[b]]
        t_values_pred = values_pred[torch.arange(len(labels)), :, labels]  # (B, num_masks_per_image)
        t_surrogate_values = surrogate_values[torch.arange(len(labels)), :, labels]  # (B, num_masks_per_image)
        t_value_loss = self.num_players * F.mse_loss(t_values_pred, t_surrogate_values)
        t_shap_values = shap_values[torch.arange(len(labels)), :, labels]  # (B, num_players)
        t_values_baseline = values_baseline[torch.arange(len(labels)), :, labels]  # (B, num_players)

        # shap_values summed over players should be close grand - null.
        efficiency_gap = self.num_players * F.mse_loss(shap_values.sum(dim=1), surrogate_grand - surrogate_null)

        # shap_values summed over classes should be close to zero, for each player?
        efficiency_class_gap = F.mse_loss(shap_values.sum(dim=2), torch.zeros_like(shap_values.sum(dim=2)))

        loss: Tensor = (
            (1 - self.hparams["target_class_lambda"]) * value_loss
            + self.hparams["target_class_lambda"] * t_value_loss
            + self.hparams["efficiency_lambda"] * efficiency_gap
            + self.hparams["efficiency_class_lambda"] * efficiency_class_gap
        )
        # loss = contrast_loss
        # loss = corr_loss

        # Root Mean Square Error in values per player:
        rmse_diff = F.mse_loss(values_pred, surrogate_values).sqrt()
        self.log(f"{phase}/v_rmse", rmse_diff, prog_bar=True)
        rmse_diff_baseline = F.mse_loss(values_baseline, surrogate_values).sqrt()
        self.log(f"{phase}/v_rmse_base", rmse_diff_baseline, prog_bar=True)
        # Mean Absolute Error in values per player:
        mae_diff = F.l1_loss(values_pred, surrogate_values)
        self.log(f"{phase}/v_mae", mae_diff, prog_bar=False)
        mae_diff_baseline = F.l1_loss(values_baseline, surrogate_values)
        self.log(f"{phase}/v_mae_base", mae_diff_baseline, prog_bar=False)

        # Same but only on target class
        rmse_diff = F.mse_loss(t_values_pred, t_surrogate_values).sqrt()
        self.log(f"{phase}/vt_rmse", rmse_diff, prog_bar=False)
        rmse_diff_baseline = F.mse_loss(t_values_baseline, t_surrogate_values).sqrt()
        self.log(f"{phase}/vt_rmse_base", rmse_diff_baseline, prog_bar=False)
        mae_diff = F.l1_loss(t_values_pred, t_surrogate_values)
        self.log(f"{phase}/vt_mae", mae_diff, prog_bar=False)
        mae_diff_baseline = F.l1_loss(t_values_baseline, t_surrogate_values)
        self.log(f"{phase}/vt_mae_base", mae_diff_baseline, prog_bar=False)

        # "Efficiency", or gaps between the explainer's sum of SHAP values and what they should be.
        self.log(f"{phase}/eff", efficiency_gap, prog_bar=False)
        self.log(f"{phase}/eff_class", efficiency_class_gap, prog_bar=False)

        # Mean value predicted for target class.
        # This is exactly (grand-null) / 2, slightly smaller than 1.1 / 2 = 0.55.
        self.log(f"{phase}/vt_mean", t_values_pred.mean(), prog_bar=False)
        # Actual surrogate output value for target class.
        # This is ~0.9 if surrogate works well on masked inputs.
        self.log(f"{phase}/vt_mean_target", t_surrogate_values.mean(), prog_bar=False)

        # Minimum, maximum, and span of explainer SHAP values for the target class.
        t_shap_min = t_shap_values.min(dim=1).values.mean()
        t_shap_max = t_shap_values.max(dim=1).values.mean()
        self.log(f"{phase}/t_shap_min", t_shap_min, prog_bar=False)
        self.log(f"{phase}/t_shap_max", t_shap_max, prog_bar=False)
        # Should be as large as possible, 0.9 / num_players is OK-ish.
        self.log(f"{phase}/t_shap_span", t_shap_max - t_shap_min, prog_bar=True)

        self.log(f"{phase}/loss", loss, prog_bar=True)
        self.log(f"{phase}/contrast_loss", contrast_loss, prog_bar=True)
        self.log(f"{phase}/contrast_loss_base", contrast_loss_baseline, prog_bar=True)
        self.log(f"{phase}/corr_loss", corr_loss, prog_bar=True)

        if phase != "train":
            macc_best = get_masked_accuracy(images, masks, labels, self.surrogate, shap_values, "best")
            macc_worst = get_masked_accuracy(images, masks, labels, self.surrogate, shap_values, "worst")
            self.log(f"{phase}/macc-best", macc_best, prog_bar=True)
            self.log(f"{phase}/macc-worst", macc_worst, prog_bar=True)

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

        backbone_p = [p for p in self.backbone.parameters() if p.requires_grad]
        head_p = list(self.convs.parameters()) + list(self.attention_blocks.parameters()) + list(self.mlp.parameters())
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

    # def state_dict(self):
    #     """Remove 'surrogate' from the state_dict (the stuff saved in checkpoints)."""
    #     return {k: v for k, v in super().state_dict().items() if not k.startswith("surrogate.")}

    @classmethod
    def load_from_latest_checkpoint(
        cls, path: Path, map_location: Any = None, strict: bool = True, **kwargs: Any
    ) -> "Explainer":
        """Load the latest checkpoint found under a given path (directories are search recursively)."""
        ckpt_path = find_latest_checkpoint(path)
        explainer = cls.load_from_checkpoint(ckpt_path, map_location=map_location, strict=strict, **kwargs)
        return cast(Explainer, explainer)


# This allows calling Surrogate.load_from_latest_checkpoint() from config files.
SurrogateFromLatestCheckpoint = class_from_function(Surrogate.load_from_latest_checkpoint)


if __name__ == "__main__":
    # Set model.num_classes and model.num_players automatically from data config.
    def parser_callback(parser: LightningArgumentParser):
        parser.link_arguments("data.num_classes", "model.num_classes", apply_on="instantiate")
        parser.link_arguments("data.generate_masks_kwargs.num_players", "model.num_players")

    lightning_main(
        Explainer,
        DataModuleWithMasks,
        experiment_name="explainer",
        monitor="val/loss",
        parser_callback=parser_callback,
        main_config_callback=lambda config: {
            "datamodule": config.data.wrapped_datamodule.class_path.split(".")[-1].removesuffix("DataModule"),
            "num_players": config.data.generate_masks_kwargs.num_players,
            "backbone": config.model.backbone.split(".")[0],
            "freeze_backbone": config.model.freeze_backbone,
            "target_model": config.model.surrogate.init_args.path,
            "n_conv": config.model.head_num_convolutions,
            "n_att": config.model.head_num_attention_blocks,
            "use_tanh": config.model.use_tanh,
            "use_softmax": config.model.use_softmax,
            "divisor": config.model.divisor,
            "eff": config.model.efficiency_lambda,
            "t": config.model.target_class_lambda,
            "lr": config.model.optimizer_kwargs["lr"],
            "lr_head": config.model.optimizer_kwargs.get("lr_head"),
            "wd": config.model.optimizer_kwargs.get("weight_decay"),
            "batch": config.data.dataloader_kwargs["batch_size"],
            "acc": config.trainer.accumulate_grad_batches,
        },
        checkpoint_filename_pattern="epoch={epoch:0>3}_val-macc-best={val/macc-best:.3f}",
        model_summary_depth=2,
    )
