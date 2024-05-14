from typing import Any, Literal, cast

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import OptimizerLRSchedulerConfig
from torch.nn import functional as F
from transformers import get_cosine_schedule_with_warmup
# from transformers.optimization import AdamW
from torch.optim import AdamW
from tqdm import tqdm

from datasets.CIFAR_10_Dataset import apply_masks_to_batch
from vit_shapley.masks import make_masks_from_player_values, get_distances_from_center


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


def compute_metrics(
    pl_module: pl.LightningModule,
    num_players: int,
    masks: torch.Tensor,
    targets: torch.Tensor,
    shap_values: torch.Tensor,
    values_target: torch.Tensor,
    surrogate_grand: torch.Tensor,
    surrogate_null: torch.Tensor,
    phase: Literal["train", "val", "test"],
) -> torch.Tensor:
    """
    Compute and log metrics, return loss.

    Args:
    - pl_module: the Explainer, used to access hparams and to log metrics.
    - num_players
    - masks: (batch, num_masks_per_image, num_players)
    - targets: ground truth classes, shape (batch,)
    - shap_values: explainer output for a batch, shape (batch, num_players, num_classes)
    - values_target: surrogate output (logits or class probabilites) for masked images,
        shape (batch, num_masks_per_image, num_classes)
    - surrogate_grand: surrogate output for the grand coalition, shape (batch, num_classes)
    - surrogate_null: surrogate output for the null coalition, shape (1, num_classes)
    - phase: "train", "val", or "test".
    """
    # Get approximation of surrogate_values from shap_values.
    # Recall surrogate_values[b,n,c] is the value (the class probability outputted by the surrogate)
    # of classifying images[b] masked with masks[b,n] as class c.
    # We want it to be well-approximated by the sum of shap values for class c of players in masks[b,n]
    # plus the null value for c.
    # That is: surrogate_null[.,c] + sum_p masks[b,n,p] * shap_values[b, p, c]
    values_pred = surrogate_null.unsqueeze(0) + masks.float() @ shap_values
    # Shapes: (1, 1, num_classes) + (batch, num_masks_per_image, num_players) @ (batch, num_players, num_classes)
    # = (batch, num_masks_per_image, num_classes).

    # values_pred should be close to values_target.
    value_loss = num_players * F.mse_loss(input=values_pred, target=values_target)

    values_pred_contrast = values_pred[:, 0, :] - values_pred[:, 1, :]
    values_target_contrast = values_target[:, 0, :] - values_target[:, 1, :]
    contrast_loss = F.mse_loss(input=values_pred_contrast, target=values_target_contrast)
    corr_loss = -(values_pred_contrast * values_target_contrast).mean()

    # A baseline to compare to: constant shap_values (that sum to grand - null)
    # shape (B, num_players, num_classes)
    baseline_shap_values = (surrogate_grand - surrogate_null).div(num_players)  # (B, num_classes)
    baseline_shap_values = baseline_shap_values.unsqueeze(dim=1).expand(-1, num_players, -1)

    values_baseline = surrogate_null.unsqueeze(0) + masks.float() @ baseline_shap_values
    values_baseline_contrast = values_baseline[:, 0, :] - values_baseline[:, 1, :]
    contrast_loss_baseline = F.mse_loss(input=values_baseline_contrast, target=values_target_contrast)

    # Values for the target class.
    # target_values_pred[b, p] := values_pred[b, p, targets[b]]
    t_values_pred = values_pred[torch.arange(len(targets)), :, targets]  # (B, num_masks_per_image)
    t_values_target = values_target[torch.arange(len(targets)), :, targets]  # (B, num_masks_per_image)
    t_value_loss = num_players * F.mse_loss(input=t_values_pred, target=t_values_target)
    t_shap_values = shap_values[torch.arange(len(targets)), :, targets]  # (B, num_players)
    t_values_baseline = values_baseline[torch.arange(len(targets)), :, targets]  # (B, num_players)

    # shap_values summed over players should be close
    # to the difference between the grand coalition and the null coalition.
    efficiency_gap = num_players * F.mse_loss(input=shap_values.sum(dim=1), target=surrogate_grand - surrogate_null)

    # shap_values summed over classes should be close to zero, for each player?
    efficiency_class_gap = F.mse_loss(input=shap_values.sum(dim=2), target=torch.zeros_like(shap_values.sum(dim=2)))

    loss = (
        (1 - pl_module.hparams["target_class_lambda"]) * value_loss
        + pl_module.hparams["target_class_lambda"] * t_value_loss
        + pl_module.hparams["efficiency_lambda"] * efficiency_gap
        + pl_module.hparams["efficiency_class_lambda"] * efficiency_class_gap
    )
    # loss = contrast_loss
    # loss = corr_loss

    # Root Mean Square Error in values per player:
    rmse_diff = F.mse_loss(input=values_pred, target=values_target, reduction="mean").sqrt()
    pl_module.log(f"{phase}/v_rmse", rmse_diff, prog_bar=True)
    rmse_diff_baseline = F.mse_loss(input=values_baseline, target=values_target, reduction="mean").sqrt()
    pl_module.log(f"{phase}/v_rmse_base", rmse_diff_baseline, prog_bar=True)
    # Mean Absolute Error in values per player:
    mae_diff = F.l1_loss(input=values_pred, target=values_target, reduction="mean")
    pl_module.log(f"{phase}/v_mae", mae_diff, prog_bar=False)
    mae_diff_baseline = F.l1_loss(input=values_baseline, target=values_target, reduction="mean")
    pl_module.log(f"{phase}/v_mae_base", mae_diff_baseline, prog_bar=False)

    # Same but only on target class
    rmse_diff = F.mse_loss(input=t_values_pred, target=t_values_target, reduction="mean").sqrt()
    pl_module.log(f"{phase}/vt_rmse", rmse_diff, prog_bar=False)
    rmse_diff_baseline = F.mse_loss(input=t_values_baseline, target=t_values_target, reduction="mean").sqrt()
    pl_module.log(f"{phase}/vt_rmse_base", rmse_diff_baseline, prog_bar=False)
    mae_diff = F.l1_loss(input=t_values_pred, target=t_values_target, reduction="mean")
    pl_module.log(f"{phase}/vt_mae", mae_diff, prog_bar=False)
    mae_diff_baseline = F.l1_loss(input=t_values_baseline, target=t_values_target, reduction="mean")
    pl_module.log(f"{phase}/vt_mae_base", mae_diff_baseline, prog_bar=False)

    # "Efficiency", or gaps between the explainer's sum of SHAP values and what they should be.
    pl_module.log(f"{phase}/eff", efficiency_gap, prog_bar=False)
    pl_module.log(f"{phase}/eff_class", efficiency_class_gap, prog_bar=False)

    # Mean value predicted for target class.
    # This is exactly (grand-null) / 2, slightly smaller than 1.1 / 2 = 0.55.
    pl_module.log(f"{phase}/vt_mean", t_values_pred.mean(), prog_bar=False)
    # Actual surrogate output value for target class.
    # This is ~0.9 if surrogate works well on masked inputs.
    pl_module.log(f"{phase}/vt_mean_target", t_values_target.mean(), prog_bar=False)

    # Minimum, maximum, and span of explainer SHAP values for the target class.
    t_shap_min = t_shap_values.min(dim=1).values.mean()
    t_shap_max = t_shap_values.max(dim=1).values.mean()
    pl_module.log(f"{phase}/t_shap_min", t_shap_min, prog_bar=False)
    pl_module.log(f"{phase}/t_shap_max", t_shap_max, prog_bar=False)
    # Should be as large as possible, 0.9 / num_players is OK-ish.
    pl_module.log(f"{phase}/t_shap_span", t_shap_max - t_shap_min, prog_bar=True)

    pl_module.log(f"{phase}/loss", loss, prog_bar=True)
    pl_module.log(f"{phase}/contrast_loss", contrast_loss, prog_bar=True)
    pl_module.log(f"{phase}/contrast_loss_base", contrast_loss_baseline, prog_bar=True)
    pl_module.log(f"{phase}/corr_loss", corr_loss, prog_bar=True)

    return cast(torch.Tensor, loss)


def remake_masks(
    images: torch.Tensor,
    masks: torch.Tensor,
    targets: torch.Tensor,
    players_to_mask: Literal["best", "worst", "central", "peripheral", "random"],
    num_players: int,
    shap_values: torch.Tensor | None
) -> torch.Tensor:
    """
    Remake masks in a given batch according to a given strategy, keeping the same percentage of 0s and 1s.

    Args:
    - images: (B, C, H, W)
    - masks: (B, num_masks_per_image, num_players)
    - targets: (B,)
    - players_to_mask:
        - "best", "worst": mask (zero out) the players with the highest/lowest SHAP values for the target class,
            as estimated by the explainer.
        - "central", "peripheral": mask the players closest to/furthest from the center of the image.
        - "random": mask random players (resampled for each item in the batch).
    - shap_values: (B, num_players, num_classes) - explainer's SHAP values for all classes;
        can be None for strategies other than "best", "worst"

    Returns new masks of shape (B, num_masks_per_image, explainer.num_players)
    """
    B = images.shape[0]

    # From the masks in the batch we only take the percaentage of 0s and 1s.
    num_zeroes = masks.shape[2] - masks.sum(dim=2)  # (B, num_masks_per_image)
    # Rescale to the wanted number of players, keeping the same percentage of 0s and 1s.
    num_zeroes = (num_zeroes * num_players / masks.shape[2]).round()

    if players_to_mask in ["best", "worst"]:
        players_to_mask = cast(Literal["best", "worst"], players_to_mask)
        assert shap_values is not None
        # shap_values[b, p] := shap_values[b, p, targets[b]]
        shap_values = shap_values[torch.arange(shap_values.shape[0]), :, targets]  # (B, num_players)
        assert shap_values.shape == (B, num_players)
        return make_masks_from_player_values(
            num_zeroes=num_zeroes, player_values=shap_values, players_to_mask=players_to_mask
        )
    elif players_to_mask in ["central", "peripheral", "random"]:
        distances = get_distances_from_center(num_players).to(images.device).expand(B, num_players)
        if players_to_mask == "central":
            players_to_mask = "worst"
        elif players_to_mask == "peripheral":
            players_to_mask = "best"
        return make_masks_from_player_values(
            num_zeroes=num_zeroes, player_values=distances, players_to_mask=players_to_mask
        )
    else:
        raise ValueError(f"{players_to_mask=}")


def get_masked_accuracy(
    images: torch.Tensor,
    masks: torch.Tensor,
    targets: torch.Tensor,
    surrogate: torch.nn.Module,
    shap_values: torch.Tensor,
    players_to_mask: Literal["best", "worst"]
) -> float:
    """
    Get the accuracy of surrogate on batch when best/worst patches are masked, according to shap_values.

    Args:
    - images: (B, C, H, W)
    - masks: (B, num_masks_per_image, num_players)
    - targets: (B,)
    - surrogate
    - shap_values: (B, num_players, num_classes)

    Returns the accuracy in 0..100.
    """
    num_players = masks.shape[2]
    remade_masks = remake_masks(images, masks, targets, players_to_mask, num_players, shap_values)
    remade_masks = remade_masks.to(images.device)
    remade_images, remade_masks, remade_targets = apply_masks_to_batch(images, remade_masks, targets)
    logits = surrogate(remade_images)
    _, predicted = logits.max(dim=1)
    correct_count: float = predicted.eq(remade_targets).sum().item()
    total = remade_targets.shape[0]
    return 100.0 * correct_count / total


def quick_test_masked(
    surrogate: torch.nn.Module,
    explainer: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: Any = "cuda"
) -> tuple[float, float]:
    """
    Test the accuraccy of surrogate when the best/worst-valued patches by explainer are masked.

    Return two 0..100 accuracies for best-masked and worst-masked cases.
    The explainer explains surrogate well if the best-masked is significantly lower than worst-masked (like 65% vs 96%).
    """
    explainer.eval()
    surrogate.eval()
    with torch.no_grad():
        explainer.to(device)
        surrogate.to(device)
        n_batches = 0
        accuracies = {"best": 0., "worst": 0.}
        with tqdm(dataloader, ) as dataloader_progress:
            for batch in dataloader_progress:
                images, masks, targets = batch['images'], batch['masks'], batch['labels']
                images, masks, targets = images.to(device), masks.to(device), targets.to(device)
                shap_values = explainer(images)

                for players_to_mask in ["best", "worst"]:
                    players_to_mask = cast(Literal["best", "worst"], players_to_mask)
                    accuracies[players_to_mask] += get_masked_accuracy(
                        images, masks, targets, surrogate, shap_values, players_to_mask
                    )
                n_batches += 1
                dataloader_progress.set_postfix_str(
                    f"Masked-best-accuracy: {accuracies['best'] / n_batches:.2f}%,"
                    + f" Masked-worst-accuracy: {accuracies['worst'] / n_batches:.2f}%")
    return accuracies['best'] / n_batches, accuracies['worst'] / n_batches
