import argparse
import logging
from typing import Optional, cast

import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import RichProgressBar
from torch.nn import functional as F

from vit_shapley.modules.surrogate import Surrogate
import os
from collections import OrderedDict

import models.t2t_vit
from vit_shapley.modules import explainer_utils
from vit_shapley.CIFAR_10_Dataset import PROJECT_ROOT, CIFAR_10_Datamodule, apply_masks_to_batch
from utils import load_checkpoint


class Explainer(pl.LightningModule):
    """
    `pytorch_lightning` module for surrogate
        output_dim: the dimension of output,

        explainer_head_num_attention_blocks:
        explainer_head_mlp_layer_ratio:
        explainer_norm:

        surrogate: 'surrogate' is a model takes masks as input
        efficiency_lambda: lambda hyperparameter for efficiency penalty.
        efficiency_class_lambda: lambda hyperparameter for efficiency penalty.
        checkpoint_metric: the metric used to determine whether
            to save the current status as checkpoints during the validation phase
        optim_type: type of optimizer for optimizing parameters
        learning_rate: learning rate of optimizer
        weight_decay: weight decay of optimizer
        decay_power: only `cosine` annealing scheduler is supported currently
        warmup_steps: parameter for the `cosine` annealing scheduler
    """

    def __init__(self, output_dim: int,
                 explainer_head_num_attention_blocks: int,
                 explainer_head_mlp_layer_ratio: int, explainer_norm: bool,
                 surrogate: pl.LightningModule, efficiency_lambda,
                 efficiency_class_lambda,
                 freeze_backbone: str,
                 checkpoint_metric: Optional[str],
                 learning_rate: Optional[float],
                 weight_decay: Optional[float],
                 decay_power: Optional[str],
                 warmup_steps: Optional[int]):
        super().__init__()
        self.save_hyperparameters(ignore=['surrogate'])
        self.surrogate = surrogate

        # self.__null = None
        self.__null: torch.Tensor

        self.logger_ = logging.getLogger(__name__)

        self.backbone = models.t2t_vit.t2t_vit_14(num_classes=output_dim)
        backbone_path = PROJECT_ROOT / "saved_models/transferred/cifar10/ckpt_0.01_0.0005_97.5.pth"
        load_checkpoint(backbone_path, self.backbone, ignore_keys=["head.weight", "head.bias"])

        # Nullify classification head built in the backbone module and rebuild.
        self.backbone.head = nn.Identity()

        self.attention_blocks = nn.ModuleList([nn.MultiheadAttention(
                                embed_dim=self.backbone.num_features, 
                                num_heads=4) for _ in range(self.hparams['explainer_head_num_attention_blocks'])])
        
        self.attention_blocks[0].norm1 = nn.Identity()

        # mlps
        mlps_list = list[nn.Module]()
        mlps_list.append(nn.LayerNorm(self.backbone.num_features))
        mid_dim = int(self.hparams["explainer_head_mlp_layer_ratio"] * self.backbone.num_features)
        mlps_list.append(nn.Linear(
                in_features=self.backbone.num_features,
                out_features=mid_dim
            ))
        mlps_list.append(self.backbone.blocks[0].mlp.act.__class__())
        mlps_list.append(nn.Linear(
                in_features=mid_dim,
                out_features=self.hparams["output_dim"]
            ))

        self.mlps = nn.Sequential(*mlps_list)

        # Set up normalization.
        # (batch, num_players, num_classes), (batch, 1, num_classes), (batch, 1, num_classes)
        self.normalization = (
            lambda pred, grand, null:
                pred + ((grand - null) - torch.sum(pred, dim=1)).unsqueeze(1) / pred.shape[1]
        )

        # Set up normalization.
        self.normalization_class = lambda pred: pred - torch.sum(pred, dim=2).unsqueeze(2) / pred.shape[2]

        # freeze backbone
        if self.hparams["freeze_backbone"] == 'all':
            for key, value in self.backbone.named_parameters():
                value.requires_grad = False

        elif self.hparams["freeze_backbone"] == 'except_last_two':
            for key, value in self.backbone.named_parameters():
                value.requires_grad = False
            for key, value in self.backbone.blocks[-2].named_parameters():
                value.requires_grad = True
            for key, value in self.backbone.blocks[-1].named_parameters():
                value.requires_grad = True
            for key, value in self.backbone.norm.named_parameters():
                value.requires_grad = True

        # Set up modules for calculating metric
        explainer_utils.set_metrics(self)

    def configure_optimizers(self):
        return explainer_utils.set_schedule(self)

    def null(self, images: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        calculate or load cached null

        Args:
            images: torch.Tensor (batch, channel, height, width)
        Returns:
            values: torch.Tensor (1, num_classes)
        """
        if hasattr(self, '__null'):
            return self.__null
        else:
            if images is not None:
                self.surrogate.eval()
                with torch.no_grad():
                    surrogate_device = cast(torch.device, self.surrogate.device)
                    images = images[0:1].to(surrogate_device)
                    masks = torch.zeros(1, cast(int, self.surrogate.num_players), device=surrogate_device)
                    
                    images_masked, _, _ = apply_masks_to_batch(images, masks)

                    logits = self.surrogate(images_masked) #['logits']

                    self.__null = torch.nn.Softmax(dim=1)(logits).to(self.device)
                    # (batch, channel, height, weight) -> (1, num_classes)
                return self.__null
            else:
                raise RuntimeError(
                    "You should call explainer.null(x) at least once to get null value."
                    + " As 'x' is just used for guessing the shape of input, any dummy variable is okay.")

    def grand(self, images):
        self.surrogate.eval()
        with torch.no_grad():
            surrogate_device = cast(torch.device, self.surrogate.device)
            images = images.to(surrogate_device)  # (batch, channel, height, weight)
            masks = torch.ones(images.shape[0], cast(int, self.surrogate.num_players), device=surrogate_device)
            
            images_masked, _, _ = apply_masks_to_batch(images, masks)
            logits = self.surrogate(images_masked) # (batch, num_players)
            
            grand = torch.nn.Softmax(dim=1)(logits).to(self.device)
                        # (1, num_classes)
        return grand

    def surrogate_multiple_masks(self, images, multiple_masks=None):
        """
        forward pass for embedded surrogate model.
        Args:
            images: torch.Tensor (batch, channel, height, width)
            multiple_masks: torch.Tensor (batch, num_mask_samples, num_players)

        Returns:
            surrogate_values: torch.Tensor (batch, num_mask_samples, num_classes)

        """
        # evaluate surrogate
        self.surrogate.eval()
        with torch.no_grad():
            # mask
            assert len(multiple_masks.shape) == 3  # (batch, num_mask_samples, num_players)
            batch_size = multiple_masks.shape[0]
            assert len(multiple_masks) == len(images)
            num_mask_samples = multiple_masks.shape[1]
            assert self.surrogate.num_players == multiple_masks.shape[2]
            surrogate_device = cast(torch.device, self.surrogate.device)

            images = images.repeat_interleave(num_mask_samples, dim=0).to(surrogate_device)
            # (batch, channel, height, weight) -> (batch * num_mask_samples, channel, height, weight)
            masks=multiple_masks.flatten(0, 1).unsqueeze(1).to(surrogate_device)

            images_masked, _, _ = apply_masks_to_batch(images, masks)

            surrogate_values = torch.nn.Softmax(dim=1)(self.surrogate(images_masked))\
                .reshape(batch_size, num_mask_samples, -1).to(self.device)

            # surrogate_values = torch.nn.Softmax(dim=1)(self.surrogate(
            #     images_masked=images.repeat_interleave(num_mask_samples, dim=0).to(surrogate_device),
            #     # (batch, channel, height, weight) -> (batch * num_mask_samples, channel, height, weight)
            #     masks=multiple_masks.flatten(0, 1).unsqueeze(1).to(surrogate_device)
            #     # (batch, num_mask_samples, num_players) -> (batch * num_mask_samples, num_players)
            # )).reshape(batch_size, num_mask_samples, -1).to(self.device)

        return surrogate_values
    
    def forward_features(self, x):
        B = x.shape[0]
        x = self.backbone.tokens_to_token(x)
        cls_tokens = self.backbone.cls_token.expand(B, -1, -1)

        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.backbone.pos_embed
        x = self.backbone.pos_drop(x)

        for blk in self.backbone.blocks:
            x = blk(x)

        x = self.backbone.norm(x)
        return x[:, :self.surrogate.num_players]

    def forward(self, images, surrogate_grand=None, surrogate_null=None):
        """
        forward pass
        Args:
            residual:
            surrogate_grand:
            surrogate_null:
            images: torch.Tensor (batch, channel, height, width)

        Returns:
            pred: torch.Tensor (batch, num_players, num_classes)
            pred_sum: torch.Tensor (batch, num_classes)

        """
        self.backbone.forward_features = self.forward_features
        embedding_all = self.backbone(x=images)
        
        for i, layer_module in enumerate(self.attention_blocks):
            layer_outputs = layer_module(embedding_all, embedding_all, embedding_all)
            embedding_all = layer_outputs[0]  

        pred = self.mlps(embedding_all)
        pred = pred.tanh()
       
        if self.normalization:
            if surrogate_grand is None:
                # (batch, channel, height, weight) -> (batch, num_classes)
                surrogate_grand = self.grand(images).to(self.device)
            if surrogate_null is None:
                # (batch, channel, height, weight) -> (1, num_classes)
                surrogate_null = self.null(images).to(self.device)
            
            pred = self.normalization(pred=pred, grand=surrogate_grand, null=surrogate_null)

        if self.normalization_class:
            pred = self.normalization_class(pred=pred)

        pred_sum = pred.sum(dim=1)  # (batch, num_players, num_classes) -> (batch, num_classes)
        pred_sum_class = pred.sum(dim=2)  # (batch, num_players, num_classes) -> (batch, num_players)

        return pred, pred_sum, pred_sum_class

    def training_step(self, batch, batch_idx):
        images, masks = batch["images"], batch["masks"]

        # evaluate surrogate
        # (batch, channel, height, width), (batch, num_mask_samples, num_players) ->
        # -> (batch, num_mask_samples, num_classes)
        surrogate_values = self.surrogate_multiple_masks(images, masks)
        surrogate_grand = self.grand(images)  # (batch, channel, height, weight) -> (batch, num_classes)
        surrogate_null = self.null(images)  # (batch, channel, height, weight) -> (1, num_classes)

        # evaluate explainer
        values_pred, value_pred_beforenorm_sum, value_pred_beforenorm_sum_class = self(
            images, surrogate_grand=surrogate_grand, surrogate_null=surrogate_null
        )  # (batch, channel, height, weight) -> (batch, num_players, num_classes), (batch, num_classes)

        value_pred_approx = surrogate_null + masks.float() @ values_pred
        # (1, num_classes) + (batch, num_mask_samples, num_players) @ (batch, num_players, num_classes) ->
        # -> (batch, num_mask_samples, num_classes)

        loss = explainer_utils.compute_metrics(self,
                                               num_players=self.surrogate.num_players,
                                               values_pred=value_pred_approx,
                                               values_target=surrogate_values,
                                               efficiency_lambda=self.hparams["efficiency_lambda"],
                                               value_pred_beforenorm_sum=value_pred_beforenorm_sum,
                                               surrogate_grand=surrogate_grand,
                                               surrogate_null=surrogate_null,
                                               efficiency_class_lambda=self.hparams["efficiency_class_lambda"],
                                               value_pred_beforenorm_sum_class=value_pred_beforenorm_sum_class,
                                               phase='train')

        return loss

    def validation_step(self, batch, batch_idx):
        images, masks = batch["images"], batch["masks"]

        # evaluate surrogate
        surrogate_values = self.surrogate_multiple_masks(images, masks)

        surrogate_grand = self.grand(images)  # (batch, channel, height, weight) -> (batch, num_classes)
        surrogate_null = self.null(images)  # (batch, channel, height, weight) -> (1, num_classes)

        # evaluate explainer
        values_pred, value_pred_beforenorm_sum, value_pred_beforenorm_sum_class = self(
            images, surrogate_grand=surrogate_grand, surrogate_null=surrogate_null
        )
        # (batch, channel, height, weight) -> (batch, num_players, num_classes), (batch, num_classes)

        value_pred_approx = surrogate_null + masks.float() @ values_pred
        # (1, num_classes) + (batch, num_mask_samples, num_players) @ (batch, num_players, num_classes) ->
        # -> (batch, num_mask_samples, num_classes)
  

        loss = explainer_utils.compute_metrics(self,
                                               num_players=self.surrogate.num_players,
                                               values_pred=value_pred_approx,
                                               values_target=surrogate_values,
                                               efficiency_lambda=self.hparams["efficiency_lambda"],
                                               value_pred_beforenorm_sum=value_pred_beforenorm_sum,
                                               surrogate_grand=surrogate_grand,
                                               surrogate_null=surrogate_null,
                                               efficiency_class_lambda=self.hparams["efficiency_class_lambda"],
                                               value_pred_beforenorm_sum_class=value_pred_beforenorm_sum_class,
                                               phase='val')

        return loss

    def test_step(self, batch, batch_idx):
        images, masks = batch["images"], batch["masks"]

        # evaluate surrogate
        surrogate_values = self.surrogate_multiple_masks(images, masks)
        surrogate_grand = self.grand(images)  # (batch, channel, height, weight) -> (batch, num_classes)
        surrogate_null = self.null(images)  # (batch, channel, height, weight) -> (1, num_classes)

        # evaluate explainer
        values_pred, value_pred_beforenorm_sum, value_pred_beforenorm_sum_class = self(
            images, surrogate_grand=surrogate_grand, surrogate_null=surrogate_null
        )
        # (batch, channel, height, weight) ->
        # -> (batch, num_players, num_classes), (batch, num_classes), (batch, num_players)

        value_pred_approx = surrogate_null + masks.float() @ values_pred
        # (1, num_classes) + (batch, num_mask_samples, num_players) @ (batch, num_players, num_classes) ->
        # -> (batch, num_mask_samples, num_classes)

        loss = explainer_utils.compute_metrics(self,
                                               num_players=self.surrogate.num_players,
                                               values_pred=value_pred_approx,
                                               values_target=surrogate_values,
                                               efficiency_lambda=self.hparams["efficiency_lambda"],
                                               value_pred_beforenorm_sum=value_pred_beforenorm_sum,
                                               surrogate_grand=surrogate_grand,
                                               surrogate_null=surrogate_null,
                                               efficiency_class_lambda=self.hparams["efficiency_class_lambda"],
                                               value_pred_beforenorm_sum_class=value_pred_beforenorm_sum_class,
                                               phase='test')
        return loss


def main() -> None:    
    torch.set_float32_matmul_precision('medium')
    
    parser = argparse.ArgumentParser(description="Explainer training")
    parser.add_argument("--label", required=False, default="", type=str, help="label for checkpoints")
    parser.add_argument("--num_players", required=True, type=int, help="number of players")
    parser.add_argument("--lr", required=False, default=1e-4, type=float, help="explainer learning rate")
    parser.add_argument("--wd", required=False, default=0.0, type=float, help="explainer weight decay")
    parser.add_argument("--lr_s", required=False, default=1e-5, type=float, help="surrogate learning rate")
    parser.add_argument("--wd_s", required=False, default=0.0, type=float, help="surrogate weight decay")
    parser.add_argument("--b", required=False, default=128, type=int, help="batch size")
    parser.add_argument("--num_workers", required=False, default=0, type=int, help="number of dataloader workers")
    parser.add_argument("--num_atts", required=False, default=1, type=int, help="number of attention blocks")
    parser.add_argument("--mlp_ratio", required=False, default=4, type=int, help="ratio for the middle layer in mlps")
    
    args = parser.parse_args()



    target_model = models.t2t_vit.t2t_vit_14(num_classes=10)
    target_model_path = PROJECT_ROOT / "saved_models/transferred/cifar10/ckpt_0.01_0.0005_97.5.pth"
    load_checkpoint(target_model_path, target_model)

    surrogate = Surrogate.load_from_checkpoint(f"{PROJECT_ROOT}/saved_models/surrogate/cifar10/_player16_lr1e-05_wd0.0_b256_epoch28.ckpt",
                                        output_dim=10,
                                        target_model=target_model,
                                        learning_rate=args.lr_s,
                                        weight_decay=args.wd_s,
                                        decay_power='cosine',
                                        warmup_steps=2,
                                        num_players=args.num_players
    )
    
    explainer = Explainer(
        output_dim=10,
        explainer_head_num_attention_blocks=args.num_atts,
        explainer_head_mlp_layer_ratio=args.mlp_ratio,
        explainer_norm=True,
        surrogate=surrogate,
        efficiency_lambda=0,
        efficiency_class_lambda=0,
        freeze_backbone='except_last_two',
        checkpoint_metric='loss',
        learning_rate=args.lr,
        weight_decay=args.wd,
        decay_power=None,
        warmup_steps=None
    )


    CIFAR_10 = CIFAR_10_Datamodule(num_players=surrogate.num_players, 
                                   num_mask_samples=2, 
                                   paired_mask_samples=True)


    datamodule = CIFAR_10_Datamodule(
        num_players=args.num_players,
        num_mask_samples=2,
        paired_mask_samples=True,
        batch_size=args.b,
        num_workers=args.num_workers,
    )

    log_and_checkpoint_dir = (
        PROJECT_ROOT
        / "checkpoints"
        / "explainer"
        / f"{args.label}_player{args.num_players}_lr{args.lr}_wd{args.wd}_b{args.b}"
    )
    log_and_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    trainer = pl.Trainer(max_epochs=100, 
                         default_root_dir=log_and_checkpoint_dir,
                         callbacks=RichProgressBar(leave=True)
    )
    trainer.fit(explainer, datamodule)
    
if __name__ == '__main__':
    main()
