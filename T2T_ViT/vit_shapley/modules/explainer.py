import logging
from typing import Optional, cast

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.nn import functional as F

from vit_shapley.modules.surrogate import Surrogate
import os
from collections import OrderedDict

import models.t2t_vit
from vit_shapley.modules import explainer_utils
from vit_shapley.CIFAR_10_Dataset import CIFAR_10_Datamodule
from utils import load_for_transfer_learning

torch.set_float32_matmul_precision('medium') # | 'high')

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

        self.backbone = models.t2t_vit.t2t_vit_14(num_classes=10)

        # Nullify classification head built in the backbone module and rebuild.

        # head_in_features = self.backbone.head.in_features
        self.backbone.head = nn.Identity()

        # if self.hparams.explainer_head_num_attention_blocks == 0:
        self.attention_blocks = nn.ModuleList([nn.MultiheadAttention(
                                embed_dim=self.backbone.num_features, 
                                num_heads=4) for _ in range(1)])
        
        self.attention_blocks[0].norm1 = nn.Identity()

        # else:
        #     self.attention_blocks = nn.ModuleList([self.backbone.blocks[0].attn
        #                                     for _ in range(self.hparams.explainer_head_num_attention_blocks)])


        # mlps
        mlps_list = list[nn.Module]()
        if self.hparams["explainer_norm"] and self.hparams["explainer_head_num_attention_blocks"] > 0:
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

        # Load checkpoints
        # if self.hparams.load_path is not None:
        #     if load_path_state_dict:
        #         state_dict = torch.load(self.hparams.load_path, map_location="cpu")
        #     else:
        #         checkpoint = torch.load(self.hparams.load_path, map_location="cpu")
        #         state_dict = checkpoint["state_dict"]
        #     ret = self.load_state_dict(state_dict, strict=False)
        #     self.logger_.info(f"Model parameters were updated from a checkpoint file {self.hparams.load_path}")
        #     self.logger_.info(f"Unmatched parameters - missing_keys:    {ret.missing_keys}")
        #     self.logger_.info(f"Unmatched parameters - unexpected_keys: {ret.unexpected_keys}")

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
                    logits = self.surrogate(images, masks) #['logits']

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
            logits = self.surrogate(images=images, masks=masks) #['logits']  # (batch, num_players)
            # logits = self.surrogate(images=images[:, 0:1], masks=masks)['logits']
            grand = torch.nn.Softmax(dim=1)(logits).to(self.device) #[:,0][0].unsqueeze(0)
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

            surrogate_values = torch.nn.Softmax(dim=1)(self.surrogate(
                images=images.repeat_interleave(num_mask_samples, dim=0).to(surrogate_device),
                # (batch, channel, height, weight) -> (batch * num_mask_samples, channel, height, weight)
                masks=multiple_masks.flatten(0, 1).unsqueeze(1).to(surrogate_device)
                # (batch, num_mask_samples, num_players) -> (batch * num_mask_samples, num_players)
            )).reshape(batch_size, num_mask_samples, -1).to(self.device) 

        return surrogate_values
    
    def forward_features(self, x):
        x = self.backbone.forward_features(x)
        return x[:, :-1]

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

        pred = pred #[:, :-1] # added to match shape 196 instead of 197

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

        mse_loss = F.mse_loss(input=value_pred_approx, target=surrogate_values, reduction='mean')
        value_diff = self.surrogate.num_players * mse_loss


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

    # def training_epoch_end(self, outs):
    def on_training_epoch_end(self):
        explainer_utils.epoch_wrapup(self, phase='train')
        # self.outs.clear()

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
        # było po prostu surrogate_null
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

    # def validation_epoch_end(self, outs):
    def on_validation_epoch_end(self):
        explainer_utils.epoch_wrapup(self, phase='val')

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
        # było surrogate_null
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

    # def test_epoch_end(self, outs):
    def on_test_epoch_end(self):
        explainer_utils.epoch_wrapup(self, phase='test')


if __name__ == '__main__':
    target_model = models.t2t_vit.t2t_vit_14(num_classes=10)

    # num_classes=10
    checkpoint = torch.load('../../checkpoint_cifar10_T2t_vit_14/ckpt_0.01_0.0005_97.5.pth')

    new_state_dict = OrderedDict()

    for k, v in checkpoint.items():
        # strip `module.` prefix
        name = k[7:] if k.startswith('module') else k
        new_state_dict[name] = v

    state_dict = new_state_dict

    target_model.load_state_dict(state_dict, strict=False)
    
    surrogate = Surrogate.load_from_checkpoint("checkpoints_surrogate/lightning_logs/version_0/checkpoints/epoch=49-step=70350.ckpt",
                                        output_dim=10,
                                        target_model=target_model,
                                        learning_rate=1e-5,
                                        weight_decay=0.0,
                                        decay_power='cosine',
                                        warmup_steps=2,
    )
    # surrogate = Surrogate(output_dim=10,
    #                       target_model=None,
    #                       learning_rate=1e-5,
    #                       weight_decay=0.,
    #                       decay_power='cosine', 
    #                       warmup_steps=2                         
    #                       )

    CIFAR_10 = CIFAR_10_Datamodule(num_players=196, 
                                   num_mask_samples=2, 
                                   paired_mask_samples=True)

    explainer = Explainer(
        output_dim=10,
        explainer_head_num_attention_blocks=1,
        explainer_head_mlp_layer_ratio=4,
        explainer_norm=True,
        surrogate=surrogate,
        efficiency_lambda=0,
        efficiency_class_lambda=0,
        freeze_backbone='except_last_two',
        checkpoint_metric='loss',
        learning_rate=1e-4,
        weight_decay=0.0,
        decay_power=None,
        warmup_steps=None
    )

    print(explainer)

    # trainer = pl.Trainer(max_epochs=100, default_root_dir="./checkpoints_explainer")# logger=False)
    # trainer.fit(explainer, CIFAR_10)
    