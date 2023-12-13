import logging
from typing import Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchvision import models as cnn_models


import models.t2t_vit
from vit_shapley.modules import surrogate_utils
from utils import load_for_transfer_learning


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

    def __init__(self,
                 output_dim: int,
                 target_model: Optional[nn.Module],
                 learning_rate: Optional[float],
                 weight_decay: Optional[float],
                 decay_power: Optional[str],
                 warmup_steps: Optional[int]):

        super().__init__()
        self.save_hyperparameters(ignore=['target_model'])
        self.target_model = target_model

        # Backbone initialization

        self.backbone = models.t2t_vit.t2t_vit_19(num_classes=10)
        # state_dict = load_state_dict(self.backbone, checkpoint_path='../models/81.5_T2T_ViT_14.pth.tar',
        #                             num_classes=10)
        # self.backbone.load_state_dict(state_dict, strict=False)
        # load_for_transfer_learning(self.backbone, checkpoint_path='../models/81.5_T2T_ViT_14.pth.tar',
        #                             num_classes=10)

        # Nullify classification head built in the backbone module and rebuild.
        head_in_features = self.backbone.head.in_features
        self.backbone.head = nn.Identity()

        self.head = nn.Linear(head_in_features, self.hparams["output_dim"])

        # Set `num_players` variable.
        self.num_players = 196  # 14 * 14

        # Set up modules for calculating metric
        surrogate_utils.set_metrics(self)

    def configure_optimizers(self):
        return surrogate_utils.set_schedule(self)

    def forward(self, images, masks):
        assert masks.shape[-1] == self.num_players

        if images.shape[2:4] == (224, 224) and masks.shape[1] == 196:
            masks = masks.reshape(-1, 14, 14)
            masks = torch.repeat_interleave(torch.repeat_interleave(masks, 16, dim=2), 16, dim=1)
            # masks = masks.reshape(-1, 4, 4)
            # masks = torch.repeat_interleave(torch.repeat_interleave(masks, 56, dim=2), 56, dim=1)
        else:
            raise NotImplementedError
        images_masked = images * masks.unsqueeze(1)
        # print(f'masks.unsqueeze(1) shape is {masks.unsqueeze(1).shape}')
        # print(f'images_masked shape is {images_masked.shape}')
        out = self.backbone(images_masked)
        logits = self.head(out)[:,0].unsqueeze(1)
        
        output = {'logits': logits}

        return output

    def training_step(self, batch, batch_idx):
        assert self.target_model is not None
        images, masks = batch["images"], batch["masks"]
        logits = self(images, masks)['logits']
        self.target_model.eval()
        with torch.no_grad():
            logits_target = self.target_model(images.to(self.target_model.device))['logits'].to(
                self.device)
        loss = surrogate_utils.compute_metrics(self, logits=logits, logits_target=logits_target, phase='train')
        return loss

    def training_epoch_end(self, outs):
        surrogate_utils.epoch_wrapup(self, phase='train')

    def validation_step(self, batch, batch_idx):
        assert self.target_model is not None
        images, masks = batch["images"], batch["masks"]
        logits = self(images, masks)['logits']
        self.target_model.eval()
        with torch.no_grad():
            logits_target = self.target_model(images.to(self.target_model.device))['logits'].to(
                self.device)
        loss = surrogate_utils.compute_metrics(self, logits=logits, logits_target=logits_target, phase='val')

    def validation_epoch_end(self, outs):
        surrogate_utils.epoch_wrapup(self, phase='val')

    def test_step(self, batch, batch_idx):
        assert self.target_model is not None
        images, masks = batch["images"], batch["masks"]
        logits = self(images, masks)['logits']
        self.target_model.eval()
        with torch.no_grad():
            logits_target = self.target_model(images.to(self.target_model.device))['logits'].to(
                self.device)
        loss = surrogate_utils.compute_metrics(self, logits=logits, logits_target=logits_target, phase='test')

    def test_epoch_end(self, outs):
        surrogate_utils.epoch_wrapup(self, phase='test')
