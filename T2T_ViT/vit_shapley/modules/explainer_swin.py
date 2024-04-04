import argparse
from typing import Literal, Optional, cast
import math

import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import RichProgressBar
from transformers import SwinConfig, \
    SwinForImageClassification


import models.t2t_vit
from utils import load_checkpoint
from vit_shapley.CIFAR_10_Dataset import PROJECT_ROOT, CIFAR_10_Datamodule, apply_masks
from vit_shapley.modules import explainer_utils
from vit_shapley.modules.surrogate import Surrogate


class Explainer(pl.LightningModule):
    """
    `pytorch_lightning` module for explainer.
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

    def __init__(
        self,
        output_dim: int,
        backbone_name: str,
        explainer_head_num_attention_blocks: int,
        explainer_head_mlp_layer_ratio: int,
        explainer_norm: bool,
        efficiency_lambda: float,
        efficiency_class_lambda: float,
        freeze_backbone: Literal["all", "except_last_two", "none"],
        checkpoint_metric: Optional[str],
        learning_rate: Optional[float],
        weight_decay: Optional[float],
        decay_power: Optional[str],
        warmup_steps: Optional[int],
        use_convolution: bool,
        surrogate: Optional[pl.LightningModule] = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["surrogate"])
        self.backbone_name = backbone_name
        assert surrogate is not None
        self.surrogate = surrogate

        self.__null: Optional[torch.Tensor] = None

        # Backbone initialization
        if self.backbone_name == 't2t_vit':
            self.backbone = models.t2t_vit.t2t_vit_14(num_classes=output_dim)
            # backbone_path = PROJECT_ROOT / "saved_models/downloaded/imagenet/81.5_T2T_ViT_14.pth"
            # backbone_path = PROJECT_ROOT / "saved_models/downloaded/cifar10/cifar10_t2t-vit_14_98.3.pth"
            backbone_path = PROJECT_ROOT / "saved_models/transferred/cifar10/ckpt_0.01_0.0005_97.5.pth"
            load_checkpoint(backbone_path, self.backbone, ignore_keys=["head.weight", "head.bias"])        
            
            # Nullify classification head built in the backbone module and rebuild.
            head_in_features = self.backbone.head.in_features
            self.backbone.head = nn.Identity()
            self.backbone.forward_features = self.backbone_forward_features
            
        elif self.backbone_name == 'swin':
            configuration = SwinConfig()
            self.backbone = SwinForImageClassification(configuration) \
                            .from_pretrained("microsoft/swin-tiny-patch4-window7-224",\
                            num_labels=10,
                            ignore_mismatched_sizes=True)\
                            .to(self.device)

            backbone_path = PROJECT_ROOT / "checkpoints/transfer/cifar10/default/swin_epoch-4_acc-96.06.pth"
            state_dict = torch.load(backbone_path)
            self.backbone.load_state_dict(state_dict, strict=False)

            # Nullify classification head built in the backbone module and rebuild.
            head_in_features = self.backbone.classifier.in_features
            self.backbone.pooling = nn.Identity()
            self.backbone.classifier = nn.Identity()
            self.head = nn.Linear(head_in_features, self.hparams["output_dim"])


        self.attention_blocks = nn.ModuleList(
            [
                nn.MultiheadAttention(embed_dim=head_in_features, num_heads=4, batch_first=True)
                for _ in range(explainer_head_num_attention_blocks)
            ]
        )
        if explainer_head_num_attention_blocks:
            self.attention_blocks[0].norm1 = nn.Identity()

        # mlps
        mid_dim = int(explainer_head_mlp_layer_ratio * head_in_features)
        self.mlps = nn.Sequential(
            nn.LayerNorm(head_in_features),
            nn.Linear(in_features=head_in_features, out_features=mid_dim),
            nn.ReLU(),
            nn.Linear(in_features=mid_dim, out_features=output_dim),
            )

        self.use_convolution = use_convolution
        # https://ezyang.github.io/convolution-visualizer/index.html
        num_players_to_conv2d_params = {
            4: dict(kernel_size=7, padding=0, stride=7),
            9: dict(kernel_size=6, padding=1, stride=5),
            16: dict(kernel_size=4, padding=1, stride=4),
            25: dict(kernel_size=4, padding=1, stride=3),  # or (2,0,3)
            36: dict(kernel_size=4, padding=1, stride=2),
            49: dict(kernel_size=2, padding=0, stride=2),
            64: dict(kernel_size=2, padding=1, stride=2),
            81: dict(kernel_size=6, padding=0, stride=1),
            100: dict(kernel_size=5, padding=0, stride=1),
            121: dict(kernel_size=4, padding=0, stride=1),
            144: dict(kernel_size=3, padding=0, stride=1),
            169: dict(kernel_size=2, padding=0, stride=1),
            196: dict(kernel_size=1, padding=0, stride=1),
        }
        conv2d_params = num_players_to_conv2d_params[self.surrogate.num_players]
        if use_convolution:
            self.conv = torch.nn.Conv2d(in_channels=self.backbone.num_features,
                out_channels=self.backbone.num_features,
                kernel_size=conv2d_params['kernel_size'],
                stride=conv2d_params['stride'],
                padding=conv2d_params['padding']).to(self.device)

        # Set up normalization.
        # First we do "additive efficient normalization",
        # which means adding a constant to ensure that (for each item in the batch and each class)
        # the predicted player contributions sum up to the value for unmasked input minus the value for null input.
        # Pred: (batch, num_players, num_classes), grand: (batch, num_classes), null: (1, num_classes)
        self.normalization = (
            lambda pred, grand, null: pred + ((grand - null) - pred.sum(dim=1)).unsqueeze(1) / pred.shape[1]
        )
        # Since grand values for all classes sum up to 1 and null values sum up to 1, predictions should sum up to 0.
        # So we can optionally force that too.
        self.normalization_class = None  # lambda pred: pred - pred.sum(dim=2).unsqueeze(2) / pred.shape[2]

        # Freeze backbone.
        if freeze_backbone == "all":
            for value in self.backbone.parameters():
                value.requires_grad = False
        elif freeze_backbone == "except_last_two":
            for value in self.backbone.parameters():
                value.requires_grad = False
            for value in self.backbone.blocks[-2].parameters():
                value.requires_grad = True
            for value in self.backbone.blocks[-1].parameters():
                value.requires_grad = True
            for value in self.backbone.norm.parameters():
                value.requires_grad = True
        elif freeze_backbone == "none":
            pass
        else:
            raise ValueError(f"Unknown freeze_backbone value: {freeze_backbone}")

    def configure_optimizers(self):
        return explainer_utils.set_schedule(self)

    def null(self, images: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Class probabilites for a fully masked input (cached after first call).

        Input: (batch, channel, height, width), only the shape is relevant.
        Output: (1, num_classes).
        """
        assert self.surrogate is not None
        if self.__null is not None:
            return self.__null
        if images is None:
            raise RuntimeError("Call explainer.null(images) at least once to get null value.")
        self.surrogate.eval()
        with torch.no_grad():
            images = images[0:1].to(self.surrogate.device)  # Limit batch to one image.
            masks = torch.zeros(1, 1, self.surrogate.num_players, device=self.surrogate.device)
            images_masked = apply_masks(images, masks)  # Mask-out everything.
            logits = self.surrogate(images_masked)  # (1, channel, height, weight) -> (1, num_classes)
            self.__null = torch.nn.Softmax(dim=1)(logits).to(self.device)  # (1, num_classes)
        return self.__null

    def grand(self, images: torch.Tensor) -> torch.Tensor:
        """Class probabilities for unmasked inputs.

        Input: (batch, channel, height, width).
        Output: (batch, num_classes).
        """
        assert self.surrogate is not None
        self.surrogate.eval()
        with torch.no_grad():
            images = images.to(self.surrogate.device)  # (batch, channel, height, weight)
            masks = torch.ones(images.shape[0], 1, self.surrogate.num_players, device=self.surrogate.device)
            images_masked = apply_masks(images, masks)  # (batch, channel, height, weight)
            logits = self.surrogate(images_masked)  # (batch, num_classes)
            grand: torch.Tensor = torch.nn.Softmax(dim=1)(logits).to(self.device)  # (batch, num_classes)
        return grand

    def surrogate_multiple_masks(self, images: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """Get class probabilites from surrogate model, with multiple masks.

        Args:
            images: (batch, channel, height, width).
            masks: (batch, num_mask_samples, num_players).
        Returns class probabilites (batch, num_mask_samples, num_classes).
        """
        assert self.surrogate is not None
        self.surrogate.eval()
        with torch.no_grad():
            batch_size, num_masks_per_image, num_players = masks.shape
            assert (
                num_players == self.surrogate.num_players
            ), f"Surrogate was trained with {self.surrogate.num_players=} != {num_players=}. It's OK but unexpected."
            images, masks = images.to(self.surrogate.device), masks.to(self.surrogate.device)
            images_masked = apply_masks(images, masks)  # (B * num_mask_samples, C, H, W)
            logits = self.surrogate(images_masked)  # (B * num_mask_samples, num_classes)
            surrogate_values: torch.Tensor = torch.nn.Softmax(dim=1)(logits)  # (B * num_mask_samples, num_classes)
        return surrogate_values.reshape(batch_size, num_masks_per_image, -1).to(self.device)

    def backbone_forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Same as self.backbone.forward_features but outputs num_players vectors instead of one.

        Input: images of shape (B, C, H, W).
        Output: embeddings of shape (B, num_players, embed_dim).
        """
        assert self.surrogate is not None
        B = x.shape[0]
        # Apply tokens-to-token layers, which shrink image to embeddings of shape (B, sequence_length, embed_dim).
        # sequence_length = 196 = 14x14 for ViT14 on input images of size 224x224.
        x = self.backbone.tokens_to_token(x)

        # Prepend a learnable token to the sequence.
        cls_tokens = self.backbone.cls_token.expand(B, -1, -1)  # Shape (B, 1, embed_dim).
        x = torch.cat((cls_tokens, x), dim=1)  # Shape (B, 1 + sequence_length, embed_dim).

        # Add position embeddings and apply dropout.
        x = x + self.backbone.pos_embed
        x = self.backbone.pos_drop(x)

        # Apply the transformer blocks (shape unchanged).
        for blk in self.backbone.blocks:
            x = blk(x)

        # Apply layer normalization.
        x = self.backbone.norm(x)[:, 1:, :]
        # Shape is now (B, sequence_length, embed_dim).

        print(f'backbone_forward_features shape is {x.shape}')

        return x

    def forward(self, images: torch.Tensor, surrogate_grand=None, surrogate_null=None, normalize: bool = True) -> torch.Tensor:
        """
        Forward pass.

        Args:
            surrogate_grand:
            surrogate_null:
            images: (batch, channel, height, width)

        Returns predictions of shape (batch, num_players, num_classes).
        """
        assert self.surrogate is not None

        if self.backbone_name == 'swin':
            x = self.backbone.swin(images).last_hidden_state

        else:
            x = self.backbone(images).logits  # (B, seq_length, embed_dim)

        print(f'x shape after backbone is {x.shape}')
        
        B, seq_length, embed_dim = x.shape
        r = int(math.sqrt(seq_length))
        assert seq_length == r * r, f"Expected {seq_length=} to be a perfect square."

        # Reshape to (B, r, r, embed_dim).
        x = x.view(B, r, r, embed_dim)

        s = int(math.sqrt(self.surrogate.num_players))
        assert s * s == self.surrogate.num_players, f"Expected {self.surrogate.num_players=} to be a perfect square."

        # Adapt to (B, s, s, embed_dim).
        if r == s:
            pass
        elif self.use_convolution:
            x = x.permute(0, 3, 1, 2)  # (B, embed_dim, r, r)
            x = self.conv(x)  # (B, embed_dim, s, s)
            assert x.shape[2] == x.shape[3] == s
            x = x.permute(0, 2, 3, 1)  # (B, s, s, embed_dim)
            # embedding_all = x
        else:
            # indices = [int((i + 0.5) * 14 / s) for i in range(s)]
            indices = [int((i + 0.5) * 7 / s) for i in range(s)]
            x = x[:, indices, :, :][:, :, indices, :]  # (B, s, s, embed_dim)
            # embedding_all = x

        # Flatten to (B, num_players, embed_dim)
        x = x.flatten(1, 2)

        for layer_module in self.attention_blocks:
            x, _ = layer_module(x, x, x, need_weights=False)
   
        pred = self.mlps(x)  # (B, num_players, embed_dim) -> # (B, num_players, num_classes)
        pred = pred.tanh()

        if normalize and self.normalization:
            if surrogate_grand is None:
                surrogate_grand = self.grand(images).to(self.device)
            if surrogate_null is None:
                surrogate_null = self.null(images).to(self.device)
            pred = self.normalization(pred=pred, grand=surrogate_grand, null=surrogate_null)

        if normalize and self.normalization_class:
            pred = self.normalization_class(pred=pred)
        
        return cast(torch.Tensor, pred)

    def training_step(self, batch, batch_idx):
        assert self.surrogate is not None
        # Images have shape (batch, channel, height, weight).
        # Masks have shape: (batch, num_masks_per_image, num_players).
        images, masks = batch["images"], batch["masks"]

        # Evaluate surrogate on masked, unmasked and fully masked inputs.
        surrogate_values = self.surrogate_multiple_masks(images, masks)  # (batch, num_masks_per_image, num_classes)
        surrogate_grand = self.grand(images)  # (batch, num_classes)
        surrogate_null = self.null(images)  # (1, num_classes)

        # Evaluate explainer to get shap_values of shape (batch, num_players, num_classes).
        shap_values = self(images, surrogate_grand=surrogate_grand, surrogate_null=surrogate_null)

        # Recall surrogate_values[b,n,c] is the value (the class probability outputted by the surrogate)
        # of classifying images[b] masked with masks[b,n] as class c.
        # We want it to be well-approximated by the sum of shap values for class c of players in masks[b,n]
        # plus the null value for c.
        # That is: surrogate_null[.,c] + sum_p masks[b,n,p] * shap_values[b, p, c]
        values_pred = surrogate_null.unsqueeze(0) + masks.float() @ shap_values
        # Shapes: (1, 1, num_classes) + (batch, num_masks_per_image, num_players) @ (batch, num_players, num_classes)
        # = (batch, num_masks_per_image, num_classes).

        loss = explainer_utils.compute_metrics(
            self,
            num_players=self.surrogate.num_players,
            shap_values=shap_values,
            values_pred=values_pred,
            values_target=surrogate_values,
            surrogate_grand=surrogate_grand,
            surrogate_null=surrogate_null,
            phase="train",
        )

        return loss

    def validation_step(self, batch, batch_idx):
        assert self.surrogate is not None
        images, masks = batch["images"], batch["masks"]

        # Evaluate surrogate.
        surrogate_values = self.surrogate_multiple_masks(images, masks)
        surrogate_grand = self.grand(images)
        surrogate_null = self.null(images)

        # Evaluate explainer.
        shap_values = self(images, surrogate_grand=surrogate_grand, surrogate_null=surrogate_null)

        # Get approximation of surrogate_values from shap_values.
        values_pred = surrogate_null + masks.float() @ shap_values

        loss = explainer_utils.compute_metrics(
            self,
            num_players=self.surrogate.num_players,
            shap_values=shap_values,
            values_pred=values_pred,
            values_target=surrogate_values,
            surrogate_grand=surrogate_grand,
            surrogate_null=surrogate_null,
            phase="val",
        )

        return loss

    def test_step(self, batch, batch_idx):
        assert self.surrogate is not None
        images, masks = batch["images"], batch["masks"]

        # Evaluate surrogate.
        surrogate_values = self.surrogate_multiple_masks(images, masks)
        surrogate_grand = self.grand(images)
        surrogate_null = self.null(images)

        # Evaluate explainer.
        shap_values = self(images, surrogate_grand=surrogate_grand, surrogate_null=surrogate_null)

        # Get approximation of surrogate_values from shap_values.
        values_pred = surrogate_null + masks.float() @ shap_values

        loss = explainer_utils.compute_metrics(
            self,
            num_players=self.surrogate.num_players,
            shap_values=shap_values,
            values_pred=values_pred,
            values_target=surrogate_values,
            surrogate_grand=surrogate_grand,
            surrogate_null=surrogate_null,
            phase="test",
        )
        return loss


def main() -> None:
    torch.set_float32_matmul_precision("medium")

    parser = argparse.ArgumentParser(description="Explainer training")
    parser.add_argument("--label", required=False, default="", type=str, help="label for checkpoints")
    parser.add_argument("--num_players", required=True, type=int, help="number of players")
    parser.add_argument("--lr", required=False, default=1e-4, type=float, help="explainer learning rate")
    parser.add_argument("--wd", required=False, default=0.0, type=float, help="explainer weight decay")
    parser.add_argument("--b", required=False, default=128, type=int, help="batch size")
    parser.add_argument("--num_workers", required=False, default=0, type=int, help="number of dataloader workers")
    parser.add_argument("--num_atts", required=False, default=1, type=int, help="number of attention blocks")
    parser.add_argument("--mlp_ratio", required=False, default=4, type=int, help="ratio for the middle layer in mlps")

    parser.add_argument("--use_conv", required=True, default=False, type=bool, 
                        help="convolutions to match dim num_players")
    parser.add_argument("--freeze_backbone", required=True, default='none', type=str, 
                        help="freeze the backbone")

    args = parser.parse_args()

    # target_model = models.t2t_vit.t2t_vit_14(num_classes=10)

    configuration = SwinConfig()
    target_model_2 = SwinForImageClassification(configuration) \
                            .from_pretrained("microsoft/swin-tiny-patch4-window7-224",\
                            num_labels=10,
                            ignore_mismatched_sizes=True)

    surrogate = Surrogate.load_from_checkpoint(
        # PROJECT_ROOT / "saved_models/surrogate/cifar10/_player16_lr1e-05_wd0.0_b256_epoch28.ckpt",
        PROJECT_ROOT / "checkpoints/surrogate/_swin_player16_lr1e-05_wd0.0_b128/lightning_logs/version_0/checkpoints/epoch=19-step=7040.ckpt",
        target_model=target_model_2,
        backbone_name='swin',
        num_players=args.num_players,
        strict=False
    )

    explainer = Explainer(
        output_dim=10,
        backbone_name='swin',
        explainer_head_num_attention_blocks=args.num_atts,
        explainer_head_mlp_layer_ratio=args.mlp_ratio,
        explainer_norm=True,
        surrogate=surrogate,
        efficiency_lambda=0,
        efficiency_class_lambda=0,
        freeze_backbone=args.freeze_backbone,
        checkpoint_metric="loss",
        learning_rate=args.lr,
        weight_decay=args.wd,
        decay_power=None,
        warmup_steps=None,
        use_convolution=args.use_conv,
    )

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
        / f"use_conv_{args.use_conv}_freeze_{args.freeze_backbone}_{args.label}_player{args.num_players}_lr{args.lr}_wd{args.wd}_b{args.b}"
    )
    log_and_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    trainer = pl.Trainer(max_epochs=50, default_root_dir=log_and_checkpoint_dir, callbacks=RichProgressBar(leave=True))
    trainer.fit(explainer, datamodule)


if __name__ == "__main__":
    main()
