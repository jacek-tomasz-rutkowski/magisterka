import argparse
import datetime
import math
from typing import Callable, Literal, Optional, cast

import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import RichProgressBar

from utils import is_true_string, load_transferred_model
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
        backbone_name: Literal["t2t_vit", "swin", "vit"],
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
        use_tanh: bool = True,
        use_softmax: bool = True,
        divisor: float = 1.0,
        target_class_lambda: float = 0.0,
        num_players: Optional[int] = None,
        surrogate: Optional[pl.LightningModule] = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["surrogate"])
        self.backbone_name = backbone_name
        surrogate_num_players = getattr(surrogate, 'num_players', None)
        assert num_players is not None, f"Please specify num_players to Explainer; {surrogate_num_players=}"
        if surrogate_num_players:
            assert num_players == surrogate_num_players, \
                f"Explainer's {num_players=} != {surrogate_num_players=}. It's technically OK but unexpected."
        assert surrogate is not None, "Please provide a surrogate model to the explainer (even for eval, for grand())."
        self.num_players = num_players
        self.surrogate = surrogate

        self.__null: Optional[torch.Tensor] = None

        # Nullify classification head built in the backbone module and rebuild.
        self.backbone: nn.Module = load_transferred_model(backbone_name)
        if backbone_name == 't2t_vit':
            head_in_features = self.backbone.head.in_features
            self.backbone.head = nn.Identity()
            self.backbone.forward_features = self.backbone_forward_features  # type: ignore
        elif backbone_name == "swin":
            head_in_features = self.backbone.classifier.in_features
            # self.backbone.pooling = nn.Identity() ??
            self.backbone.classifier = nn.Identity()
        elif backbone_name == "vit":
            head_in_features = self.backbone.classifier.in_features
            self.backbone = torch.nn.Sequential(self.backbone.vit.embeddings, self.backbone.vit.encoder)
            # output of the backbone is (b, 197, 768)
        else:
            raise ValueError(f"Unexpected backbone_name: {backbone_name}")

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
        num_players_to_conv2d_params = {  # assumes an input sequence of length 196 = 14 * 14.
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
        conv2d_params = num_players_to_conv2d_params[num_players]
        if use_convolution:
            self.conv = torch.nn.Conv2d(in_channels=head_in_features,
                out_channels=head_in_features,
                kernel_size=conv2d_params['kernel_size'],
                stride=conv2d_params['stride'],
                padding=conv2d_params['padding']).to(self.device)

        # Set up normalization.
        # First we do "additive efficient normalization",
        # which means adding a constant to ensure that (for each item in the batch and each class)
        # the predicted player contributions sum up to the value for unmasked input minus the value for null input.
        # Pred: (batch, num_players, num_classes), grand: (batch, num_classes), null: (1, num_classes)
        self.normalization: Callable | None
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

    # def state_dict(self):
    #     """Remove 'surrogate' from the state_dict (the stuff saved in checkpoints)."""
    #     return {k: v for k, v in super().state_dict().items() if not k.startswith("surrogate.")}

    def null(self, images: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Class probabilites for a fully masked input (cached after first call).

        Input: (batch, channel, height, width), only the shape is relevant.
        Output: (1, num_classes).
        """
        if self.__null is not None:
            return self.__null
        if images is None:
            raise RuntimeError("Call explainer.null(images) at least once to get null value.")
        assert self.surrogate is not None
        self.surrogate.eval()
        with torch.no_grad():
            images = images[0:1].to(self.surrogate.device)  # Limit batch to one image.
            masks = torch.zeros(1, 1, self.num_players, device=self.surrogate.device)
            images_masked = apply_masks(images, masks)  # Mask-out everything.
            logits = self.surrogate(images_masked)  # (1, channel, height, weight) -> (1, num_classes)
            if self.hparams["use_softmax"]:
                self.__null = torch.nn.Softmax(dim=1)(logits).to(self.device)  # (1, num_classes)
            else:
                self.__null = logits.to(self.device)
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
            masks = torch.ones(images.shape[0], 1, self.num_players, device=self.surrogate.device)
            images_masked = apply_masks(images, masks)  # (batch, channel, height, weight)
            logits = self.surrogate(images_masked)  # (batch, num_classes)
            grand: torch.Tensor
            if self.hparams["use_softmax"]:
                grand = torch.nn.Softmax(dim=1)(logits).to(self.device)  # (batch, num_classes)
            else:
                grand = logits.to(self.device)
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
                num_players == self.num_players
            ), f"Explainer was inited with {self.num_players=} but got {num_players=} from dataloader."
            images, masks = images.to(self.surrogate.device), masks.to(self.surrogate.device)
            images_masked = apply_masks(images, masks)  # (B * num_mask_samples, C, H, W)
            logits = self.surrogate(images_masked)  # (B * num_mask_samples, num_classes)
            surrogate_values: torch.Tensor
            if self.hparams["use_softmax"]:
                surrogate_values = torch.nn.Softmax(dim=1)(logits)  # (B * num_mask_samples, num_classes)
            else:
                surrogate_values = logits
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

        return x

    def forward(
        self,
        images: torch.Tensor,
        surrogate_grand: torch.Tensor | None = None,
        surrogate_null: torch.Tensor | None = None, normalize: bool = True
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            images: (batch, channel, height, width)
            surrogate_grand: optional (batch, num_classes).
            surrogate_null: optional (1, num_classes).

        Returns predictions of shape (batch, num_players, num_classes).
        """
        assert self.surrogate is not None

        if self.backbone_name == 'swin':
            # x = self.backbone.swin(images).last_hidden_state
            x = self.backbone.swin(images, output_hidden_states=True).hidden_states[-3]
            # shape of x is now (B, sequence_len=196, hidden_size=384).
            x = torch.repeat_interleave(x, repeats=2, dim=-1)  # (B, seq_length, embed_dim=768)

        elif self.backbone_name == 'vit':
            x = self.backbone(images).last_hidden_state[:, 1:]
            # output of vit is of size (b,197,768), we want (b,196,768)

        elif self.backbone_name == 't2t_vit':
            x = self.backbone(images)  # (B, seq_length, embed_dim)

        else:
            raise ValueError(f"Unexpected backbone_name: {self.backbone_name}")

        assert x.shape[1] == 196, f"Expected {x.shape=} to be (batch, 196, embed_dim)."

        B, seq_length, embed_dim = x.shape
        r = int(math.sqrt(seq_length))
        assert seq_length == r * r, f"Expected {seq_length=} to be a perfect square."

        # Reshape to (B, r, r, embed_dim).
        x = x.view(B, r, r, embed_dim)

        s = int(math.sqrt(self.num_players))
        assert s * s == self.num_players, f"Expected {self.num_players=} to be a perfect square."

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
            indices = [int((i + 0.5) * 14 / s) for i in range(s)]
            # indices = [int((i + 0.5) * 7 / s) for i in range(s)]
            x = x[:, indices, :, :][:, :, indices, :]  # (B, s, s, embed_dim)
            # embedding_all = x

        # Flatten to (B, num_players, embed_dim)
        x = x.flatten(1, 2)

        for layer_module in self.attention_blocks:
            x, _ = layer_module(x, x, x, need_weights=False)

        pred = self.mlps(x)  # (B, num_players, embed_dim) -> # (B, num_players, num_classes)
        if self.hparams["use_tanh"]:
            pred = pred.tanh()

        pred = pred / self.hparams["divisor"]

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
        # Labels have shape: (batch,).
        images, masks, labels = batch["images"], batch["masks"], batch["labels"]

        # Evaluate surrogate on masked, unmasked and fully masked inputs.
        surrogate_values = self.surrogate_multiple_masks(images, masks)  # (batch, num_masks_per_image, num_classes)
        surrogate_grand = self.grand(images)  # (batch, num_classes)
        surrogate_null = self.null(images)  # (1, num_classes)

        # Evaluate explainer to get shap_values of shape (batch, num_players, num_classes).
        shap_values = self(images, surrogate_grand=surrogate_grand, surrogate_null=surrogate_null)

        loss = explainer_utils.compute_metrics(
            self,
            targets=labels,
            masks=masks,
            num_players=self.num_players,
            shap_values=shap_values,
            values_target=surrogate_values,
            surrogate_grand=surrogate_grand,
            surrogate_null=surrogate_null,
            phase="train",
        )

        return loss

    def validation_step(self, batch, batch_idx):
        assert self.surrogate is not None
        images, masks, labels = batch["images"], batch["masks"], batch["labels"]

        # Evaluate surrogate.
        surrogate_values = self.surrogate_multiple_masks(images, masks)
        surrogate_grand = self.grand(images)
        surrogate_null = self.null(images)

        # Evaluate explainer.
        shap_values = self(images, surrogate_grand=surrogate_grand, surrogate_null=surrogate_null)

        loss = explainer_utils.compute_metrics(
            self,
            num_players=self.num_players,
            targets=labels,
            masks=masks,
            shap_values=shap_values,
            values_target=surrogate_values,
            surrogate_grand=surrogate_grand,
            surrogate_null=surrogate_null,
            phase="val",
        )

        self.log(
            "val/macc-best",
            explainer_utils.get_masked_accuracy(images, masks, labels, self.surrogate, shap_values, "best"),
            prog_bar=True
        )
        self.log(
            "val/macc-worst",
            explainer_utils.get_masked_accuracy(images, masks, labels, self.surrogate, shap_values, "worst"),
            prog_bar=True
        )

        return loss

    def test_step(self, batch, batch_idx):
        assert self.surrogate is not None
        images, masks, labels = batch["images"], batch["masks"], batch["labels"]

        # Evaluate surrogate.
        surrogate_values = self.surrogate_multiple_masks(images, masks)
        surrogate_grand = self.grand(images)
        surrogate_null = self.null(images)

        # Evaluate explainer.
        shap_values = self(images, surrogate_grand=surrogate_grand, surrogate_null=surrogate_null)

        loss = explainer_utils.compute_metrics(
            self,
            num_players=self.num_players,
            targets=labels,
            masks=masks,
            shap_values=shap_values,
            values_target=surrogate_values,
            surrogate_grand=surrogate_grand,
            surrogate_null=surrogate_null,
            phase="test",
        )

        self.log(
            "test/macc-best",
            explainer_utils.get_masked_accuracy(images, masks, labels, self.surrogate, shap_values, "best"),
            prog_bar=True
        )
        self.log(
            "test/macc-worst",
            explainer_utils.get_masked_accuracy(images, masks, labels, self.surrogate, shap_values, "worst"),
            prog_bar=True
        )

        return loss


def main() -> None:
    torch.set_float32_matmul_precision("medium")

    parser = argparse.ArgumentParser(description="Explainer training")
    parser.add_argument("--label", default="", type=str, help="label for checkpoints")
    parser.add_argument("--num_players", required=True, type=int, help="number of players")
    parser.add_argument("--lr", default=1e-4, type=float, help="explainer learning rate")
    parser.add_argument("--wd", default=0.0, type=float, help="explainer weight decay")
    parser.add_argument("--b", default=1024, type=int, help="batch size")
    parser.add_argument("--num_workers", default=2, type=int, help="number of dataloader workers")
    parser.add_argument("--num_atts", default=0, type=int, help="number of attention blocks")
    parser.add_argument("--mlp_ratio", default=4, type=int, help="ratio for the middle layer in mlps")
    parser.add_argument("--t_lambda", default=0, type=float, help="ratio for target class in loss")
    parser.add_argument("--use_surg", default=True, type=is_true_string,
                        help="use surrogate (rather than model trained without masks)")
    parser.add_argument("--use_conv", default=False, type=is_true_string,
                        help="convolutions to match dim num_players")
    parser.add_argument("--use_tanh", default=True, type=is_true_string,
                        help="use tanh at explainer end")
    parser.add_argument("--use_softmax", default=True, type=is_true_string,
                        help="use softmax in explainer surrogate calls")
    parser.add_argument("--divisor", default=1.0, type=float,
                        help="divisor for pre-normalization shap values, should be somewhat smaller than num_players")

    parser.add_argument("--target_model_name", required=True, default='vit', type=str, help="name of the target model")
    parser.add_argument("--backbone_name", required=True, default='vit', type=str, help="name of the backbone")
    parser.add_argument("--freeze_backbone", default='except_last_two', type=str,
                        help="freeze the backbone")
    parser.add_argument("--mode", default='uniform', type=str, help="uniform or shapley mask sampling")

    args = parser.parse_args()

    if args.use_surg:
        surrogate_dir = PROJECT_ROOT / "saved_models/surrogate/cifar10"
        if args.target_model_name == 't2t_vit':
            surrogate_path = surrogate_dir / f"v2/player{args.num_players}/t2t_vit.ckpt"
        elif args.target_model_name == 'swin':
            surrogate_path = surrogate_dir / f"v2/player{args.num_players}/swin.ckpt"
        elif args.target_model_name == 'vit':
            surrogate_path = surrogate_dir / f"v2/player{args.num_players}/vit.ckpt"
        else:
            raise ValueError(f"Unexpected target model name: {args.target_model_name}")

        # For older checkpoints, it's OK to set 'strict=False' and
        # ignore Surrogate's "target_model.*" being saved checkpoint but not in Surrogate for evaluation.
        target_model = Surrogate.load_from_checkpoint(
            surrogate_path,
            map_location="cuda",
            strict=True,
            # backbone_name="t2t_vit"  # Needs to be specified for very old checkpoints.
        )
    else:
        target_model = load_transferred_model(args.target_model_name)

    explainer = Explainer(
        output_dim=10,
        backbone_name=args.backbone_name,
        explainer_head_num_attention_blocks=args.num_atts,
        explainer_head_mlp_layer_ratio=args.mlp_ratio,
        explainer_norm=True,
        surrogate=target_model,
        efficiency_lambda=0,
        efficiency_class_lambda=0,
        freeze_backbone=args.freeze_backbone,
        checkpoint_metric="loss",
        learning_rate=args.lr,
        weight_decay=args.wd,
        decay_power=None,
        warmup_steps=None,
        target_class_lambda=args.t_lambda,
        use_convolution=args.use_conv,
        num_players=args.num_players,
        use_tanh=args.use_tanh,
        use_softmax=args.use_softmax,
        divisor=args.divisor
    )

    datamodule = CIFAR_10_Datamodule(
        num_players=args.num_players,
        num_mask_samples=2,
        paired_mask_samples=True,
        batch_size=args.b,
        num_workers=args.num_workers,
        train_mode=args.mode,
        val_mode="uniform",
        test_mode="uniform"
    )

    log_name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M_")
    if args.label:
        log_name += f"{args.label}_"
    if args.use_conv:
        log_name += "conv_"
    log_name += f"{args.backbone_name}_"
    if args.target_model_name != args.backbone_name:
        log_name += f"{args.target_model_name}_"
    if args.freeze_backbone != "except_last_two":
        log_name += f"freeze-{args.freeze_backbone}_"
    if not args.use_surg:
        log_name += "nosurg_"
    log_name += f"{args.mode}_p{args.num_players}_lr{args.lr}_"
    if args.wd:
        log_name += f"wd{args.wd}_"
    if args.b != 1024:
        log_name += f"b{args.b}_"
    log_name += f"t{args.t_lambda}_"
    if args.divisor != 1.0:
        log_name += f"d{args.divisor}_"
    if not args.use_tanh:
        log_name += "notanh_"
    if not args.use_softmax:
        log_name += "nosoftmax_"
    log_name = log_name.removesuffix("_")
    print(log_name)

    log_and_checkpoint_dir = (PROJECT_ROOT / "checkpoints" / "explainer" / log_name)
    log_and_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(str(log_and_checkpoint_dir))

    trainer = pl.Trainer(max_epochs=500, default_root_dir=log_and_checkpoint_dir, callbacks=RichProgressBar(leave=True))
    trainer.fit(explainer, datamodule)


if __name__ == "__main__":
    main()
