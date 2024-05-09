from copy import deepcopy
from pathlib import Path

import torch

from vit_shapley.modules.surrogate import Surrogate
from vit_shapley.modules.explainer_swin import Explainer
from vit_shapley.CIFAR_10_Dataset import CIFAR_10_Datamodule, PROJECT_ROOT

import pytorch_lightning as pl
from pytorch_lightning.callbacks import RichProgressBar


def cleanup_checkpoint(source: Path, target: Path, prefix: str) -> None:
    ckpt = torch.load(source)
    keys = [k for k in ckpt["state_dict"]]
    for k in keys:
        if k.startswith(prefix):
            del ckpt["state_dict"][k]
    torch.save(ckpt, target)


def main() -> None:
    tmp_checkpoint_path = PROJECT_ROOT / "checkpoints" / "tmp.ckpt"

    surrogate_dir = PROJECT_ROOT / "saved_models/surrogate/cifar10"
    surrogate_path = (
        surrogate_dir / "v2/player196/t2t_vit.ckpt"
        # surrogate_dir / "_t2t_vit_player16_lr0.0001_wd0.0_b256_epoch19.ckpt"
        # surrogate_dir / "_vit_player16_lr0.0001_wd0.0_b256_epoch19.ckpt"
        # surrogate_dir / "_swin_player16_lr0.0002_wd0.0_b256_epoch19.ckpt"
        # surrogate_dir / "_player196_lr1e-05_wd0.0_b128_epoch49.ckpt"
        # surrogate_dir / "_player16_lr1e-05_wd0.0_b256_epoch28.ckpt"
    )
    cleanup_checkpoint(surrogate_path, tmp_checkpoint_path, "target_model.")
    surrogate = Surrogate.load_from_checkpoint(
        tmp_checkpoint_path,
        map_location="cuda",
        strict=True,  # It's OK to ignore "target_model.*" being saved checkpoint but not in Surrogate for evaluation.
        # backbone_name="t2t_vit",  # Needs to be specified for older checkpoints.
    )
    surrogate.eval()
    surrogate_ckpt = torch.load(tmp_checkpoint_path)
    print("Surrogate loaded")

    explainer_dir = PROJECT_ROOT / "saved_models/explainer"
    explainer_path = (
        # PROJECT_ROOT / 'checkpoints/explainer/use_conv_True_t2t_vit_freeze_none_use_surgTrue_player196_lr0.0001_wd0.0_b256/lightning_logs/version_2/checkpoints/epoch=49-step=8800.ckpt' # medium poor
        # PROJECT_ROOT / 'checkpoints/explainer/use_conv_True_t2t_vit_freeze_none_use_surgTrue_player16_lr5e-05_wd0.0_b256/lightning_logs/version_4/checkpoints/epoch=7-step=1408.ckpt'
        # PROJECT_ROOT / 'checkpoints/explainer/use_conv_True_t2t_vit_freeze_none_use_surgTrue_player196_lr5e-05_wd0.0_b256/lightning_logs/version_0/checkpoints/epoch=9-step=1760.ckpt'  # poorly
        explainer_dir / "_player196_lr5e-05_wd0.0_b256.ckpt"  # performs well, even explaining "v2/player196/t2t_vit.ckpt", even on 16 players.
        # explainer_dir / "_player16_lr5e-05_wd0.0_b256.ckpt",
        # explainer_dir / "t2t_vit_freeze_none_use_surgTrue_player16_lr0.0001_wd0.0_b128.ckpt",  # performs poorly
    )

    cleanup_checkpoint(Path(explainer_path), tmp_checkpoint_path, "surrogate.target_model.")

    # Override surrogate in explainer.surrogate checkpoint.
    if False:
        ckpt = torch.load(tmp_checkpoint_path)
        keys = [k for k in ckpt["state_dict"]]
        for k in keys:
            if k.startswith("surrogate."):
                ckpt["state_dict"][k] = surrogate_ckpt["state_dict"][k.removeprefix("surrogate.")]
        torch.save(ckpt, tmp_checkpoint_path)

    explainer = Explainer.load_from_checkpoint(
        tmp_checkpoint_path,
        map_location="cuda",
        surrogate=deepcopy(surrogate),
        strict=True,
        backbone_name="t2t_vit",  # Needs to be specified for older checkpoints.
        use_convolution=False,    # Needs to be specified for older checkpoints.
        num_players=196,          # Needs to be specified for older checkpoints.
    )
    explainer.eval()
    print("Explainer loaded")

    datamodule = CIFAR_10_Datamodule(
        num_players=196,
        num_mask_samples=2,
        paired_mask_samples=True,
        batch_size=256,
        num_workers=2,
    )
    datamodule.setup()

    trainer = pl.Trainer(
        default_root_dir=PROJECT_ROOT / "checkpoints" / "tmplog",
        callbacks=RichProgressBar(leave=True),
        limit_val_batches=0.25,
        limit_test_batches=0.25,
    )
    # trainer.validate(explainer, datamodule, ckpt_path=tmp_checkpoint_path)
    trainer.test(explainer, datamodule, ckpt_path=str(tmp_checkpoint_path))


if __name__ == "__main__":
    main()
