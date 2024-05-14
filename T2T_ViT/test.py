import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import RichProgressBar

from utils import load_transferred_model, find_latest_checkpoint, find_latest_checkpoints
from vit_shapley.modules.surrogate import Surrogate
from vit_shapley.modules.explainer import Explainer
from datasets.CIFAR_10_Dataset import CIFAR_10_Datamodule, PROJECT_ROOT


def cleanup_checkpoint(source: Path, target: Path, prefix: str) -> None:
    ckpt = torch.load(source)
    keys = [k for k in ckpt["state_dict"]]
    for k in keys:
        if k.startswith(prefix):
            del ckpt["state_dict"][k]
    torch.save(ckpt, target)


def main(explainer_path: Path) -> None:
    torch.set_float32_matmul_precision("medium")
    tmp_checkpoint_path = PROJECT_ROOT / "checkpoints" / "tmp.ckpt"

    use_surg = True
    if use_surg:
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
            strict=True,  # It's OK to ignore "target_model.*" (in checkpoint but not in Surrogate) for evaluation.
            # backbone_name="t2t_vit",  # Needs to be specified for older checkpoints.
        )
    else:
        surrogate = load_transferred_model("t2t_vit", device="cuda")
    surrogate.eval()
    surrogate_state_dict = dict(surrogate.state_dict())
    print("Surrogate loaded")

    # Get latest checkpoint from the run:
    explainer_path = find_latest_checkpoint(explainer_path.parent)

    cleanup_checkpoint(Path(explainer_path), tmp_checkpoint_path, "surrogate.target_model.")

    # Force override of surrogate in explainer.surrogate checkpoint.
    if True:
        cleanup_checkpoint(Path(explainer_path), tmp_checkpoint_path, "surrogate.")
        ckpt = torch.load(tmp_checkpoint_path)
        for k in surrogate_state_dict:
            ckpt["state_dict"]["surrogate." + k] = surrogate_state_dict[k]
        # Stuff that needs to be specified for older checkpoints:
        defaults = dict(
            backbone_name="t2t_vit",
            use_convolution=False,
            num_players=196,
            use_surg=True
        )
        for k, v in defaults.items():
            if k not in ckpt["hyper_parameters"]:
                ckpt["hyper_parameters"][k] = v
        torch.save(ckpt, tmp_checkpoint_path)

    explainer = Explainer.load_from_checkpoint(
        tmp_checkpoint_path,
        map_location="cuda",
        surrogate=deepcopy(surrogate),
        strict=True,
        # Needs to be specified for older checkpoints:
        # backbone_name="t2t_vit",
        # use_convolution=False,
        # num_players=196,
        # use_surg=False
    )
    explainer.eval()
    print("Explainer loaded")

    datamodule = CIFAR_10_Datamodule(
        num_players=196,
        num_mask_samples=2,
        paired_mask_samples=True,
        batch_size=256,
        num_workers=2,
        train_mode="uniform",
        val_mode="uniform",
        test_mode="uniform",
    )
    datamodule.setup()

    trainer = pl.Trainer(
        default_root_dir=PROJECT_ROOT / "checkpoints" / "tmplog",
        callbacks=RichProgressBar(leave=True),
        limit_val_batches=0.25,
        limit_test_batches=0.25,
    )
    # trainer.validate(explainer, datamodule)
    results: dict[str, Any] = dict(trainer.test(explainer, datamodule)[0])  # , ckpt_path=str(tmp_checkpoint_path))
    results['path'] = str(explainer_path)
    print(explainer_path)
    with (explainer_path.parent / "results.txt").open("a") as f:
        json.dump(results, f, indent=4, sort_keys=True)


if __name__ == "__main__":
    for p in find_latest_checkpoints(PROJECT_ROOT / "checkpoints/explainer"):
        try:
            print(p)
            main(p)
        except Exception as e:
            print(e)
