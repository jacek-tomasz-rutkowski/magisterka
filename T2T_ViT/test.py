import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import lightning as L
import torch
from lightning.pytorch.callbacks import RichProgressBar

import models.t2t_vit  # noqa: F401
from datasets.datamodules import CIFAR10DataModule, DataModuleWithMasks, GenerateMasksKwargs
from lightning_modules.classifier import Classifier
from lightning_modules.explainer import Explainer
from lightning_modules.surrogate import Surrogate
from utils import PROJECT_ROOT, find_latest_checkpoint, find_latest_checkpoints


def cleanup_checkpoint(source: Path, target: Path, prefix: str) -> None:
    ckpt = torch.load(source)
    keys = [k for k in ckpt["state_dict"]]
    for k in keys:
        if k.startswith(prefix):
            del ckpt["state_dict"][k]
    torch.save(ckpt, target)


def main(explainer_path: Path) -> None:
    torch.set_float32_matmul_precision("medium")

    # Load target/surrogate model.
    use_surg = True
    model: L.LightningModule
    if use_surg:
        surrogate_dir = PROJECT_ROOT / "saved_models/surrogate/cifar10"
        surrogate_path = (
            surrogate_dir / "v4/player196/swin_tiny_patch4_window7_224"
            # surrogate_dir / "v4/player196/vit_small_patch16_224"
        )
        model = Surrogate.load_from_latest_checkpoint(
            surrogate_path,
            map_location="cuda"
        )
    else:
        model = Classifier.load_from_latest_checkpoint(
            PROJECT_ROOT / "saved_models/classifier/cifar10/v4/swin_tiny_patch4_window7_224",
            map_location="cuda"
        )
    model.eval()
    model_state_dict = dict(model.state_dict())
    print("Target/surrogate model loaded")

    # Get latest checkpoint from the run:
    explainer_path = find_latest_checkpoint(explainer_path.parent)

    # Force override of surrogate in explainer.surrogate checkpoint.
    if True:
        tmp_checkpoint_path = PROJECT_ROOT / "checkpoints" / "tmp.ckpt"
        cleanup_checkpoint(Path(explainer_path), tmp_checkpoint_path, "surrogate.")
        ckpt = torch.load(tmp_checkpoint_path)
        for k in model_state_dict:
            ckpt["state_dict"]["surrogate." + k] = model_state_dict[k]
        torch.save(ckpt, tmp_checkpoint_path)
    else:
        tmp_checkpoint_path = explainer_path

    explainer = Explainer.load_from_checkpoint(
        tmp_checkpoint_path,
        map_location="cuda",
        surrogate=deepcopy(model)
    )
    explainer.eval()
    print("Explainer loaded")

    datamodule = DataModuleWithMasks(
        CIFAR10DataModule(),
        GenerateMasksKwargs(num_players=explainer.num_players),
        dict(batch_size=32)
    )
    datamodule.setup("test")

    trainer = L.Trainer(
        default_root_dir=PROJECT_ROOT / "checkpoints" / "tmplog",
        callbacks=RichProgressBar(leave=True),
        limit_val_batches=0.25,
        limit_test_batches=0.25
    )
    results: dict[str, Any] = dict(trainer.test(explainer, datamodule)[0])
    # results: dict[str, Any] = dict(trainer.validate(explainer, datamodule)[0])
    results['path'] = str(explainer_path)
    print(explainer_path)
    with (explainer_path.parent.parent / "test_results.txt").open("a") as f:
        json.dump(results, f, indent=4, sort_keys=True)
        print("", file=f)


if __name__ == "__main__":
    for p in find_latest_checkpoints(PROJECT_ROOT / "checkpoints/explainer"):
        try:
            print(p)
            main(p)
        except Exception as e:
            print(f"Exception ({type(e)})\n", e)
