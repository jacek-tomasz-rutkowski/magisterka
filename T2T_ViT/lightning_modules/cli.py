import os
import pprint
import warnings
from functools import partial
from typing import Any, Callable, Type

import jsonargparse
import torch
import torch.nn
import torch.utils.data
import lightning as L
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
)
from lightning.pytorch.cli import LightningArgumentParser, LightningCLI, SaveConfigCallback
# from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger  # noqa: F401
from lightning.pytorch.loggers import Logger as LightningLogger
from utils import PROJECT_ROOT


def lightning_main(
    lmodule_class: Type[L.LightningModule] | None = None,
    datamodule_class: Type[L.LightningDataModule] | None = None,
    experiment_name: str = "unknown",
    monitor: str = "val/loss",
    parser_callback: Callable[[LightningArgumentParser], None] = lambda x: None,
    main_config_callback: Callable[[jsonargparse.Namespace], dict[str, Any]] = lambda x: {},
    checkpoint_filename_pattern: str = "epoch={epoch:0>3}",
    model_summary_depth: int = 1,
) -> None:
    """
    Run the lightning CLI. This parses command line arguments and fits/validates/evaluates/predicts.

    See: https://lightning.ai/docs/pytorch/stable/cli/lightning_cli_intermediate.html

    Args:
    - lmodule_class: a tuple of LightningModule classes that can be selected.
    - datamodule_class: a tuple of LightningDataModule classes that can be selected.
    - experiment_name: the name under the 'checkpoints/' directory where the logs and checkpoints will be saved.
    - monitor: the metric to monitor for checkpointing.
    - parser_callback: a function that takes a LightningArgumentParser.
        Use it to link configuration params (e.g. set model.num_classes := data.num_classes) or add custom arguments.
        See: https://pytorch-lightning.readthedocs.io/en/2.2.3/pytorch/cli/lightning_cli_expert.html#cli-link-arguments
    - main_config_callback: a function that takes a full config and returns the most relevant stuff.
        These are printed, and logged as hyperparameters, which allows e.g. tensorboard to see them as columns.
    - checkpoint_filename_pattern: a python format string for the checkpoint filenames (without .cktp).
        Any logged metrics can be used, e.g. "epoch={epoch:0>3}_val-acc={val/accuracy:.3f}".
    - model_summary_depth: how deep to look into the model for the table summary of model size and parameters.
    """
    torch.set_float32_matmul_precision("medium")

    # Ignore a few spurious warnings from Lightning.
    # warnings.filterwarnings("ignore", message="The `srun` command is available on your system")
    warnings.filterwarnings("ignore", message="Experiment logs directory .* exists")
    warnings.filterwarnings("ignore", message="The number of training batches .* is smaller than the logging interval")

    # Choose where to save logs and checkpoints.
    checkpoints_dir = PROJECT_ROOT / "checkpoints"
    (checkpoints_dir / experiment_name).mkdir(exist_ok=True, parents=True)
    version: str = "s" + os.environ.get("SLURM_JOB_ID", "")
    if not version:
        # If not running with slurm, set the version to the next available number, prefixed with "v" instead of "s".
        versions = [int(p.name[1:]) for p in (checkpoints_dir / experiment_name).iterdir() if p.name.startswith("v")]
        version = f"v{max(versions, default=0) + 1}"
    print(str(checkpoints_dir / experiment_name / version))

    save_config_callback = partial(_LoggerSaveConfigCallback, main_config_callback=main_config_callback)

    _MyLightningCLI(
        lmodule_class,
        datamodule_class,
        trainer_defaults=dict(
            default_root_dir=checkpoints_dir / experiment_name,
            logger=[
                dict(class_path=c, init_args=dict(save_dir=str(checkpoints_dir), name=experiment_name, version=version))
                for c in ["TensorBoardLogger", "CSVLogger"]
            ],
            callbacks=[
                # DeviceStatsMonitor(cpu_stats=True),
                LearningRateMonitor(),
                ModelCheckpoint(
                    filename=checkpoint_filename_pattern,
                    auto_insert_metric_name=False,
                    monitor=monitor,
                    save_top_k=1,
                    verbose=False,
                ),
                RichProgressBar(
                    leave=True,
                    console_kwargs=dict(force_terminal=True, force_interactive=True, width=250),
                ),
                RichModelSummary(max_depth=model_summary_depth),
            ],
        ),
        parser_callback=parser_callback,
        # Type ignored because LightningCLI thinks it needs a class, while a constructor Callable is enough.
        save_config_callback=save_config_callback,  # type: ignore
        save_config_kwargs={"overwrite": True}  # Otherwise continue doesn't work.
    )


class _MyLightningCLI(LightningCLI):
    """LightningCLI subclass in which `add_arguments_to_parser` is defined as a callback."""

    def __init__(self, *args, parser_callback: Callable[[LightningArgumentParser], None], **kwargs):
        self.parser_callback = parser_callback
        super().__init__(*args, **kwargs)

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        self.parser_callback(parser)


class _LoggerSaveConfigCallback(SaveConfigCallback):
    """SaveConfigCallback subclass that logs the config as hyperparameters."""
    def __init__(self, *args, main_config_callback: Callable[[jsonargparse.Namespace], dict[str, Any]], **kwargs):
        super().__init__(*args, **kwargs)
        self.main_config_callback = main_config_callback

    def save_config(self, trainer: L.Trainer, pl_module: L.LightningModule, stage: str) -> None:
        if isinstance(trainer.logger, LightningLogger):
            main_config = self.main_config_callback(self.config)
            pprint.pp(main_config, indent=8, width=200)
            full_config = self.config
            full_config.pop("config.trainer.logger")
            full_config.pop("config.trainer.callbacks")
            trainer.logger.log_hyperparams({**main_config, "config": full_config})
