import torch
from torch.nn import functional as F
from torchmetrics import MeanMetric
from transformers import get_cosine_schedule_with_warmup
from transformers.optimization import AdamW


def set_schedule(pl_module):
    optimizer = AdamW(params=pl_module.parameters(), lr=pl_module.hparams.learning_rate,
                          weight_decay=pl_module.hparams.weight_decay)

    if pl_module.trainer.max_steps is None or pl_module.trainer.max_steps == -1:
        n_train_batches_per_epoch = len(pl_module.trainer.datamodule.train_dataloader())
        n_train_batches_total = n_train_batches_per_epoch * pl_module.trainer.max_epochs
        max_steps = n_train_batches_total // pl_module.trainer.accumulate_grad_batches
    else:
        max_steps = pl_module.trainer.max_steps

    if pl_module.hparams.decay_power == "cosine":
        scheduler = {"scheduler": get_cosine_schedule_with_warmup(optimizer,
                                                                  num_warmup_steps=pl_module.hparams.warmup_steps,
                                                                  num_training_steps=max_steps),
                     "interval": "step"}
    else:
        NotImplementedError("Only cosine scheduler is implemented for now")

    return ([optimizer], [scheduler],)


def set_metrics(pl_module):
    for phase in ["train", "val", "test"]:
        setattr(pl_module, f"{phase}_loss", MeanMetric())


def epoch_wrapup(pl_module, phase):
    loss = getattr(pl_module, f"{phase}_loss").compute()
    getattr(pl_module, f"{phase}_loss").reset()
    pl_module.log(f"{phase}/epoch_loss", loss)

    checkpoint_metric = -loss
    pl_module.log(f"{phase}/checkpoint_metric", checkpoint_metric)


def compute_metrics(pl_module, logits, logits_target, phase):
    loss = F.kl_div(input=torch.log_softmax(logits, dim=1),
                        target=torch.softmax(logits_target, dim=1),
                        reduction='batchmean',
                        log_target=False)

    loss = getattr(pl_module, f"{phase}_loss")(loss)

    pl_module.log(f"{phase}/loss", loss)

    return loss
