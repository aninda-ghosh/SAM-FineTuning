# encoding: utf-8
"""
@author:  Aninda Ghosh
@contact: aghosh57@asu.edu
"""

import torch

def build_lrSchedular(cfg, optimizer):
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=2, factor=0.01, verbose=False, min_lr=cfg.SOLVER.MIN_LR
    )

    lr_scheduler_config = {
        # REQUIRED: The scheduler instance
        "scheduler": scheduler,
        # The unit of the scheduler's step size, could also be 'step'.
        # 'epoch' updates the scheduler on epoch end whereas 'step'
        # updates it after a optimizer update.
        "interval": "epoch",
        # How many epochs/steps should pass between calls to
        # `scheduler.step()`. 1 corresponds to updating the learning
        # rate after every epoch/step.
        "frequency": 1,
        # Metric to to monitor for schedulers like `ReduceLROnPlateau`
        "monitor": "valid_loss",
        # If set to `True`, will enforce that the value specified 'monitor'
        # is available when the scheduler is updated, thus stopping
        # training if not found. If set to `False`, it will only produce a warning
        "strict": True,
        # If using the `LearningRateMonitor` callback to monitor the
        # learning rate progress, this keyword can be used to specify
        # a custom logged name
        "name": None,
    }

    return lr_scheduler_config