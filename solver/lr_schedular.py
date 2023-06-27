# encoding: utf-8
"""
@author:  Aninda Ghosh
@contact: aghosh57@asu.edu
"""

import torch
from torch.optim.lr_scheduler import StepLR


def build_lrSchedular(cfg, optimizer):
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, mode="min", patience=3, factor=0.01, verbose=False, min_lr=cfg.SOLVER.MIN_LR
    # )

    scheduler = StepLR( optimizer, 
                        step_size = 4, # Period of learning rate decay
                        gamma = 0.5 # Multiplicative factor of learning rate decay
                    ) 

    return scheduler