# encoding: utf-8
"""
@author:  Aninda Ghosh
@contact: aghosh57@asu.edu
"""

import torch

def build_lrSchedular(cfg, optimizer):
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.01, verbose=False
    )

    return scheduler