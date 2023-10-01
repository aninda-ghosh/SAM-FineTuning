# encoding: utf-8
"""
@author:  Aninda Ghosh
@contact: aghosh57@asu.edu
"""

import torch


def build_optimizer(cfg, parameters):
    
    # Using Adam optimizer for SAM Model
    optimizer = torch.optim.Adam(
        params=parameters, 
        lr=cfg.SOLVER.START_LR, 
        weight_decay=cfg.SOLVER.WEIGHT_DECAY
    )
    
    return optimizer

def build_custom_optimizer(cfg, model):
    
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        params += [{"params": [value], "lr": cfg.SOLVER.START_LR, "weight_decay": cfg.SOLVER.WEIGHT_DECAY}]
    optimizer = getattr(torch.optim, 'Adam')(params)
    return optimizer
