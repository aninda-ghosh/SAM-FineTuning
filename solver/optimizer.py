# encoding: utf-8
"""
@author:  Aninda Ghosh
@contact: aghosh57@asu.edu
"""

import torch


def build_optimizer(cfg, model):
    
    # if Model.grad is required to be False, then we will not calculate the gradient of the model
    # This is useful when we are using a pretrained model and we do not want to change the weights of the model
    # for name, param in model.named_parameters():
    #     if name.startswith('image_encoder') or name.startswith('prompt_encoder'):
    #         param.requires_grad = False

    # Using SGD optimizer for SAM Model
    # optimizer = torch.optim.SGD(
    #     model.mask_decoder.parameters(),
    #     lr=cfg.SOLVER.BASE_LR,
    #     momentum=cfg.SOLVER.MOMENTUM,
    #     weight_decay=cfg.SOLVER.WEIGHT_DECAY
    # ) 

    # Using Adam optimizer for SAM Model
    optimizer = torch.optim.Adam(
        model.mask_decoder.parameters(), 
        lr=cfg.SOLVER.BASE_LR, 
        weight_decay=cfg.SOLVER.WEIGHT_DECAY
    )
    
    return optimizer
