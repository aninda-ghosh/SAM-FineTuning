# encoding: utf-8
"""
@author:  Aninda Ghosh
@contact: aghosh57@asu.edu
"""

import argparse
from datetime import datetime
import os
import sys
from os import mkdir

import torch.nn.functional as F

sys.path.append('.')
from config import cfg

from data.build import SAM_Dataloader
from modeling.segment_anything import prepare_sam
from solver.optimizer import build_optimizer
from solver.lr_schedular import build_lrSchedular
from solver.loss import FocalLoss, DiceLoss

from utils.logger import setup_logger

from config import cfg  # Import the default config file

import torch
import numpy as np

from solver.loss import FocalLoss, DiceLoss, IoULoss
from engine.sam_trainer import do_train



# Freeze the config file 
cfg.freeze()

# Create a folder to save the model and the logs based on the current time
current_time = datetime.now().strftime('%b%d_%H-%M-%S')
output_dir = cfg.OUTPUT_DIR + current_time
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    os.makedirs(output_dir + "/model_checkpoints")

# Setup the logger
logger = setup_logger('SAM', cfg.LOGGER.LEVEL, output_dir)



def main():
    # Get the model and download the checkpoint if needed
    model = prepare_sam(checkpoint=cfg.MODEL.CHECKPOINT, model_type = cfg.MODEL.TYPE)
    device = cfg.MODEL.DEVICE
    epochs = cfg.SOLVER.MAX_EPOCHS

    #Set the portion of the model to be trained (We will train only the mask_decoder part)
    for name, param in model.named_parameters():
        if name.startswith('image_encoder') or name.startswith('prompt_encoder'):
            param.requires_grad = False
    
    # Get the dataloader and prepare the train, validation and test dataloader
    sam_dataloader = SAM_Dataloader(cfg)
    train_loader, valid_loader, test_loader = sam_dataloader.build_dataloader()

    # Get the optimizer and scheduler
    optimizer = build_optimizer(cfg, model)
    scheduler = build_lrSchedular(cfg=cfg, optimizer=optimizer)

    Focal_Loss = FocalLoss()
    Dice_Loss = DiceLoss()
    Iou_Loss = IoULoss()

    #Put the model in the train mode
    model.to(device)    # Put the model on GPU

    logger.info(f'Config:\n{cfg}')

    # Start the training
    logger.info('Start Training Epochs')
    do_train(
        cfg = cfg,
        logger = logger,
        model = model,
        device = device,
        train_dataloader = train_loader,
        valid_dataloader = valid_loader,
        test_dataloader = test_loader,
        optimizer = optimizer,
        scheduler = scheduler,
        focal_loss = Focal_Loss,
        dice_loss = Dice_Loss,
        iou_loss = Iou_Loss,
        epochs = epochs,
        output_dir = output_dir
    )
    

if __name__ == '__main__':
    main()
