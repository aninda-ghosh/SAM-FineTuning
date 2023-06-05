# encoding: utf-8
"""
@author:  Aninda Ghosh
@contact: aghosh57@asu.edu
"""

import logging

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.handlers import ModelCheckpoint, Timer
from ignite.metrics import Accuracy, Loss, RunningAverage

from tqdm import tqdm
import torch

def do_train(
        cfg,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        loss_fn,
):
    # Get the parameters from the config file for training
    #log_period = cfg.SOLVER.LOG_PERIOD
    #checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    #output_dir = cfg.OUTPUT_DIR
    device = cfg.MODEL.DEVICE
    epochs = cfg.SOLVER.MAX_EPOCHS

    #Put the model in the train mode
    model.to(device)    # Put the model on GPU
    model.train()       # Set the model to training mode

    for epoch in range(cfg.SOLVER.MAX_EPOCHS):
        epoch_loss = []
        progress_bar = tqdm(train_loader, total=len(train_loader))

        for data in progress_bar:
            #print(data)
            image = data["image"].to(device)
            bbox_prompts = data["bbox_prompts"].to(device)
            point_prompts = data["point_prompts"].to(device)
            labels = data["labels"].to(device)

            # Get the single mask prediction
            with torch.inference_mode():
                image_embeddings = model.image_encoder(image)  # (B,256,64,64)

                sparse_embeddings, dense_embeddings = model.prompt_encoder(
                    points=point_prompts,
                    boxes=bbox_prompts,
                    masks=None,
                )

            # mask_predictions, _ = model.mask_decoder(
            #     image_embeddings=image_embeddings.to(device),  # (B, 256, 64, 64)
            #     image_pe=model.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
            #     sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
            #     dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
            #     multimask_output=False,
            # )

            pass




