# encoding: utf-8
"""
@author: Aninda Ghosh, Manthan Satish
@contact: aghosh57@asu.edu, mcsatish@asu.edu
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from solver.optimizer import build_optimizer
from solver.lr_schedular import build_lrSchedular
from solver.loss import FocalLoss, DiceLoss, IoULoss

import lightning.pytorch as pl

from config import cfg


class SAMTrainer(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.Focal_Loss = FocalLoss()
        self.Dice_Loss = DiceLoss()
        self.Iou_Loss = IoULoss()

    def forward(self, x):
        self.process_step(x)

    def training_step(self, batchx, batch_idx):
        focal_loss, dice_loss, iou_loss = self.process_step(batchx)

        loss = (cfg.LOSS.FOCAL_LOSS_WEIGHT * focal_loss) + (cfg.LOSS.DICE_LOSS_WEIGHT * dice_loss)
        self.log("train_loss", loss)
        self.log("train_focal_loss", focal_loss)
        self.log("train_dice_loss", dice_loss)
        self.log("train_iou_loss", iou_loss)

        return loss

    def validation_step(self, batchx, batch_idx):
        focal_loss, dice_loss, iou_loss = self.process_step(batchx)

        loss = (cfg.LOSS.FOCAL_LOSS_WEIGHT * focal_loss) + (cfg.LOSS.DICE_LOSS_WEIGHT * dice_loss)
        self.log("train_loss", loss)
        self.log("train_focal_loss", focal_loss)
        self.log("train_dice_loss", dice_loss)
        self.log("train_iou_loss", iou_loss)

        return loss

    def configure_optimizers(self):
        optimizer = build_optimizer(cfg, self.model)
        scheduler = build_lrSchedular(cfg=cfg, optimizer=optimizer)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}
    
    def process_step(self, batchx):
        instances = 0
        focal_loss = 0
        dice_loss = 0
        iou_loss = 0

        for batch in batchx:
            x1, x2, y, image_size = batch['image'], batch['bbox_prompts'], batch['gt_masks'], batch['image_size']
            image_embeddings, sparse_embeddings, dense_embeddings = self.encode(x1, x2)
            low_res_masks, iou_predictions = self.model.mask_decoder(
                image_embeddings=image_embeddings.to(self.device),  # (B, 256, 64, 64)
                image_pe= self.model.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
                sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
                dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
                multimask_output=False,
            )
            upscaled_masks = self.model.postprocess_masks(low_res_masks, (1024, 1024), image_size).to(self.device)
            high_res_masks = F.normalize(F.threshold(upscaled_masks, 0.0, 0)).to(self.device).to(self.device).to(torch.float32)

            if len(y) == 0:   # if no masks are provided then generate boolean pixel masks with image original size
                _pixel_masks = []
                # generate boolean pixel masks with image original size, use False as the mask value
                for i in range(len(high_res_masks)):
                    _pixel_masks.append(np.full((image_size[0], image_size[1]), False, dtype=bool))
                pixel_masks_tensor = torch.as_tensor(np.array(_pixel_masks).astype(float)).to(self.device).to(torch.float32)
            else:
                pixel_masks_tensor = torch.as_tensor(np.array(y).astype(float)).to(self.device).to(torch.float32)

            for i in range(len(high_res_masks)):
                focal_loss += self.Focal_Loss(high_res_masks[i], pixel_masks_tensor[i])
                dice_loss += self.Dice_Loss(high_res_masks[i], pixel_masks_tensor[i])
                iou_loss += self.Iou_Loss(high_res_masks[i], pixel_masks_tensor[i], iou_predictions[i][0].to(self.device))
                instances += 1

        return (focal_loss/instances, dice_loss/instances, iou_loss/instances)
    
    
    def encode(self, x1, x2):
        """
		Encode the input image and the prompts
		"""
        with torch.no_grad():
            image_embeddings = self.model.image_encoder(x1.to(self.device))
            sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                points=None,
                boxes=x2.to(self.device),
                masks=None,
            )

        return image_embeddings, sparse_embeddings, dense_embeddings
