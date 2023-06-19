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

from modeling.segment_anything.build_sam import prepare_sam
from modeling.segment_anything.utils.transforms import ResizeLongestSide
from utils.data_support import get_bbox_point_and_inputlabel_prompts

class SAMTrainer(pl.LightningModule):
    def __init__(self, cfg, model, device):
        super().__init__()
        self.model = model
        self.Focal_Loss = FocalLoss()
        self.Dice_Loss = DiceLoss()
        self.Iou_Loss = IoULoss()
        self.cfg = cfg
        self.device = device

    def forward(self, x):
        self.process_step(x)

    def training_step(self, batchx, batch_idx):
        focal_loss, dice_loss, iou_loss = self.process_step(batchx)

        loss = (self.cfg.LOSS.FOCAL_LOSS_WEIGHT * focal_loss) + (self.cfg.LOSS.DICE_LOSS_WEIGHT * dice_loss)
        self.log("train_loss", loss)
        self.log("train_focal_loss", focal_loss)
        self.log("train_dice_loss", dice_loss)
        self.log("train_iou_loss", iou_loss)

        return loss

    def validation_step(self, batchx, batch_idx):
        focal_loss, dice_loss, iou_loss = self.process_step(batchx)

        loss = (self.cfg.LOSS.FOCAL_LOSS_WEIGHT * focal_loss) + (self.cfg.LOSS.DICE_LOSS_WEIGHT * dice_loss)
        self.log("train_loss", loss)
        self.log("train_focal_loss", focal_loss)
        self.log("train_dice_loss", dice_loss)
        self.log("train_iou_loss", iou_loss)

        return loss

    def configure_optimizers(self):
        optimizer = build_optimizer(self.cfg, self.model)
        scheduler = build_lrSchedular(cfg=self.cfg, optimizer=optimizer)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

    def process_step(self, batchx):
        instances = 0
        focal_loss = 0
        dice_loss = 0
        iou_loss = 0

        for batch in batchx:
            id, img, gt_masks, image_size, scale_factor = batch['parcel_id'], batch['image'], batch['gt_masks'], batch['image_size'], batch['scale_factor']
            image_embeddings, sparse_embeddings, dense_embeddings = self.encode(self.cfg, img, gt_masks, image_size, scale_factor)
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

            del high_res_masks, pixel_masks_tensor, low_res_masks, iou_predictions, upscaled_masks, image_embeddings, sparse_embeddings, dense_embeddings   

        return (focal_loss/instances, dice_loss/instances, iou_loss/instances)
    
    
    def encode(self, cfg, img, gt_masks, image_size, scale_factor):
        """
		Encode the input image and the prompts
		"""
        # Image needs to be resized to 1024*1024 and necessary preprocessing should be done
        sam_transform = ResizeLongestSide(self.model.image_encoder.img_size)
        resize_img = sam_transform.apply_image(img)
        resize_img = torch.as_tensor(resize_img.transpose(2, 0, 1)).to(self.device)
        resize_img = self.model.preprocess(resize_img[None,:,:,:]) # (1, 3, 1024, 1024)

        # Get the bbox, point prompts and input labels and transform accordingly
        bbox_prompts, point_prompts, input_labels_prompts = get_bbox_point_and_inputlabel_prompts(gt_masks, image_size[0], image_size[1], cfg.BBOX.NUMBER, cfg.BBOX.MIN_DISTANCE, cfg.BBOX.SIZE_REF)
        #scale the bbox prompts and point prompts according to the scale factor
        bbox_prompts = np.around(np.array(bbox_prompts) * scale_factor)
        point_prompts = np.around(np.array(point_prompts) * scale_factor)

        bbox_prompts = torch.as_tensor(bbox_prompts).to(self.device)

        # limit the number of prompts to the box limiter value
        if len(bbox_prompts) > cfg.BBOX.BOX_LIMITER:
            bbox_prompts = bbox_prompts[:cfg.BBOX.BOX_LIMITER]
            point_prompts = point_prompts[:cfg.BBOX.BOX_LIMITER]
            input_labels_prompts = input_labels_prompts[:cfg.BBOX.BOX_LIMITER]

        with torch.no_grad():
            # Get the image embeddings, sparse embeddings and dense embeddings from the image encoder and prompt encoder
            image_embeddings = self.model.image_encoder(resize_img.to(self.device))
            sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                points=None,
                boxes=bbox_prompts.to(self.device),
                masks=None,
            )

        return image_embeddings, sparse_embeddings, dense_embeddings
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        torch.cuda.empty_cache()
