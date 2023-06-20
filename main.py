# encoding: utf-8
"""
@author:  Aninda Ghosh
@contact: aghosh57@asu.edu
"""

import argparse
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
import gc

from tqdm import tqdm
import torch

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from modeling.segment_anything.utils.transforms import ResizeLongestSide
import torch.nn.functional as F
from torch.nn.functional import threshold, normalize

from solver.loss import FocalLoss, DiceLoss, IoULoss

from utils.data_support import get_bbox_point_and_inputlabel_prompts





def train(model, device, train_loader, optimizer, Focal_Loss, Dice_Loss, Iou_Loss):
    epoch_loss = 0
    # Set the model to training mode
    model.train()       
    # Iterate over the batches and acculmulate the loss for each batch and update the model
    for batch in tqdm(train_loader, total=len(train_loader)):
        instances = 0
        focal_loss = 0
        dice_loss = 0
        iou_loss = 0
        
        # Zero the gradients for the optimizer
        optimizer.zero_grad()

        # This is one item(image) in the batch, here we will calculate the loss for each image and each mask in the image
        for item in batch:
            
            image =  item["image"]
            pixel_masks = item["gt_masks"]
            scale_factor = item["scale_factor"]
            original_image_size = item["image_size"]
            
            # Image needs to be resized to 1024*1024 and necessary preprocessing should be done
            sam_transform = ResizeLongestSide(model.image_encoder.img_size)
            resize_img = sam_transform.apply_image(image)
            resize_img = torch.as_tensor(resize_img.transpose(2, 0, 1)).to(device)
            resize_img = model.preprocess(resize_img[None,:,:,:]) # (1, 3, 1024, 1024)

            # Get the bbox, point prompts and input labels and transform accordingly
            bbox_prompts, point_prompts, input_labels_prompts = get_bbox_point_and_inputlabel_prompts(pixel_masks, original_image_size[0], original_image_size[1], cfg.BBOX.NUMBER, cfg.BBOX.MIN_DISTANCE, cfg.BBOX.SIZE_REF)
            #scale the bbox prompts and point prompts according to the scale factor
            bbox_prompts = np.around(np.array(bbox_prompts) * scale_factor)
            point_prompts = np.around(np.array(point_prompts) * scale_factor)

            bbox_prompts = torch.as_tensor(bbox_prompts).to(device)

            # limit the number of prompts to the box limiter value
            if len(bbox_prompts) > cfg.BBOX.BOX_LIMITER:
                bbox_prompts = bbox_prompts[:cfg.BBOX.BOX_LIMITER]
                point_prompts = point_prompts[:cfg.BBOX.BOX_LIMITER]
                input_labels_prompts = input_labels_prompts[:cfg.BBOX.BOX_LIMITER]
        

            with torch.no_grad():
                # Obtain the image embeddings from the image encoder, image encoder here is run with inference mode (Strict version of no grad)
                # This will be done only once
                image_embeddings = model.image_encoder(resize_img)  # (B,256,64,64)
                #TODO: Store the image embeddings in torch cache for later use
                    

                # Obtain the sparse and dense embeddings from the prompt encoder, prompt encoder here is run with inference mode (Strict version of no grad)
                # if points are provided then providing labels are mandatory
                # This will repeat mutiple times and for all the masks we will calculate the losses
                sparse_embeddings, dense_embeddings = model.prompt_encoder(
                    points=None,
                    boxes=bbox_prompts,
                    masks=None,
                )
            
            # Run the mask decoder with the image embeddings, sparse embeddings and dense embeddings
            low_res_masks, iou_predictions = model.mask_decoder(
                image_embeddings=image_embeddings.to(device),  # (B, 256, 64, 64)
                image_pe=model.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
                sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
                dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
                multimask_output=False,
            )

            # Postprocess and retrieve the predicted mask as a binary mask
            # high_res_masks = np.squeeze(_post_process_masks_pt(masks = low_res_masks, original_size = original_image_size, reshaped_input_sizes = [1024, 1024], mask_threshold=0.0, binarize=False, pad_size=None))
            
            upscaled_masks = model.postprocess_masks(low_res_masks, (1024, 1024), original_image_size).to(device)
            high_res_masks = normalize(threshold(upscaled_masks, 0.0, 0)).to(device).float()

            # Plot the masks and the image (Only For Visualization Purposes)
            # for i in range(len(high_res_masks)):
            #     fig, ax = plt.subplots(1, 3)
            #     ax[0].imshow(image)
            #     ax[0].set_title("Original Image")
            #     ax[1].imshow(high_res_masks[i].cpu())
            #     ax[1].set_title("Predicted Mask")
            #     ax[2].imshow(pixel_masks[i])
            #     ax[2].set_title("GT Mask")
            #     plt.show()

            # Calculate the loss between each instance of the mask and the predicted mask and accumulate the loss
            
            if len(pixel_masks) == 0:   # if no masks are provided then generate boolean pixel masks with image original size
                _pixel_masks = []
                # generate boolean pixel masks with image original size, use False as the mask value
                for i in range(len(high_res_masks)):
                    _pixel_masks.append(np.full((original_image_size[0], original_image_size[1]), False, dtype=bool))
                pixel_masks_tensor = torch.as_tensor(np.array(_pixel_masks).astype(float)).to(device).float()
            else:
                pixel_masks_tensor = torch.as_tensor(np.array(pixel_masks).astype(float)).to(device).float()

            for i in range(len(high_res_masks)):
                focal_loss += Focal_Loss(high_res_masks[i], pixel_masks_tensor[i])
                dice_loss += Dice_Loss(high_res_masks[i], pixel_masks_tensor[i])
                iou_loss += Iou_Loss(high_res_masks[i], pixel_masks_tensor[i], iou_predictions[i][0].to(device))
                instances += 1
            
        
            # delete all the local variables and cuda cache after each image is processed, only leave the global loss variable that can be deleted after one batch is processed
            del image, pixel_masks, pixel_masks_tensor, scale_factor, original_image_size, sam_transform, resize_img, bbox_prompts, point_prompts, input_labels_prompts, image_embeddings, sparse_embeddings, dense_embeddings, low_res_masks, iou_predictions, high_res_masks
            
        
        # Update the model after each batch            
        focal_loss = (focal_loss/instances) 
        dice_loss = (dice_loss/instances)   
        iou_loss = (iou_loss/instances)

        # Update the model with the loss
        loss = (cfg.LOSS.FOCAL_LOSS_WEIGHT * focal_loss) + (cfg.LOSS.DICE_LOSS_WEIGHT * dice_loss)
        loss.backward()
        optimizer.step()

        torch.cuda.empty_cache()
        gc.collect()
    
        epoch_loss += loss.item()

    return epoch_loss



def validate(model, device, valid_loader, Focal_Loss, Dice_Loss, Iou_Loss):
    epoch_loss = 0
    # Set the model to training mode
    model.eval()       
    # Iterate over the batches and acculmulate the loss for each batch and update the model
    for batch in tqdm(valid_loader, total=len(valid_loader)):
        instances = 0
        focal_loss = 0
        dice_loss = 0
        iou_loss = 0
        
        # This is one item(image) in the batch, here we will calculate the loss for each image and each mask in the image
        for item in batch:
            
            image =  item["image"]
            pixel_masks = item["gt_masks"]
            scale_factor = item["scale_factor"]
            original_image_size = item["image_size"]
            
            # Image needs to be resized to 1024*1024 and necessary preprocessing should be done
            sam_transform = ResizeLongestSide(model.image_encoder.img_size)
            resize_img = sam_transform.apply_image(image)
            resize_img = torch.as_tensor(resize_img.transpose(2, 0, 1)).to(device)
            resize_img = model.preprocess(resize_img[None,:,:,:]) # (1, 3, 1024, 1024)

            # Get the bbox, point prompts and input labels and transform accordingly
            bbox_prompts, point_prompts, input_labels_prompts = get_bbox_point_and_inputlabel_prompts(pixel_masks, original_image_size[0], original_image_size[1], cfg.BBOX.NUMBER, cfg.BBOX.MIN_DISTANCE, cfg.BBOX.SIZE_REF)
            #scale the bbox prompts and point prompts according to the scale factor
            bbox_prompts = np.around(np.array(bbox_prompts) * scale_factor)
            point_prompts = np.around(np.array(point_prompts) * scale_factor)

            bbox_prompts = torch.as_tensor(bbox_prompts).to(device)

            # limit the number of prompts to the box limiter value
            if len(bbox_prompts) > cfg.BBOX.BOX_LIMITER:
                bbox_prompts = bbox_prompts[:cfg.BBOX.BOX_LIMITER]
                point_prompts = point_prompts[:cfg.BBOX.BOX_LIMITER]
                input_labels_prompts = input_labels_prompts[:cfg.BBOX.BOX_LIMITER]
        

            with torch.no_grad():
                # Obtain the image embeddings from the image encoder, image encoder here is run with inference mode (Strict version of no grad)
                # This will be done only once
                image_embeddings = model.image_encoder(resize_img)  # (B,256,64,64)
                #TODO: Store the image embeddings in torch cache for later use
                    

                # Obtain the sparse and dense embeddings from the prompt encoder, prompt encoder here is run with inference mode (Strict version of no grad)
                # if points are provided then providing labels are mandatory
                # This will repeat mutiple times and for all the masks we will calculate the losses
                sparse_embeddings, dense_embeddings = model.prompt_encoder(
                    points=None,
                    boxes=bbox_prompts,
                    masks=None,
                )
            
                # Run the mask decoder with the image embeddings, sparse embeddings and dense embeddings
                low_res_masks, iou_predictions = model.mask_decoder(
                    image_embeddings=image_embeddings.to(device),  # (B, 256, 64, 64)
                    image_pe=model.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
                    sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
                    dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
                    multimask_output=False,
                )

            # Postprocess and retrieve the predicted mask as a binary mask
            # high_res_masks = np.squeeze(_post_process_masks_pt(masks = low_res_masks, original_size = original_image_size, reshaped_input_sizes = [1024, 1024], mask_threshold=0.0, binarize=False, pad_size=None))
            
            upscaled_masks = model.postprocess_masks(low_res_masks, (1024, 1024), original_image_size).to(device)
            high_res_masks = normalize(threshold(upscaled_masks, 0.0, 0)).to(device).float()

            # Plot the masks and the image (Only For Visualization Purposes)
            # for i in range(len(high_res_masks)):
            #     fig, ax = plt.subplots(1, 3)
            #     ax[0].imshow(image)
            #     ax[0].set_title("Original Image")
            #     ax[1].imshow(high_res_masks[i].cpu())
            #     ax[1].set_title("Predicted Mask")
            #     ax[2].imshow(pixel_masks[i])
            #     ax[2].set_title("GT Mask")
            #     plt.show()

            # Calculate the loss between each instance of the mask and the predicted mask and accumulate the loss
            if len(pixel_masks) == 0:   # if no masks are provided then generate boolean pixel masks with image original size
                _pixel_masks = []
                # generate boolean pixel masks with image original size, use False as the mask value
                for i in range(len(high_res_masks)):
                    _pixel_masks.append(np.full((original_image_size[0], original_image_size[1]), False, dtype=bool))
                pixel_masks_tensor = torch.as_tensor(np.array(_pixel_masks).astype(float)).to(device).float()
            else:
                pixel_masks_tensor = torch.as_tensor(np.array(pixel_masks).astype(float)).to(device).float()

            for i in range(len(high_res_masks)):
                focal_loss += Focal_Loss(high_res_masks[i], pixel_masks_tensor[i])
                dice_loss += Dice_Loss(high_res_masks[i], pixel_masks_tensor[i])
                iou_loss += Iou_Loss(high_res_masks[i], pixel_masks_tensor[i], iou_predictions[i][0].to(device))
                instances += 1
            
        
            # delete all the local variables and cuda cache after each image is processed, only leave the global loss variable that can be deleted after one batch is processed
            del image, pixel_masks, pixel_masks_tensor, scale_factor, original_image_size, sam_transform, resize_img, bbox_prompts, point_prompts, input_labels_prompts, image_embeddings, sparse_embeddings, dense_embeddings, low_res_masks, iou_predictions, high_res_masks
            
        
        # Update the model after each batch            
        focal_loss = (focal_loss/instances) 
        dice_loss = (dice_loss/instances)   
        iou_loss = (iou_loss/instances)

        # Update the model with the loss
        loss = (cfg.LOSS.FOCAL_LOSS_WEIGHT * focal_loss) + (cfg.LOSS.DICE_LOSS_WEIGHT * dice_loss)
        
        torch.cuda.empty_cache()
        gc.collect()
    
        epoch_loss += loss.item()

    return epoch_loss
    

def test(model, device, test_loader, Focal_Loss, Dice_Loss, Iou_Loss):
    epoch_loss = 0
    # Set the model to training mode
    model.eval()       
    # Iterate over the batches and acculmulate the loss for each batch and update the model
    for batch in tqdm(test_loader, total=len(test_loader)):
        instances = 0
        focal_loss = 0
        dice_loss = 0
        iou_loss = 0
        
        # This is one item(image) in the batch, here we will calculate the loss for each image and each mask in the image
        for item in batch:
            
            image =  item["image"]
            pixel_masks = item["gt_masks"]
            scale_factor = item["scale_factor"]
            original_image_size = item["image_size"]
            
            # Image needs to be resized to 1024*1024 and necessary preprocessing should be done
            sam_transform = ResizeLongestSide(model.image_encoder.img_size)
            resize_img = sam_transform.apply_image(image)
            resize_img = torch.as_tensor(resize_img.transpose(2, 0, 1)).to(device)
            resize_img = model.preprocess(resize_img[None,:,:,:]) # (1, 3, 1024, 1024)

            # Get the bbox, point prompts and input labels and transform accordingly
            bbox_prompts, point_prompts, input_labels_prompts = get_bbox_point_and_inputlabel_prompts(pixel_masks, original_image_size[0], original_image_size[1], cfg.BBOX.NUMBER, cfg.BBOX.MIN_DISTANCE, cfg.BBOX.SIZE_REF)
            #scale the bbox prompts and point prompts according to the scale factor
            bbox_prompts = np.around(np.array(bbox_prompts) * scale_factor)
            point_prompts = np.around(np.array(point_prompts) * scale_factor)

            bbox_prompts = torch.as_tensor(bbox_prompts).to(device)

            # limit the number of prompts to the box limiter value
            if len(bbox_prompts) > cfg.BBOX.BOX_LIMITER:
                bbox_prompts = bbox_prompts[:cfg.BBOX.BOX_LIMITER]
                point_prompts = point_prompts[:cfg.BBOX.BOX_LIMITER]
                input_labels_prompts = input_labels_prompts[:cfg.BBOX.BOX_LIMITER]
        

            with torch.inference_mode():
                # Obtain the image embeddings from the image encoder, image encoder here is run with inference mode (Strict version of no grad)
                # This will be done only once
                image_embeddings = model.image_encoder(resize_img)  # (B,256,64,64)
                #TODO: Store the image embeddings in torch cache for later use
                    

                # Obtain the sparse and dense embeddings from the prompt encoder, prompt encoder here is run with inference mode (Strict version of no grad)
                # if points are provided then providing labels are mandatory
                # This will repeat mutiple times and for all the masks we will calculate the losses
                sparse_embeddings, dense_embeddings = model.prompt_encoder(
                    points=None,
                    boxes=bbox_prompts,
                    masks=None,
                )
            
                # Run the mask decoder with the image embeddings, sparse embeddings and dense embeddings
                low_res_masks, iou_predictions = model.mask_decoder(
                    image_embeddings=image_embeddings.to(device),  # (B, 256, 64, 64)
                    image_pe=model.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
                    sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
                    dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
                    multimask_output=False,
                )

            # Postprocess and retrieve the predicted mask as a binary mask
            # high_res_masks = np.squeeze(_post_process_masks_pt(masks = low_res_masks, original_size = original_image_size, reshaped_input_sizes = [1024, 1024], mask_threshold=0.0, binarize=False, pad_size=None))
            
            upscaled_masks = model.postprocess_masks(low_res_masks, (1024, 1024), original_image_size).to(device)
            high_res_masks = normalize(threshold(upscaled_masks, 0.0, 0)).to(device).float()

            # Plot the masks and the image (Only For Visualization Purposes)
            # for i in range(len(high_res_masks)):
            #     fig, ax = plt.subplots(1, 3)
            #     ax[0].imshow(image)
            #     ax[0].set_title("Original Image")
            #     ax[1].imshow(high_res_masks[i].cpu())
            #     ax[1].set_title("Predicted Mask")
            #     ax[2].imshow(pixel_masks[i])
            #     ax[2].set_title("GT Mask")
            #     plt.show()

            # Calculate the loss between each instance of the mask and the predicted mask and accumulate the loss
            if len(pixel_masks) == 0:   # if no masks are provided then generate boolean pixel masks with image original size
                _pixel_masks = []
                # generate boolean pixel masks with image original size, use False as the mask value
                for i in range(len(high_res_masks)):
                    _pixel_masks.append(np.full((original_image_size[0], original_image_size[1]), False, dtype=bool))
                pixel_masks_tensor = torch.as_tensor(np.array(_pixel_masks).astype(float)).to(device).float()
            else:
                pixel_masks_tensor = torch.as_tensor(np.array(pixel_masks).astype(float)).to(device).float()

            for i in range(len(high_res_masks)):
                focal_loss += Focal_Loss(high_res_masks[i], pixel_masks_tensor[i])
                dice_loss += Dice_Loss(high_res_masks[i], pixel_masks_tensor[i])
                iou_loss += Iou_Loss(high_res_masks[i], pixel_masks_tensor[i], iou_predictions[i][0].to(device))
                instances += 1
            
        
            # delete all the local variables and cuda cache after each image is processed, only leave the global loss variable that can be deleted after one batch is processed
            del image, pixel_masks, pixel_masks_tensor, scale_factor, original_image_size, sam_transform, resize_img, bbox_prompts, point_prompts, input_labels_prompts, image_embeddings, sparse_embeddings, dense_embeddings, low_res_masks, iou_predictions, high_res_masks
            
        
        # Update the model after each batch            
        focal_loss = (focal_loss/instances) 
        dice_loss = (dice_loss/instances)   
        iou_loss = (iou_loss/instances)

        # Update the model with the loss
        loss = (cfg.LOSS.FOCAL_LOSS_WEIGHT * focal_loss) + (cfg.LOSS.DICE_LOSS_WEIGHT * dice_loss)
        
        torch.cuda.empty_cache()
        gc.collect()
    
        epoch_loss += loss.item()

    return epoch_loss



def main():
    
    cfg.freeze()

    # Get the model and download the checkpoint if needed
    model = prepare_sam(checkpoint=cfg.MODEL.CHECKPOINT, model_type = 'large')
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


    training_losses = []
    validation_losses = []
    for epoch in range(1, epochs+1):
        print(f'\nEpoch {epoch}:')

        print('Training the model...')
        # Train the model
        epoch_train_loss = train(model, device, train_loader, optimizer, Focal_Loss, Dice_Loss, Iou_Loss)
        epoch_train_loss /= (epoch)
        scheduler.step(metrics=epoch_train_loss)
        training_losses.append(epoch_train_loss)

        print('Validating the model...')
        # Validate the model
        epoch_valid_loss = validate(model, device, valid_loader, Focal_Loss, Dice_Loss, Iou_Loss)
        epoch_valid_loss /= (epoch)
        validation_losses.append(epoch_valid_loss)

        print(f'Train Loss: {epoch_train_loss}, Valid Loss: {epoch_valid_loss}')

        if epoch % 30 == 0:
            print(f'\nSaving the model at epoch {epoch}\n')
            torch.save(model.state_dict(), f"modeling/model_checkpoints/sam_checkpoint_{epoch}.pth")

    # Test the model
    test_loss = test(model, device, test_loader, Focal_Loss, Dice_Loss, Iou_Loss)
    print(f'\nTest Loss: {test_loss}')

if __name__ == '__main__':
    main()
