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


def _build_point_grid(n_per_side: int) -> np.ndarray:
    """Generates a 2D grid of points evenly spaced in [0,1]x[0,1]."""
    offset = 1 / (2 * n_per_side)
    points_one_side = np.linspace(offset, 1 - offset, n_per_side)
    points_x = np.tile(points_one_side[None, :], (n_per_side, 1))
    points_y = np.tile(points_one_side[:, None], (1, n_per_side))
    points = np.stack([points_x, points_y], axis=-1).reshape(-1, 2)
    return points

def _get_bbox_point_and_inputlabel_prompts(pixel_masks, image_width, image_height, num_boxes, min_distance, box_percentage):
    bbox_prompts = []
    point_prompts = []
    input_labels_prompts = []
    
    if len(pixel_masks) == 0:
        # TODO: We have to create few bounding boxes here with just blank background
        bbox_prompts = generate_random_bounding_boxes(image_width, image_height, num_boxes, min_distance, box_percentage)
    else:
        for mask in pixel_masks:
            # get bounding box from mask
            y_indices, x_indices = np.where(mask > 0)
            x_min, x_max = np.min(x_indices), np.max(x_indices)
            y_min, y_max = np.min(y_indices), np.max(y_indices)
            # add perturbation to bounding box coordinates
            H, W = mask.shape
            x_min = max(0, x_min - np.random.randint(0, 30))
            x_max = min(W, x_max + np.random.randint(0, 30))
            y_min = max(0, y_min - np.random.randint(0, 30))
            y_max = min(H, y_max + np.random.randint(0, 30))
            bbox = [x_min, y_min, x_max, y_max]
            bbox_prompts.append(bbox)

            # Get grid points within the bounding box
            point_grid_per_crop= _build_point_grid(10)
            #This forms a 2D grid points and we need to normalize it to the bbox size
            point_grid_per_crop[:, 0] = point_grid_per_crop[:, 0] * (x_max - x_min) + x_min
            point_grid_per_crop[:, 1] = point_grid_per_crop[:, 1] * (y_max - y_min) + y_min
            #Convert the grid points to a integer
            point_grid_per_crop = np.around(point_grid_per_crop).astype(np.float64)
            points_per_crop = np.array([np.array([point_grid_per_crop])])
            points_per_crop = np.transpose(points_per_crop, axes=(0, 2, 1, 3))
            point_prompts.append(points_per_crop)

            input_labels = np.ones_like(points_per_crop[:, :, :, 0], dtype=np.int64)
            input_labels_prompts.append(input_labels)

    return bbox_prompts, point_prompts, input_labels_prompts



# To Generate the Random Points and Bounding Boxes using https://www.jasondavies.com/poisson-disc/ algorithm
def generate_random_bounding_boxes(image_width, image_height, number_boxes, min_distance, box_percentage):
    def generate_random_points(image_width, image_height, min_distance, num_points, k=30):
        image_size = (image_width, image_height)
        cell_size = min_distance / np.sqrt(2)
        grid_width = int(np.ceil(image_width / cell_size))
        grid_height = int(np.ceil(image_height / cell_size))
        grid = np.empty((grid_width, grid_height), dtype=np.int32)
        grid.fill(-1)

        points = []
        active_points = []

        def generate_random_point():
            return np.random.uniform(0, image_width), np.random.uniform(0, image_height)

        def get_neighboring_cells(point):
            x, y = point
            x_index = int(x / cell_size)
            y_index = int(y / cell_size)

            cells = []
            for i in range(max(0, x_index - 2), min(grid_width, x_index + 3)):
                for j in range(max(0, y_index - 2), min(grid_height, y_index + 3)):
                    cells.append((i, j))

            return cells

        def is_point_valid(point):
            x, y = point
            if x < 0 or y < 0 or x >= image_width or y >= image_height:
                return False

            x_index = int(x / cell_size)
            y_index = int(y / cell_size)

            cells = get_neighboring_cells(point)
            for cell in cells:
                if grid[cell] != -1:
                    cell_points = points[grid[cell]]
                    if np.any(np.linalg.norm(np.array(cell_points) - np.array(point), axis=None) < min_distance):
                        return False

            return True

        def add_point(point):
            x, y = point
            x_index = int(x / cell_size)
            y_index = int(y / cell_size)

            points.append(point)
            index = len(points) - 1
            grid[x_index, y_index] = index
            active_points.append(point)

        start_point = generate_random_point()
        add_point(start_point)

        while active_points and len(points) < num_points:
            random_index = np.random.randint(len(active_points))
            random_point = active_points[random_index]
            added_new_point = False

            for _ in range(k):
                angle = 2 * np.pi * np.random.random()
                radius = min_distance + min_distance * np.random.random()
                new_point = (random_point[0] + radius * np.cos(angle), random_point[1] + radius * np.sin(angle))
                if is_point_valid(new_point):
                    add_point(new_point)
                    added_new_point = True

            if not added_new_point:
                active_points.pop(random_index)

        return points
    

    points = generate_random_points(image_width, image_height, min_distance, number_boxes)
    
    
    box_width = int(image_width * box_percentage)
    box_height = int(image_height * box_percentage)

    bounding_boxes = []
    for point in points:
        x = int(point[0] - box_width / 2)
        y = int(point[1] - box_height / 2)

        # Adjust the coordinates to keep the bounding box within the image
        x = max(0, min(x, image_width - box_width))
        y = max(0, min(y, image_height - box_height))

        bounding_boxes.append([x, y, x+box_width, y+box_height])

    return bounding_boxes

# def visualize_points_and_boxes(image_width, image_height, bounding_boxes):
#     # Create a figure and axis
#     fig, ax = plt.subplots(1)
#     ax.set_xlim(0, image_width)
#     ax.set_ylim(0, image_height)

#     # # Plot the points
#     # for point in points:
#     #     ax.plot(point[0], point[1], 'ro')

#     # Plot the bounding boxes
#     for bbox in bounding_boxes:
#         rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1], linewidth=1, edgecolor='g', facecolor='none')
#         ax.add_patch(rect)

#     # Display the image
#     plt.show()




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
            bbox_prompts, point_prompts, input_labels_prompts = _get_bbox_point_and_inputlabel_prompts(pixel_masks, original_image_size[0], original_image_size[1], cfg.BBOX.NUMBER, cfg.BBOX.MIN_DISTANCE, cfg.BBOX.SIZE_REF)
            #scale the bbox prompts and point prompts according to the scale factor
            bbox_prompts = np.around(np.array(bbox_prompts) * scale_factor)
            point_prompts = np.around(np.array(point_prompts) * scale_factor)

            bbox_prompts = torch.as_tensor(bbox_prompts).to(device)

            # limit the number of prompts to 50, let's expect that we will have maximum 50 prompts for each image
            if len(bbox_prompts) > 30:
                bbox_prompts = bbox_prompts[:30]
                point_prompts = point_prompts[:30]
                input_labels_prompts = input_labels_prompts[:30]
        

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
                # generate boolean pixel masks with image original size, use False as the mask value
                for i in range(len(high_res_masks)):
                    pixel_masks.append(np.zeros((1, original_image_size[0], original_image_size[1])))
                pixel_masks = torch.as_tensor(pixel_masks).to(device).float()
            else:
                pixel_masks = torch.as_tensor(pixel_masks.astype(float)).to(device).float()

            for i in range(len(high_res_masks)):
                focal_loss += Focal_Loss(high_res_masks[i], pixel_masks[i])
                dice_loss += Dice_Loss(high_res_masks[i], pixel_masks[i])
                iou_loss += Iou_Loss(high_res_masks[i], pixel_masks[i], iou_predictions[i][0].to(device))
                instances += 1
            
        
            # delete all the local variables and cuda cache after each image is processed, only leave the global loss variable that can be deleted after one batch is processed
            del image, pixel_masks, scale_factor, original_image_size, sam_transform, resize_img, bbox_prompts, point_prompts, input_labels_prompts, image_embeddings, sparse_embeddings, dense_embeddings, low_res_masks, iou_predictions, high_res_masks
            
        
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
            bbox_prompts, point_prompts, input_labels_prompts = _get_bbox_point_and_inputlabel_prompts(pixel_masks, original_image_size[0], original_image_size[1], cfg.BBOX.NUMBER, cfg.BBOX.MIN_DISTANCE, cfg.BBOX.SIZE_REF)
            #scale the bbox prompts and point prompts according to the scale factor
            bbox_prompts = np.around(np.array(bbox_prompts) * scale_factor)
            point_prompts = np.around(np.array(point_prompts) * scale_factor)

            bbox_prompts = torch.as_tensor(bbox_prompts).to(device)

            # limit the number of prompts to 30
            if len(bbox_prompts) > 30:
                bbox_prompts = bbox_prompts[:30]
                point_prompts = point_prompts[:30]
                input_labels_prompts = input_labels_prompts[:30]
        

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
            pixel_masks = torch.as_tensor(pixel_masks.astype(float)).to(device).float()

            for i in range(len(high_res_masks)):
                focal_loss += Focal_Loss(high_res_masks[i], pixel_masks[i])
                dice_loss += Dice_Loss(high_res_masks[i], pixel_masks[i])
                iou_loss += Iou_Loss(high_res_masks[i], pixel_masks[i], iou_predictions[i][0].to(device))
                instances += 1
            
        
            # delete all the local variables and cuda cache after each image is processed, only leave the global loss variable that can be deleted after one batch is processed
            del image, pixel_masks, scale_factor, original_image_size, sam_transform, resize_img, bbox_prompts, point_prompts, input_labels_prompts, image_embeddings, sparse_embeddings, dense_embeddings, low_res_masks, iou_predictions, high_res_masks
            
        
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
            bbox_prompts, point_prompts, input_labels_prompts = _get_bbox_point_and_inputlabel_prompts(pixel_masks, original_image_size[0], original_image_size[1], cfg.BBOX.NUMBER, cfg.BBOX.MIN_DISTANCE, cfg.BBOX.SIZE_REF)
            #scale the bbox prompts and point prompts according to the scale factor
            bbox_prompts = np.around(np.array(bbox_prompts) * scale_factor)
            point_prompts = np.around(np.array(point_prompts) * scale_factor)

            bbox_prompts = torch.as_tensor(bbox_prompts).to(device)

            # limit the number of prompts to 30
            if len(bbox_prompts) > 30:
                bbox_prompts = bbox_prompts[:30]
                point_prompts = point_prompts[:30]
                input_labels_prompts = input_labels_prompts[:30]
        

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
            pixel_masks = torch.as_tensor(pixel_masks.astype(float)).to(device).float()

            for i in range(len(high_res_masks)):
                focal_loss += Focal_Loss(high_res_masks[i], pixel_masks[i])
                dice_loss += Dice_Loss(high_res_masks[i], pixel_masks[i])
                iou_loss += Iou_Loss(high_res_masks[i], pixel_masks[i], iou_predictions[i][0].to(device))
                instances += 1
            
        
            # delete all the local variables and cuda cache after each image is processed, only leave the global loss variable that can be deleted after one batch is processed
            del image, pixel_masks, scale_factor, original_image_size, sam_transform, resize_img, bbox_prompts, point_prompts, input_labels_prompts, image_embeddings, sparse_embeddings, dense_embeddings, low_res_masks, iou_predictions, high_res_masks
            
        
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
    model = prepare_sam(checkpoint=cfg.MODEL.CHECKPOINT, model_type = 'base')
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
        
        # Train the model
        epoch_train_loss = train(model, device, train_loader, optimizer, Focal_Loss, Dice_Loss, Iou_Loss)
        epoch_train_loss /= (epoch)
        scheduler.step(metrics=epoch_train_loss)
        training_losses.append(epoch_train_loss)


        # Validate the model
        epoch_valid_loss = validate(model, device, valid_loader, Focal_Loss, Dice_Loss, Iou_Loss)
        epoch_valid_loss /= (epoch)
        validation_losses.append(epoch_valid_loss)

        print(f'EPOCH: {epoch}, Train Loss: {epoch_train_loss}, Valid Loss: {epoch_valid_loss}\n')

        if epoch % 30 == 0:
            print(f'Saving the model at epoch {epoch}\n')
            torch.save(model.state_dict(), f"modeling/model_checkpoints/sam_checkpoint_{epoch}.pth")

    # Test the model
    test_loss = test(model, device, test_loader, Focal_Loss, Dice_Loss, Iou_Loss)
    print(f'Test Loss: {test_loss}')

if __name__ == '__main__':
    main()
