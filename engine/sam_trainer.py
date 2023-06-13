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

import numpy as np
import matplotlib.pyplot as plt
from modeling.segment_anything.utils.transforms import ResizeLongestSide
import torch.nn.functional as F

def _build_point_grid(n_per_side: int) -> np.ndarray:
    """Generates a 2D grid of points evenly spaced in [0,1]x[0,1]."""
    offset = 1 / (2 * n_per_side)
    points_one_side = np.linspace(offset, 1 - offset, n_per_side)
    points_x = np.tile(points_one_side[None, :], (n_per_side, 1))
    points_y = np.tile(points_one_side[:, None], (1, n_per_side))
    points = np.stack([points_x, points_y], axis=-1).reshape(-1, 2)
    return points

def _get_bbox_point_and_inputlabel_prompts(pixel_masks):
    bbox_prompts = []
    point_prompts = []
    input_labels_prompts = []
    for mask in pixel_masks:
        # get bounding box from mask
        y_indices, x_indices = np.where(mask > 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        # add perturbation to bounding box coordinates
        H, W = mask.shape
        x_min = max(0, x_min - np.random.randint(0, 20))
        x_max = min(W, x_max + np.random.randint(0, 20))
        y_min = max(0, y_min - np.random.randint(0, 20))
        y_max = min(H, y_max + np.random.randint(0, 20))
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

def _post_process_masks_pt(masks, original_size, reshaped_input_sizes, mask_threshold=0.0, binarize=True, pad_size=None):
        target_image_size = (1024, 1024)
        
        interpolated_masks = F.interpolate(masks, target_image_size, mode="bilinear", align_corners=False)
        interpolated_masks = interpolated_masks[..., : reshaped_input_sizes[0], : reshaped_input_sizes[1]]
        interpolated_masks = F.interpolate(interpolated_masks, original_size, mode="bilinear", align_corners=False)
        if binarize:
            interpolated_masks = interpolated_masks > mask_threshold
        
        return interpolated_masks


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

        for batch in progress_bar:
            for item in batch:

                image =  item["image"]
                pixel_masks = item["gt_masks"]
                scale_factor = item["scale_factor"]
                original_image_size = item["image_size"]
                
                # TODO: Image needs to be resized to 1024*1024 and necessary preprocessing should be done
                sam_transform = ResizeLongestSide(model.image_encoder.img_size)
                resize_img = sam_transform.apply_image(image)
                resize_img = torch.as_tensor(resize_img.transpose(2, 0, 1)).to(device)
                resize_img = model.preprocess(resize_img[None,:,:,:]) # (1, 3, 1024, 1024)

                # TODO: Get the bbox, point prompts and input labels and transform accordingly
                bbox_prompts, point_prompts, input_labels_prompts = _get_bbox_point_and_inputlabel_prompts(pixel_masks)
                #scale the bbox prompts and point prompts according to the scale factor
                bbox_prompts = np.around(np.array(bbox_prompts) * scale_factor)
                point_prompts = np.around(np.array(point_prompts) * scale_factor)

                bbox_prompts = torch.as_tensor(bbox_prompts).to(device)
            

                # TODO: Obtain the image embeddings from the image encoder, image encoder here is run with inference mode (Strict version of no grad)
                # This will be done only once
                with torch.inference_mode():
                    image_embeddings = model.image_encoder(resize_img)  # (B,256,64,64)

                # TODO: Obtain the sparse and dense embeddings from the prompt encoder, prompt encoder here is run with inference mode (Strict version of no grad)
                # if points are provided then providing labels are mandatory
                # This will repeat mutiple times and for all the masks we will calculate the losses
                with torch.inference_mode():
                    sparse_embeddings, dense_embeddings = model.prompt_encoder(
                        points=None,
                        boxes=bbox_prompts,
                        masks=None,
                    )
                
                with torch.inference_mode():    #Just keeping it here to reduce memory consumption in the GPU
                    low_res_masks, iou_predictions = model.mask_decoder(
                        image_embeddings=image_embeddings.to(device),  # (B, 256, 64, 64)
                        image_pe=model.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
                        sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
                        dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
                        multimask_output=False,
                    )

                # TODO: Postprocess and retrieve the predicted mask as a binary mask
                high_res_masks = np.squeeze(_post_process_masks_pt(masks = low_res_masks, original_size = original_image_size, reshaped_input_sizes = [1024, 1024], mask_threshold=0.0, binarize=True, pad_size=None))
                
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

                # TODO: Calculate the loss between each instance of the mask and the predicted mask 

                # delete all the local variables and cuda cache
                del image_embeddings, sparse_embeddings, dense_embeddings
                del image, pixel_masks, scale_factor, original_image_size
                del bbox_prompts, point_prompts, input_labels_prompts
                del resize_img
                del low_res_masks, iou_predictions
                torch.cuda.empty_cache()




