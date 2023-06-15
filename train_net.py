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

# from engine.sam_trainer import do_train

from data.build import SAM_Dataloader
from modeling.segment_anything import prepare_sam
from solver.optimizer import build_optimizer
from solver.lr_schedular import build_lrSchedular
from solver.loss import FocalLoss, DiceLoss

from utils.logger import setup_logger

from config import cfg  # Import the default config file

from tqdm import tqdm
import torch

import numpy as np
import matplotlib.pyplot as plt
from modeling.segment_anything.utils.transforms import ResizeLongestSide
import torch.nn.functional as F

from solver.loss import FocalLoss, DiceLoss, IoULoss


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
        x_min = max(0, x_min - np.random.randint(0, 100))
        x_max = min(W, x_max + np.random.randint(0, 100))
        y_min = max(0, y_min - np.random.randint(0, 100))
        y_max = min(H, y_max + np.random.randint(0, 100))
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





def train(cfg):
    # Get the model and download the checkpoint if needed
    model = prepare_sam(checkpoint=cfg.MODEL.CHECKPOINT)
    device = cfg.MODEL.DEVICE
    epochs = cfg.SOLVER.MAX_EPOCHS

    #Set the portion of the model to be trained (We will train only the mask_decoder part)
    for name, param in model.named_parameters():
        if name.startswith('image_encoder') or name.startswith('mask_decoder'):
            param.requires_grad = False
    
    # Get the dataloader and prepare the train, validation and test dataloader
    sam_dataloader = SAM_Dataloader(cfg)
    train_loader, val_loader, _ = sam_dataloader.build_dataloader()

    # Get the optimizer and scheduler
    optimizer = build_optimizer(cfg, model)
    scheduler = build_lrSchedular(cfg=None, optimizer=optimizer)

    Focal_Loss = FocalLoss()
    Dice_Loss = DiceLoss()
    Iou_Loss = IoULoss()

    #Put the model in the train mode
    model.to(device)    # Put the model on GPU
    model.train()       # Set the model to training mode

    epochs_loss = []
    for epoch in range(cfg.SOLVER.MAX_EPOCHS):
        # Iterate over the batches and acculmulate the loss for each batch and update the model
        for batch in tqdm(train_loader, total=len(train_loader)):
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
                bbox_prompts, point_prompts, input_labels_prompts = _get_bbox_point_and_inputlabel_prompts(pixel_masks)
                #scale the bbox prompts and point prompts according to the scale factor
                bbox_prompts = np.around(np.array(bbox_prompts) * scale_factor)
                point_prompts = np.around(np.array(point_prompts) * scale_factor)

                bbox_prompts = torch.as_tensor(bbox_prompts).to(device)
            

                # Obtain the image embeddings from the image encoder, image encoder here is run with inference mode (Strict version of no grad)
                # This will be done only once
                with torch.no_grad():
                    image_embeddings = model.image_encoder(resize_img)  # (B,256,64,64)
                    #TODO: Store the image embeddings in torch cache for later use
                     

                # Obtain the sparse and dense embeddings from the prompt encoder, prompt encoder here is run with inference mode (Strict version of no grad)
                # if points are provided then providing labels are mandatory
                # This will repeat mutiple times and for all the masks we will calculate the losses
                # with torch.inference_mode():
                sparse_embeddings, dense_embeddings = model.prompt_encoder(
                    points=None,
                    boxes=bbox_prompts,
                    masks=None,
                )
                
                with torch.no_grad():    #Just keeping it here to reduce memory consumption in the GPU
                    low_res_masks, iou_predictions = model.mask_decoder(
                        image_embeddings=image_embeddings.to(device),  # (B, 256, 64, 64)
                        image_pe=model.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
                        sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
                        dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
                        multimask_output=False,
                    )

                # Postprocess and retrieve the predicted mask as a binary mask
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

                # TODO: Calculate the loss between each instance of the mask and the predicted mask and accumulate the loss
                high_res_masks = torch.as_tensor(high_res_masks).to(device)
                pixel_masks = torch.as_tensor(pixel_masks).to(device)
                
                instances = 0
                focal_loss = 0
                dice_loss = 0
                iou_loss = 0
                for i in range(len(high_res_masks)):
                    focal_loss += Focal_Loss(high_res_masks[i], pixel_masks[i])
                    dice_loss += Dice_Loss(high_res_masks[i], pixel_masks[i])
                    iou_loss += Iou_Loss(high_res_masks[i], pixel_masks[i], iou_predictions[i][0].to(device))
                    instances += 1
                
                focal_loss = (focal_loss/instances) 
                dice_loss = (dice_loss/instances)   
                iou_loss = (iou_loss/instances)
                
                # delete all the local variables and cuda cache after each image is processed, only leave the global loss variable that can be deleted after one batch is processed
                # del image, pixel_masks, scale_factor, original_image_size, sam_transform, resize_img, bbox_prompts, point_prompts, input_labels_prompts, image_embeddings, sparse_embeddings, dense_embeddings, low_res_masks, iou_predictions, high_res_masks
                
                #torch.cuda.empty_cache()

                

                # TODO: Update the model with the loss
                optimizer.zero_grad()
                loss = (cfg.LOSS.FOCAL_LOSS_WEIGHT * focal_loss) + (cfg.LOSS.DICE_LOSS_WEIGHT * dice_loss)
                loss.backward()
                optimizer.step()
            

































def main():
    # parser = argparse.ArgumentParser(description="PyTorch Template MNIST Training")
    # parser.add_argument(
    #     "--config_file", default="", help="path to config file", type=str
    # )
    # parser.add_argument("opts", help="Modify config options using the command-line", default=None,
    #                     nargs=argparse.REMAINDER)

    # args = parser.parse_args()

    # num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

    # if args.config_file != "":
    #     cfg.merge_from_file(args.config_file)
    # cfg.merge_from_list(args.opts)
    cfg.freeze()

    # output_dir = cfg.OUTPUT_DIR
    # if output_dir and not os.path.exists(output_dir):
    #     mkdir(output_dir)

    # logger = setup_logger("template_model", output_dir, 0)
    # logger.info("Using {} GPUS".format(num_gpus))
    # logger.info(args)

    # if args.config_file != "":
    #     logger.info("Loaded configuration file {}".format(args.config_file))
    #     with open(args.config_file, 'r') as cf:
    #         config_str = "\n" + cf.read()
    #         logger.info(config_str)
    # logger.info("Running with config:\n{}".format(cfg))

    print(cfg)


    train(cfg)


if __name__ == '__main__':
    main()
