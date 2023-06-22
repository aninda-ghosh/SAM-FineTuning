# encoding: utf-8
"""
@author:  Aninda Ghosh
@contact: aghosh57@asu.edu
"""

from modeling.segment_anything.utils.transforms import ResizeLongestSide
from utils.data_support import get_bbox_point_and_inputlabel_prompts

import torch
import torch.nn.functional as F
from torch.nn.functional import threshold, normalize

from tqdm import tqdm
import numpy as np
import gc


# Define the epoch step function
def epoch_step(mode, cfg, logger, model, device, data_loader, optimizer, focal_loss, dice_loss, iou_loss):
    _epoch_loss = 0

    if mode == 'TRAIN':
        model.train()       
    else:
        model.eval()

    # Iterate over the batches and acculmulate the loss for each batch and update the model
    for batch in tqdm(data_loader, total=len(data_loader)):
        _instances = 0
        _focal_loss = 0
        _dice_loss = 0
        _iou_loss = 0
        
        if mode == 'TRAIN':
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
            

            if mode == 'TRAIN':
                # Run the mask decoder with the image embeddings, sparse embeddings and dense embeddings
                low_res_masks, iou_predictions = model.mask_decoder(
                    image_embeddings=image_embeddings.to(device),  # (B, 256, 64, 64)
                    image_pe=model.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
                    sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
                    dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
                    multimask_output=False,
                )
            else:
                with torch.no_grad():
                    # Run the mask decoder with the image embeddings, sparse embeddings and dense embeddings
                    low_res_masks, iou_predictions = model.mask_decoder(
                        image_embeddings=image_embeddings.to(device),  # (B, 256, 64, 64)
                        image_pe=model.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
                        sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
                        dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
                        multimask_output=False,
                    )                    
            
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
                try:
                    _focal_loss += focal_loss(high_res_masks[i], pixel_masks_tensor[i])
                    _dice_loss += dice_loss(high_res_masks[i], pixel_masks_tensor[i])
                    _iou_loss += iou_loss(high_res_masks[i], pixel_masks_tensor[i], iou_predictions[i][0].to(device))
                    _instances += 1
                except:
                    # Catch the error and continue
                    logger.debug(f'Error in calculating the loss for mask {i}')
                    continue
            
        
            # delete all the local variables and cuda cache after each image is processed, only leave the global loss variable that can be deleted after one batch is processed
            del image, pixel_masks, pixel_masks_tensor, scale_factor, original_image_size, sam_transform, resize_img, bbox_prompts, point_prompts, input_labels_prompts, image_embeddings, sparse_embeddings, dense_embeddings, low_res_masks, iou_predictions, high_res_masks

        logger.debug(f'Batch Loss: Instances: {_instances}, Avg. Focal Loss: {_focal_loss/_instances}, Avg. Dice Loss: {_dice_loss/_instances}, Avg. IoU Loss: {_iou_loss/_instances}')    
        
        # Update the model after each batch            
        _focal_loss = (_focal_loss/_instances) 
        _dice_loss = (_dice_loss/_instances)   
        _iou_loss = (_iou_loss/_instances)

        # Update the model with the loss
        loss = (cfg.LOSS.FOCAL_LOSS_WEIGHT * _focal_loss) + (cfg.LOSS.DICE_LOSS_WEIGHT * _dice_loss)
        
        if mode == 'TRAIN':
            loss.backward()
            optimizer.step()

        torch.cuda.empty_cache()
        gc.collect()
    
        _epoch_loss += loss.item()

    return _epoch_loss





def do_train(
        cfg,
        logger,
        model,
        device,
        train_dataloader,
        valid_dataloader,
        test_dataloader,
        optimizer,
        scheduler,
        focal_loss,
        dice_loss,
        iou_loss,
        epochs,
        output_dir
):
    training_losses = []
    validation_losses = []
    for epoch in range(1, epochs+1):
        logger.info(f'\nEpoch {epoch}:')

        logger.info('Train Step')
        # Train the model
        epoch_train_loss = epoch_step(
                                        mode = "TRAIN", 
                                        cfg = cfg, 
                                        logger = logger, 
                                        model = model, 
                                        device= device, 
                                        data_loader = train_dataloader, 
                                        optimizer = optimizer, 
                                        focal_loss = focal_loss, 
                                        dice_loss = dice_loss, 
                                        iou_loss = iou_loss
                                    )
        
        epoch_train_loss /= (epoch)
        training_losses.append(epoch_train_loss)

        logger.info('Validation Step')
        # Validate the model
        epoch_valid_loss = epoch_step(
                                        mode = "VALIDATE", 
                                        cfg = cfg, 
                                        logger = logger, 
                                        model = model, 
                                        device = device, 
                                        data_loader = valid_dataloader, 
                                        optimizer = None, 
                                        focal_loss = focal_loss, 
                                        dice_loss = dice_loss, 
                                        iou_loss = iou_loss
                                    )
        epoch_valid_loss /= (epoch)
        validation_losses.append(epoch_valid_loss)

        scheduler.step(metrics=epoch_valid_loss)

        logger.info(f'Train Loss: {epoch_train_loss}, Validation Loss: {epoch_valid_loss}')

        if epoch % 2 == 0:
            logger.info(f'\nSaving the model at epoch {epoch}\n')
            torch.save(model.state_dict(), f"{output_dir}/model_checkpoints/sam_checkpoint_{epoch}.pth")

    # Test the model
    logger.info('Test Step')
    test_loss = epoch_step(
                            mode = "TEST", 
                            cfg = cfg, 
                            logger = logger, 
                            model = model, 
                            device = device, 
                            data_loader = test_dataloader, 
                            optimizer = None, 
                            focal_loss = focal_loss, 
                            dice_loss = dice_loss, 
                            iou_loss = iou_loss
                        )
    logger.info(f'\nTest Loss: {test_loss}')

    # Store the training and validation losses as numpy arrays in the output directory
    np.save(f"{output_dir}/training_losses.npy", np.array(training_losses))
    np.save(f"{output_dir}/validation_losses.npy", np.array(validation_losses))
    
    # Save the model for the final time in the output directory
    logger.info(f'\nSaving the final model\n')
    torch.save(model.state_dict(), f"{output_dir}/model_checkpoints/sam_checkpoint_final.pth")
