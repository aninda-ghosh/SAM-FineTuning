# encoding: utf-8
"""
@author:  Aninda Ghosh
@contact: aghosh57@asu.edu
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

ALPHA = 0.8
GAMMA = 2


class FocalLoss(nn.Module):

    def __init__(self, weight=None, size_average=True):
        super().__init__()

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, alpha=ALPHA, gamma=GAMMA):
        smooth=1e-6   # Used to prevent division by zero.

        targets = targets.view(-1)
        focal_loss = []
        
        for input in inputs:
            #flatten label and prediction tensors
            input = input.view(-1)
            BCE = F.binary_cross_entropy(input, targets, reduction='mean')
            BCE_EXP = torch.exp(-BCE)
            focal_loss.append(alpha * (1 - BCE_EXP)**gamma * BCE)

        return focal_loss


class DiceLoss(nn.Module):

    def __init__(self, weight=None, size_average=True):
        super().__init__()

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        smooth=1e-6   # Used to prevent division by zero.

        targets = targets.view(-1)
        dice_loss = []
        for input in inputs:
            #flatten label and prediction tensors
            input = input.view(-1)
        
            intersection = (input * targets).sum()
            dice_score = (2. * intersection + smooth) / (input.sum() + targets.sum() + smooth)

            dice_loss.append(1 - dice_score)

        return dice_loss
    

class MSELoss(nn.Module):

    def __init__(self, weight=None, size_average=True):
        super().__init__()

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        targets = targets.view(-1)
        mse_loss = []
        for input in inputs:
            input = input.view(-1)
            MSE_LOSS = nn.MSELoss()
            mse_loss.append(MSE_LOSS(input, targets))
        
        return mse_loss


class IoULoss(nn.Module):

    def __init__(self, weight=None, size_average=True):
        super().__init__()

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, pred_iou: torch.Tensor):
        smooth=1e-6   # Used to prevent division by zero.

        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()
        union = inputs.sum() + targets.sum() - intersection
        iou = (intersection + smooth) / (union + smooth)
        
        loss = iou - pred_iou

        return loss