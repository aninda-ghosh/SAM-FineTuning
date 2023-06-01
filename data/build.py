# encoding: utf-8
"""
@author:  Aninda Ghosh
@contact: aghosh57@asu.edu
"""

from torch.utils import data
import numpy as np
from .datasets.planetscope import ParcelDataset


class SAM_Dataloader:
    def __init__(self, cfg):
        self.cfg = cfg
        # This dictionary stores the data loaders for training, validation and testing
        self.dataloaders = {
            "train": None,
            "valid": None,
            "test": None
        }

        #ParcelDataset: This class is used to load the dataset from the disk. Using Image paths and pixel masks.
        """
        Data format: 
        Numpy Array (images)
        Numpy Array (pixel_masks)
        """
        # TODO: Change the path for the dataset
        self.dataset = ParcelDataset(path=cfg.DATASETS.ROOT_DIR) 

        # Get the dataset length
        self.dataset_length = len(self.dataset)
        # Find the nearest multiple of 2 for the dataset length (This will help in data parallelization)
        self.dataset_length = self.dataset_length + (2 - (self.dataset_length % 2))

        # Trim the dataset based on the dataset length
        self.dataset = self.dataset[:self.dataset_length]


    def build_dataloader(self):
        """
        This function builds the data loaders for training, validation and testing.
        """

        # Split the data into train, validation and test sets
        train_size = int(self.cfg.DATALOADER.TRAIN_DATA * len(self.dataset))   
        test_size = len(self.dataset) - train_size
        train_dataset, test_dataset = data.random_split(self.dataset, [train_size, test_size])

        train_size = int(self.cfg.DATALOADER.TRAIN_DATA * len(train_dataset))
        valid_size = len(train_dataset) - train_size
        train_dataset, valid_dataset = data.random_split(train_dataset, [train_size, valid_size])

        # Create the data loaders
        train_loader = data.DataLoader(
            train_dataset, batch_size=self.cfg.SOLVER.ITEMS_PER_BATCH, shuffle=True, num_workers=self.cfg.DATALOADER.NUM_WORKERS
        )

        valid_loader = data.DataLoader(
            valid_dataset, batch_size=self.cfg.VALID.ITEMS_PER_BATCH, shuffle=True, num_workers=self.cfg.DATALOADER.NUM_WORKERS
        )

        test_loader = data.DataLoader(
            test_dataset, batch_size=self.cfg.TEST.ITEMS_PER_BATCH, shuffle=True, num_workers=self.cfg.DATALOADER.NUM_WORKERS
        )

        # Store the data loaders in the dictionary
        self.dataloaders["train"] = train_loader
        self.dataloaders["valid"] = valid_loader
        self.dataloaders["test"] = test_loader

        # Return the data loaders for training, validation and testing
        return self.dataloaders
