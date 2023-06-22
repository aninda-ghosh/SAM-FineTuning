# encoding: utf-8
"""
@author:  Aninda Ghosh
@contact: aghosh57@asu.edu
"""

from torch.utils import data
from .datasets.planetscope import ParcelDataset
from torch.utils.data import DataLoader

class SAM_Dataloader(DataLoader):
    def __init__(self, cfg):
        """
        SAM Dataloader is a custom dataloader which prepares the inputs to feed to SAM for finetuning.

        DataFormat for each items from the dataset
        image,                  bbox_prompts,   point_prompts,  labels
        image as numpy array,   numpy array,    numpy array,    numpy array
        """

        self.cfg = cfg
        # Dataset constructor
        self.train_valid_dataset = ParcelDataset(path=cfg.DATASETS.ROOT_DIR, train_data=True)
        self.test_dataset = ParcelDataset(path=cfg.DATASETS.ROOT_DIR, train_data=False) 
        # Get the dataset length
        self.train_valid_dataset_length = len(self.train_valid_dataset)
        self.test_dataset_length = len(self.test_dataset)

    def build_dataloader(self):
        """
        This function builds the data loaders for training, validation and testing.
        """

        # Split the data into train, validation and test sets
        train_size = int(self.cfg.DATALOADER.TRAIN_DATA * len(self.train_valid_dataset))   
        valid_size = len(self.train_valid_dataset) - train_size
        train_dataset, valid_dataset = data.random_split(self.Dataset, [train_size, valid_size])

        # Create the data loaders
        train_loader = data.DataLoader(
            train_dataset, batch_size=self.cfg.SOLVER.ITEMS_PER_BATCH, shuffle=True, num_workers=self.cfg.DATALOADER.NUM_WORKERS, collate_fn=lambda x: x
        )

        valid_loader = data.DataLoader(
            valid_dataset, batch_size=self.cfg.VALID.ITEMS_PER_BATCH, shuffle=False, num_workers=self.cfg.DATALOADER.NUM_WORKERS, collate_fn=lambda x: x
        )

        test_loader = data.DataLoader(
            self.test_dataset, batch_size=self.cfg.TEST.ITEMS_PER_BATCH, shuffle=False, num_workers=self.cfg.DATALOADER.NUM_WORKERS, collate_fn=lambda x: x
        )
        
        return train_loader, valid_loader, test_loader
