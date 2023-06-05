from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,
# or _TEST for a test-specific parameter.
# For example, the number of images during training will be
# IMAGES_PER_BATCH_TRAIN, while the number of images for testing will be
# IMAGES_PER_BATCH_TEST

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

_C.MODEL = CN()
_C.MODEL.DEVICE = "cuda"
_C.MODEL.CHECKPOINT = "/home/legion-ubuntu/Research/Kerner-Lab/SAM-FineTuning/modeling/model_checkpoints/sam_vit_b_01ec64.pth"

# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
# Root directory of dataset
_C.DATASETS.ROOT_DIR = "/home/legion-ubuntu/Research/Kerner-Lab/datasets/images2/"

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 0   # Add any positive integer for multiple worker nodes
# Batch size of the dataset
_C.DATALOADER.BATCH_SIZE = 4
# Number of images per batch during training, validation and testing
_C.DATALOADER.TRAIN_DATA = 0.8  # 80% of the batched data for training
_C.DATALOADER.VALID_DATA = 0.2  # 20% of the batched data for validation
_C.DATALOADER.TEST_DATA = 0.2  # 20% of the batched data for testing



# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #

# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.SOLVER = CN()
_C.SOLVER.MAX_EPOCHS = 5
_C.SOLVER.ITEMS_PER_BATCH = 4
_C.SOLVER.BASE_LR = 0.01
_C.SOLVER.WEIGHT_DECAY = 0.001

# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.VALID = CN()
_C.VALID.ITEMS_PER_BATCH = 2

# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.TEST = CN()
_C.TEST.ITEMS_PER_BATCH = 2

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.OUTPUT_DIR = ""
