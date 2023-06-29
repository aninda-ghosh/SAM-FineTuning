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

# -----------------------------------------------------------------------------
# Logger
_C.LOGGER = CN()
_C.LOGGER.LEVEL = "DEBUG"

# -----------------------------------------------------------------------------
# MODEL
_C.MODEL = CN()
_C.MODEL.DEVICE = "cuda"

# For Large Model
# _C.MODEL.CHECKPOINT = "./modeling/model_checkpoints/sam_vit_l_0b3195.pth"
# _C.MODEL.TYPE = "large"

# For Base Model
# _C.MODEL.CHECKPOINT = "./modeling/model_checkpoints/sam_vit_b_01ec64.pth"
_C.MODEL.CHECKPOINT = "/home/aghosh57/Kerner-Lab/SAM-FineTuning/logs/Jun28_21-40-50/model_checkpoints/sam_checkpoint_final.pth"
_C.MODEL.TYPE = "base"

# Allows epochs to pass before saving the model
_C.MODEL.SAVE_INTERVAL = 1

# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
# Root directory of dataset
_C.DATASETS.ROOT_DIR = "/home/aghosh57/Kerner-Lab/all_dataset/"

# This is used to generate the bboxes for non labeled images
_C.BBOX = CN()
_C.BBOX.NUMBER = 30
_C.BBOX.MIN_DISTANCE = 50
_C.BBOX.SIZE_REF = 0.25

# This is used to control the per image masks instances to play with
_C.BBOX.BOX_LIMITER = 50

# This is used to remove the small masks from the dataset
_C.MASKS = CN()
_C.MASKS.MIN_AREA = 1000

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 8   # Add any positive integer for multiple worker nodes
# Number of images per batch during training, validation and testing
_C.DATALOADER.TRAIN_DATA = 0.8  # 80% of the batched data for training
_C.DATALOADER.VALID_DATA = 0.2  # 20% of the batched data for validation


# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #

# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.SOLVER = CN()
_C.SOLVER.MAX_EPOCHS = 1
_C.SOLVER.ITEMS_PER_BATCH = 16
_C.SOLVER.START_LR = 0.01
_C.SOLVER.MIN_LR = 0.000001
_C.SOLVER.WEIGHT_DECAY = 0.0001
_C.SOLVER.MOMENTUM = 0.9

# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.VALID = CN()
_C.VALID.ITEMS_PER_BATCH = 4

# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.TEST = CN()
_C.TEST.ITEMS_PER_BATCH = 1

# Loss function parameters
_C.LOSS = CN()
_C.LOSS.FOCAL_LOSS_WEIGHT = 1
_C.LOSS.DICE_LOSS_WEIGHT = 1

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.OUTPUT_DIR = "./logs/"
