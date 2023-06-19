import sys
sys.path.append('.')

from config import cfg
from modeling.segment_anything import prepare_sam

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import RichProgressBar
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger
from engine.sam_trainer import SAMTrainer
from data.build import SAMDataLoader

def main():
    cfg.freeze()

    # Get the model and download the checkpoint if needed
    model = prepare_sam(checkpoint=cfg.MODEL.CHECKPOINT, model_type = 'base')
    device = cfg.MODEL.DEVICE
    epochs = cfg.SOLVER.MAX_EPOCHS

    # Set the portion of the model to be trained (We will train only the mask_decoder part)
    for name, param in model.named_parameters():
        if name.startswith('image_encoder') or name.startswith('prompt_encoder'):
            param.requires_grad = False

    sam_datamodule = SAMDataLoader(cfg)
    # wandb_logger = WandbLogger(project='SAM', entity='sam')
    tensorborad_logger = TensorBoardLogger('logs')
    trainer = Trainer(
        callbacks=[RichProgressBar()], 
        # logger=wandb_logger, 
        logger=tensorborad_logger,
        max_epochs=epochs
    )

    modelx = SAMTrainer(model)
    trainer.fit(modelx, sam_datamodule)


if __name__ == '__main__':
    main()