# Finetuning SAM
In order to use the Foundational Model SAM (Segment Anything Model) from META, we have to identify how to fine-tune the different parts of the model. The Model mainly comprises of Image Encoder (ViT Architecture Based), a Prompt Encoder, and a Lightweight Decoder. Initially, we target to train the decoder with DICE and Focal Loss. Later we can expand this to train other parts of the model as well. But Since the Image Encoder is already trained with 1 Billion Images (From META), we can ignore this for the moment.

# Requirements
- [yacs](https://github.com/rbgirshick/yacs) (Yet Another Configuration System)
- [PyTorch](https://pytorch.org/) (An open-source deep learning platform) 
- [ignite](https://github.com/pytorch/ignite) (High-level library to help with training neural networks in PyTorch) --> Currently being developed for training workflow.

# Table Of Contents
-  [In a Nutshell](#in-a-nutshell)
-  [In Details](#in-details)
-  [Future Work](#future-work)
-  [Scope of Work](#Scope-of-Work)

# In a Nutshell   
- In `modeling`  folder we create a model file to describe the model.   
- In `engine`  folder we create a model trainer function and inference function. In the trainer function, write the logic of the training process, Later we will use PyTorch ignite to simplify the process.

```python
# trainer
def do_train(cfg, model, train_loader, val_loader, optimizer, scheduler, loss_fn):
 """
 implement the logic of epoch:
 -loop on the number of iterations in the config and call the train step
 -add any summaries you want using the summary
 """
pass

# inference
def inference(cfg, model, val_loader):
"""
implement the logic of the train step
- run the tensorflow session
- return any metrics you need to summarize
 """
pass
```

- In `tools`  folder, you create the `train.py` .  In this file, we need to get the instances of the following objects "Model",  "DataLoader”, “Optimizer”, and config
```python
# create instance of the model you want
model = build_model(cfg)

# create your data generator
train_loader = make_data_loader(cfg, is_train=True)
val_loader = make_data_loader(cfg, is_train=False)

# create your model optimizer
optimizer = make_optimizer(cfg, model)
```

- Pass the all these objects to the function `do_train` , and start your training
```python
# here you train your model
do_train(cfg, model, train_loader, val_loader, optimizer, None, F.cross_entropy)
```

# In Details
```
├── config
│   └── defaults.py         - here's the default config file. 
├── data  
│   └── datasets            - here's the datasets folder that is responsible for all data handling.
│   └── transforms          - here's the data preprocess folder that is responsible for all data augmentation.
│   └── build.py  	        - here's the file to make dataloader.
│   └── collate_batch.py    - here's the file that is responsible for merges a list of samples to form a mini-batch.
├── engine
│   ├── trainer.py          - this file contains the train loops.
│   └── inference.py        - this file contains the inference process.
├── modeling                - this folder contains segment anything model definitions.
│   └── segment_anything
├── solver                  - this folder contains optimizer of the project.
│   └── build.py
│   └── lr_scheduler.py
├── tools                   - here's the train/test model of the project.
│   └── train_net.py        - here's an example of train model that is responsible for the whole pipeline.
└── utils
    └── logger.py
```

# Future Work

- Use PyTorch Ignite to simplify end-to-end training workflow.
- Training the Entire Network (3 parts end to end) for larger available datasets.


# Scope of Work

- Dataloader for Parcel Data    - DONE
- Training Engine               - DONE
- Inference Engine              - DONE
- Multi GPU support             - TODO
- Custom Modelling              - Done
- Custom dataset size           - DONE
