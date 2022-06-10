
## Standard libraries
from typing import Any, List, Optional, Tuple
import argparse

## PyTorch
import torch
import os

# Pytorch lightning
import pytorch_lightning as pl
# Callbacks 
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping, RichModelSummary,RichProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins import DDPPlugin

# load network
from utils.vit import VisionTransformer
# load datamodule
from utils.cifar10datamodule import CIFAR10DataModule
# load plmodule
from utils.cifar10plmodule import CIFARModule
#
from utils.helper import dir_path

# # Set the visible GPUs, in case of multi-GPU device, otherwise comment it
# # you can use `nvidia-smi` in terminal to see the available GPUS
os.environ["CUDA_VISIBLE_DEVICES"]="11,12"

######################################################################
# Set the Global values
######################################################################
# Transform argument
IMAGE_SIZE = (32,32)  # H X W
SCALE_BOUNDS = (0.8,1.0) # lower & upper bounds
ASPECT_BOUNDS = (0.9,1.1) # lower & upper bounds
                                     

# Set the device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

## classes
NAME_CLASSES = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
NUM_CLASSES = len(NAME_CLASSES)
DATA_MEAN = [0.49421428, 0.48513139, 0.45040909]
DATA_STD = [0.24665252, 0.24289226, 0.26159238]


######################################################################
# PL Trainer 
######################################################################
def train_model(dm=None, checkpoint_path=None, save_name=None, 
                gpus=1, strategy=None, sync_batchnorm=False,
                max_epochs=150, device='cpu', logger=None,
                **kwargs):
    
    if save_name is None:
        save_name = 'model'
    
    callbacks = [ModelCheckpoint(save_weights_only=True, mode="max", monitor="val/acc"), 
                LearningRateMonitor("epoch"),
                RichModelSummary(max_depth=-1),
                RichProgressBar()
                ]

    # Create a PyTorch Lightning trainer with the generation callback
    trainer = pl.Trainer(default_root_dir=os.path.join(checkpoint_path, save_name),
                        strategy=strategy,
                        accelerator="gpu" if str(device)=="cuda" else "",
                        gpus=gpus if str(device)=="cuda" else 0,
                        sync_batchnorm=sync_batchnorm,
                        max_epochs=max_epochs,
                        callbacks=callbacks,
                        logger=logger,
                        progress_bar_refresh_rate=1)
    
    trainer.logger._log_graph = True         # If True, we plot the computation graph in tensorboard
    trainer.logger._default_hp_metric = None # Optional logging argument that we don't need
    
    pl.seed_everything(42) # To be reproducable
    model = CIFARModule(**kwargs)
    trainer.fit(model, datamodule=dm)
        
    # # Test best model on validation
    trainer.test(model, datamodule=dm)
    
    return


######################################################################
# Main function
######################################################################
def main():
    ######################################################################
    # input cmdline arguments
    ######################################################################
    parser=argparse.ArgumentParser()
    # Required parameters
    parser.add_argument('--run_name',required=True,type=str)
    parser.add_argument('--save_name',default='test',required=True,type=str)
    parser.add_argument('--random_seed',default=42,type=int)             
    parser.add_argument('-ep','--epochs',default=5,type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument('-bs','--batch_size',default=128,type=int)
    parser.add_argument('-nw','--num_workers',default=4,type=int)
    parser.add_argument('-ng','--gpus',default=1,type=int)
    
    # optimizer parameters
    parser.add_argument('-ls', "--learning_rate", default=0.1, type=float,
                        help="The initial learning rate for SGD.")
    parser.add_argument('-ms', "--milestones", nargs="+", default=[50, 100], type=int,
                        help='2 milestone values for MultistepLR, eg --milestones 50 100')
    parser.add_argument("--gamma", default=0.1, type=float,
                        help="gamma value for MultiStepLR.")
    
    # paths
    parser.add_argument("--dataset_path", default='../data', type=dir_path,
                        help="Path to dataset folder")
    parser.add_argument("--log_path", default='./logs', type=dir_path,
                        help="Path to save logs")
    parser.add_argument("--checkpoint_path", default='../saved_models', type=dir_path,
                        help="Path to save model")
    
    args=parser.parse_args()
    
    # print input arguments
    print(f'Random seed: {args.random_seed}')
    print(f'Run name : {args.run_name}')
    print(f'Save name: {args.save_name}')
    
    print(f'Num of training epochs: {args.epochs}')
    print(f'Batch size: {args.batch_size}')
    print(f'Num workers: {args.num_workers}')
    print(f'Num of GPUs: {args.gpus}')
    
    print(f'Learning rate: {args.learning_rate}')
    print(f'milestones: {args.milestones}')
    print(f'gamma: {args.gamma}')

    print(f'Dataset_path: {args.dataset_path}')
    print(f'Log_path: {args.log_path}')
    print(f'Checkpoint_path: {args.log_path}')

    # Start the logger
    logger = TensorBoardLogger(save_dir=args.log_path, name=args.run_name)
    # set the random seed
    pl.seed_everything(args.random_seed)

    ######################################################################
    # Train and val datasets
    ######################################################################
    print("\nUsing device: ", device)
    dm = CIFAR10DataModule(data_dir=args.dataset_path,
                            batch_size=args.batch_size,
                            num_workers=args.num_workers,
                            num_classes=NUM_CLASSES,
                            data_mean=DATA_MEAN,
                            data_std=DATA_STD,
                            image_size=IMAGE_SIZE,
                            scale_bounds=SCALE_BOUNDS,
                            aspect_bounds=ASPECT_BOUNDS)

    
    ######################################################################
    # Training
    ######################################################################
    train_model(dm=dm,
                logger=logger, checkpoint_path= args.checkpoint_path, save_name=args.save_name,
                gpus=args.gpus, strategy='ddp', sync_batchnorm=True,
                max_epochs=args.epochs, device=device,
                name_classes=NAME_CLASSES,
                data_mean=DATA_MEAN,
                data_std=DATA_STD,
                model_kwargs={'num_classes': NUM_CLASSES,
                                'num_heads': 8,
                                'num_layers': 6,
                                'num_channels': 3,
                                'num_patches': 64,
                                'patch_size': 4,
                                'embed_dim': 256,
                                'hidden_dim': 512,
                                'dropout': 0.2,
                                },
                optimizer_hparams={"lr": args.learning_rate},
                scheduler_hparams={"milestones": args.milestones,
                                    "gamma": args.gamma
                                    })

if __name__=='__main__':
    main()