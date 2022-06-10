import pytorch_lightning as pl
import torchvision
from torchvision.datasets import CIFAR10
from torchvision import transforms
from typing import Any, List, Optional, Tuple
from torch.utils.data import DataLoader

######################################################################
# Pytorch lightning Dataclass
######################################################################
class CIFAR10DataModule(pl.LightningDataModule):
    """
    A DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    def __init__(
        self,
        data_dir: str = "data/",
        batch_size: int = 128,
        num_workers: int = 4,
        pin_memory: bool = True,
        num_classes: int = 10,
        data_mean: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        data_std: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        image_size: Tuple[int, int] = (32, 32), 
        scale_bounds: Tuple[float, float] = (0.8, 1.0),
        aspect_bounds: Tuple[float, float] = (0.9, 1.1)
        ):

        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)

        # data transformations
        self.train_transforms = transforms.Compose([
                                        transforms.RandomHorizontalFlip(),
                                        transforms.RandomResizedCrop(image_size,scale=scale_bounds,ratio=aspect_bounds),
                                        transforms.ToTensor(),
                                        transforms.Normalize(data_mean, data_std)
                                     ])

        self.test_transforms = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize(data_mean, data_std)
                                     ])
    
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        self.tmp_num_classes = num_classes

    @property
    def num_classes(self) -> int:
        return self.tmp_num_classes

    def prepare_data(self):
        """Download data if needed.

        This method is called only from a single GPU.
        Do not use it to assign state (self.x = y).
        """
        CIFAR10(self.hparams.data_dir, train=True, download=True)
        CIFAR10(self.hparams.data_dir, train=False, download=True)

   
    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning when doing `trainer.fit()` and `trainer.test()`,
        so be careful not to execute the random split twice! The `stage` can be used to
        differentiate whether it's called before trainer.fit()` or `trainer.test()`.
        """

        # load datasets only if they're not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            trainset = CIFAR10(self.hparams.data_dir,
                            train=True, 
                            transform=self.train_transforms)
            valset = CIFAR10(self.hparams.data_dir,
                            train=False, 
                            transform=self.test_transforms)
            testset = CIFAR10(self.hparams.data_dir,
                            train=False, 
                            transform=self.test_transforms)
                            
            self.data_train = trainset
            self.data_val = valset
            self.data_test = testset

        
    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=2*self.hparams.num_workers,
            shuffle=False,
            drop_last=False
        )


    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=2*self.hparams.num_workers,
            shuffle=False,
            drop_last=False
        )