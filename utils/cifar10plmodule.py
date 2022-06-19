## PyTorch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

# Torchvision
from torchvision.datasets import CIFAR10
from torchvision import transforms

# Pytorch lightning
import pytorch_lightning as pl

# Metrics
from torchmetrics import MaxMetric
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics.functional import confusion_matrix

## Standard libraries
from typing import Any, List, Optional, Tuple
import numpy as np 

# load helper functions 
from utils.plotter import plot_cm, plot_preds
from utils.vit import VisionTransformer
import matplotlib.pyplot as plt
import wandb

######################################################################
# Pytorch lightning model class
######################################################################
class CIFARModule(pl.LightningModule):
    def __init__(self,
                name_classes,
                data_mean,
                data_std,
                model_kwargs, 
                optimizer_hparams, 
                scheduler_hparams):
        """
        Inputs:
            name_classes - CIFAR10 classes name
            model_kwargs - model parameters
            optimizer_hparams - Hyperparameters for the optimizer, as dictionary. This includes learning rate, weight decay, etc.
            scheduler_hparams - Hyperparameters for the scheduler, as dictionary. This includes learning rate, weight decay, etc.
        """

        super().__init__()
        # Exports the hyperparameters to a YAML file, and create "self.hparams" namespace
        self.save_hyperparameters()
        # Create model
        self.model = VisionTransformer(**model_kwargs)
        # Create loss module
        self.criterion = nn.CrossEntropyLoss()
        # Example input for visualizing the graph in Tensorboard
        self.example_input_array = torch.zeros((1, 3, 32, 32), dtype=torch.float32)
        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()

        # for logging best so far validation accuracy
        self.val_acc_best = MaxMetric()
        
        # for plot
        self.name_classes = name_classes
        self.data_mean = data_mean
        self.data_std = data_std
        self.num_classes = len(name_classes)

    def forward(self, imgs):
        # Forward function that is run when visualizing the graph
        return self.model(imgs)

    def configure_optimizers(self):
        # We will use SGD optimizer
        optimizer = optim.AdamW(self.parameters(), **self.hparams.optimizer_hparams)
        
        # We will reduce the learning rate by 0.1 after 100 and 150 epochs
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, **self.hparams.scheduler_hparams)
        return [optimizer], [scheduler]


    def step(self, batch: Any):
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    def training_step(self, batch, batch_idx):
        # "batch" is the output of the training data loader.
        loss, preds, targets = self.step(batch)

        # Logs the accuracy per epoch to tensorboard (weighted average over batches)
        acc = self.train_acc(preds, targets)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()`` below
        # remember to always return loss from `training_step()` or else backpropagation will fail!
        return {"loss": loss, "preds": preds, "targets": targets}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        # plot the confusion matrix at the end of each epoch
        preds = torch.cat([tmp['preds'] for tmp in outputs])
        targets = torch.cat([tmp['targets'] for tmp in outputs])
        
        # plot confusion matrix
        cm = confusion_matrix(targets, preds, self.num_classes)
        fig_ = plot_cm(cm, self.name_classes)
        plt.close(fig_)
        self.logger.experiment.log({"confusion_matrix_train": wandb.Image(fig_)})
        
    def validation_step(self, batch, batch_idx):
        # "batch" is the output of the training data loader.
        loss, preds, targets = self.step(batch)

        # plot figures
        if batch_idx == 0:
            images, _ = batch
            fig_ = plot_preds(images.cpu().numpy(), 
                            targets.cpu().numpy(), 
                            preds.cpu().numpy(), 
                            self.name_classes,
                            nimg=32,
                            ncols=8,
                            data_mean=self.data_mean,
                            data_std=self.data_std)
            self.logger.experiment.log({"example_val_batches": wandb.Image(fig_)})
        
        # Logs the accuracy per epoch to tensorboard (weighted average over batches)
        acc = self.val_acc(preds, targets)
        # By default logs it per epoch (weighted average over batches)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}


    def validation_epoch_end(self, outputs: List[Any]):
        acc = self.val_acc.compute()  # get val accuracy from current epoch
        self.val_acc_best.update(acc)
        self.log("val/acc_best", self.val_acc_best.compute(), on_epoch=True, prog_bar=True)

        # plot the confusion matrix at the end of each epoch
        preds = torch.cat([tmp['preds'] for tmp in outputs])
        targets = torch.cat([tmp['targets'] for tmp in outputs])
        cm = confusion_matrix(targets, preds, self.num_classes)
        fig_ = plot_cm(cm, self.name_classes)
        plt.close(fig_)
        self.logger.experiment.log({"confusion_matrix_val": wandb.Image(fig_)})
        

    def test_step(self, batch, batch_idx):
        loss, preds, targets = self.step(batch)

         # plot figures
        if batch_idx == 0:
            images, _ = batch
            fig_ = plot_preds(images.cpu().numpy(), 
                            targets.cpu().numpy(), 
                            preds.cpu().numpy(), 
                            self.name_classes,
                            nimg=32,
                            ncols=8,
                            data_mean=self.data_mean,
                            data_std=self.data_std)
            self.logger.experiment.log({"example_test_batches": wandb.Image(fig_)})

        # log test metrics
        acc = self.test_acc(preds, targets)
        # By default logs it per epoch (weighted average over batches), and returns it afterwards
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/acc", acc, on_step=False, on_epoch=True)

        return {"loss": loss, "preds": preds, "targets": targets}
        
    def test_epoch_end(self, outputs: List[Any]):
        # plot the confusion matrix at the end of each epoch
        preds = torch.cat([tmp['preds'] for tmp in outputs])
        targets = torch.cat([tmp['targets'] for tmp in outputs])
        cm = confusion_matrix(targets, preds, self.num_classes)
        fig_ = plot_cm(cm, self.name_classes)
        plt.close(fig_)
        
        self.logger.experiment.log({"confusion_matrix_test": wandb.Image(fig_)})

    def on_epoch_end(self):
        # reset metrics at the end of every epoch
        self.train_acc.reset()
        self.test_acc.reset()
        self.val_acc.reset()