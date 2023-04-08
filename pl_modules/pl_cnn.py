## PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
# PyTorch Lightning
import pytorch_lightning as pl

from modules.cnn import CNN

class LitCNN(pl.LightningModule):
    
    def __init__(self, 
                 base_channel_size: int, 
                 num_labs: int = 10, 
                 seed : int = 42,
                 linear_hidden_dims : list = [],
                 num_input_channels: int = 3, 
                 width: int = 32, 
                 height: int = 32,
                 epochs=20):
        super().__init__()
        
        # Saving hyperparameters of autoencoder
        self.save_hyperparameters() 
        
        pl.seed_everything(seed)

        # Creating encoder and decoder
        self.net = CNN(num_input_channels, 
                        base_channel_size, 
                        num_labs,
                        linear_hidden_dims)
                
        self.loss_module = nn.CrossEntropyLoss()
        
        self.epochs = epochs
        
        # Example input array needed for visualizing the graph of the network
        self.example_input_array = torch.zeros(2, num_input_channels, width, height)
    
    def forward(self, x):
       
        return self.net(x) 
        
    def detailed_forward(self, x):
        """
        The forward function takes in an image and returns a list of all the activations
        """
        
        return self.net.detailed_forward(x)
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        # Using a scheduler is optional but can be helpful.
        # The scheduler reduces the LR if the validation performance hasn't improved for the last N epochs
        
        """scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer,
                                          base_lr=5e-5,
                                          max_lr=1e-3,
                                          step_size_up=(4*NUM_BATCHES_TRAIN)//2,
                                          cycle_momentum=False,
                                          mode="triangular2")"""
                                          
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                max_lr=1e-3,
                                                total_steps=self.epochs)
        
       
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
    
    def training_step(self, batch, batch_idx):
        # "batch" is the output of the training data loader.
        imgs, labels = batch
        preds = self.net(imgs)
        loss = self.loss_module(preds, labels)
        acc = (preds.argmax(dim=-1) == labels).float().mean()

        # Logs the accuracy per epoch to tensorboard (weighted average over batches)
        self.log("train_acc", acc, prog_bar=True)
        self.log("train_loss", loss, prog_bar=True)
        return loss  # Return tensor to call ".backward" on

    def validation_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self.net(imgs)
        loss = self.loss_module(preds, labels)
        acc = (preds.argmax(dim=-1) == labels).float().mean()

        # Logs the accuracy per epoch to tensorboard (weighted average over batches)
        self.log("val_acc", acc, prog_bar=True)
        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self.net(imgs).argmax(dim=-1)
        acc = (labels == preds).float().mean()
        # By default logs it per epoch (weighted average over batches), and returns it afterwards
        self.log("test_acc", acc)