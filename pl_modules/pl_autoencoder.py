## PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
# PyTorch Lightning
import pytorch_lightning as pl
import torch.nn.functional as F


from modules.autoencoder import Autoencoder, Encoder, Decoder

class LitAutoencoder(pl.LightningModule):
    
    def __init__(self, 
                 base_channel_size: int, 
                 latent_dim: int, 
                 seed : int =42,
                 linear_hidden_dims : list = [],
                 encoder_class : object = Encoder,
                 decoder_class : object = Decoder,
                 num_input_channels: int = 3, 
                 width: int = 32, 
                 height: int = 32,
                 epochs=300):
        super().__init__()
        
        # Saving hyperparameters of autoencoder
        self.save_hyperparameters() 
        
        pl.seed_everything(seed)

        # Creating encoder and decoder
        self.net = Autoencoder(base_channel_size, 
                               latent_dim,
                               linear_hidden_dims,
                               encoder_class,
                               decoder_class,
                               num_input_channels)
        
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
    
    def _get_reconstruction_loss(self, batch):
        """
        Given a batch of images, this function returns the reconstruction loss (MSE in our case)
        """
        x, _ = batch # We do not need the labels
        x_hat = self.forward(x)
        loss = F.mse_loss(x, x_hat, reduction="none")
        loss = loss.sum(dim=[1,2,3]).mean(dim=[0])
        return loss
    
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
        loss = self._get_reconstruction_loss(batch)                             
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log('val_loss', loss, prog_bar=True)
    
    def test_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log('test_loss', loss)