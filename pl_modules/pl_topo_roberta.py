## PyTorch
import torch.nn as nn
# PyTorch Lightning
import pytorch_lightning as pl
import torch.optim as optim
from modules.relRoberta import RelRoberta
import transformers
from pl_modules.pl_roberta import roberta_base_AdamW_LLRD


class LitTopoRelRoberta(pl.LightningModule):
    
    def __init__(self, 
                 num_labels,
                 transformer_model,
                 anchor_dataloader,
                 train_load,
                 topo_load,
                 epochs_mix=1,
                 hidden_size=768,
                 similarity_mode="inner",
                 normalization_mode=None,
                 output_normalization_mode=None,
                 dropout_prob=0.1,
                 seed=42,
                 steps=20,
                 weight_decay=0.0,
                 head_lr=1e-3,
                 encoder_lr=3.6e-6,
                 layer_decay=0.65,
                 scheduler_act=True,
                 freq_anchors=100,
                 device="cpu",
                 fine_tune=False):
        super().__init__()
        
        # Saving hyperparameters of autoencoder
        self.save_hyperparameters() 
        
        pl.seed_everything(seed)

        # Creating encoder and decoder
        self.net = RelRoberta(num_labels=num_labels,
                              transformer_model=transformer_model,
                              anchor_dataloader=anchor_dataloader,
                              hidden_size=hidden_size,
                              similarity_mode=similarity_mode,
                              normalization_mode=normalization_mode,
                              output_normalization_mode=output_normalization_mode,
                              dropout_prob=dropout_prob,
                              freq_anchors=freq_anchors,
                              device=device,
                              fine_tune=fine_tune
                             )
                
        self.loss_module = nn.CrossEntropyLoss()
        self.aux_loss = nn.L1Loss()
        
        self.scheduler_act = scheduler_act
        self.steps = steps
        self.layer_decay = layer_decay
        self.weight_decay = weight_decay
        self.head_lr = head_lr
        self.encoder_lr = encoder_lr
        
        self.train_load = train_load
        self.topo_load = topo_load
        self.epochs_mix = epochs_mix
        
        
    def forward(self, x):
       
        return self.net(**x)["prediction"]
        
    def detailed_forward(self, x):
        """
        The forward function takes in an image and returns a list of all the activations
        """
        
        return self.net(x)
    
    def configure_optimizers(self):
        config = {"optimizer": roberta_base_AdamW_LLRD(self.net,
                                            encoder_lr=self.encoder_lr,
                                            head_lr=self.head_lr,
                                            layer_decay=self.layer_decay,
                                            weight_decay=self.weight_decay
                                            )
                  }
        
        if self.scheduler_act:
            config["lr_scheduler"] = {
            "scheduler": transformers.get_cosine_schedule_with_warmup(   
                                                    optimizer = config["optimizer"],
                                                    num_warmup_steps=int(self.steps*0.1),
                                                    num_training_steps=self.steps
                                                    ),
                "interval": "step",

            }
        
        
        return  config
    
    def training_step(self, batch, batch_idx):
        # "batch" is the output of the training data loader.
        tokens, labels = batch
        preds = self.net(batch_idx=batch_idx, **tokens)["prediction"]
        loss = self.loss_module(preds, labels)
        prediction = preds.argmax(dim=-1)
        acc = (prediction == labels).float().mean()
        mae = self.aux_loss(prediction.float(), labels.float())*100

        # Logs the accuracy per epoch to tensorboard (weighted average over batches)
        self.log("train_acc", acc, prog_bar=True)
        self.log("train_loss", loss)
        self.log("train_mae", mae, prog_bar=True)
        return loss  # Return tensor to call ".backward" on

    def validation_step(self, batch, batch_idx):
        tokens, labels = batch
        preds = self.net(batch_idx=batch_idx, **tokens)["prediction"]
        loss = self.loss_module(preds, labels)
        prediction = preds.argmax(dim=-1)
        acc = (prediction == labels).float().mean()
        mae = self.aux_loss(prediction.float(), labels.float())*100


        # Logs the accuracy per epoch to tensorboard (weighted average over batches)
        self.log("val_acc", acc, prog_bar=True)
        self.log("val_mae", mae, prog_bar=True)
        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        tokens, labels = batch
        preds = self.net(batch_idx=batch_idx, **tokens)["prediction"].argmax(dim=-1)
        acc = (labels == preds).float().mean()
        # By default logs it per epoch (weighted average over batches), and returns it afterwards
        self.log("test_acc", acc)

    def train_dataloader(self):
        if self.trainer.current_epoch < self.epochs_mix:
            print("Using mix loader")
            return self.train_load
        else:
            print("Using grouped loader")
            return self.topo_load


