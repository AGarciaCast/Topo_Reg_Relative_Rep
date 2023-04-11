## PyTorch
import torch.nn as nn
# PyTorch Lightning
import pytorch_lightning as pl
from transformers import get_cosine_schedule_with_warmup, AdamW
from modules.relRoberta import RelRoberta


def roberta_base_AdamW_LLRD(model, init_lr=1e-3, head_lr=1e-3, layer_decay=0.65, weight_decay=0.01):
    """
    https://towardsdatascience.com/advanced-techniques-for-fine-tuning-transformers-82e4e61e16e
    """
    
    opt_parameters = []    # To be passed to the optimizer (only parameters of the layers you want to update).
    named_parameters = list(model.named_parameters()) 
        
    # According to AAAMLP book by A. Thakur, we generally do not use any decay 
    # for bias and LayerNorm.weight layers.
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    init_lr = 1e-3 
    head_lr = 1e-3
    lr = init_lr
    
    # === Pooler and regressor ======================================================  
    
    params_0 = [p for n,p in named_parameters if ("pooler" in n or "regressor" in n) 
                and any(nd in n for nd in no_decay)]
    params_1 = [p for n,p in named_parameters if ("pooler" in n or "regressor" in n)
                and not any(nd in n for nd in no_decay)]
    
    head_params = {"params": params_0, "lr": head_lr, "weight_decay": 0.0}    
    opt_parameters.append(head_params)
        
    head_params = {"params": params_1, "lr": head_lr, "weight_decay": weight_decay}    
    opt_parameters.append(head_params)
                
    # === 12 Hidden layers ==========================================================
    
    for layer in range(11,-1,-1):        
        params_0 = [p for n,p in named_parameters if f"encoder.layer.{layer}." in n 
                    and any(nd in n for nd in no_decay)]
        params_1 = [p for n,p in named_parameters if f"encoder.layer.{layer}." in n 
                    and not any(nd in n for nd in no_decay)]
        
        layer_params = {"params": params_0, "lr": lr, "weight_decay": 0.0}
        opt_parameters.append(layer_params)   
                            
        layer_params = {"params": params_1, "lr": lr, "weight_decay": weight_decay}
        opt_parameters.append(layer_params)       
        
        lr *= layer_decay     
        
    # === Embeddings layer ==========================================================
    
    params_0 = [p for n,p in named_parameters if "embeddings" in n 
                and any(nd in n for nd in no_decay)]
    params_1 = [p for n,p in named_parameters if "embeddings" in n
                and not any(nd in n for nd in no_decay)]
    
    embed_params = {"params": params_0, "lr": lr, "weight_decay": 0.0} 
    opt_parameters.append(embed_params)
        
    embed_params = {"params": params_1, "lr": lr, "weight_decay": weight_decay} 
    opt_parameters.append(embed_params)        
    
    return AdamW(opt_parameters, lr=init_lr)


class LitRelRoberta(pl.LightningModule):
    
    def __init__(self, 
                 num_labels,
                 transformer_model,
                 anchor_dataloader,
                 hidden_size=768,
                 similarity_mode="inner",
                 normalization_mode="l2",
                 output_normalization_mode=None,
                 dropout_prob=0.1,
                 seed=42,
                 epochs=20,
                 weight_decay=0.0,
                 lr_init=1e-3,
                 layer_decay=0.65,
                 warmup_steps=10,
                 device="cpu"):
        super().__init__()
        
        # Saving hyperparameters of autoencoder
        self.save_hyperparameters() 
        
        pl.seed_everything(seed)

        # Creating encoder and decoder
        self.net = RelRoberta(num_labels,
                              transformer_model,
                              anchor_dataloader,
                              hidden_size,
                              similarity_mode,
                              normalization_mode,
                              output_normalization_mode,
                              dropout_prob,
                              device=device)
                
        self.loss_module = nn.CrossEntropyLoss()
        
        self.warmup_steps = warmup_steps
        self.epochs = epochs
        self.layer_decay = layer_decay
        self.weight_decay = weight_decay
        self.lr_init = lr_init
        
        
    def forward(self, x):
       
        return self.net(x)["prediction"]
        
    def detailed_forward(self, x):
        """
        The forward function takes in an image and returns a list of all the activations
        """
        
        return self.net(x)
    
    def configure_optimizers(self):
        config = {"optimizer": roberta_base_AdamW_LLRD(self.net,
                                            init_lr=self.lr_init,
                                            head_lr=self.lr_init,
                                            layer_decay=self.layer_decay,
                                            weight_decay=self.weight_decay
                                            )
                  }

        if self.warmup_steps is not None:
          config["scheduler"] = get_cosine_schedule_with_warmup(
                                      config["optimizer"],
                                      num_warmup_steps=self.warmup_steps,
                                      num_training_steps=self.epochs
                                      )
        
        
        return config
    
    def training_step(self, batch, batch_idx):
        # "batch" is the output of the training data loader.
        tokens, labels = batch
        preds = self.net(**tokens)["prediction"]
        loss = self.loss_module(preds, labels)
        acc = (preds.argmax(dim=-1) == labels).float().mean()

        # Logs the accuracy per epoch to tensorboard (weighted average over batches)
        self.log("train_acc", acc, prog_bar=True)
        self.log("train_loss", loss, prog_bar=True)
        return loss  # Return tensor to call ".backward" on

    def validation_step(self, batch, batch_idx):
        tokens, labels = batch
        preds = self.net(**tokens)["prediction"]
        loss = self.loss_module(preds, labels)
        acc = (preds.argmax(dim=-1) == labels).float().mean()

        # Logs the accuracy per epoch to tensorboard (weighted average over batches)
        self.log("val_acc", acc, prog_bar=True)
        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        tokens, labels = batch
        preds = self.net(**tokens)["prediction"].argmax(dim=-1)
        acc = (labels == preds).float().mean()
        # By default logs it per epoch (weighted average over batches), and returns it afterwards
        self.log("test_acc", acc)