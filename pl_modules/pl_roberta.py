## PyTorch
import torch.nn as nn
# PyTorch Lightning
import pytorch_lightning as pl
import torch.optim as optim
from modules.relRoberta import RelRoberta
import transformers


def roberta_base_AdamW_LLRD(model, encoder_lr=3.6e-6, head_lr=1e-3, layer_decay=0.65, weight_decay=0.01):
    """
    https://towardsdatascience.com/advanced-techniques-for-fine-tuning-transformers-82e4e61e16e
    """
    
    opt_parameters = []    # To be passed to the optimizer (only parameters of the layers you want to update).
    named_parameters = list(model.named_parameters()) 
        
    # According to AAAMLP book by A. Thakur, we generally do not use any decay 
    # for bias and LayerNorm.weight layers.
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    lr = encoder_lr
    
    # === Pooler and regressor ======================================================  
    
    params_0 = [p for n,p in named_parameters if "decoder" in n
                and any(nd in n for nd in no_decay)]
    params_1 = [p for n,p in named_parameters if "decoder" in n
                and not any(nd in n for nd in no_decay)]
    
    head_params = {"params": params_0, "lr": head_lr, "weight_decay": 0.0}    
    opt_parameters.append(head_params)
        
    head_params = {"params": params_1, "lr": head_lr, "weight_decay": 0.01}    
    opt_parameters.append(head_params)
                
    # === 12 Hidden layers ==========================================================
    
    for layer in range(11,-1,-1):        
        params_0 = [p for n,p in named_parameters if f"encoder.layer.{layer}." in n 
                    and any(nd in n for nd in no_decay) and p.requires_grad]
        params_1 = [p for n,p in named_parameters if f"encoder.layer.{layer}." in n 
                    and not any(nd in n for nd in no_decay) and p.requires_grad]
        
        layer_params = {"params": params_0, "lr": lr, "weight_decay": 0.0}
        opt_parameters.append(layer_params)   
                            
        layer_params = {"params": params_1, "lr": lr, "weight_decay": 0.01}
        opt_parameters.append(layer_params)       
        
        lr *= 0.9     
        
    # === Embeddings layer ==========================================================
    
    params_0 = [p for n,p in named_parameters if "embeddings" in n 
                and any(nd in n for nd in no_decay) and p.requires_grad]
    params_1 = [p for n,p in named_parameters if "embeddings" in n
                and not any(nd in n for nd in no_decay) and p.requires_grad]
    
    embed_params = {"params": params_0, "lr": lr, "weight_decay": 0.0} 
    opt_parameters.append(embed_params)
        
    embed_params = {"params": params_1, "lr": lr, "weight_decay": 0.01} 
    opt_parameters.append(embed_params)        
    
    return optim.AdamW(opt_parameters)


class LitRelRoberta(pl.LightningModule):
    
    def __init__(self, 
                 num_labels,
                 transformer_model,
                 anchor_dataloader,
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
            "scheduler": transformers.get_constant_schedule_with_warmup(   
                                                    optimizer = config["optimizer"],
                                                    num_warmup_steps=int(self.steps*0.1),
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


