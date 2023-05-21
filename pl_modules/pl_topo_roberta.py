## PyTorch
import torch
import torch.nn as nn
# PyTorch Lightning
import pytorch_lightning as pl
import torch.optim as optim
from modules.relRoberta import RelRoberta
import transformers
from pl_modules.pl_roberta import roberta_base_AdamW_LLRD
from utils.pershom import TopoRegLoss
import numpy as np

# +
POS2RES = {
    "pre": "batch_latent",
    "post_no_norm": "similarities",
    "post": "norm_similarities"
}

def frange_cycle_linear(start, stop, scale, n_epoch, n_cycle=4, ratio=0.5):
    L = np.ones(n_epoch)*scale
    period = n_epoch/n_cycle
    step = (stop-start)/(period*ratio) # linear schedule

    for c in range(n_cycle):

        v , i = start , 0
        while v <= stop and (int(i+c*period) < n_epoch):
            L[int(i+c*period)] = v*scale
            v += step
            i += 1
    return L   


def dfs_freeze(model):
    for module in model.modules():
        # print(module)
        if isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.LayerNorm):
            """
            for param in module.parameters():
                param.requires_grad = True
            """
            module.train()
            
        if isinstance(module, nn.Linear) or isinstance(module, nn.LayerNorm):
            for param in module.parameters():
                param.requires_grad = False
                
                
def dfs_unfreeze(model):
    for module in model.modules():
        # print(module)
        if isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.LayerNorm):
            """
            for param in module.parameters():
                param.requires_grad = False
            """
            module.eval()
            
        if isinstance(module, nn.Linear) or isinstance(module, nn.LayerNorm):
            for param in module.parameters():
                param.requires_grad = True


# -

class LitTopoRelRoberta(pl.LightningModule):
    
    def __init__(self, 
                 num_labels,
                 transformer_model,
                 anchor_dataloader,
                 topo_par=("pre", "L_1", 8, 0.1, "L_1"), # "post_no_norm", "post"
                 hidden_size=768,
                 similarity_mode="inner",
                 normalization_mode=None,
                 output_normalization_mode=None,
                 dropout_prob=0.1,
                 seed=42,
                 steps=20,
                 epochs=40,
                 weight_decay=0.0,
                 head_lr=1e-3,
                 encoder_lr=3.6e-6,
                 layer_decay=0.65,
                 scheduler_act=True,
                 freq_anchors=100,
                 device="cpu",
                 fine_tune=False,
                 linear=True
                ):
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
                              fine_tune=fine_tune,
                              linear=linear
                             )
                
                
        self.loss_module = nn.CrossEntropyLoss()
        self.aux_loss = nn.L1Loss()
        
        self.scheduler_act = scheduler_act
        self.steps = steps
        self.num_labels = num_labels
        self.layer_decay = layer_decay
        self.weight_decay = weight_decay
        self.head_lr = head_lr
        self.encoder_lr = encoder_lr
        self.reg_loss = None
        if topo_par is not None:
            self.latent_pos = topo_par[0]
            if self.net.anchor_dataloader is None:
                assert self.latent_pos == "pre"
            self.reg_loss = TopoRegLoss(top_scale=topo_par[2],
                                        pers=topo_par[1],
                                        loss_type=topo_par[4]
                                       )
            self.w_loss= topo_par[3]
        
        
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
                                                    num_warmup_steps=int(self.steps*0.1/self.trainer.max_epochs),
                                                    num_training_steps=self.steps
                                                    ),
                "interval": "step",

            }
            
            if self.reg_loss is not None:
                self.w_loss = iter(frange_cycle_linear(0, 1, self.w_loss, self.steps, n_cycle=4, ratio=0.75))
        
        
        return  config
    
    def training_step(self, batch, batch_idx):
        # "batch" is the output of the training data loader.
        tokens, labels = batch
      
        aux = batch_idx%(self.num_labels+1)
        if aux==0:
            dfs_freeze(self.net)
        elif aux==1:
            dfs_unfreeze(self.net)
        
        res = self.net(batch_idx=batch_idx, **tokens)
        
        if aux>=1:
            preds = res["prediction"]
            loss = self.loss_module(preds, labels)
            prediction = preds.argmax(dim=-1)
            acc = (prediction == labels).float().mean()
            mae = self.aux_loss(prediction.float(), labels.float())*100

            # Logs the accuracy per epoch to tensorboard (weighted average over batches)
            self.log("cls_loss", loss, prog_bar=True)
            self.log("train_acc", acc, prog_bar=True)
            self.log("train_mae", mae, prog_bar=True)

            if self.reg_loss is not None:
                if self.latent_pos=="pre" or self.latent_pos=="both":
                    latent_pre = res[POS2RES["pre"]]
                    loss_r_pre = self.reg_loss(latent_pre)
                if self.latent_pos=="post" or self.latent_pos=="both":
                    latent_post = res[POS2RES["post"]]
                    loss_r_post = self.reg_loss(latent_post)
                
                if self.latent_pos=="pre":
                    loss_r = loss_r_pre
                elif self.latent_pos=="post":
                    loss_r = loss_r_post
                elif self.latent_pos=="post_no_normor":
                    latent= res[POS2RES["post_no_normor"]]
                    loss_r = self.reg_loss(latent)
                else:
                    loss_r = 0.25*loss_r_pre + 0.75*loss_r_post
                    self.log("reg_pre", loss_r_pre, prog_bar=True)
                    self.log("reg_post", loss_r_post, prog_bar=True)
                    
                loss_r = next(self.w_loss)*loss_r
                self.log("reg_loss", loss_r, prog_bar=True)
                loss+=loss_r
        else:
            loss = torch.tensor(0.0, requires_grad=True).to(self.device)

        return loss # Return tensor to call ".backward" on

    def validation_step(self, batch, batch_idx):
        batch_idx = int(batch_idx>0)
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
        batch_idx = int(batch_idx>0)
        tokens, labels = batch
        preds = self.net(batch_idx=batch_idx, **tokens)["prediction"].argmax(dim=-1)
        acc = (labels == preds).float().mean()
        # By default logs it per epoch (weighted average over batches), and returns it afterwards
        self.log("test_acc", acc)



