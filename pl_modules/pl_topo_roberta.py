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
from utils.tensor_ops import dfs_freeze, dfs_unfreeze
import numpy as np

# +
POS2RES = {
    "pre": "batch_latent",
    "post_no_norm": "similarities",
    "post": "norm_similarities"
}

def frange_cycle_linear(start, stop, scale, n_epoch, n_cycle=4, ratio=0.5):
    """
    Generates a cyclical learning rate schedule with linearly increasing values.

    Args:
        start (float): Starting value of the learning rate.
        stop (float): Ending value of the learning rate.
        scale (float): Scaling factor for the learning rate.
        n_epoch (int): Total number of epochs.
        n_cycle (int, optional): Number of cycles. Defaults to 4.
        ratio (float, optional): Ratio of the increasing phase within each cycle. Defaults to 0.5.

    Returns:
        np.ndarray: Array of learning rate values with linearly increasing sections.

    Note:
        This function is based on the cyclical learning rate implementation from the following source:
        https://github.com/haofuml/cyclical_annealing/tree/master
    """
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


# -

class LitTopoRelRoberta(pl.LightningModule):
    
    def __init__(self, 
                 num_labels,
                 transformer_model,
                 anchor_dataloader,
                 topo_par=("pre", "L_2", 8, 0.1, "L_1"), # "post_no_norm", "post"
                 hidden_size=768,
                 similarity_mode="inner",
                 normalization_mode="batchnorm",
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
        """
        Lightning module for RelRoberta using Hoffer's Topological regularization.

        Args:
            num_labels (int): Number of labels.
            transformer_model: Transformer model for Relative RoBERTa.
            anchor_dataloader: Dataloader for anchor points.
            topo_par (tuple): Topological regularization parameters. Defaults to ("pre", "L_2", 8, 0.1, "L_1").
            hidden_size (int): Hidden size of the model. Defaults to 768.
            similarity_mode (str): Similarity mode for Relative RoBERTa. Defaults to "inner".
            normalization_mode: Normalization mode for Relative RoBERTa. Defaults to "batchnorm".
            output_normalization_mode: Output normalization mode for Relative RoBERTa. Defaults to None.
            dropout_prob (float): Dropout probability. Defaults to 0.1.
            seed (int): Random seed. Defaults to 42.
            steps (int): Number of steps. Defaults to 20.
            epochs (int): Number of epochs. Defaults to 40.
            weight_decay (float): Weight decay. Defaults to 0.0.
            head_lr (float): Learning rate for the model's head. Defaults to 1e-3.
            encoder_lr (float): Learning rate for the model's encoder. Defaults to 3.6e-6.
            layer_decay (float): Layer decay rate. Defaults to 0.65.
            scheduler_act (bool): Whether to activate the learning rate scheduler. Defaults to True.
            freq_anchors (int): Frequency of anchor updates. Defaults to 100.
            device (str): Device to use (e.g., "cpu", "cuda"). Defaults to "cpu".
            fine_tune (bool): Whether to fine-tune the model. Defaults to False.
            linear (bool): Whether to use linear classification head. Defaults to True.
        """
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
                
            if self.latent_pos == "both":
                
                if type(topo_par[2]) is tuple:
                    top_scale_pre = topo_par[2][0]
                    top_scale_post = topo_par[2][1]
                else:
                    top_scale_pre = topo_par[2]
                    top_scale_post = topo_par[2] 
                    
                self.reg_loss = (TopoRegLoss(top_scale=top_scale_pre,
                                        pers="L_2",
                                        loss_type=topo_par[4]
                                       ),
                             TopoRegLoss(top_scale=top_scale_post,
                                        pers=topo_par[1],
                                        loss_type=topo_par[4]
                                       )
                            )
                
            else:
                self.reg_loss = TopoRegLoss(top_scale=topo_par[2],
                                        pers=topo_par[1],
                                        loss_type=topo_par[4]
                                       )
            self.w_loss= topo_par[3]
        
        
    def forward(self, x):
        """
        Performs forward pass of the model.

        Args:
            x (dict): Input data dictionary.

        Returns:
            torch.Tensor: Predicted output tensor.
        """
        
        return self.net(**x)["prediction"]
        
    def detailed_forward(self, x):
        """
        Performs detailed forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            dict: Dictionary containing the prediction, the batch latent representation, similarities, normalized similarities,  original anchors, normalized anchors, and normalized batch.

        """
        
        return self.net(x)
    
    def configure_optimizers(self):
        """
        Configures the optimizers, learning rate and weight schedulers for training.

        Returns:
            dict: Dictionary containing the optimizer and optional learning rate scheduler.
        """
        
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
        """
        Performs a single training step.

        Args:
            batch (tuple): Input batch data.
            batch_idx (int): Batch index.

        Returns:
            torch.Tensor: Training loss.
        """
        
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
                    
                    if self.latent_pos=="both":
                        loss_r_pre = self.reg_loss[0](latent_pre)
                    else:
                        loss_r_pre = self.reg_loss(latent_pre)
                        
                if self.latent_pos=="post" or self.latent_pos=="both":
                    latent_post = res[POS2RES["post"]]
                    
                    if self.latent_pos=="both":  
                        loss_r_post = self.reg_loss[1](latent_post)
                    else:
                        loss_r_post = self.reg_loss(latent_post)
                        
                if self.latent_pos=="pre":
                    loss_r = loss_r_pre
                elif self.latent_pos=="post":
                    loss_r = loss_r_post
                elif self.latent_pos=="post_no_normor":
                    latent= res[POS2RES["post_no_normor"]]
                    loss_r = self.reg_loss(latent)
                else:
                    loss_r = 0.1*loss_r_pre + 0.9*loss_r_post
                    self.log("reg_pre", loss_r_pre, prog_bar=True)
                    self.log("reg_post", loss_r_post, prog_bar=True)
                    
                loss_r = next(self.w_loss)*loss_r
                self.log("reg_loss", loss_r, prog_bar=True)
                loss+=loss_r
        else:
            loss = torch.tensor(0.0, requires_grad=True).to(self.device)

        return loss # Return tensor to call ".backward" on

    def validation_step(self, batch, batch_idx):
        """
        Performs a single validation step.

        Args:
            batch (tuple): Input batch data.
            batch_idx (int): Batch index.
        """
        
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
        """
        Performs a single test step.

        Args:
            batch (tuple): Input batch data.
            batch_idx (int): Batch index.
        """
        
        batch_idx = int(batch_idx>0)
        tokens, labels = batch
        preds = self.net(batch_idx=batch_idx, **tokens)["prediction"].argmax(dim=-1)
        acc = (labels == preds).float().mean()
        # By default logs it per epoch (weighted average over batches), and returns it afterwards
        self.log("test_acc", acc)



