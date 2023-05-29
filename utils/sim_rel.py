

import numpy as np
from torch.nn.functional import pdist
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from warnings import warn
import matplotlib.pyplot as plt
import seaborn as sns 
from collections import defaultdict

# +
import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from utils.pershom import pers2fn


# -

def plot_topo_2(relative, pre_topo, pre_topo_max, post_topo, post_topo_max, title, save_path):
    fig, ax = plt.subplots(2, 2, figsize=(15, 5))
    
    pre_mean_1 = np.round(np.mean(pre_topo["model1"]), 2)
    pre_max_mean_1 = np.round(np.mean(pre_topo_max["model1"]), 2)
    extra_tit_1 = f"Pre mean {pre_mean_1}"
    extra_tit_max_1 = f"Pre max mean {pre_max_mean_1}"
    
    pre_mean_2 = np.round(np.mean(pre_topo["model2"]), 2)
    pre_max_mean_2 = np.round(np.mean(pre_topo_max["model2"]), 2)
    extra_tit_2 = f"Pre mean {pre_mean_2}"
    extra_tit_max_2 = f"Pre max mean {pre_max_mean_2}"
    
    if relative:
        post_mean_1 = np.round(np.mean(post_topo["model1"]), 2)
        post_max_mean_1 = np.round(np.mean(post_topo_max["model1"]), 2)
        extra_tit_1 += f". Post mean {post_mean_1}"
        extra_tit_max_1 += f". Post max mean {post_max_mean_1}"
        
        post_mean_2 = np.round(np.mean(post_topo["model2"]), 2)
        post_max_mean_2 = np.round(np.mean(post_topo_max["model2"]), 2)
        extra_tit_2 += f". Post mean {post_mean_2}"
        extra_tit_max_2 += f". Post max mean {post_max_mean_2}"
        
        sns.kdeplot(post_topo["model1"], linewidth=2, fill=True, label="Post-rel", ax=ax[0,0])
        sns.kdeplot(post_topo["model2"], linewidth=2, fill=True, label="Post-rel", ax=ax[1,0])
        
    sns.kdeplot(pre_topo["model1"], linewidth=2, fill=True, label="Pre-rel", ax=ax[0,0])
    sns.kdeplot(pre_topo["model2"], linewidth=2, fill=True, label="Pre-rel", ax=ax[1,0])
    ax[0,0].set_title(f"Model 1: All death times ({extra_tit_1})")
    ax[1,0].set_title(f"Model 2: All death times ({extra_tit_2})")
    
    if relative:
        sns.kdeplot(post_topo_max["model1"], linewidth=2, fill=True, label="Post-rel", ax=ax[0,1])
        sns.kdeplot(post_topo_max["model2"], linewidth=2, fill=True, label="Post-rel", ax=ax[1,1])
        
    sns.kdeplot(pre_topo_max["model1"], linewidth=2, fill=True, label="Pre-rel", ax=ax[0,1])
    sns.kdeplot(pre_topo_max["model2"], linewidth=2, fill=True, label="Pre-rel", ax=ax[1,1])
    ax[0,1].set_title(f"Model 1: Max death times ({extra_tit_max_1})")
    ax[1,1].set_title(f"Model 2: Max death times ({extra_tit_max_2})")

    if relative:
        ax[0,1].legend()
        
    ax[0,0].grid()
    ax[1,0].grid()
    ax[0,1].grid()
    ax[1,1].grid()
    fig.suptitle(title)
    plt.tight_layout()
    fig.savefig(save_path)
    plt.show()
    
    plt.close()


# +
def compare_topo_models(model1, model2, device,
            dataloader1: DataLoader,
            save_path,  pers="L_2", title="", relative=True,
            dataloader2: DataLoader = None):
    """
    Computes the feature similarity between the models on the
    given datasets.
    :param dataloader1: (DataLoader)
    :param dataloader2: (DataLoader) If given, model 2 will run on this
                        dataset. (default = None)
    """
    if type(pers) is tuple:
        pers_fn = pers2fn(pers[0])
        pers_fn_2 = pers2fn(pers[1])
    else:
        pers_fn = pers2fn(pers)
        if relative:
            pers_fn_2 = pers2fn(pers)
    
    if dataloader2 is None:
        warn("Dataloader for Model 2 is not given. Using the same dataloader for both models.")
        dataloader2 = dataloader1
    
    
    model1.to(device)
    model1.eval()
    model2.to(device)
    model2.eval()
    
    pre_topo = defaultdict(list)
    pre_topo_max = defaultdict(list)
    post_topo = None
    post_topo_max = None
    if relative:
        post_topo = defaultdict(list)
        post_topo_max = defaultdict(list)

    
    with torch.no_grad():
        batch_idx = 0
        for batch1, batch2 in tqdm(zip(dataloader1, dataloader2), position=0, leave=True, desc="Computing"):
            batch1.to(device)
            res1 = model1(batch_idx=batch_idx, **batch1)
            X = res1["batch_latent"]
            aux = pers_fn(X.contiguous())[0][0][:, 1].tolist()
            pre_topo["model1"] += aux
            pre_topo_max["model1"].append(np.max(aux))
            
            
            batch2.to(device)
            res2 = model1(batch_idx=batch_idx, **batch2)
            Y = res2["batch_latent"]
            aux = pers_fn(Y.contiguous())[0][0][:, 1].tolist()
            pre_topo["model2"] += aux
            pre_topo_max["model2"].append(np.max(aux))
            
            
            if relative:
                X = res1["norm_similarities"]
                aux = pers_fn_2(X.contiguous())[0][0][:, 1].tolist()
                post_topo["model1"] += aux
                post_topo_max["model1"].append(np.max(aux))
                
                Y = res2["norm_similarities"]
                aux = pers_fn_2(Y.contiguous())[0][0][:, 1].tolist()
                post_topo["model2"] += aux
                post_topo_max["model2"].append(np.max(aux))
                

            
            batch_idx = 1
            
            
  
    plot_topo_2(relative, pre_topo, pre_topo_max, post_topo, post_topo_max, title, save_path)
    
    

    
def compare_CKA_models(model1, model2, device,
            dataloader1: DataLoader,
            relative=True,
            dataloader2: DataLoader = None):
    """
    Computes the feature similarity between the models on the
    given datasets.
    :param dataloader1: (DataLoader)
    :param dataloader2: (DataLoader) If given, model 2 will run on this
                        dataset. (default = None)
    """
    
    def HSIC(K, L):
        """
        Computes the unbiased estimate of HSIC metric.

        Reference: https://arxiv.org/pdf/2010.15327.pdf Eq (3)
        """
        N = K.shape[0]
        ones = torch.ones(N, 1).to(device)
        result = torch.trace(K @ L)
        result += ((ones.T @ K @ ones @ ones.T @ L @ ones) / ((N - 1) * (N - 2))).item()
        result -= ((ones.T @ K @ L @ ones) * 2 / (N - 2)).item()
        return (1 / (N * (N - 3)) * result).item()
    
    if dataloader2 is None:
        warn("Dataloader for Model 2 is not given. Using the same dataloader for both models.")
        dataloader2 = dataloader1
    
    
    model1.to(device)
    model1.eval()
    model2.to(device)
    model2.eval()
    

    num_batches = min(len(dataloader1), len(dataloader2))
    sim_pre = np.zeros(3)
    sim_post = np.zeros(3)
    
    with torch.no_grad():
        batch_idx = 0
        for batch1, batch2 in tqdm(zip(dataloader1, dataloader2), position=0, leave=True, desc="Computing"):
            if type(batch1) is list:
                batch1, _ = batch1
                
            batch1.to(device)
            res1 = model1(batch_idx=batch_idx, **batch1)
            X = res1["batch_latent"]
            K = X @ X.T
            K.fill_diagonal_(0.0)
            
            if type(batch2) is list:
                batch2, _ = batch2
                
            batch2.to(device)
            res2 = model1(batch_idx=batch_idx, **batch2)
            batch_idx = 1
            Y = res2["batch_latent"]
            L = Y @ Y.T
            L.fill_diagonal_(0.0)
            
            sim_pre[0] += HSIC(K, L) / num_batches
            sim_pre[1] += HSIC(K, K) / num_batches
            sim_pre[2] += HSIC(L, L)/ num_batches
            
            if relative:
                X = res1["norm_similarities"]
                K = X @ X.T
                K.fill_diagonal_(0.0)
                
                Y = res2["norm_similarities"]
                L = Y @ Y.T
                L.fill_diagonal_(0.0)
                
                sim_post[0] += HSIC(K, L) / num_batches
                sim_post[1] += HSIC(K, K) / num_batches
                sim_post[2] += HSIC(L, L)/ num_batches
            
            
            

    res_pre = sim_pre[0] / (np.sqrt(sim_pre[1]) * np.sqrt(sim_pre[2]))
            
    if relative:
        res_post = sim_post[0] / (np.sqrt(sim_post[1]) * np.sqrt(sim_post[2]))
        return res_pre, res_post
    else:
        return res_pre

    
def minFrobDist_distM(self, points_A, points_B):
        unnorm_dist_A = pdist(points_A, p=self.metric)
        unnorm_dist_B = pdist(points_B, p=self.metric)
        dist_A, _ = torch.sort(unnorm_dist_A/torch.sqrt((unnorm_dist_A**2).sum()))
        dist_B, _ =torch.sort(unnorm_dist_B/torch.sqrt((unnorm_dist_B**2).sum()))

        cost = (dist_A-dist_B)**2
        minimum_dist = torch.sqrt(cost.sum())/torch.sqrt(dist_A.shape[0]*dist_A.shape[1])
        
        return minimum_dist.item()

def compare_Frob_models(model1, model2, device,
            dataloader1: DataLoader,
            relative=True,
            dataloader2: DataLoader = None,
            metric=2):
    """
    Computes the feature similarity between the models on the
    given datasets.
    :param dataloader1: (DataLoader)
    :param dataloader2: (DataLoader) If given, model 2 will run on this
                        dataset. (default = None)
    """

    if dataloader2 is None:
        warn("Dataloader for Model 2 is not given. Using the same dataloader for both models.")
        dataloader2 = dataloader1



    sim_pre = 0.0
    sim_post = 0.0

    num_batches = min(len(dataloader1), len(dataloader2))
    batch_idx=0
    for batch1, batch2 in tqdm(zip(dataloader1, dataloader2), position=0, leave=True, desc="Computing"):
        if type(batch1) is list:
            batch1, _ = batch1

        batch1.to(device)
        res1 = model1(batch_idx=batch_idx, **batch1)
        X = res1["batch_latent"]
        unnorm_dist_A = pdist(X, p=metric)
        dist_A = unnorm_dist_A/torch.sqrt((unnorm_dist_A**2).sum())
        dist_A = torch.sort(dist_A)[0]

        if type(batch2) is list:
            batch2, _ = batch2

        batch2.to(device)
        res2 = model1(batch_idx=batch_idx, **batch2)
        batch_idx = 1
        Y = res2["batch_latent"]
        unnorm_dist_B = pdist(Y, p=metric)
        dist_B = unnorm_dist_B/torch.sqrt((unnorm_dist_B**2).sum())
        dist_B = torch.sort(dist_B)[0]

        cost = (dist_A - dist_B)**2
        minimum_dist = torch.sqrt(cost.sum()).item()

        sim_pre += minimum_dist / num_batches

        if relative:
            X = res1["norm_similarities"]
            unnorm_dist_A = pdist(X, p=metric)
            dist_A = unnorm_dist_A/torch.sqrt((unnorm_dist_A**2).sum())
            dist_A = torch.sort(dist_A)[0]

            Y = res2["norm_similarities"]
            unnorm_dist_B = pdist(Y, p=metric)
            dist_B = unnorm_dist_B/torch.sqrt((unnorm_dist_B**2).sum())
            dist_B = torch.sort(dist_B)[0]

            cost = (dist_A - dist_B)**2
            minimum_dist = torch.sqrt(cost.sum()).item()

            sim_post += minimum_dist / num_batches

            
    if relative:
        return sim_pre, sim_post
    else:
        return sim_pre


# -

def topo_model(model, device, dataloader, save_path, title="", pers="L_2", plot_topo=False, relative=True):
    model.to(device)
    model.eval()
    
    if type(pers) is tuple:
        pers_fn = pers2fn(pers[0])
        pers_fn_2 = pers2fn(pers[1])
    else:
        pers_fn = pers2fn(pers)
        if relative:
            pers_fn_2 = pers2fn(pers)
    
    pre_topo = []
    pre_topo_max = []
    if relative:
        post_topo = []
        post_topo_max = []
        
    with torch.no_grad():
        batch_idx = 0
        for batch in tqdm(dataloader, position=0, leave=True, desc="Computing"):
            batch.to(device)
            res = model(batch_idx=batch_idx, **batch)
            aux = pers_fn(res["batch_latent"].contiguous())[0][0][:, 1].tolist()
            pre_topo += aux
            pre_topo_max.append(np.max(aux))
            if relative:
                aux = pers_fn_2(res["norm_similarities"].contiguous())[0][0][:, 1].tolist()
                post_topo += aux
                post_topo_max.append(np.max(aux))
            
            batch_idx = 1
    
    pre_mean = np.round(np.mean(pre_topo), 2)
    pre_max_mean = np.round(np.mean(pre_topo_max), 2)
    extra_tit = f"Pre mean {pre_mean}"
    extra_tit_max = f"Pre max mean {pre_max_mean}"
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    if relative:
        post_mean = np.round(np.mean(post_topo), 2)
        post_max_mean = np.round(np.mean(post_topo_max), 2)
        extra_tit += f". Post mean {post_mean}"
        extra_tit_max += f". Post max mean {post_max_mean}"
        sns.kdeplot(post_topo, linewidth=2, fill=True, label="Post-rel", ax=ax1)
        
    sns.kdeplot(pre_topo, linewidth=2, fill=True, label="Pre-rel", ax=ax1)
    ax1.set_title(f"All death times ({extra_tit})")
    
    if relative:
        sns.kdeplot(post_topo_max, linewidth=2, fill=True, label="Post-rel", ax=ax2)
        
    sns.kdeplot(pre_topo_max, linewidth=2, fill=True, label="Pre-rel", ax=ax2)
    ax2.set_title(f"Max death times ({extra_tit_max})")

    if relative:
        plt.legend()
        
    ax1.grid()
    ax2.grid()
    fig.suptitle(title)
    plt.tight_layout()
    fig.savefig(save_path)
    if plot_topo:
        plt.show()
    
    plt.close()

    if relative:
        return pre_topo, pre_topo_max, post_topo, post_topo_max
    else:
        return pre_topo, pre_topo_max
