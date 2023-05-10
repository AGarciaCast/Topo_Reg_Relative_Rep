

import numpy as np
from torch.nn.functional import pdist
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from warnings import warn
import matplotlib.pyplot as plt
import seaborn as sns 
from collections import defaultdict



def plot_topo(relative, pre_topo, pre_topo_max, post_topo, post_topo_max, title, save_path):
    fig, ax = plt.subplots(2, 2, figsize=(15, 5))
    if relative:
        sns.kdeplot(post_topo["model1"], linewidth=2, fill=True, label="Post-rel", ax=ax[0,0])
        sns.kdeplot(post_topo["model2"], linewidth=2, fill=True, label="Post-rel", ax=ax[1,0])
        
    sns.kdeplot(pre_topo["model1"], linewidth=2, fill=True, label="Pre-rel", ax=ax[0,0])
    sns.kdeplot(pre_topo["model2"], linewidth=2, fill=True, label="Pre-rel", ax=ax[1,0])
    ax[0,0].set_title("All death times (model 1)")
    ax[1,0].set_title("All death times (model 2)")
    
    if relative:
        sns.kdeplot(post_topo_max["model1"], linewidth=2, fill=True, label="Post-rel", ax=ax[0,1])
        sns.kdeplot(post_topo_max["model2"], linewidth=2, fill=True, label="Post-rel", ax=ax[1,1])
        
    sns.kdeplot(pre_topo_max["model1"], linewidth=2, fill=True, label="Pre-rel", ax=ax[0,1])
    sns.kdeplot(pre_topo_max["model2"], linewidth=2, fill=True, label="Pre-rel", ax=ax[1,1])
    ax[0,1].set_title("Max death times (model 1)")
    ax[1,1].set_title("MAx death times (model 2)")

    if relative:
        plt.legend()
        
    ax[0,0].grid()
    ax[1,0].grid()
    ax[0,1].grid()
    ax[1,1].grid()
    fig.suptitle(title)
    plt.tight_layout()
    # fig.savefig(save_path)
    if plot_topo:
        plt.show()
    
    plt.close()


def compare_topo_models(model1, model2, device,
            dataloader1: DataLoader,
            save_path, pers_fn, title="", plot_topo=False, relative=True,
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
        result += ((ones.t() @ K @ ones @ ones.t() @ L @ ones) / ((N - 1) * (N - 2))).item()
        result -= ((ones.t() @ K @ L @ ones) * 2 / (N - 2)).item()
        return (1 / (N * (N - 3)) * result).item()
    
    if dataloader2 is None:
        warn("Dataloader for Model 2 is not given. Using the same dataloader for both models.")
        dataloader2 = dataloader1
    
    
    model1.to(device)
    model1.eval()
    model2.to(device)
    model2.eval()
    
    pre_topo = defaultdict(list)
    pre_topo_max = defaultdict(list)
    if relative:
        post_topo = defaultdict(list)
        post_topo_max = defaultdict(list)

    num_batches = min(len(dataloader1), len(dataloader1))
    sim_pre = np.zeros([0.0, 0.0, 0.0])
    sim_post = np.zeros([0.0, 0.0, 0.0])
    
    with torch.no_grad():
        batch_idx = 0
        for batch1, batch2 in tqdm(zip(dataloader1, dataloader2), position=0, leave=True, desc="Computing"):
            batch1.to(device)
            res1 = model1(batch_idx=batch_idx, **batch1)
            X = res1["batch_latent"].contiguous()
            aux = pers_fn(X)[0][0][:, 1].tolist()
            pre_topo["model1"] += aux
            pre_topo_max["model1"].append(np.max(aux))
            
            K = X @ X.t()
            K.fill_diagonal_(0.0)
            sim_pre[0] += HSIC(K, K) / num_batches
            
            batch2.to(device)
            res2 = model1(batch_idx=batch_idx, **batch2)
            Y = res2["batch_latent"].contiguous()
            aux = pers_fn(Y)[0][0][:, 1].tolist()
            pre_topo["model2"] += aux
            pre_topo_max["model2"].append(np.max(aux))
            
            L = Y @ Y.t()
            L.fill_diagonal_(0)
            sim_pre[1] += HSIC(K, L) / num_batches
            sim_pre[2] += HSIC(L, L)/ num_batches
            
            if relative:
                X = res1["batch_latent"].contiguous()
                aux = pers_fn(X)[0][0][:, 1].tolist()
                post_topo["model1"] += aux
                post_topo_max["model1"].append(np.max(aux))
                
                K = X @ X.t()
                K.fill_diagonal_(0.0)
                sim_post[0] += HSIC(K, K) / num_batches
                
                Y = res2["norm_similarities"].contiguous()
                aux = pers_fn(Y)[0][0][:, 1].tolist()
                post_topo["model2"] += aux
                post_topo_max["model2"].append(np.max(aux))
                
                L = Y @ Y.t()
                L.fill_diagonal_(0)
                sim_post[1] += HSIC(K, L) / num_batches
                sim_post[2] += HSIC(L, L)/ num_batches
            
            batch_idx = 1
            

    sim_pre = sim_pre[1] / (np.sqrt(sim_pre[0]) * np.sqrt(sim_pre[2]))
            
    print(f"Pre mean model1 {np.mean(pre_topo['model1'])}. Pre max mean model1 {np.mean(pre_topo_max['model1'])}")
    print(f"Pre mean model2 {np.mean(pre_topo['model2'])}. Pre max mean model2 {np.mean(pre_topo_max['model2'])}")
    plot_topo(relative, pre_topo, pre_topo_max, post_topo, post_topo_max, title, save_path)
    
    if relative:
        print(f"Post mean model1 {np.mean(post_topo['model1'])}. Post max mean model1 {np.mean(post_topo_max['model2'])}")
        print(f"Post mean model1 {np.mean(post_topo['model2'])}. Prost max mean model2 {np.mean(post_topo_max['model2'])}")

        sim_post = sim_post[1] / (np.sqrt(sim_post[0]) * np.sqrt(sim_post[2]))
        return sim_pre, sim_post
    else:
        return sim_pre