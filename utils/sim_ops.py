# -*- coding: utf-8 -*-

import numpy as np
from scipy.optimize import linear_sum_assignment
from torch.nn.functional import pdist
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from functools import partial
from warnings import warn
from typing import List, Dict
import matplotlib.pyplot as plt
from mpl_toolkits import axes_grid1


def add_colorbar(im, aspect=10, pad_fraction=0.5, **kwargs):
    """Add a vertical color bar to an image plot."""
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1./aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_ax)
    return im.axes.figure.colorbar(im, cax=cax, **kwargs)


class SimLayers:
    def __init__(self,
                 model1: nn.Module,
                 model2: nn.Module,
                 model1_name: str = None,
                 model2_name: str = None,
                 model1_layers: List[str] = None,
                 model2_layers: List[str] = None,
                 device: str ='cpu'):
        """

        :param model1: (nn.Module) Neural Network 1
        :param model2: (nn.Module) Neural Network 2
        :param model1_name: (str) Name of model 1
        :param model2_name: (str) Name of model 2
        :param model1_layers: (List) List of layers to extract features from
        :param model2_layers: (List) List of layers to extract features from
        :param device: Device to run the model
        """

        self.model1 = model1
        self.model2 = model2

        self.device = device

        self.model1_info = {}
        self.model2_info = {}

        if model1_name is None:
            self.model1_info['Name'] = model1.__repr__().split('(')[0]
        else:
            self.model1_info['Name'] = model1_name

        if model2_name is None:
            self.model2_info['Name'] = model2.__repr__().split('(')[0]
        else:
            self.model2_info['Name'] = model2_name

        if self.model1_info['Name'] == self.model2_info['Name']:
            warn(f"Both model have identical names - {self.model2_info['Name']}. " \
                 "It may cause confusion when interpreting the results. " \
                 "Consider giving unique names to the models :)")

        self.model1_info['Layers'] = []
        self.model2_info['Layers'] = []

        self.model1_features = {}
        self.model2_features = {}

        if len(list(model1.modules())) > 150 and model1_layers is None:
            warn("Model 1 seems to have a lot of layers. " \
                 "Consider giving a list of layers whose features you are concerned with " \
                 "through the 'model1_layers' parameter. Your CPU/GPU will thank you :)")

        self.model1_layers = model1_layers

        if len(list(model2.modules())) > 150 and model2_layers is None:
            warn("Model 2 seems to have a lot of layers. " \
                 "Consider giving a list of layers whose features you are concerned with " \
                 "through the 'model2_layers' parameter. Your CPU/GPU will thank you :)")

        self.model2_layers = model2_layers

        self._insert_hooks()
        self.model1 = self.model1.to(self.device)
        self.model2 = self.model2.to(self.device)

        self.model1.eval()
        self.model2.eval()

    def _log_layer(self,
                   model: str,
                   name: str,
                   layer: nn.Module,
                   inp: torch.Tensor,
                   out: torch.Tensor):

        if model == "model1":
            self.model1_features[name] = out

        elif model == "model2":
            self.model2_features[name] = out

        else:
            raise RuntimeError("Unknown model name for _log_layer.")

    def _insert_hooks(self):
        # Model 1
        for name, layer in self.model1.named_modules():
            if self.model1_layers is not None:
                if name in self.model1_layers:
                    self.model1_info['Layers'] += [name]
                    layer.register_forward_hook(partial(self._log_layer, "model1", name))
            else:
                self.model1_info['Layers'] += [name]
                layer.register_forward_hook(partial(self._log_layer, "model1", name))

        # Model 2
        for name, layer in self.model2.named_modules():
            if self.model2_layers is not None:
                if name in self.model2_layers:
                    self.model2_info['Layers'] += [name]
                    layer.register_forward_hook(partial(self._log_layer, "model2", name))
            else:

                self.model2_info['Layers'] += [name]
                layer.register_forward_hook(partial(self._log_layer, "model2", name))


    def compare(self,
                dataloader1: DataLoader,
                dataloader2: DataLoader = None) -> None:
        """
        Computes the feature similarity between the models on the
        given datasets.
        :param dataloader1: (DataLoader)
        :param dataloader2: (DataLoader) If given, model 2 will run on this
                            dataset. (default = None)
        """


    def export(self) -> Dict:
        """
        Exports the CKA data along with the respective model layer names.
        :return:
        """
        return {
            "model1_name": self.model1_info['Name'],
            "model2_name": self.model2_info['Name'],
            "Sim": self.sim_matrix,
            "model1_layers": self.model1_info['Layers'],
            "model2_layers": self.model2_info['Layers'],
            "dataset1_name": self.model1_info['Dataset'],
            "dataset2_name": self.model2_info['Dataset'],

        }

    def plot_results(self,
                     save_path: str = None,
                     title: str = None,
                     ax_ = None,
                     cmap="magma"):
        
        if ax_ is None:
            fig, ax = plt.subplots()
        else:
            ax = ax_
            
        im = ax.imshow(self.sim_matrix, origin='lower', cmap=cmap)
        
        if ax_ is None:
            
            ax.set_xlabel(f"Layers {self.model2_info['Name']}", fontsize=15)
            ax.set_ylabel(f"Layers {self.model1_info['Name']}", fontsize=15)

            if title is not None:
                ax.set_title(f"{title}", fontsize=18)
            else:
                ax.set_title(f"{self.model1_info['Name']} vs {self.model2_info['Name']}", fontsize=18)

            add_colorbar(im)
        
            plt.tight_layout()

            if save_path is not None:
                plt.savefig(save_path, dpi=300)

            plt.show()


class CKA(SimLayers):
    # Modified from: https://github.com/AntixK/PyTorch-Model-Compare 
    
    def __init__(self,
                 model1: nn.Module,
                 model2: nn.Module,
                 model1_name: str = None,
                 model2_name: str = None,
                 model1_layers: List[str] = None,
                 model2_layers: List[str] = None,
                 device: str ='cpu'):
        """

        :param model1: (nn.Module) Neural Network 1
        :param model2: (nn.Module) Neural Network 2
        :param model1_name: (str) Name of model 1
        :param model2_name: (str) Name of model 2
        :param model1_layers: (List) List of layers to extract features from
        :param model2_layers: (List) List of layers to extract features from
        :param device: Device to run the model
        """
        super().__init__(model1, model2,
                         model1_name, model2_name,
                         model1_layers, model2_layers,
                         device)
       

    def _HSIC(self, K, L):
        """
        Computes the unbiased estimate of HSIC metric.

        Reference: https://arxiv.org/pdf/2010.15327.pdf Eq (3)
        """
        N = K.shape[0]
        ones = torch.ones(N, 1).to(self.device)
        result = torch.trace(K @ L)
        result += ((ones.t() @ K @ ones @ ones.t() @ L @ ones) / ((N - 1) * (N - 2))).item()
        result -= ((ones.t() @ K @ L @ ones) * 2 / (N - 2)).item()
        return (1 / (N * (N - 3)) * result).item()

    def compare(self,
                dataloader1: DataLoader,
                dataloader2: DataLoader = None) -> None:
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

        self.model1_info['Dataset'] = dataloader1.dataset.__repr__().split('\n')[0]
        self.model2_info['Dataset'] = dataloader2.dataset.__repr__().split('\n')[0]

        N = len(self.model1_layers) if self.model1_layers is not None else len(list(self.model1.modules()))
        M = len(self.model2_layers) if self.model2_layers is not None else len(list(self.model2.modules()))

        self.sim_matrix = torch.zeros(N, M, 3)

        num_batches = min(len(dataloader1), len(dataloader2))

        for (x1, *_), (x2, *_) in tqdm(zip(dataloader1, dataloader2), desc="| Comparing features |", total=num_batches):
            aux_L = {}
            self.model1_features = {}
            self.model2_features = {}
            _ = self.model1(x1.to(self.device))
            _ = self.model2(x2.to(self.device))

            for i, (name1, feat1) in enumerate(self.model1_features.items()):
                X = feat1.flatten(1)
                K = X @ X.t()
                K.fill_diagonal_(0.0)
                self.sim_matrix[i, :, 0] += self._HSIC(K, K) / num_batches

                for j, (name2, feat2) in enumerate(self.model2_features.items()):
                    if name2 not in aux_L:
                        Y = feat2.flatten(1)
                        L = Y @ Y.t()
                        L.fill_diagonal_(0)
                        assert K.shape == L.shape, f"Feature shape mistach! {K.shape}, {L.shape}"
                        aux_HSIC = self._HSIC(L, L)
                        aux_L[name2] = (L, aux_HSIC)

                    self.sim_matrix[i, j, 1] += self._HSIC(K, aux_L[name2][0]) / num_batches
                    self.sim_matrix[i, j, 2] += aux_L[name2][1] / num_batches

        self.sim_matrix = self.sim_matrix[:, :, 1] / (self.sim_matrix[:, :, 0].sqrt() *
                                                        self.sim_matrix[:, :, 2].sqrt())

        # assert not torch.isnan(self.sim_matrix).any(), "HSIC computation resulted in NANs"


class MinFrob(SimLayers):
    # Modified from: https://github.com/AntixK/PyTorch-Model-Compare 
    
    def __init__(self,
                 model1: nn.Module,
                 model2: nn.Module,
                 model1_name: str = None,
                 model2_name: str = None,
                 model1_layers: List[str] = None,
                 model2_layers: List[str] = None,
                 device: str ='cpu',
                 metric=2):
        """

        :param model1: (nn.Module) Neural Network 1
        :param model2: (nn.Module) Neural Network 2
        :param model1_name: (str) Name of model 1
        :param model2_name: (str) Name of model 2
        :param model1_layers: (List) List of layers to extract features from
        :param model2_layers: (List) List of layers to extract features from
        :param device: Device to run the model
        """
        super().__init__(model1, model2,
                         model1_name, model2_name,
                         model1_layers, model2_layers,
                         device)
        
        self.metric = metric
       

    
    def minFrobDist_distM(self, points_A, points_B):
        unnorm_dist_A = pdist(points_A, p=self.metric)
        unnorm_dist_B = pdist(points_B, p=self.metric)
        dist_A, _ = torch.sort(unnorm_dist_A/torch.sqrt((unnorm_dist_A**2).sum()))
        dist_B, _ =torch.sort(unnorm_dist_B/torch.sqrt((unnorm_dist_B**2).sum()))

        cost = (dist_A-dist_B)**2
        minimum_dist = torch.sqrt(cost.sum())/torch.sqrt(dist_A.shape[0]*dist_A.shape[1])
        
        return minimum_dist.item()

    def compare(self,
                dataloader1: DataLoader,
                dataloader2: DataLoader = None) -> None:
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

        self.model1_info['Dataset'] = dataloader1.dataset.__repr__().split('\n')[0]
        self.model2_info['Dataset'] = dataloader2.dataset.__repr__().split('\n')[0]

        N = len(self.model1_layers) if self.model1_layers is not None else len(list(self.model1.modules()))
        M = len(self.model2_layers) if self.model2_layers is not None else len(list(self.model2.modules()))

        self.sim_matrix = torch.zeros(N, M)

        num_batches = min(len(dataloader1), len(dataloader2))
        

        for (x1, *_), (x2, *_) in tqdm(zip(dataloader1, dataloader2), desc="| Comparing features |", total=num_batches):
            
            dist_2 = {}
            
            self.model1_features = {}
            self.model2_features = {}
            
            _ = self.model1(x1.to(self.device))
            _ = self.model2(x2.to(self.device))

            for i, (name1, feat1) in enumerate(self.model1_features.items()):
                X = feat1.flatten(1)
                unnorm_dist_A = pdist(X, p=self.metric)
                dist_A = unnorm_dist_A/torch.sqrt((unnorm_dist_A**2).sum())
                dist_A = torch.sort(dist_A)[0]
                
                for j, (name2, feat2) in enumerate(self.model2_features.items()):
                   
                    if name2 not in dist_2:
                        Y = feat2.flatten(1)
                        unnorm_dist_B = pdist(Y, p=self.metric)
                        dist_B = unnorm_dist_B/torch.sqrt((unnorm_dist_B**2).sum())
                        dist_2[name2] = torch.sort(dist_B)[0]
                    
                    cost = (dist_A - dist_2[name2])**2
                    minimum_dist = torch.sqrt(cost.sum()).item()
                    
                    self.sim_matrix[i, j] += minimum_dist / num_batches




