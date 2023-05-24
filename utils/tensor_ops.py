"""
Extracted from https://openreview.net/attachment?id=SrC-nwieGJ&name=supplementary_material
"""
from typing import Any, Optional, Tuple

import torch
from torch import nn
from torchvision import models
import numpy as np

def set_seed(seed):
    """
    Extracted from https://lightning.ai/docs/pytorch/latest/notebooks/course_UvA-DL/02-activation-functions.html
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # GPU operation have separate seed
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def dfs_freeze(model):
    """
    Freezes the parameters of Linear and LayerNorm modules in the model, and sets BatchNorm1d and LayerNorm to train mode.

    Args:
        model (nn.Module): The model to freeze.
    """
    for module in model.modules():
        if isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.LayerNorm):
            module.train()
            
        if isinstance(module, nn.Linear) or isinstance(module, nn.LayerNorm):
            for param in module.parameters():
                param.requires_grad = False
                
                
def dfs_unfreeze(model):
    """
    Unfreezes  the parameters of Linear and LayerNorm modules in the model, and sets BatchNorm1d and LayerNorm to eval mode.

    Args:
        model (nn.Module): The model to freeze.
    """
    for module in model.modules():
        if isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.LayerNorm):
            module.eval()
            
        if isinstance(module, nn.Linear) or isinstance(module, nn.LayerNorm):
            for param in module.parameters():
                param.requires_grad = True
