"""
Modified from https://github.com/c-hofer/topologically_densified_distributions
"""

import torch
from torch import nn
from torchph.pershom import vr_persistence_l1, vr_persistence

EPSILON = 0.000001


class VrPersistenceL_1:
    def __call__(self, point_cloud):

        return vr_persistence_l1(point_cloud, 0, 0)


class VrPersistenceL_2:
    def __call__(self, point_cloud):
        D = (point_cloud.unsqueeze(0) - point_cloud.unsqueeze(1)).pow(2).sum(2)
        tmp = torch.zeros_like(D)
        tmp[D == 0.0] = EPSILON
        D = D + tmp
        D = D.sqrt()
        return vr_persistence(D, 0, 0)


class VrPersistenceL_p:
    """
    IMPORTANT: This handles 0 disstance differently than VrPersistenceL_2.
    Since p < 0 is possible we have to handle zero-instances at already at 
    *coordinate* (oposed to VrPersistenceL_2 where this is done after coordinate
    summation). 
    """

    def __init__(self, p):
        self.p = float(p)

    def __call__(self, point_cloud):
        D = (point_cloud.unsqueeze(0) - point_cloud.unsqueeze(1)).abs()

        tmp = torch.zeros_like(D)
        tmp[D < EPSILON] = EPSILON
        D = D + tmp

        D = D.pow(self.p)
        D = D.sum(2)

        D = D.pow(1./self.p)

        return vr_persistence(D, 0, 0)


class VrPersistenceL_inf:
    def __call__(self, point_cloud):
        D = (point_cloud.unsqueeze(0) - point_cloud.unsqueeze(1)).abs()
        D = D.max(dim=-1)[0]

        return vr_persistence(D, 0, 0)


class VrPersistenceF_p:
    def __init__(self, p):
        self.p = float(p)

    def __call__(self, point_cloud):
        D = (point_cloud.unsqueeze(0) - point_cloud.unsqueeze(1)).abs()

        tmp = torch.zeros_like(D)
        tmp[D < EPSILON] = EPSILON
        D = D + tmp

        D = D.pow(self.p)
        D = D.sum(2)

        return vr_persistence(D, 0, 0)


class VrPersistenceF_inf:
    def __call__(self, point_cloud):
        D = (point_cloud.unsqueeze(0) - point_cloud.unsqueeze(1))
        D = D.abs().max(dim=2)[0]
        return vr_persistence(D, 0, 0)


class VrPersistenceF_0:
    def __call__(self, point_cloud):
        D = (point_cloud.unsqueeze(0) - point_cloud.unsqueeze(1)).abs()

        D = D / (1. + D)
        D = D.sum(2)

        return vr_persistence(D, 0, 0)

def pers2fn(pers):
    pers = pers.split("_")
    if pers[0]=="L":
        if pers[1]=="1":
            pers_fn = VrPersistenceL_1()
        elif pers[1]=="2":
            pers_fn = VrPersistenceL_2()
        elif pers[1]=="inf":
            pers_fn = VrPersistenceL_inf()
        else:
            pers_fn = VrPersistenceL_p(int(pers[1]))
    else:
        # TODO: better control of input
        if pers[1]=="0":
            pers_fn = VrPersistenceF_0()
        elif pers[1]=="inf":
            pers_fn = VrPersistenceF_inf()
        else:
            pers_fn = VrPersistenceF_p(int(pers[1]))
    
    return pers_fn


class TopoRegLoss(nn.Module):
    def __init__(self, top_scale, pers="L_1", loss_type="L_1"):
        super().__init__()
        self.pers_fn = pers2fn(pers)
        self.top_scale = top_scale
        self.relu = nn.ReLU()
        self.loss_type = loss_type
                 
    
    def __call__(self, point_cloud):
        z_sample = point_cloud.contiguous()
        lt = self.pers_fn(z_sample)[0][0][:, 1]
        
        aux = lt - self.top_scale
        if self.loss_type == "L_1":
            loss = aux.abs().sum()
        elif self.loss_type == "L_2":
            loss = (aux**2).sum()
        elif self.loss_type == "relu":
            loss = self.relu(aux).sum()
        else:
            loss = torch.tensor(0.0, requires_grad=True)
  
        return loss
