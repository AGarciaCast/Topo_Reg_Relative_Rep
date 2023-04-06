from typing import Dict

import torch
from torch import nn


class StitchingModule(nn.Module):
    def __init__(self, module1, module2):
        super().__init__()
        self.module1 = module1
        self.module2 = module2

    def forward(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        encoding = self.module1.encode(*args, **kwargs)
        return self.module2.decode(**encoding)
