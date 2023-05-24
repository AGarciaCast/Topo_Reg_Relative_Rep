"""
Modified from https://openreview.net/attachment?id=SrC-nwieGJ&name=supplementary_material
"""
from typing import Dict

import torch
from torch import nn


class StitchingModule(nn.Module):
    def __init__(self, module1, module2):
        """
        Initializes an instance of the StitchingModule class.

        Args:
            module1: The first module to be used for encoding.
            module2: The second module to be used for decoding.
        """
        super().__init__()
        self.module1 = module1
        self.module2 = module2

    def forward(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Performs a forward pass through the stitching module.

        Args:
            *args: Variable length arguments to be passed to the encode method of module1.
            **kwargs: Keyword arguments to be passed to the encode method of module1.

        Returns:
            A dictionary containing the output of the decode method of module2.

        Note:
            The encode method of module1 is called with the provided arguments and keyword arguments,
            and the resulting encoding is passed to the decode method of module2.
        """
        encoding = self.module1.encode(*args, **kwargs)
        return self.module2.decode(**encoding)
