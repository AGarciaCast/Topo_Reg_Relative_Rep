import math

import torch
import torch.nn.functional as F
from torch import nn


class RelativeAttention(nn.Module):
    def __init__(
        self,
        n_anchors: int,
        similarity_mode="inner",
        normalization_mode="l2",
        output_normalization_mode=None,
    ):
        """Relative attention block.

        Args:
            n_anchors: number of anchors
            similarity_mode: how to compute similarities: inner, basis_change
            normalization_mode: normalization to apply to the anchors and batch before computing the transformation
            output_normalization_mode: normalization to apply to the relative transformation
        """
        super().__init__()

        self.n_anchors = n_anchors
        self.similarity_mode = similarity_mode
        self.normalization_mode = normalization_mode
        self.output_normalization_mode = output_normalization_mode

        if self.output_normalization_mode == "batchnorm":
            self.outnorm = nn.BatchNorm1d(num_features=self.output_dim)
        elif self.output_normalization_mode == "layernorm":
            self.outnorm = nn.LayerNorm(normalized_shape=self.output_dim)
        elif self.output_normalization_mode == "instancenorm":
            self.outnorm = nn.InstanceNorm1d(num_features=self.output_dim, affine=True)

    def encode(self, x: torch.Tensor, anchors: torch.Tensor):
        original_anchors = anchors
        if x.shape[-1] != anchors.shape[-1]:
            raise ValueError(f"Inconsistent dimensions between batch and anchors: {x.shape}, {anchors.shape}")


        # Normalize latents
        if self.normalization_mode is None:
            pass
        elif self.normalization_mode == "l2":
            x = F.normalize(x, p=2, dim=-1)
            anchors = F.normalize(anchors, p=2, dim=-1)
        else:
            raise ValueError(f"Normalization mode not supported: {self.normalization_mode}")

        # Compute queries-keys similarities
        if self.similarity_mode == "inner":
            similarities = torch.einsum("bm, am -> ba", x, anchors)
            
            if self.normalization_mode is None:
                similarities = similarities / math.sqrt(x.shape[-1])
                
        elif self.similarity_mode == "basis_change":
            similarities = torch.linalg.lstsq(anchors.T, x.T)[0].T
            
        else:
            raise ValueError(f"Similarity mode not supported: {self.similarity_mode}")
       

        return {
            "similarities": similarities,
            "original_anchors": original_anchors,
            "norm_anchors": anchors,
            "norm_batch": x
        }

    def decode(self, similarities: torch.Tensor, **kwargs):
        
        output = similarities

        # Normalize the output
        if self.output_normalization_mode is None:
            pass
        elif self.output_normalization_mode == "l2":
            output = F.normalize(output, p=2, dim=-1)
        elif self.output_normalization_mode == "batchnorm":
            output = self.outnorm(output)
        elif self.output_normalization_mode == "layernorm":
            output = self.outnorm(output)
        elif self.output_normalization_mode == "instancenorm":
            output = torch.einsum("lc -> cl", output)
            output = self.outnorm(output)
            output = torch.einsum("cl -> lc", output)
        else:
            assert False

        return {
            "output": output,
            "similarities": similarities,
            "original_anchors": kwargs["original_anchors"],
            "norm_anchors": kwargs["norm_anchors"],
            "norm_batch": kwargs["norm_batch"]
        }

    def forward(self, x: torch.Tensor, anchors: torch.Tensor):
        """Forward pass.

        Args:
            x: [batch_size, hidden_dim]
            anchors: [num_anchors, hidden_dim]
            anchors_targets: [num_anchors]
        """
        encoding = self.encode(x=x, anchors=anchors)
        return self.decode(**encoding)

    @property
    def output_dim(self) -> int:
        return self.n_anchors
      