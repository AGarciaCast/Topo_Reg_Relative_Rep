"""
Modified from https://openreview.net/attachment?id=SrC-nwieGJ&name=supplementary_material
"""

import math

import torch
import torch.nn.functional as F
from torch import nn


class RelativeAttention(nn.Module):
    def __init__(
        self,
        n_anchors: int,
        similarity_mode="inner",
        normalization_mode=None,
        output_normalization_mode=None,
        in_features = None,
    ):
        """
        Relative attention block.

        Args:
            n_anchors: Number of anchors.
            similarity_mode: How to compute similarities: inner, basis_change.
            normalization_mode: Normalization to apply to the anchors and batch before computing the transformation.
            output_normalization_mode: Normalization to apply to the relative transformation.
            in_features: Number of input features for normalization (if normalization_mode is batchnorm).
        """
        super().__init__()

        self.n_anchors = n_anchors
        self.similarity_mode = similarity_mode
        self.normalization_mode = normalization_mode
        self.output_normalization_mode = output_normalization_mode
        
        if normalization_mode is not None and normalization_mode == "batchnorm" and in_features is not None:
            self.anchor_norm = nn.BatchNorm1d(num_features=in_features, affine=False, momentum=0.3)
            self.x_norm = nn.BatchNorm1d(num_features=in_features, affine=False, momentum=0.3)
        
        if self.output_normalization_mode == "batchnorm":
            self.outnorm = nn.BatchNorm1d(num_features=self.output_dim)
        elif self.output_normalization_mode == "layernorm":
            self.outnorm = nn.LayerNorm(normalized_shape=self.output_dim)
        elif self.output_normalization_mode == "instancenorm":
            self.outnorm = nn.InstanceNorm1d(num_features=self.output_dim, affine=True)

    def encode(self, x: torch.Tensor, anchors: torch.Tensor):
        """
        Encode the input tensor and anchors to compute the queries-keys similarities.

        Args:
            x: Input tensor of shape [batch_size, hidden_dim].
            anchors: Anchors tensor of shape [num_anchors, hidden_dim].

        Returns:
            Dictionary containing the encoded information including similarities, original anchors, normalized anchors, and normalized batch.
        """
        
        original_anchors = anchors
        if x.shape[-1] != anchors.shape[-1]:
            raise ValueError(f"Inconsistent dimensions between batch and anchors: {x.shape}, {anchors.shape}")
        
        
       
        # Normalize latents
        if self.normalization_mode is None:
            pass
        elif self.normalization_mode == "batchnorm":
            x = self.anchor_norm(x)
            anchors = self.anchor_norm(anchors)
        else:
            raise ValueError(f"Normalization mode not supported: {self.normalization_mode}")
        
        x = F.normalize(x, p=2, dim=-1)
        anchors = F.normalize(anchors, p=2, dim=-1)


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
        """
        Decode the similarities and normalize the output.

        Args:
            similarities: Similarities tensor.

        Returns:
            Dictionary containing the decoded information including the output, similarities, original anchors, normalized anchors, and normalized batch.
        """
        
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
        """
        Forward pass of the RelativeAttention module.

        Args:
            x: Input tensor of shape [batch_size, hidden_dim].
            anchors: Anchors tensor of shape [num_anchors, hidden_dim].

        Returns:
            Dictionary containing the decoded information including the output, similarities, original anchors, normalized anchors, and normalized batch.
        """
        encoding = self.encode(x=x, anchors=anchors)
        return self.decode(**encoding)

    @property
    def output_dim(self) -> int:
        """
        Returns the output dimension of the RelativeAttention module.

        Returns:
            Output dimension.
        """
        return self.n_anchors

