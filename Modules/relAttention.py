import abc
import logging
import math
from enum import auto
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn


#try:
# be ready for 3.10 when it drops
from enum import StrEnum
#except ImportError:
#from backports.strenum import StrEnum

pylogger = logging.getLogger(__name__)

class NormalizationMode(StrEnum):
    NONE = auto()
    L2 = auto()

class RelativeEmbeddingMethod(StrEnum):
    BASIS_CHANGE = auto()
    INNER = auto()


class OutputNormalization(StrEnum):
    NONE = auto()
    L2 = auto()
    BATCHNORM = auto()
    INSTANCENORM = auto()
    LAYERNORM = auto()


class AttentionOutput(StrEnum):
    NON_QUANTIZED_SIMILARITIES = auto()
    SUBSPACE_OUTPUTS = auto()
    ANCHORS_TARGETS = auto()
    ORIGINAL_ANCHORS = auto()
    ANCHORS = auto()
    NORM_BATCH = auto()
    ANCHORS_LATENT = auto()
    OUTPUT = auto()
    SIMILARITIES = auto()
    UNTRASFORMED_ATTENDED = auto()


class AbstractRelativeAttention(nn.Module, abc.ABC):
    @property
    @abc.abstractmethod
    def output_dim(self) -> int:
        raise NotImplementedError


class RelativeAttention(AbstractRelativeAttention):
    def __init__(
        self,
        n_anchors: int,
        similarity_mode: RelativeEmbeddingMethod,
        normalization_mode: Optional[NormalizationMode] = None,
        output_normalization_mode: Optional[OutputNormalization] = None,
    ):
        """Relative attention block.

        If values_mode = TRAINABLE we are invariant to batch-anchors rotations, if it is false we are equivariant
        to batch-anchors rotations

        Args:
            n_anchors: number of anchors
            similarity_mode: how to compute similarities: inner, basis_change
            normalization_mode: normalization to apply to the anchors and batch before computing the transformation
            output_normalization_mode: normalization to apply to the relative transformation
        """
        super().__init__()
        pylogger.info(f"Instantiating <{self.__class__.__qualname__}>")

        self.n_anchors = n_anchors
        self.similarity_mode = similarity_mode

        # Parameter validation
        self.normalization_mode = normalization_mode if normalization_mode is not None else NormalizationMode.NONE

        self.output_normalization_mode = (
            output_normalization_mode if output_normalization_mode is not None else OutputNormalization.NONE
        )

        if self.output_normalization_mode not in set(OutputNormalization):
            raise ValueError(f"Unknown output normalization mode {self.output_normalization_mode}")

        if self.similarity_mode == RelativeEmbeddingMethod.BASIS_CHANGE and (
            self.n_anchors_sampling_per_class is not None and self.n_anchors_sampling_per_class > 1
        ):
            raise ValueError("The basis change is not deterministic with possibly repeated basis vectors")

        # End Parameter validation

        if self.output_normalization_mode == OutputNormalization.BATCHNORM:
            self.outnorm = nn.BatchNorm1d(num_features=self.output_dim)
        elif self.output_normalization_mode == OutputNormalization.LAYERNORM:
            self.outnorm = nn.LayerNorm(normalized_shape=self.output_dim)
        elif self.output_normalization_mode == OutputNormalization.INSTANCENORM:
            self.outnorm = nn.InstanceNorm1d(num_features=self.output_dim, affine=True)

    def encode(
        self,
        x: torch.Tensor,
        anchors: torch.Tensor,
    ):
        original_anchors = anchors
        if x.shape[-1] != anchors.shape[-1]:
            raise ValueError(f"Inconsistent dimensions between batch and anchors: {x.shape}, {anchors.shape}")


        # Normalize latents
        if self.normalization_mode == NormalizationMode.NONE:
            pass
        elif self.normalization_mode == NormalizationMode.L2:
            x = F.normalize(x, p=2, dim=-1)
            anchors = F.normalize(anchors, p=2, dim=-1)
        else:
            raise ValueError(f"Normalization mode not supported: {self.normalization_mode}")

        # Compute queries-keys similarities
        if self.similarity_mode == RelativeEmbeddingMethod.INNER:
            similarities = torch.einsum("bm, am -> ba", x, anchors)
            if self.normalization_mode == NormalizationMode.NONE:
                similarities = similarities / math.sqrt(x.shape[-1])
        elif self.similarity_mode == RelativeEmbeddingMethod.BASIS_CHANGE:
            similarities = torch.linalg.lstsq(anchors.T, x.T)[0].T
        else:
            raise ValueError(f"Similarity mode not supported: {self.similarity_mode}")
       

        return {
            AttentionOutput.SIMILARITIES: similarities,
            AttentionOutput.ANCHORS: anchors,
            AttentionOutput.ORIGINAL_ANCHORS: original_anchors,
            AttentionOutput.NORM_BATCH: x,
        }

    def decode(
        self,
        similarities,
        **kwargs,
    ):

        output = similarities
        

        # Normalize the output
        if self.output_normalization_mode == OutputNormalization.NONE:
            pass
        elif self.output_normalization_mode == OutputNormalization.L2:
            output = F.normalize(output, p=2, dim=-1)
        elif self.output_normalization_mode == OutputNormalization.BATCHNORM:
            output = self.outnorm(output)
        elif self.output_normalization_mode == OutputNormalization.LAYERNORM:
            output = self.outnorm(output)
        elif self.output_normalization_mode == OutputNormalization.INSTANCENORM:
            output = torch.einsum("lc -> cl", output)
            output = self.outnorm(output)
            output = torch.einsum("cl -> lc", output)
        else:
            assert False

        return {
            AttentionOutput.OUTPUT: output,
            AttentionOutput.UNTRASFORMED_ATTENDED: output,
            AttentionOutput.SIMILARITIES: similarities,
            AttentionOutput.ANCHORS_LATENT: kwargs[AttentionOutput.ANCHORS_LATENT],
            AttentionOutput.NORM_BATCH: kwargs[AttentionOutput.NORM_BATCH],
        }

    def forward(
        self,
        x: torch.Tensor,
        anchors: torch.Tensor,
        anchors_targets: Optional[torch.Tensor] = None,
    ) -> Dict[AttentionOutput, torch.Tensor]:
        """Forward pass.

        Args:
            x: [batch_size, hidden_dim]
            anchors: [num_anchors, hidden_dim]
            anchors_targets: [num_anchors]
        """
        encoding = self.encode(x=x, anchors=anchors, anchors_targets=anchors_targets)
        return self.decode(**encoding)

    @property
    def output_dim(self) -> int:
        return self.n_anchors
      