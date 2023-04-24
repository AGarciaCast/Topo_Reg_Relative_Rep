"""
Extracted from https://openreview.net/attachment?id=SrC-nwieGJ&name=supplementary_material
"""
from enum import auto


try:
# be ready for 3.10 when it drops
    from enum import StrEnum
except ImportError:
    from backports.strenum import StrEnum


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

class Output(StrEnum):
    NON_QUANTIZED_SIMILARITIES = auto()
    SIMILARITIES = auto()
    LOGITS = auto()
    RECONSTRUCTION = auto()
    DEFAULT_LATENT = auto()
    DEFAULT_LATENT_NORMALIZED = auto()
    BATCH_LATENT = auto()
    LATENT_MU = auto()
    LATENT_LOGVAR = auto()
    ANCHORS_LATENT = auto()
    INV_LATENTS = auto()
    LOSS = auto()
    BATCH = auto()
    ANCHORS_OUT = auto()
    INT_PREDICTIONS = auto()
    STR_PREDICTIONS = auto()
