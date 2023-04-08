
import torch
from torch import Tensor, nn

from relAttention import RelativeAttention

class RobertaClassificationHead(nn.Module):
    """
    https://github.com/huggingface/transformers/blob/68287689f2f0d8b7063c400230b3766987abf18d/src/transformers/models/roberta/modeling_roberta.py#L1439
    Head for sentence-level classification tasks.
    """

    def __init__(self, hidden_size, num_labels, hidden_dropout_prob=0):
        super().__init__()
        
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.out_proj = nn.Linear(hidden_size, num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

    

class RelRoberta(nn.Module):
    def __init__(
        self,
        input_size,
        num_labels,
        transformer_model,
        anchors,
        similarity_mode="inner",
        normalization_mode="l2",
        output_normalization_mode=None,
        hidden_dropout_prob=0
    ) -> None:
        
        super().__init__()

        self.input_size = input_size
        self.latent_dim = anchors.shape[0]
        
        self.encoder = transformer_model
        
        self.decoder = RobertaClassificationHead(
            hidden_size = self.latent_dim,
            num_labels=num_labels,
            hidden_dropout_prob=hidden_dropout_prob
        )
        
        
        self.relative_attention=RelativeAttention(
            n_anchors=self.latent_dim,
            similarity_mode=similarity_mode,
            normalization_mode=normalization_mode,
            output_normalization_mode=output_normalization_mode
        )

        self.anchors = anchors
        
        with torch.no_grad():
            self.anchors_latent = self.embed(self.anchors)
      

    def embed(self, encoding):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(**encoding)["hidden_states"][-1]
        return result
    
    
    def encode(self, x): 
        
        x_embedded = self.embed(x)

        attention_encoding = self.relative_attention.encode(
            x=x_embedded,
            anchors=self.anchors_latent,
        )
        return {
            **attention_encoding,
            "batch_latent": x_embedded,
        }
        
    
    def decode(self, **kwargs):

        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        attention_output = self.relative_attention.decode(**kwargs)
        result = self.decoder(attention_output["output"])
        return {
            "prediction": result,
            "similarities": attention_output["similarities"],
            "original_anchors": attention_output["original_anchors"],
            "norm_anchors": attention_output["norm_anchors"],
            "batch_latent": attention_output["batch_latent"],
            "norm_batch": attention_output["norm_batch"]
        }
    
    def forward(self, x: torch.Tensor):
        """Forward pass.

        Args:
            x: [batch_size, hidden_dim]
            anchors: [num_anchors, hidden_dim]
            anchors_targets: [num_anchors]
        """
        encoding = self.encode(x)
        return self.decode(**encoding)



class VanillaRelRoberta(nn.Module):
    def __init__(
        self,
        input_size,
        num_labels,
        transformer_model,
        hidden_size,
        hidden_dropout_prob=0
    ) -> None:
        
        super().__init__()

        self.input_size = input_size
        self.latent_dim = hidden_size
        
        self.encoder = transformer_model
        
        self.decoder = RobertaClassificationHead(
            hidden_size = self.latent_dim,
            num_labels=num_labels,
            hidden_dropout_prob=hidden_dropout_prob
        )
        
      

    def embed(self, encoding):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(**encoding)["hidden_states"][-1]
        return result
    
    
    def encode(self, x): 
        
        x_embedded = self.embed(x)

      
        return {
            "batch_latent": x_embedded,
        }
        
    
    def decode(self, input):

        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder(input)
        return {
            "prediction": result
        }
    
    def forward(self, x: torch.Tensor):
        """Forward pass.

        Args:
            x: [batch_size, hidden_dim]
            anchors: [num_anchors, hidden_dim]
            anchors_targets: [num_anchors]
        """
        encoding = self.encode(x)
        return self.decode(encoding)