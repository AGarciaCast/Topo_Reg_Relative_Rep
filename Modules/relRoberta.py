
import torch
from torch import nn
from typing import Optional, List
from tqdm import tqdm

from modules.relAttention import RelativeAttention
from transformers import RobertaModel, AutoConfig


class RobertaClassificationHead(nn.Module):
    """
    https://github.com/huggingface/transformers/blob/68287689f2f0d8b7063c400230b3766987abf18d/src/transformers/models/roberta/modeling_roberta.py#L1439
    Head for sentence-level classification tasks.
    """

    def __init__(self, hidden_size, num_labels, hidden_dropout_prob=0.1):
        super().__init__()
        
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.out_proj = nn.Linear(hidden_size, num_labels)

    def forward(self, features, **kwargs):
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class RelRoberta(nn.Module):
    def __init__(
        self,
        num_labels,
        transformer_model,
        anchor_dataloader,
        hidden_size=768,
        similarity_mode="inner",
        normalization_mode="l2",
        output_normalization_mode=None,
        dropout_prob=0.1,
        device="cpu"
    ) -> None:
        
        super().__init__()

        self.latent_dim = hidden_size
        
        configuration = AutoConfig.from_pretrained(transformer_model)
        configuration.hidden_dropout_prob = dropout_prob
        configuration.attention_probs_dropout_prob = dropout_prob
            
        self.encoder = RobertaModel.from_pretrained(
                            pretrained_model_name_or_path = transformer_model, 
                            config = configuration,
                            add_pooling_layer=False
                        ).to(device)

        
        self.decoder = RobertaClassificationHead(
            hidden_size = self.latent_dim,
            num_labels=num_labels,
            hidden_dropout_prob=dropout_prob
        )
        
        if anchor_dataloader is not None:
            self.relative_attention=RelativeAttention(
                n_anchors=self.latent_dim,
                similarity_mode=similarity_mode,
                normalization_mode=normalization_mode,
                output_normalization_mode=output_normalization_mode
            )

            anchors = []
            anchors_latent = []
            
            with torch.no_grad():
                for batch in tqdm(anchor_dataloader, desc="Computing latents anchors"):
                    anchors.append(batch)
                    batch.to(device)
                    batch_latents = self.embed(**batch)
                    anchors_latent.append(batch_latents)
                                
            self.anchors = torch.cat(anchors, dim=0)
            self.anchors_latent = torch.cat(anchors_latent, dim=0)
      

    def embed(self,
              input_ids: Optional[torch.Tensor] = None,
              attention_mask: Optional[torch.Tensor] = None,
              token_type_ids: Optional[torch.Tensor] = None,
              position_ids: Optional[torch.Tensor] = None,
              head_mask: Optional[torch.Tensor] = None,
              inputs_embeds: Optional[torch.Tensor] = None,
              encoder_hidden_states: Optional[torch.Tensor] = None,
              encoder_attention_mask: Optional[torch.Tensor] = None,
              past_key_values: Optional[List[torch.FloatTensor]] = None,
              use_cache: Optional[bool] = None,
              output_attentions: Optional[bool] = None,
              output_hidden_states: Optional[bool] = None,
              return_dict: Optional[bool] = None):
        
        result = self.encoder(input_ids,
                              attention_mask,
                              token_type_ids,
                              position_ids,
                              head_mask,
                              inputs_embeds,
                              encoder_hidden_states,
                              encoder_attention_mask,
                              past_key_values,
                              use_cache,
                              output_attentions,
                              output_hidden_states,
                              return_dict)[0]
        return result[:, 0, :]
    
    
    def encode(self, 
               input_ids: Optional[torch.Tensor] = None,
               attention_mask: Optional[torch.Tensor] = None,
               token_type_ids: Optional[torch.Tensor] = None,
               position_ids: Optional[torch.Tensor] = None,
               head_mask: Optional[torch.Tensor] = None,
               inputs_embeds: Optional[torch.Tensor] = None,
               encoder_hidden_states: Optional[torch.Tensor] = None,
               encoder_attention_mask: Optional[torch.Tensor] = None,
               past_key_values: Optional[List[torch.FloatTensor]] = None,
               use_cache: Optional[bool] = None,
               output_attentions: Optional[bool] = None,
               output_hidden_states: Optional[bool] = None,
               return_dict: Optional[bool] = None): 
        
        x_embedded = self.embed(input_ids,
                                attention_mask,
                                token_type_ids,
                                position_ids,
                                head_mask,
                                inputs_embeds,
                                encoder_hidden_states,
                                encoder_attention_mask,
                                past_key_values,
                                use_cache,
                                output_attentions,
                                output_hidden_states,
                                return_dict)
        
        attention_encoding = {}
        
        if self.anchors is not None:
            attention_encoding = self.relative_attention.encode(
                x=x_embedded,
                anchors=self.anchors_latent,
            )
            
        return {
            **attention_encoding,
            "batch_latent": x_embedded,
        }
        
    
    def decode(self, batch_latent, **kwargs):
        
        attention_dict = {}
        if self.anchors is not None:
            attention_output = self.relative_attention.decode(**kwargs)
            
            input_dec = attention_output["output"]
            attention_dict={
                "similarities": attention_output["similarities"],
                "original_anchors": attention_output["original_anchors"],
                "norm_anchors": attention_output["norm_anchors"],
                "batch_latent": attention_output["batch_latent"],
                "norm_batch": attention_output["norm_batch"]
            }
            
        else:
            input_dec = batch_latent
            
        result = self.decoder(input_dec)
        return {
            "prediction": result,
            **attention_dict
        }
    
    
    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                token_type_ids: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None,
                head_mask: Optional[torch.Tensor] = None,
                inputs_embeds: Optional[torch.Tensor] = None,
                encoder_hidden_states: Optional[torch.Tensor] = None,
                encoder_attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[torch.FloatTensor]] = None,
                use_cache: Optional[bool] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None): 
        """Forward pass."""
        
        encoding = self.encode(input_ids,
                               attention_mask,
                               token_type_ids,
                               position_ids,
                               head_mask,
                               inputs_embeds,
                               encoder_hidden_states,
                               encoder_attention_mask,
                               past_key_values,
                               use_cache,
                               output_attentions,
                               output_hidden_states,
                               return_dict)
        
        return self.decode(**encoding)
