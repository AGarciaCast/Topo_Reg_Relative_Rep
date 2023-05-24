"""
Inspired from https://openreview.net/attachment?id=SrC-nwieGJ&name=supplementary_material
"""
import torch
from torch import nn
from typing import Optional, List
from tqdm import tqdm
import torch.nn.functional as F

from modules.relAttention import RelativeAttention
from transformers import RobertaModel, AutoConfig



class RobertaClassificationHead(nn.Module):
    def __init__(self, hidden_size, num_labels, linear=False, hidden_dropout_prob=0.1):
        """
        Roberta classification head module.

        Args:
            hidden_size: Size of the hidden layer.
            num_labels: Number of output labels.
            linear: Whether to use a linear layer for classification or a fully-connected network.
            hidden_dropout_prob: Dropout probability for the hidden layers.
        """
        super().__init__()
        self.linear = linear
        if not linear:
            self.pooler = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.Tanh(),
                nn.Dropout(hidden_dropout_prob)
            )

            self.net = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
                nn.Dropout(hidden_dropout_prob),
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
                nn.Dropout(hidden_dropout_prob),
                nn.Linear(hidden_size, num_labels)
            )

        else:
            self.net = nn.Sequential(
                nn.Linear(hidden_size, num_labels),
                )


    def forward(self, x, **kwargs):
        """
        Forward pass of the RobertaClassificationHead module.

        Args:
            x: Input tensor.
            **kwargs: Additional keyword arguments.

        Returns:
            Output tensor.
        """
        if not self.linear:
            x = self.pooler(x)
                
        x = self.net(x)
        return x


# -
class RelRoberta(nn.Module):
    def __init__(
        self,
        num_labels,
        transformer_model,
        anchor_dataloader,
        hidden_size=768,
        similarity_mode="inner",
        normalization_mode="batchnorm",
        output_normalization_mode=None,
        dropout_prob=0.1,
        freq_anchors=100,
        device="cpu",
        fine_tune=False,
        linear=False
    ) -> None:
        """
        Relative attention-based RoBERTa model for classification tasks.

        Args:
            num_labels: Number of output labels.
            transformer_model: Pretrained RoBERTa model name or path.
            anchor_dataloader: DataLoader for anchor samples.
            hidden_size: Size of the hidden layer (default: 768).
            similarity_mode: Mode for computing similarities (default: "inner").
            normalization_mode: Normalization mode for anchors and batch (default: "batchnorm").
            output_normalization_mode: Normalization mode for the relative transformation (default: None).
            dropout_prob: Dropout probability for the hidden layers (default: 0.1).
            freq_anchors: Frequency of updating anchor samples (default: 100).
            device: Device to use (default: "cpu").
            fine_tune: Whether to fine-tune the encoder (default: False).
            linear: Whether to use a linear layer for classification or a fully-connected network (default: False).
        """
        super().__init__()

        self.latent_dim = hidden_size
        
        configuration = AutoConfig.from_pretrained(transformer_model)
        configuration.hidden_dropout_prob = dropout_prob
        configuration.attention_probs_dropout_prob = dropout_prob
        configuration.output_hidden_states=True
        configuration.return_dict=True
            
        self.encoder = RobertaModel.from_pretrained(
                            pretrained_model_name_or_path = transformer_model, 
                            config = configuration,
                            add_pooling_layer=False)
        
        self.fine_tune = fine_tune
        if fine_tune:
            self.encoder = self.encoder.requires_grad_(False)

        
        self.decoder = RobertaClassificationHead(
            hidden_size = self.latent_dim,
            num_labels=num_labels,
            linear=linear,
            hidden_dropout_prob=dropout_prob
        )
        
        self.device = device
        self.anchor_dataloader = anchor_dataloader
        
        if anchor_dataloader is not None:
            self.relative_attention=RelativeAttention(
                n_anchors=self.latent_dim,
                similarity_mode=similarity_mode,
                normalization_mode=normalization_mode,
                output_normalization_mode=output_normalization_mode,
                in_features=self.latent_dim
            )
                        
            self.freq_anchors = freq_anchors

      

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
        """
        Extracts the hidden states from RoBERTa model.

        Args:
            input_ids: Input token IDs.
            attention_mask: Attention mask.
            token_type_ids: Token type IDs.
            position_ids: Positional IDs.
            head_mask: Head mask.
            inputs_embeds: Embedded inputs.
            encoder_hidden_states: Encoder hidden states.
            encoder_attention_mask: Encoder attention mask.
            past_key_values: Past key values.
            use_cache: Whether to use cache.
            output_attentions: Whether to output attentions.
            output_hidden_states: Whether to output hidden states.
            return_dict: Whether to return a dictionary.

        Returns:
            The extracted hidden states.
        """
        
        
        if self.fine_tune:
            self.encoder.eval()
        
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
                              return_dict)["hidden_states"][-1]
        
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
               return_dict: Optional[bool] = None,
               batch_idx=0): 
        """
        Encodes the input tokens and computes attention encoding.

        Args:
            input_ids: Input token IDs.
            attention_mask: Attention mask.
            token_type_ids: Token type IDs.
            position_ids: Positional IDs.
            head_mask: Head mask.
            inputs_embeds: Embedded inputs.
            encoder_hidden_states: Encoder hidden states.
            encoder_attention_mask: Encoder attention mask.
            past_key_values: Past key values.
            use_cache: Whether to use cache.
            output_attentions: Whether to output attentions.
            output_hidden_states: Whether to output hidden states.
            return_dict: Whether to return a dictionary.
            batch_idx: Batch index.

        Returns:
           Dictionary containing the batch latent representation, similarities, original anchors, normalized anchors, and normalized batch.
        """
        
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
        
        if self.anchor_dataloader is not None:
            
            if batch_idx % self.freq_anchors == 0:
                anchors_list = []
                with torch.no_grad():
                    for batch in self.anchor_dataloader:
                        batch.to(self.device)
                        batch_latents = self.embed(**batch)
                        anchors_list.append(batch_latents)

                self.anchors_latents = torch.cat(anchors_list, dim=0).to(self.device)
                
            attention_encoding = self.relative_attention.encode(
                x=x_embedded,
                anchors=self.anchors_latents,
            )
            
        return {
            **attention_encoding,
            "batch_latent": x_embedded,
        }
        
    
    def decode(self, batch_latent, **kwargs):
        """
        Decodes the batch relative latent representation and computes the prediction.

        Args:
            batch_latent: Batch latent representation.
            **kwargs: Additional keyword arguments.

        Returns:
            Dictionary containing the prediction, the batch latent representation, similarities, normalized similarities,  original anchors, normalized anchors, and normalized batch.
        """
        
        attention_dict = {}
        if self.anchor_dataloader is not None:
            attention_output = self.relative_attention.decode(**kwargs)
            
            input_dec = attention_output["output"]
            attention_dict={
                "similarities": attention_output["similarities"],
                "norm_similarities": input_dec,
                "original_anchors": attention_output["original_anchors"],
                "norm_anchors": attention_output["norm_anchors"],
                "norm_batch": attention_output["norm_batch"]
            }
            
        else:
            input_dec = batch_latent
            
        result = self.decoder(input_dec)
        return {
            "prediction": result,
            "batch_latent": batch_latent,
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
                return_dict: Optional[bool] = None,
                batch_idx=0): 
        """
        Forward pass of the RelRoberta model.

        Args:
            input_ids: Input token IDs.
            attention_mask: Attention mask.
            token_type_ids: Token type IDs.
            position_ids: Positional IDs.
            head_mask: Head mask.
            inputs_embeds: Embedded inputs.
            encoder_hidden_states: Encoder hidden states.
            encoder_attention_mask: Encoder attention mask.
            past_key_values: Past key values.
            use_cache: Whether to use cache.
            output_attentions: Whether to output attentions.
            output_hidden_states: Whether to output hidden states.
            return_dict: Whether to return a dictionary.
            batch_idx: Batch index.

        Returns:
            Dictionary containing the prediction, the batch latent representation, similarities, normalized similarities,  original anchors, normalized anchors, and normalized batch.
        """
        
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
                               return_dict,
                               batch_idx)
        
        return self.decode(**encoding)
