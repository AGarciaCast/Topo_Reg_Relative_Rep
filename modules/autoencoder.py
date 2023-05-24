
from torch import nn


class Encoder(nn.Module):
    
    def __init__(self, 
                 num_input_channels : int, 
                 base_channel_size : int, 
                 latent_dim : int,
                 linear_hidden_dims: list, 
                 act_fn : object = nn.GELU):
        """
        Initializes the Encoder module.
        
        Inputs:
            - num_input_channels: Number of input channels of the image. For CIFAR, this parameter is 3.
            - base_channel_size: Number of channels used in the first convolutional layers. Deeper layers might use a duplicate of it.
            - latent_dim: Dimensionality of the latent representation z.
            - linear_hidden_dims: List of sizes for the hidden linear layers.
            - act_fn: Activation function used throughout the encoder network. Default is nn.GELU.
        """
        super().__init__()
        c_hid = base_channel_size
        layers =  [nn.Conv2d(num_input_channels, c_hid, kernel_size=3, padding=3, stride=2), # 28x28 => 16x16
            act_fn(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.Conv2d(c_hid, 2*c_hid, kernel_size=3, padding=1, stride=2), # 16x16 => 8x8
            act_fn(),
            nn.Conv2d(2*c_hid, 2*c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.Conv2d(2*c_hid, 2*c_hid, kernel_size=3, padding=1, stride=2), # 8x8 => 4x4
            act_fn(),
            nn.Flatten(), # Image grid to single feature vector
            ]
        
        layer_sizes = [2*16*c_hid] + linear_hidden_dims
        for layer_index in range(1, len(layer_sizes)):
            layers += [nn.Linear(layer_sizes[layer_index-1],
                                 layer_sizes[layer_index]),
                        act_fn()]
        
        layers += [nn.Linear(layer_sizes[-1], latent_dim),
                   nn.Identity()]
        
        self.net = nn.ModuleList(layers)
    
    
    def forward(self, x):
        """
        Forward pass of the Encoder module.
        
        Inputs:
            - x: Input tensor.
        
        Returns:
            - Output tensor after passing through the Encoder module.
        """
        
        for blk in self.net:
            x = blk(x)
            
        return x
    
    def detailed_forward(self, x):
        """
        Detailed forward pass of the Encoder module.
        
        Inputs:
            - x: Input tensor.
        
        Returns:
            - List of intermediate tensors at each step of the forward pass.
        """
        
        res = [x]
        for blk in self.net:
            x = blk(x)
            res.append(x.detach().reshape(x.shape[0], -1))
            
        return res


class Decoder(nn.Module):
    
    def __init__(self, 
                 num_input_channels : int, 
                 base_channel_size : int, 
                 latent_dim : int, 
                 linear_hidden_dims : list,
                 act_fn : object = nn.GELU):
        """
        Initializes the Decoder module.
        
        Inputs:
            - num_input_channels: Number of channels of the image to reconstruct. For CIFAR, this parameter is 3.
            - base_channel_size: Number of channels used in the last convolutional layers. Early layers might use a duplicate of it.
            - latent_dim: Dimensionality of the latent representation z.
            - linear_hidden_dims: List of sizes for the hidden linear layers.
            - act_fn: Activation function used throughout the decoder network. Default is nn.GELU.
        """
        super().__init__()
        c_hid = base_channel_size
        
        layer_sizes =  [latent_dim] + linear_hidden_dims + [2*16*c_hid]
        linear_layers = []
        for layer_index in range(1, len(layer_sizes)):
            linear_layers += [nn.Linear(layer_sizes[layer_index-1],
                                 layer_sizes[layer_index]),
                        act_fn()]
        
        self.linear = nn.ModuleList(linear_layers)
        
        self.net = nn.ModuleList([
            nn.ConvTranspose2d(2*c_hid, 2*c_hid, kernel_size=3, output_padding=1, padding=1, stride=2), # 4x4 => 8x8
            act_fn(),
            nn.Conv2d(2*c_hid, 2*c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.ConvTranspose2d(2*c_hid, c_hid, kernel_size=3, output_padding=1, padding=1, stride=2), # 8x8 => 16x16
            act_fn(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.ConvTranspose2d(c_hid, num_input_channels, kernel_size=3, output_padding=1, padding=3, stride=2), # 16x16 => 28x28
            nn.Tanh() # The input images is scaled between -1 and 1, hence the output has to be bounded as well
        ])

    def forward(self, x):
        """
        Forward pass of the Decoder module.
        
        Inputs:
            - x: Input tensor.
        
        Returns:
            - Output tensor after passing through the Decoder module.
        """
        
        for blk in self.linear:
            x = blk(x)
        
        x = x.reshape(x.shape[0], -1, 4, 4)
        
        for blk in self.net:
            x = blk(x)

        return x
    
    def detailed_forward(self, x):
        """
        Detailed forward pass of the Decoder module.
        
        Inputs:
            - x: Input tensor.
        
        Returns:
            - List of intermediate tensors at each step of the forward pass.
        """
        
        res = []
        
        for blk in self.linear:
            x = blk(x)
            res.append(x.detach())
            
        
        x = x.reshape(x.shape[0], -1, 4, 4)
        
        for blk in self.net:
            x = blk(x)
            res.append(x.detach().reshape(x.shape[0], -1))
            
        return res


class Autoencoder(nn.Module):
    
    def __init__(self, 
                 base_channel_size: int, 
                 latent_dim: int, 
                 linear_hidden_dims : list = [],
                 encoder_class : object = Encoder,
                 decoder_class : object = Decoder,
                 num_input_channels: int = 3, 
                 ):
        """
        Initializes the Autoencoder module.
        
        Inputs:
            - base_channel_size: Number of channels used in the encoder and decoder modules.
            - latent_dim: Dimensionality of the latent representation z.
            - linear_hidden_dims: List of sizes for the hidden linear layers in the encoder and decoder.
            - encoder_class: Encoder class to use for the Autoencoder. Default is Encoder.
            - decoder_class: Decoder class to use for the Autoencoder. Default is Decoder.
            - num_input_channels: Number of input channels of the image. For CIFAR, this parameter is 3.
        """
        super().__init__()
        
        # Creating encoder and decoder
        self.encoder = encoder_class(num_input_channels, base_channel_size, latent_dim, linear_hidden_dims)
        self.decoder = decoder_class(num_input_channels, base_channel_size, latent_dim, linear_hidden_dims[::-1])
    
    def forward(self, x):
        """
        Forward pass of the Autoencoder module.
        
        Inputs:
            - x: Input tensor.
        
        Returns:
            - Reconstructed input tensor.
        """
        
        z = self.encoder(x)
        x_hat = self.decoder(z)
        
        return x_hat 
        
    def detailed_forward(self, x):
        """
        Detailed forward pass of the Autoencoder module.
        
        Inputs:
            - x: Input tensor.
        
        Returns:
            - List of intermediate tensors from the encoder and the reconstructed input tensor.
        """
        
        z = self.encoder.detailed_forward(x)
        x_hat = self.decoder.detailed_forward(z[-1])
        return z + x_hat