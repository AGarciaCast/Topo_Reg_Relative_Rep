
from torch import nn


class CNN(nn.Module):
    
    def __init__(self, 
                 num_input_channels : int, 
                 base_channel_size : int, 
                 num_labs: int,
                 linear_hidden_dims: list,
                 act_fn : object = nn.GELU):
        """
        Inputs: 
            - num_input_channels : Number of input channels of the image. For CIFAR, this parameter is 3
            - base_channel_size : Number of channels we use in the first convolutional layers. Deeper layers might use a duplicate of it.
            - latent_dim : Dimensionality of latent representation z
            - act_fn : Activation function used throughout the encoder network
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
        
        layers += [nn.Linear(layer_sizes[-1], num_labs)]
        
        self.net = nn.ModuleList(layers)
    
    
    def forward(self, x):
        for blk in self.net:
            x = blk(x)
            
        return x
    
    def detailed_forward(self, x):
        res = [x]
        for blk in self.net:
            x = blk(x)
            res.append(x.detach().reshape(x.shape[0], -1))
            
        return res