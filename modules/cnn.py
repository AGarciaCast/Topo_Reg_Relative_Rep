
import torch
from torch import nn


class CNN(nn.Module):
    
    def __init__(self, 
                 num_input_channels : int, 
                 base_channel_size : int, 
                 num_labs: int,
                 linear_hidden_dims: list,
                 act_fn : object = nn.GELU):
        """
        Initializes the CNN module.
        
        Inputs:
            - num_input_channels: Number of input channels of the image. For CIFAR, this parameter is 3.
            - base_channel_size: Number of channels used in the first convolutional layers. Deeper layers might use a duplicate of it.
            - num_labs: Number of output labels.
            - linear_hidden_dims: List of sizes for the hidden linear layers.
            - act_fn: Activation function used throughout the CNN network. Default is nn.GELU.
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
        """
        Forward pass of the CNN module.
        
        Inputs:
            - x: Input tensor.
        
        Returns:
            - Output tensor after passing through the CNN module.
        """
        for blk in self.net:
            x = blk(x)
            
        return x
    
    def detailed_forward(self, x):
        """
        Detailed forward pass of the CNN module.
        
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



class SimpleCNN13(nn.Module):
    """
    Simple CNN model with 13 convolutional layers.
    
    Reference: https://github.com/c-hofer/topologically_densified_distributions/blob/master/core/models/simple_cnn.py
    """
    def __init__(self,
                 num_classes: int):
        """
        Initializes the SimpleCNN13 module.
        
        Inputs:
            - num_classes: Number of output classes.
        """
        super().__init__()
       
        self.feat_ext = nn.Sequential(
            nn.Conv2d(3, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, stride=2, padding=0),
            nn.Dropout(0.5),
            #
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, stride=2, padding=0),
            nn.Dropout(0.5),
            #
            nn.Conv2d(256, 512, 3, padding=0),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 256, 1, padding=0),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 128, 1, padding=0),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.AvgPool2d(6, stride=2, padding=0),
        )

        self.cls = nn.Linear(128, num_classes)


    def forward(self, x):
        """
        Forward pass of the SimpleCNN13 module.
        
        Inputs:
            - x: Input tensor.
        
        Returns:
            - Tuple containing the predicted class scores and the intermediate feature representation.
        """
        
        z = self.feat_ext(x)
        z = torch.flatten(z, 1)
        y_hat = self.cls(z)

        return y_hat, z
