import torch as torch 
import torch.nn as nn


import torch 
import torch.nn as nn
import numpy as np


class MLP(nn.Module):
    def __init__(self, input_size:int, action_size:int, hidden_size:int=512,non_linear:nn.Module=nn.ReLU):
        """
        input: tuple[int]
            The input size of the image, of shape (channels, height, width)
        action_size: int
            The number of possible actions
        hidden_size: int
            The number of neurons in the hidden layer

        This is a seperate class because it may be useful for the bonus questions
        """
        super(MLP, self).__init__()
        # ========== YOUR CODE HERE ==========
        # TODO:
        # self.linear1 = 
        # self.output = 
        # self.non_linear = 
        # ====================================
        
        self.linear1 = nn.Linear(in_features=input_size, out_features=hidden_size)
        self.non_linear = non_linear()
        self.output = nn.Linear(in_features=hidden_size, out_features=action_size)

        # ========== YOUR CODE ENDS ==========

    def forward(self, x:torch.Tensor)->torch.Tensor:
        # ========== YOUR CODE HERE ==========
        
        x = self.linear1(x)        # shape: (B, hidden_size)
        x = self.non_linear(x)
        x = self.output(x)         # shape: (B, action_size)

        # ========== YOUR CODE ENDS ==========
        return x

class Nature_Paper_Conv(nn.Module):
    """
    A class that defines a neural network with the following architecture:
    - 1 convolutional layer with 32 8x8 kernels with a stride of 4x4 w/ ReLU activation
    - 1 convolutional layer with 64 4x4 kernels with a stride of 2x2 w/ ReLU activation
    - 1 convolutional layer with 64 3x3 kernels with a stride of 1x1 w/ ReLU activation
    - 1 fully connected layer with 512 neurons and ReLU activation. 
    Based on 2015 paper 'Human-level control through deep reinforcement learning' by Mnih et al
    """
    def __init__(self, input_size:tuple[int], action_size:int,**kwargs):
        """
        input: tuple[int]
            The input size of the image, of shape (channels, height, width)
        action_size: int
            The number of possible actions
        **kwargs: dict
            additional kwargs to pass for stuff like dropout, etc if you would want to implement it
        """
        super(Nature_Paper_Conv, self).__init__()
        # ========== YOUR CODE HERE ==========
        c, h, w = input_size

        self.CNN = nn.Sequential(
            nn.Conv2d(in_channels=c,   out_channels=32, kernel_size=8, stride=4),  # CNN.0
            nn.ReLU(),                                                               # CNN.1
            nn.Conv2d(in_channels=32,  out_channels=64, kernel_size=4, stride=2),   # CNN.2
            nn.ReLU(),                                                               # CNN.3
            nn.Conv2d(in_channels=64,  out_channels=64, kernel_size=3, stride=1),   # CNN.4
            nn.ReLU(),                                                               # CNN.5
            nn.Flatten()                                                             # CNN.6
        )

        # Determine how many features come out of CNN before the FC layers
        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            out = self.CNN(dummy)
            flattened_size = out.shape[1]  # should be 64*7*7 = 3136 for (4,84,84) input


        self.MLP = MLP(input_size=flattened_size, action_size=action_size)
        # ========== YOUR CODE ENDS ==========

    def forward(self, x:torch.Tensor)->torch.Tensor:
        # ========== YOUR CODE HERE ==========

        x = self.CNN(x)   #  shape (batch, 3136)
        x = self.MLP(x)   #  shape (batch, action_size)
    
        # ========== YOUR CODE ENDS ==========
        return x
