# Copyright 2025 The MathWorks, Inc.
import torch.nn as nn
import numpy as np

class DeepRxResidualBlock(nn.Module):
    """
    The residual blocks combine ideas from ResNet-v2 and Xception:
    
    1. ResNet with full pre-activation (ResNet-v2):
       - Reference: He, K., Zhang, X., Ren, S., and Sun, J. 
       - Title: "Identity mappings in deep residual networks"
       - Source: European Conference on Computer Vision, pp. 630–645, Springer, 2016.
    
    2. Xception:
       - Reference: F. Chollet
       - Title: "Xception: Deep Learning with Depthwise Separable Convolutions"
       - Source: 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 
         Honolulu, HI, USA, 2017, pp. 1800-1807.
       - DOI: 10.1109/CVPR.2017.195
    """
    def __init__(self, idx, list_channels, list_dilation, is_projection=False, projection_factor=2):
        """
        Construct the DeepRx architecture based on Table I in [1], featuring 11 residual blocks.
        The depth multiplier value for grouped convolutions is set to 2.
        
        Reference:
            [1] M. Honkala, D. Korpi, and J. M. J. Huttunen, "DeepRx: Fully Convolutional Deep Learning Receiver,"
            IEEE Transactions on Wireless Communications, vol. 20, no. 6, pp. 3925-3940, June 2021.
            DOI: 10.1109/TWC.2021.3054520.
        
        Args:
            idx (int): Index for the number of channels and dilation factor.
            list_channels (list of int): List specifying the number of channels for each residual block.
            list_dilations (list of int): List specifying the dilation factors for each residual block.
            is_projection (bool): Flag indicating whether to use a projection shortcut.
            projection_factor (float): Factor used in the projection shortcut.
        
        Returns:
            torch.nn.Module: The constructed DeepRx model.
        """
        super(DeepRxResidualBlock, self).__init__()

        # Common input channels and dilation factors
        in_channel = list_channels[idx-1]
        d_factor   = list_dilation[idx-1]
        
        # Connection input channels
        prev_channel = is_projection*(list_channels[idx-2])+(1-is_projection)*in_channel
        
        # Main path components
        self.relu         = nn.ReLU() # common activation
        self.relu_res     = nn.ReLU() # connection activation
        self.bn           = nn.BatchNorm2d(in_channel) # common bn
        self.bn_res       = nn.BatchNorm2d(prev_channel) # connection bn
        self.conv1_3x3sep = nn.Conv2d(prev_channel, 2*prev_channel, 3, groups=prev_channel, dilation=d_factor, stride=1, padding='same')
        self.conv2_1x1    = nn.Conv2d(2*prev_channel, in_channel, 1, stride=1, padding=0)
        self.conv3_3x3sep = nn.Conv2d(in_channel, 2*in_channel, 3, groups=in_channel, dilation=d_factor, stride=1, padding='same')
        self.conv4_1x1    = nn.Conv2d(2*in_channel, in_channel, 1, stride=1, padding=0)

        # Shortcut path components (projection or identity)
        if is_projection:
            self.projection = True
            self.shortcut = nn.Conv2d(prev_channel, in_channel, 1, stride=1, padding=0) # projection shortcut
        else:
            self.projection = False
    
    def forward(self, x):

        # Forward pass
        if self.projection:
            # Main path
            y = self.bn_res(x)
            z = self.relu_res(y)
            y = self.conv1_3x3sep(z)
            y = self.conv2_1x1(y)
            y = self.bn(y)
            y = self.relu(y)
            y = self.conv3_3x3sep(y)
            y = self.conv4_1x1(y)

            # Projection shortcut
            shortcut = self.shortcut(z)
        else:
            # Main path
            y = self.bn_res(x)
            y = self.relu_res(y)
            y = self.conv1_3x3sep(y)
            y = self.conv2_1x1(y)
            y = self.bn(y)
            y = self.relu(y)
            y = self.conv3_3x3sep(y)
            y = self.conv4_1x1(y)

            # Identity shortcut
            shortcut = x

        # Addition
        y += shortcut
        
        return y

class DeepRx(nn.Module):
    """
    DeepRx Model

    This model implements the DeepRx architecture, which processes input data through a series
    of convolutional and residual blocks to produce an output suitable for deep learning receivers.

    Model Input and Output Shapes:
    
    Model Input:
        [n_subcarriers, n_symbols, 4*n_rx+2]
        Example: [312, 14, 10]
    
    Model Output:
        [n_subcarriers, n_symbols, n_bits]
        Example: [312, 14, 4]
    """
    def __init__(self, in_size:np.int32, out_size:int):
        super().__init__()
        
        # Input convolution
        self.conv_in = nn.Conv2d(in_size[-1], 64, 3, stride=1, padding='same') # Filter: (3,3) | Dilation: (1,1)

        # DeepRx Residual blocks
        num_filters = [64, 64, 128, 128, 256, 256, 256, 128, 128, 64, 64]
        dilation_factors = [(1, 1), (1, 1), (2, 3), (2, 3), (2, 3), (3, 6), (2, 3), (2, 3), (2, 3), (1, 1), (1, 1)]

        # DeepRxResidualBlock(num_filters, dilation_factor, is_projection)
        self.resnet_block_1  = DeepRxResidualBlock(1,  num_filters, dilation_factors, is_projection=False)
        self.resnet_block_2  = DeepRxResidualBlock(2,  num_filters, dilation_factors, is_projection=False)
        self.resnet_block_3  = DeepRxResidualBlock(3,  num_filters, dilation_factors, is_projection=True)
        self.resnet_block_4  = DeepRxResidualBlock(4,  num_filters, dilation_factors, is_projection=False)
        self.resnet_block_5  = DeepRxResidualBlock(5,  num_filters, dilation_factors, is_projection=True)
        self.resnet_block_6  = DeepRxResidualBlock(6,  num_filters, dilation_factors, is_projection=False)
        self.resnet_block_7  = DeepRxResidualBlock(7,  num_filters, dilation_factors, is_projection=False)
        self.resnet_block_8  = DeepRxResidualBlock(8,  num_filters, dilation_factors, is_projection=True)
        self.resnet_block_9  = DeepRxResidualBlock(9,  num_filters, dilation_factors, is_projection=False)
        self.resnet_block_10 = DeepRxResidualBlock(10, num_filters, dilation_factors, is_projection=True)
        self.resnet_block_11 = DeepRxResidualBlock(11, num_filters, dilation_factors, is_projection=False)
        
        # Output layers
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.conv_out = nn.Conv2d(64, out_size[-1], 1) # Filter: (1,1) | Dilation: (1,1)

    def forward(self, x):
        x = self.conv_in(x) # Input convolution
        x = self.resnet_block_1(x) # begin residual blocks
        x = self.resnet_block_2(x)
        x = self.resnet_block_3(x)
        x = self.resnet_block_4(x)
        x = self.resnet_block_5(x)
        x = self.resnet_block_6(x)
        x = self.resnet_block_7(x)
        x = self.resnet_block_8(x)
        x = self.resnet_block_9(x)
        x = self.resnet_block_10(x)
        x = self.resnet_block_11(x) # end residual blocks
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv_out(x) # Output convolution
        return x