"""
Atrous Spatial Pyramid Pooling (ASPP) module for semantic segmentation.
Can be inserted at the bottleneck between encoder and decoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ASPPConv(nn.Sequential):
    """Atrous convolution block with batch norm and activation."""
    
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    """Image-level features with global average pooling and upsampling."""
    
    def __init__(self, in_channels, out_channels):
        modules = [
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        super(ASPPPooling, self).__init__(*modules)

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling module.
    
    ASPP uses multiple atrous (dilated) convolutions with different rates to capture
    multi-scale contextual information. This is particularly useful at the bottleneck
    of encoder-decoder networks.
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        atrous_rates (list): List of atrous convolution rates. Default: [6, 12, 18]
    """
    
    def __init__(self, in_channels, out_channels, atrous_rates=[6, 12, 18]):
        super(ASPP, self).__init__()
        
        modules = []
        
        # 1x1 convolution
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ))
        
        # 3x3 atrous convolutions with different rates
        rates = atrous_rates
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))
        
        # Image pooling
        modules.append(ASPPPooling(in_channels, out_channels))
        
        self.convs = nn.ModuleList(modules)
        
        # Project concatenated features
        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, C, H, W)
        
        Returns:
            Output tensor of shape (B, out_channels, H, W)
        """
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)
