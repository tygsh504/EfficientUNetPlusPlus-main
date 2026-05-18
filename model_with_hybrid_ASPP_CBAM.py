"""
Custom EfficientUNetPlusPlus model with Hybrid ASPP CBAM module at the bottleneck.
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

# Monkey-patch for timm 0.3.2 compatibility with PyTorch 2.0+
if not hasattr(torch, '_six'):
    import types, collections.abc
    torch._six = types.ModuleType('torch._six')
    torch._six.container_abcs = collections.abc
    sys.modules['torch._six'] = torch._six

from typing import Optional, Union, List
import segmentation_models_pytorch.segmentation_models_pytorch as smp
from segmentation_models_pytorch.segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.segmentation_models_pytorch.efficientunetplusplus.decoder import EfficientUnetPlusPlusDecoder
from segmentation_models_pytorch.segmentation_models_pytorch.base import SegmentationHead, SegmentationModel

class ChannelAttention(nn.Module):
    """Channel attention module to focus on 'what' is meaningful."""
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Shared Multi-Layer Perceptron (MLP)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        # Combine both pooled features
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """Spatial attention module to focus on 'where' is an informative part."""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Max and Avg pooling along the channel dimension
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        
        out = self.conv(x_cat)
        return self.sigmoid(out)


class CBAM(nn.Module):
    """Convolutional Block Attention Module"""
    def __init__(self, in_channels, reduction_ratio=16, spatial_kernel=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_channels, reduction_ratio)
        self.sa = SpatialAttention(spatial_kernel)

    def forward(self, x):
        # Apply channel attention then spatial attention
        out = x * self.ca(x)
        out = out * self.sa(out)
        return out


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


class AttentionASPP(nn.Module):
    """
    Hybrid ASPP Module with CBAM Attention.
    """
    def __init__(self, in_channels, out_channels, atrous_rates=[6, 12, 18], reduction_ratio=16):
        super(AttentionASPP, self).__init__()
        
        modules = []
        
        # 1x1 convolution
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ))
        
        # 3x3 atrous convolutions with different rates
        for rate in atrous_rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))
        
        # Image pooling
        modules.append(ASPPPooling(in_channels, out_channels))
        
        self.convs = nn.ModuleList(modules)
        
        # Number of channels after concatenation
        concat_channels = len(self.convs) * out_channels
        
        self.attention = CBAM(concat_channels, reduction_ratio=reduction_ratio)
        
        # Final Projection
        self.project = nn.Sequential(
            nn.Conv2d(concat_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        
        res = torch.cat(res, dim=1)
        res = self.attention(res)
        
        return self.project(res)


class EfficientUNetPlusPlusWithHybridASPPCBAM(SegmentationModel):
    def __init__(
        self,
        encoder_name: str = "timm-efficientnet-b0",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        squeeze_ratio: int = 1,
        expansion_ratio: int = 1,
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[Union[str, callable]] = None,
        aux_params: Optional[dict] = None,
        aspp_out_channels: Optional[int] = None,
        aspp_rates: List[int] = [6, 12, 18],
        spatial_dropout: float = 0.0,
    ):
        super().__init__()
        
        self.classes = classes
        
        self.encoder = get_encoder(encoder_name, in_channels=in_channels, depth=encoder_depth, weights=encoder_weights)
        if aspp_out_channels is None:
            aspp_out_channels = self.encoder.out_channels[-1]
        
        self.hybrid_aspp = AttentionASPP(in_channels=self.encoder.out_channels[-1], out_channels=aspp_out_channels, atrous_rates=aspp_rates)
        encoder_channels_with_hybrid_aspp = list(self.encoder.out_channels[:-1]) + [aspp_out_channels]
        self.decoder = EfficientUnetPlusPlusDecoder(encoder_channels=encoder_channels_with_hybrid_aspp, decoder_channels=decoder_channels, n_blocks=encoder_depth, squeeze_ratio=squeeze_ratio, expansion_ratio=expansion_ratio)
        self.spatial_dropout = nn.Dropout2d(spatial_dropout) if spatial_dropout > 0.0 else nn.Identity()
        self.segmentation_head = SegmentationHead(in_channels=decoder_channels[-1], out_channels=classes, activation=activation, kernel_size=3)
        self.classification_head = None
        
        self.name = f"EfficientUNet++-HybridASPPCBAM-{encoder_name}"
        self.initialize()

    def forward(self, x):
        features = self.encoder(x)
        features_list = list(features)
        
        hybrid_aspp_out = self.hybrid_aspp(features[-1])
        hybrid_aspp_out = self.spatial_dropout(hybrid_aspp_out)
            
        features_list[-1] = hybrid_aspp_out
        decoder_output = self.decoder(*tuple(features_list))
        
        return self.segmentation_head(decoder_output)