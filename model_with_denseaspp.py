"""
Custom EfficientUNetPlusPlus model with DenseASPP module at the bottleneck.
"""

import sys
import torch

# Monkey-patch for timm 0.3.2 compatibility with PyTorch 2.0+
if not hasattr(torch, '_six'):
    import types, collections.abc
    torch._six = types.ModuleType('torch._six')
    torch._six.container_abcs = collections.abc
    sys.modules['torch._six'] = torch._six

from typing import Optional, Union, List
import torch.nn as nn
import segmentation_models_pytorch.segmentation_models_pytorch as smp
from segmentation_models_pytorch.segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.segmentation_models_pytorch.efficientunetplusplus.decoder import EfficientUnetPlusPlusDecoder
from segmentation_models_pytorch.segmentation_models_pytorch.base import SegmentationHead, SegmentationModel
from segmentation_models_pytorch.segmentation_models_pytorch.efficientunetplusplus.model import CBAM

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y) 
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        out = identity * a_w * a_h
        return out

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, max(1, channels // reduction), kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(1, channels // reduction), channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.fc(y)
        return x * y

class DenseASPPBlock(nn.Module):
    """Atrous convolution block that concatenates input with its output."""
    def __init__(self, in_channels, inter_channels, dilation):
        super(DenseASPPBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, inter_channels, 3, padding=dilation, dilation=dilation, bias=False)
        )

    def forward(self, x):
        out = self.conv(x)
        # Concatenate the input and the dilated convolution's output along the channel dimension
        return torch.cat([x, out], dim=1)


class DenseASPP(nn.Module):
    """
    Dense Atrous Spatial Pyramid Pooling module.
    
    Features are passed through a series of dilated convolutions, where each
    layer receives the concatenated outputs of all previous layers. This forms
    a dense feature pyramid, capturing contextual information at various scales.
    """
    def __init__(self, in_channels, out_channels, atrous_rates=[3, 6, 12, 18], d_feature0=256, d_feature1=64, dropout=0.1):
        super(DenseASPP, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, d_feature0, 1, bias=False),
            nn.BatchNorm2d(d_feature0),
            nn.ReLU(inplace=True)
        )
        
        self.blocks = nn.ModuleList()
        current_channels = d_feature0
        for rate in atrous_rates:
            self.blocks.append(DenseASPPBlock(current_channels, d_feature1, rate))
            current_channels += d_feature1
            
        self.project = nn.Sequential(
            nn.Conv2d(current_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.conv1(x)
        for block in self.blocks:
            x = block(x)
        return self.project(x)


class EfficientUNetPlusPlusWithDenseASPP(SegmentationModel):
    """
    EfficientUNetPlusPlus model with DenseASPP module inserted at the bottleneck.
    """
    
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
        denseaspp_out_channels: Optional[int] = None,
        denseaspp_rates: List[int] = [3, 6, 12, 18],
        attention_type: str = 'cbam',
        spatial_dropout: float = 0.0,
    ):
        super().__init__()
        
        self.attention_type = attention_type.lower() if attention_type else 'none'
        self.classes = classes
        
        # Initialize encoder
        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )
        
        if denseaspp_out_channels is None:
            denseaspp_out_channels = self.encoder.out_channels[-1]
        
        # Initialize DenseASPP module at bottleneck
        denseaspp_in_channels = self.encoder.out_channels[-1]
        self.dense_aspp = DenseASPP(
            in_channels=denseaspp_in_channels,
            out_channels=denseaspp_out_channels,
            atrous_rates=denseaspp_rates,
        )
        
        encoder_channels_with_denseaspp = list(self.encoder.out_channels[:-1]) + [denseaspp_out_channels]
        
        # Initialize decoder
        self.decoder = EfficientUnetPlusPlusDecoder(
            encoder_channels=encoder_channels_with_denseaspp,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            squeeze_ratio=squeeze_ratio,
            expansion_ratio=expansion_ratio,
        )
        
        if self.attention_type == 'cbam':
            self.attention = CBAM(denseaspp_out_channels)
        elif self.attention_type == 'ca':
            self.attention = CoordAtt(denseaspp_out_channels, denseaspp_out_channels)
        elif self.attention_type == 'se':
            self.attention = SEBlock(denseaspp_out_channels)
        else:
            self.attention = None
            
        self.spatial_dropout = nn.Dropout2d(spatial_dropout) if spatial_dropout > 0.0 else nn.Identity()
        
        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )
        
        self.classification_head = None
        
        self.name = f"EfficientUNet++-DenseASPP-{encoder_name}"
        self.initialize()

    def forward(self, x):
        features = self.encoder(x)
        features_list = list(features)
        dense_out = self.dense_aspp(features[-1])
        
        if self.attention is not None:
            dense_out = self.attention(dense_out)
            
        dense_out = self.spatial_dropout(dense_out)
            
        features_list[-1] = dense_out
        features = tuple(features_list)
        decoder_output = self.decoder(*features)
        masks = self.segmentation_head(decoder_output)
        return masks