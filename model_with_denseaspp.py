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
    ):
        super().__init__()
        
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
        
        self.cbam = CBAM(decoder_channels[-1])
        
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
        features_list[-1] = self.dense_aspp(features[-1])
        features = tuple(features_list)
        decoder_output = self.decoder(*features)
        decoder_output = self.cbam(decoder_output)
        masks = self.segmentation_head(decoder_output)
        return masks