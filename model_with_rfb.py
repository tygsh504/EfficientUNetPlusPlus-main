"""
Custom EfficientUNetPlusPlus model with Receptive Field Block (RFB) at the bottleneck.
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
from segmentation_models_pytorch.segmentation_models_pytorch.efficientunetplusplus.model import CBAM


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes) if bn else nn.Identity()
        self.relu = nn.ReLU(inplace=True) if relu else nn.Identity()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class RFB(nn.Module):
    """Receptive Field Block"""
    def __init__(self, in_channels, out_channels):
        super(RFB, self).__init__()
        inter_channels = in_channels // 4

        # Branch 0: 1x1 conv
        self.branch0 = BasicConv(in_channels, inter_channels, kernel_size=1)

        # Branch 1: 1x1 conv + 3x3 conv
        self.branch1 = nn.Sequential(
            BasicConv(in_channels, inter_channels, kernel_size=1),
            BasicConv(inter_channels, inter_channels, kernel_size=3, padding=1)
        )

        # Branch 2: 1x1 conv + 3x3 conv + 3x3 dilated conv (dilation=3)
        self.branch2 = nn.Sequential(
            BasicConv(in_channels, inter_channels, kernel_size=1),
            BasicConv(inter_channels, inter_channels, kernel_size=3, padding=1),
            BasicConv(inter_channels, inter_channels, kernel_size=3, padding=3, dilation=3)
        )

        # Branch 3: 1x1 conv + 3x3 conv + 3x3 conv + 3x3 dilated conv (dilation=5)
        self.branch3 = nn.Sequential(
            BasicConv(in_channels, inter_channels, kernel_size=1),
            BasicConv(inter_channels, inter_channels, kernel_size=3, padding=1),
            BasicConv(inter_channels, inter_channels, kernel_size=3, padding=1),
            BasicConv(inter_channels, inter_channels, kernel_size=3, padding=5, dilation=5)
        )

        self.conv_cat = BasicConv(inter_channels * 4, out_channels, kernel_size=1, relu=False)
        self.conv_res = BasicConv(in_channels, out_channels, kernel_size=1, relu=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        
        out = torch.cat((x0, x1, x2, x3), dim=1)
        out = self.conv_cat(out)
        res = self.conv_res(x)
        return self.relu(out + res)


class EfficientUNetPlusPlusWithRFB(SegmentationModel):
    """
    EfficientUNetPlusPlus model with Receptive Field Block (RFB) module
    inserted at the bottleneck between encoder and decoder.
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
        rfb_out_channels: Optional[int] = None,
        use_cbam: bool = True,
    ):
        super().__init__()
        
        self.use_cbam = use_cbam
        self.classes = classes
        
        # Initialize encoder
        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )
        
        # Set RFB output channels to match the deepest encoder channel if not specified
        if rfb_out_channels is None:
            rfb_out_channels = self.encoder.out_channels[-1]
        
        # Initialize RFB module at bottleneck
        rfb_in_channels = self.encoder.out_channels[-1]
        self.rfb = RFB(
            in_channels=rfb_in_channels,
            out_channels=rfb_out_channels,
        )
        
        # Modify encoder channels list to account for RFB output
        encoder_channels_with_rfb = list(self.encoder.out_channels[:-1]) + [rfb_out_channels]
        
        # Initialize decoder
        self.decoder = EfficientUnetPlusPlusDecoder(
            encoder_channels=encoder_channels_with_rfb,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            squeeze_ratio=squeeze_ratio,
            expansion_ratio=expansion_ratio,
        )
        
        if self.use_cbam:
            # CBAM module immediately after RFB
            self.cbam = CBAM(rfb_out_channels)

        # Segmentation head
        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )
        
        # Optional classification head
        if aux_params is not None:
            from segmentation_models_pytorch.segmentation_models_pytorch.base import ClassificationHead
            self.classification_head = ClassificationHead(
                in_channels=self.encoder.out_channels[-1], **aux_params
            )
        else:
            self.classification_head = None
        
        self.name = f"EfficientUNet++-RFB-{encoder_name}"
        self.initialize()

    def forward(self, x):
        # Get features from encoder
        features = self.encoder(x)
        
        # Apply RFB to the deepest (bottleneck) feature
        features_list = list(features)
        rfb_out = self.rfb(features[-1])
        
        if self.use_cbam:
            rfb_out = self.cbam(rfb_out)
            
        features_list[-1] = rfb_out
        features = tuple(features_list)
        
        # Pass modified features to decoder
        decoder_output = self.decoder(*features)
        
        # Get segmentation output
        masks = self.segmentation_head(decoder_output)
        
        if self.classification_head is not None:
            labels = self.classification_head(features[-1])
            return masks, labels
        
        return masks
