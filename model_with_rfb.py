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
        
        if self.attention_type == 'cbam':
            self.attention = CBAM(rfb_out_channels)
        elif self.attention_type == 'ca':
            self.attention = CoordAtt(rfb_out_channels, rfb_out_channels)
        elif self.attention_type == 'se':
            self.attention = SEBlock(rfb_out_channels)
        else:
            self.attention = None
            
        self.spatial_dropout = nn.Dropout2d(spatial_dropout) if spatial_dropout > 0.0 else nn.Identity()

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
        
        if self.attention is not None:
            rfb_out = self.attention(rfb_out)
            
        rfb_out = self.spatial_dropout(rfb_out)
            
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
