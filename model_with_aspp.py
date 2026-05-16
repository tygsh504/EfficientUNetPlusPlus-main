"""
Custom EfficientUNetPlusPlus model with ASPP module at the bottleneck.
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
from aspp import ASPP
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

class EfficientUNetPlusPlusWithASPP(SegmentationModel):
    """
    EfficientUNetPlusPlus model with ASPP (Atrous Spatial Pyramid Pooling) module
    inserted at the bottleneck between encoder and decoder.
    
    The ASPP module captures multi-scale contextual information, which helps improve
    segmentation performance especially for leaf disease detection with varying leaf sizes.
    
    Args:
        encoder_name: Name of the classification model used as encoder
        encoder_depth: Number of stages of the encoder (3-5)
        encoder_weights: Pretrained weights ("imagenet" or None)
        decoder_channels: List of output channels for decoder blocks
        squeeze_ratio: Squeeze ratio for inverted residual blocks
        expansion_ratio: Expansion ratio for inverted residual blocks
        in_channels: Number of input channels (default: 3 for RGB)
        classes: Number of output classes (default: 1 for binary segmentation)
        activation: Activation function after final conv
        aux_params: Auxiliary classification head parameters
        aspp_out_channels: Output channels for ASPP module (default: same as deepest encoder channel)
        aspp_rates: List of atrous convolution rates for ASPP (default: [6, 12, 18])
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
        aspp_out_channels: Optional[int] = None,
        aspp_rates: List[int] = [6, 12, 18],
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
        
        # Set ASPP output channels to match the deepest encoder channel if not specified
        if aspp_out_channels is None:
            aspp_out_channels = self.encoder.out_channels[-1]
        
        # Initialize ASPP module at bottleneck
        # ASPP input is the deepest encoder feature
        aspp_in_channels = self.encoder.out_channels[-1]
        self.aspp = ASPP(
            in_channels=aspp_in_channels,
            out_channels=aspp_out_channels,
            atrous_rates=aspp_rates,
        )
        
        # Modify encoder channels list to account for ASPP output
        # The decoder receives ASPP output instead of direct encoder output at bottleneck
        encoder_channels_with_aspp = list(self.encoder.out_channels[:-1]) + [aspp_out_channels]
        
        # Initialize decoder
        self.decoder = EfficientUnetPlusPlusDecoder(
            encoder_channels=encoder_channels_with_aspp,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            squeeze_ratio=squeeze_ratio,
            expansion_ratio=expansion_ratio,
        )
        
        if self.attention_type == 'cbam':
            self.attention = CBAM(aspp_out_channels)
        elif self.attention_type == 'ca':
            self.attention = CoordAtt(aspp_out_channels, aspp_out_channels)
        elif self.attention_type == 'se':
            self.attention = SEBlock(aspp_out_channels)
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
        
        self.name = f"EfficientUNet++-ASPP-{encoder_name}"
        self.initialize()

    def forward(self, x):
        """
        Forward pass with ASPP at bottleneck.
        
        Flow:
        1. Encoder extracts features at multiple scales
        2. Deepest feature is passed through ASPP for multi-scale context
        3. All features (with ASPP-processed deepest feature) go to decoder
        4. Decoder upsamples and refines features
        5. Segmentation head produces final output
        
        Args:
            x: Input tensor (B, C, H, W)
            
        Returns:
            Output masks (B, classes, H, W)
            Optionally also returns classification labels if aux_params used
        """
        # Get features from encoder
        features = self.encoder(x)
        
        # Apply ASPP to the deepest (bottleneck) feature
        # features is a tuple of feature maps from different scales
        # We modify the deepest one (features[-1]) through ASPP
        features_list = list(features)
        aspp_out = self.aspp(features[-1])
        
        if self.attention is not None:
            aspp_out = self.attention(aspp_out)
            
        aspp_out = self.spatial_dropout(aspp_out)
            
        features_list[-1] = aspp_out
        features = tuple(features_list)
        
        # Pass modified features to decoder
        decoder_output = self.decoder(*features)
        
        # Get segmentation output
        masks = self.segmentation_head(decoder_output)
        
        # Optional: get classification output if auxiliary head exists
        if self.classification_head is not None:
            labels = self.classification_head(features[-1])
            return masks, labels
        
        return masks
