import torch
import torch.nn as nn
from typing import Optional, Union, List
from .decoder import EfficientUnetPlusPlusDecoder
from ..encoders import get_encoder
from ..base import SegmentationModel
from ..base import SegmentationHead, ClassificationHead
from torchvision import transforms

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        reduced_planes = max(1, in_planes // ratio)
        self.fc = nn.Sequential(nn.Conv2d(in_planes, reduced_planes, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(reduced_planes, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x

class EfficientUnetPlusPlus(SegmentationModel):
    """The EfficientUNet++ is a fully convolutional neural network for ordinary and medical image semantic segmentation. 
    Consists of an *encoder* and a *decoder*, connected by *skip connections*. The encoder extracts features of 
    different spatial resolutions, which are fed to the decoder through skip connections. The decoder combines its 
    own feature maps with the ones from skip connections to produce accurate segmentations masks.  The EfficientUNet++ 
    decoder architecture is based on the UNet++, a model composed of nested U-Net-like decoder sub-networks. To 
    increase performance and computational efficiency, the EfficientUNet++ replaces the UNet++'s blocks with 
    inverted residual blocks with depthwise convolutions and embedded spatial and channel attention mechanisms.
    Synergizes well with EfficientNet encoders. Due to their efficient visual representations (i.e., using few channels
    to represent extracted features), EfficientNet encoders require few computation from the decoder.

    Args:
        encoder_name: Name of the classification model that will be used as an encoder (a.k.a backbone) to extract features
        encoder_depth: Number of stages of the encoder, in range [3 ,5]. Each stage generate features two times smaller, 
            in spatial dimensions, than the previous one (e.g., for depth=0 features will haves shapes [(N, C, H, W)]), 
            for depth 1 features will have shapes [(N, C, H, W), (N, C, H // 2, W // 2)] and so on).
            Default is 5
        encoder_weights: One of **None** (random initialization), **"imagenet"** (pre-training on ImageNet) and 
            other pretrained weights (see table with available weights for each encoder_name)
        decoder_channels: List of integers which specify **in_channels** parameter for convolutions used in the decoder.
            Length of the list should be the same as **encoder_depth**
        in_channels: The number of input channels of the model, default is 3 (RGB images)
        classes: The number of classes of the output mask. Can be thought of as the number of channels of the mask
        activation: An activation function to apply after the final convolution layer.
            Available options are **"sigmoid"**, **"softmax"**, **"logsoftmax"**, **"tanh"**, **"identity"**, **callable** and **None**.
            Default is **None**
        aux_params: Dictionary with parameters of the auxiliary output (classification head). Auxiliary output is built 
            on top of encoder if **aux_params** is not **None** (default). Supported params:
                - classes (int): A number of classes
                - pooling (str): One of "max", "avg". Default is "avg"
                - dropout (float): Dropout factor in [0, 1)
                - activation (str): An activation function to apply "sigmoid"/"softmax" (could be **None** to return logits)
    Returns:
        ``torch.nn.Module``: **EfficientUnet++**

    Reference:
        https://arxiv.org/abs/2106.11447
    """

    def __init__(
        self,
        encoder_name: str = "timm-efficientnet-b5",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        squeeze_ratio: int = 1,
        expansion_ratio: int = 1,
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[Union[str, callable]] = None,
        aux_params: Optional[dict] = None,
    ):
        super().__init__()
        self.classes = classes
        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )

        self.decoder = EfficientUnetPlusPlusDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            squeeze_ratio=squeeze_ratio,
            expansion_ratio=expansion_ratio
        )

        self.cbam = CBAM(decoder_channels[-1])

        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )

        if aux_params is not None:
            self.classification_head = ClassificationHead(
                in_channels=self.encoder.out_channels[-1], **aux_params
            )
        else:
            self.classification_head = None

        self.name = "EfficientUNet++-{}".format(encoder_name)
        self.initialize()

    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        features = self.encoder(x)
        decoder_output = self.decoder(*features)
        
        # Apply CBAM before the segmentation head
        decoder_output = self.cbam(decoder_output)

        masks = self.segmentation_head(decoder_output)

        if self.classification_head is not None:
            labels = self.classification_head(features[-1])
            return masks, labels

        return masks

    def predict(self, x):
        """Inference method. Switch model to `eval` mode, call `.forward(x)` with `torch.no_grad()`

        Args:
            x: 4D torch tensor with shape (batch_size, channels, height, width)

        Return:
            prediction: 4D torch tensor with shape (batch_size, classes, height, width)

        """
        if self.training:
            self.eval()

        with torch.no_grad():
            output = self.forward(x)

        if self.classes > 1:
            probs = torch.softmax(output, dim=1)
        else:
            probs = torch.sigmoid(output)

        probs = probs.squeeze(0)
        tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(x.size[1]),
                transforms.ToTensor()
            ]
        )
        full_mask = tf(probs.cpu())   

        return full_mask