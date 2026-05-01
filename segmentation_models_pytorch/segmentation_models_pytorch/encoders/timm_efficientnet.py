import torch
import torch.nn as nn

from timm.models.efficientnet import EfficientNet
from timm.models.efficientnet import decode_arch_def, round_channels, default_cfgs
try:
    # For older timm versions
    from timm.models.layers.activations import Swish
except ImportError:
    try:
        # For newer timm versions (0.9.0+)
        from timm.layers import Swish
    except ImportError:
        # Fallback to PyTorch's native Swish equivalent
        import torch.nn as nn
        Swish = nn.SiLU

from ._base import EncoderMixin


def get_efficientnet_kwargs(channel_multiplier=1.0, depth_multiplier=1.0, drop_rate=0.2):
    """Creates an EfficientNet model.
    Ref impl: https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/efficientnet_model.py
    Paper: https://arxiv.org/abs/1905.11946
    EfficientNet params
    name: (channel_multiplier, depth_multiplier, resolution, dropout_rate)
    'efficientnet-b0': (1.0, 1.0, 224, 0.2),
    'efficientnet-b1': (1.0, 1.1, 240, 0.2),
    'efficientnet-b2': (1.1, 1.2, 260, 0.3),
    'efficientnet-b3': (1.2, 1.4, 300, 0.3),
    'efficientnet-b4': (1.4, 1.8, 380, 0.4),
    'efficientnet-b5': (1.6, 2.2, 456, 0.4),
    'efficientnet-b6': (1.8, 2.6, 528, 0.5),
    'efficientnet-b7': (2.0, 3.1, 600, 0.5),
    'efficientnet-b8': (2.2, 3.6, 672, 0.5),
    'efficientnet-l2': (4.3, 5.3, 800, 0.5),
    Args:
      channel_multiplier: multiplier to number of channels per layer
      depth_multiplier: multiplier to number of repeats per stage
    """
    arch_def = [
        ['ds_r1_k3_s1_e1_c16_se0.25'],
        ['ir_r2_k3_s2_e6_c24_se0.25'],
        ['ir_r2_k5_s2_e6_c40_se0.25'],
        ['ir_r3_k3_s2_e6_c80_se0.25'],
        ['ir_r3_k5_s1_e6_c112_se0.25'],
        ['ir_r4_k5_s2_e6_c192_se0.25'],
        ['ir_r1_k3_s1_e6_c320_se0.25'],
    ]
    model_kwargs = dict(
        block_args=decode_arch_def(arch_def, depth_multiplier),
        num_features=round_channels(1280, channel_multiplier, 8, None),
        stem_size=32,
        channel_multiplier=channel_multiplier,
        act_layer=Swish,
        norm_kwargs={},  # TODO: check
        drop_rate=drop_rate,
        drop_path_rate=0.2,
    )
    return model_kwargs

def gen_efficientnet_lite_kwargs(channel_multiplier=1.0, depth_multiplier=1.0, drop_rate=0.2):
    """Creates an EfficientNet-Lite model.

    Ref impl: https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet/lite
    Paper: https://arxiv.org/abs/1905.11946

    EfficientNet params
    name: (channel_multiplier, depth_multiplier, resolution, dropout_rate)
      'efficientnet-lite0': (1.0, 1.0, 224, 0.2),
      'efficientnet-lite1': (1.0, 1.1, 240, 0.2),
      'efficientnet-lite2': (1.1, 1.2, 260, 0.3),
      'efficientnet-lite3': (1.2, 1.4, 280, 0.3),
      'efficientnet-lite4': (1.4, 1.8, 300, 0.3),

    Args:
      channel_multiplier: multiplier to number of channels per layer
      depth_multiplier: multiplier to number of repeats per stage
    """
    arch_def = [
        ['ds_r1_k3_s1_e1_c16'],
        ['ir_r2_k3_s2_e6_c24'],
        ['ir_r2_k5_s2_e6_c40'],
        ['ir_r3_k3_s2_e6_c80'],
        ['ir_r3_k5_s1_e6_c112'],
        ['ir_r4_k5_s2_e6_c192'],
        ['ir_r1_k3_s1_e6_c320'],
    ]
    model_kwargs = dict(
        block_args=decode_arch_def(arch_def, depth_multiplier, fix_first_last=True),
        num_features=1280,
        stem_size=32,
        fix_stem=True,
        channel_multiplier=channel_multiplier,
        act_layer=nn.ReLU6,
        norm_kwargs={},
        drop_rate=drop_rate,
        drop_path_rate=0.2,
    )
    return model_kwargs

class EfficientNetBaseEncoder(EfficientNet, EncoderMixin):

    def __init__(self, stage_idxs, out_channels, depth=5, **kwargs):
        # Filter kwargs for compatibility with timm 0.9.2
        # Only pass parameters that EfficientNet actually accepts
        valid_params = ['block_args', 'num_features', 'in_chans', 'stem_size', 'fix_stem', 'channel_divisor',
                        'channel_min', 'output_stride',
                        'act_layer', 'norm_layer', 'drop_rate', 'drop_path_rate',
                        'global_pool', 'resynthesizer']
        
        filtered_kwargs = {}
        for key, value in kwargs.items():
            if key in valid_params:
                filtered_kwargs[key] = value
        
        try:
            super().__init__(**filtered_kwargs)
        except TypeError as e:
            # If still fails, try with minimal params
            print(f"Warning: EfficientNet init failed with filtered kwargs: {e}")
            try:
                super().__init__(
                    block_args=filtered_kwargs.get('block_args'),
                    num_features=filtered_kwargs.get('num_features', 1280),
                    in_chans=filtered_kwargs.get('in_chans', 3),
                    stem_size=filtered_kwargs.get('stem_size', 32),
                )
            except TypeError:
                # Last resort: use default init
                super().__init__()

        self._stage_idxs = stage_idxs
        self._out_channels = out_channels
        self._depth = depth
        self._in_channels = 3

        del self.classifier

    def get_stages(self):
        stem_modules = []
        if hasattr(self, 'conv_stem'):
            stem_modules.append(self.conv_stem)
        if hasattr(self, 'bn1'):
            stem_modules.append(self.bn1)
        if hasattr(self, 'act1'):
            stem_modules.append(self.act1)
            
        return [
            nn.Identity(),
            nn.Sequential(*stem_modules),
            self.blocks[:self._stage_idxs[0]],
            self.blocks[self._stage_idxs[0]:self._stage_idxs[1]],
            self.blocks[self._stage_idxs[1]:self._stage_idxs[2]],
            self.blocks[self._stage_idxs[2]:],
        ]

    def forward(self, x):
        stages = self.get_stages()

        features = []
        for i in range(self._depth + 1):
            x = stages[i](x)
            features.append(x)

        return features

    def load_state_dict(self, state_dict, **kwargs):
        state_dict.pop("classifier.bias")
        state_dict.pop("classifier.weight")
        super().load_state_dict(state_dict, **kwargs)


class EfficientNetEncoder(EfficientNetBaseEncoder):

    def __init__(self, stage_idxs, out_channels, depth=5, channel_multiplier=1.0, depth_multiplier=1.0, drop_rate=0.2):
        kwargs = get_efficientnet_kwargs(channel_multiplier, depth_multiplier, drop_rate)
        super().__init__(stage_idxs, out_channels, depth, **kwargs)


class EfficientNetLiteEncoder(EfficientNetBaseEncoder):

    def __init__(self, stage_idxs, out_channels, depth=5, channel_multiplier=1.0, depth_multiplier=1.0, drop_rate=0.2):
        kwargs = gen_efficientnet_lite_kwargs(channel_multiplier, depth_multiplier, drop_rate)
        super().__init__(stage_idxs, out_channels, depth, **kwargs)


def prepare_settings(settings, arch_name=None):
    # Fallback URLs for modern timm versions that removed the 'url' field from configs
    fallback_urls = {
        "tf_efficientnet_b0": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b0_aa-827b6e33.pth",
        "tf_efficientnet_b1": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b1_aa-ea7a6ee0.pth",
        "tf_efficientnet_b2": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b2_aa-60c94f97.pth",
        "tf_efficientnet_b3": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b3_aa-84b4657e.pth",
        "tf_efficientnet_b4": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b4_aa-818f208c.pth",
        "tf_efficientnet_b5": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b5_ra-9a3e5369.pth",
        "tf_efficientnet_b6": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b6_aa-80bd178a.pth",
        "tf_efficientnet_b7": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b7_ra-6c08e654.pth",
    }

    # Support both older timm (dict) and newer timm (DefaultCfg object)
    mean = settings["mean"] if isinstance(settings, dict) else getattr(settings, 'mean', (0.485, 0.456, 0.406))
    std = settings["std"] if isinstance(settings, dict) else getattr(settings, 'std', (0.229, 0.224, 0.225))
    url = settings["url"] if isinstance(settings, dict) else getattr(settings, 'url', '')
    
    # If url is missing (timm 0.9.0+), find the matching architecture fallback
    if not url:
        if arch_name and arch_name in fallback_urls:
            url = fallback_urls[arch_name]
        else:
            hf_id = settings.get('hf_hub_id', '') if isinstance(settings, dict) else getattr(settings, 'hf_hub_id', '')
            arch = settings.get('architecture', '') if isinstance(settings, dict) else getattr(settings, 'architecture', '')
            for key, fb_url in fallback_urls.items():
                if key in hf_id or key == arch or key.replace('tf_', '') == arch:
                    url = fb_url
                    break
        
        # Absolute fallback if we couldn't match (prevents FileNotFoundError from empty url)
        if not url:
            url = fallback_urls.get(arch_name, fallback_urls["tf_efficientnet_b0"])

    return {
        "mean": mean,
        "std": std,
        "url": url,
        "input_range": (0, 1),
        "input_space": "RGB",
    }


timm_efficientnet_encoders = {

    "timm-efficientnet-b0": {
        "encoder": EfficientNetEncoder,
        "pretrained_settings": {
            "imagenet": prepare_settings(default_cfgs["tf_efficientnet_b0"], "tf_efficientnet_b0"),
            "advprop": prepare_settings(default_cfgs["tf_efficientnet_b0_ap"], "tf_efficientnet_b0"),
            "noisy-student": prepare_settings(default_cfgs["tf_efficientnet_b0_ns"], "tf_efficientnet_b0"),
        },
        "params": {
            "out_channels": (3, 32, 24, 40, 112, 320),
            "stage_idxs": (2, 3, 5),
            "channel_multiplier": 1.0,
            "depth_multiplier": 1.0,
            "drop_rate": 0.2,
        },
    },

    "timm-efficientnet-b1": {
        "encoder": EfficientNetEncoder,
        "pretrained_settings": {
            "imagenet": prepare_settings(default_cfgs["tf_efficientnet_b1"], "tf_efficientnet_b1"),
            "advprop": prepare_settings(default_cfgs["tf_efficientnet_b1_ap"], "tf_efficientnet_b1"),
            "noisy-student": prepare_settings(default_cfgs["tf_efficientnet_b1_ns"], "tf_efficientnet_b1"),
        },
        "params": {
            "out_channels": (3, 32, 24, 40, 112, 320),
            "stage_idxs": (2, 3, 5),
            "channel_multiplier": 1.0,
            "depth_multiplier": 1.1,
            "drop_rate": 0.2,
        },
    },

    "timm-efficientnet-b2": {
        "encoder": EfficientNetEncoder,
        "pretrained_settings": {
            "imagenet": prepare_settings(default_cfgs["tf_efficientnet_b2"], "tf_efficientnet_b2"),
            "advprop": prepare_settings(default_cfgs["tf_efficientnet_b2_ap"], "tf_efficientnet_b2"),
            "noisy-student": prepare_settings(default_cfgs["tf_efficientnet_b2_ns"], "tf_efficientnet_b2"),
        },
        "params": {
            "out_channels": (3, 32, 24, 48, 120, 352),
            "stage_idxs": (2, 3, 5),
            "channel_multiplier": 1.1,
            "depth_multiplier": 1.2,
            "drop_rate": 0.3,
        },
    },

    "timm-efficientnet-b3": {
        "encoder": EfficientNetEncoder,
        "pretrained_settings": {
            "imagenet": prepare_settings(default_cfgs["tf_efficientnet_b3"], "tf_efficientnet_b3"),
            "advprop": prepare_settings(default_cfgs["tf_efficientnet_b3_ap"], "tf_efficientnet_b3"),
            "noisy-student": prepare_settings(default_cfgs["tf_efficientnet_b3_ns"], "tf_efficientnet_b3"),
        },
        "params": {
            "out_channels": (3, 40, 32, 48, 136, 384),
            "stage_idxs": (2, 3, 5),
            "channel_multiplier": 1.2,
            "depth_multiplier": 1.4,
            "drop_rate": 0.3,
        },
    },

    "timm-efficientnet-b4": {
        "encoder": EfficientNetEncoder,
        "pretrained_settings": {
            "imagenet": prepare_settings(default_cfgs["tf_efficientnet_b4"], "tf_efficientnet_b4"),
            "advprop": prepare_settings(default_cfgs["tf_efficientnet_b4_ap"], "tf_efficientnet_b4"),
            "noisy-student": prepare_settings(default_cfgs["tf_efficientnet_b4_ns"], "tf_efficientnet_b4"),
        },
        "params": {
            "out_channels": (3, 48, 32, 56, 160, 448),
            "stage_idxs": (2, 3, 5),
            "channel_multiplier": 1.4,
            "depth_multiplier": 1.8,
            "drop_rate": 0.4,
        },
    },

    "timm-efficientnet-b5": {
        "encoder": EfficientNetEncoder,
        "pretrained_settings": {
            "imagenet": prepare_settings(default_cfgs["tf_efficientnet_b5"], "tf_efficientnet_b5"),
            "advprop": prepare_settings(default_cfgs["tf_efficientnet_b5_ap"], "tf_efficientnet_b5"),
            "noisy-student": prepare_settings(default_cfgs["tf_efficientnet_b5_ns"], "tf_efficientnet_b5"),
        },
        "params": {
            "out_channels": (3, 48, 40, 64, 176, 512),
            "stage_idxs": (2, 3, 5),
            "channel_multiplier": 1.6,
            "depth_multiplier": 2.2,
            "drop_rate": 0.4,
        },
    },

    "timm-efficientnet-b6": {
        "encoder": EfficientNetEncoder,
        "pretrained_settings": {
            "imagenet": prepare_settings(default_cfgs["tf_efficientnet_b6"], "tf_efficientnet_b6"),
            "advprop": prepare_settings(default_cfgs["tf_efficientnet_b6_ap"], "tf_efficientnet_b6"),
            "noisy-student": prepare_settings(default_cfgs["tf_efficientnet_b6_ns"], "tf_efficientnet_b6"),
        },
        "params": {
            "out_channels": (3, 56, 40, 72, 200, 576),
            "stage_idxs": (2, 3, 5),
            "channel_multiplier": 1.8,
            "depth_multiplier": 2.6,
            "drop_rate": 0.5,
        },
    },

    "timm-efficientnet-b7": {
        "encoder": EfficientNetEncoder,
        "pretrained_settings": {
            "imagenet": prepare_settings(default_cfgs["tf_efficientnet_b7"], "tf_efficientnet_b7"),
            "advprop": prepare_settings(default_cfgs["tf_efficientnet_b7_ap"], "tf_efficientnet_b7"),
            "noisy-student": prepare_settings(default_cfgs["tf_efficientnet_b7_ns"], "tf_efficientnet_b7"),
        },
        "params": {
            "out_channels": (3, 64, 48, 80, 224, 640),
            "stage_idxs": (2, 3, 5),
            "channel_multiplier": 2.0,
            "depth_multiplier": 3.1,
            "drop_rate": 0.5,
        },
    },

    "timm-efficientnet-b8": {
        "encoder": EfficientNetEncoder,
        "pretrained_settings": {
            "imagenet": prepare_settings(default_cfgs["tf_efficientnet_b8"], "tf_efficientnet_b8"),
            "advprop": prepare_settings(default_cfgs["tf_efficientnet_b8_ap"], "tf_efficientnet_b8"),
        },
        "params": {
            "out_channels": (3, 72, 56, 88, 248, 704),
            "stage_idxs": (2, 3, 5),
            "channel_multiplier": 2.2,
            "depth_multiplier": 3.6,
            "drop_rate": 0.5,
        },
    },

    "timm-efficientnet-l2": {
        "encoder": EfficientNetEncoder,
        "pretrained_settings": {
            "noisy-student": prepare_settings(default_cfgs["tf_efficientnet_l2_ns"]),
        },
        "params": {
            "out_channels": (3, 136, 104, 176, 480, 1376),
            "stage_idxs": (2, 3, 5),
            "channel_multiplier": 4.3,
            "depth_multiplier": 5.3,
            "drop_rate": 0.5,
        },
    },

    "timm-tf_efficientnet_lite0": {
        "encoder": EfficientNetLiteEncoder,
        "pretrained_settings": {
            "imagenet": prepare_settings(default_cfgs["tf_efficientnet_lite0"]),
        },
        "params": {
            "out_channels": (3, 32, 24, 40, 112, 320),
            "stage_idxs": (2, 3, 5),
            "channel_multiplier": 1.0,
            "depth_multiplier": 1.0,
            "drop_rate": 0.2,
        },
    },

    "timm-tf_efficientnet_lite1": {
        "encoder": EfficientNetLiteEncoder,
        "pretrained_settings": {
            "imagenet": prepare_settings(default_cfgs["tf_efficientnet_lite1"]),
        },
        "params": {
            "out_channels": (3, 32, 24, 40, 112, 320),
            "stage_idxs": (2, 3, 5),
            "channel_multiplier": 1.0,
            "depth_multiplier": 1.1,
            "drop_rate": 0.2,
        },
    },

    "timm-tf_efficientnet_lite2": {
        "encoder": EfficientNetLiteEncoder,
        "pretrained_settings": {
            "imagenet": prepare_settings(default_cfgs["tf_efficientnet_lite2"]),
        },
        "params": {
            "out_channels": (3, 32, 24, 48, 120, 352),
            "stage_idxs": (2, 3, 5),
            "channel_multiplier": 1.1,
            "depth_multiplier": 1.2,
            "drop_rate": 0.3,
        },
    },

    "timm-tf_efficientnet_lite3": {
        "encoder": EfficientNetLiteEncoder,
        "pretrained_settings": {
            "imagenet": prepare_settings(default_cfgs["tf_efficientnet_lite3"]),
        },
        "params": {
            "out_channels": (3, 32, 32, 48, 136, 384),
            "stage_idxs": (2, 3, 5),
            "channel_multiplier": 1.2,
            "depth_multiplier": 1.4,
            "drop_rate": 0.3,
        },
    },

    "timm-tf_efficientnet_lite4": {
        "encoder": EfficientNetLiteEncoder,
        "pretrained_settings": {
            "imagenet": prepare_settings(default_cfgs["tf_efficientnet_lite4"]),
        },
        "params": {
            "out_channels": (3, 32, 32, 56, 160, 448),
            "stage_idxs": (2, 3, 5),
            "channel_multiplier": 1.4,
            "depth_multiplier": 1.8,
            "drop_rate": 0.4,
        },
    },
}
