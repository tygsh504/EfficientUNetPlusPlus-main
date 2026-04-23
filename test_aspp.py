"""
Quick test script to verify ASPP integration works correctly.
Run this to check if ASPP model can be instantiated and forward pass works.
"""

import torch
import sys

# Test imports
try:
    from aspp import ASPP
    print("✓ ASPP module imported successfully")
except ImportError as e:
    print(f"✗ Failed to import ASPP: {e}")
    sys.exit(1)

try:
    from model_with_aspp import EfficientUNetPlusPlusWithASPP
    print("✓ EfficientUNetPlusPlusWithASPP model imported successfully")
except ImportError as e:
    print(f"✗ Failed to import EfficientUNetPlusPlusWithASPP: {e}")
    sys.exit(1)

# Test ASPP module instantiation and forward pass
print("\n--- Testing ASPP Module ---")
try:
    aspp = ASPP(in_channels=320, out_channels=256, atrous_rates=[6, 12, 18])
    print(f"✓ ASPP module instantiated: {aspp}")
    
    # Test forward pass
    x = torch.randn(2, 320, 32, 32)  # Batch size 2, 320 channels, 32x32 spatial
    y = aspp(x)
    print(f"✓ ASPP forward pass successful")
    print(f"  Input shape:  {x.shape}")
    print(f"  Output shape: {y.shape}")
    assert y.shape == (2, 256, 32, 32), f"Output shape mismatch: expected (2, 256, 32, 32), got {y.shape}"
    print(f"✓ Output shape is correct")
except Exception as e:
    print(f"✗ ASPP test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test Model instantiation
print("\n--- Testing EfficientUNetPlusPlusWithASPP Model ---")
try:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Using device: {device}")
    
    # Note: This will download imagenet weights, might take a moment
    model = EfficientUNetPlusPlusWithASPP(
        encoder_name='timm-efficientnet-b0',
        encoder_depth=5,
        encoder_weights='imagenet',
        in_channels=3,
        classes=1,
        aspp_rates=[6, 12, 18]
    )
    print(f"✓ Model instantiated successfully")
    
    model.to(device)
    model.eval()
    
    # Test forward pass
    x = torch.randn(1, 3, 256, 256).to(device)  # Batch size 1, 3 channels (RGB), 256x256
    with torch.no_grad():
        y = model(x)
    print(f"✓ Model forward pass successful")
    print(f"  Input shape:  {x.shape}")
    print(f"  Output shape: {y.shape}")
    assert y.shape == (1, 1, 256, 256), f"Output shape mismatch: expected (1, 1, 256, 256), got {y.shape}"
    print(f"✓ Output shape is correct")
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
except Exception as e:
    print(f"✗ Model test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*60)
print("✓ All tests passed! ASPP integration is working correctly.")
print("="*60)
print("\nYou can now train with ASPP using:")
print("  python train.py -ti <train_imgs> -tm <train_masks> -vi <val_imgs> -vm <val_masks> --use-aspp")
