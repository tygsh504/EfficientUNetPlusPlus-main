# testing_rfb.py
import os
import sys
import logging
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader

# ── Monkey-patch: copied verbatim from train.py ──────────────────────────────
if not hasattr(torch, '_six'):
    import types, collections.abc
    torch._six = types.ModuleType('torch._six')
    torch._six.container_abcs = collections.abc
    sys.modules['torch._six'] = torch._six
# ─────────────────────────────────────────────────────────────────────────────

try:
    import segmentation_models_pytorch.segmentation_models_pytorch as smp
except ImportError:
    import segmentation_models_pytorch as smp

# Import your custom dataset exactly as train.py does
from utils.dataset import PaddyBinaryDataset
from model_with_rfb import EfficientUNetPlusPlusWithRFB


# ═════════════════════════════════════════════════════════════════════════════
#  USER CONFIGURATION  ── edit these before running
# ═════════════════════════════════════════════════════════════════════════════

MODEL_PATH    = r"b0_rfb_boundary.pth"
BASE_DATA_PATH = r"D:\Testing\Testing Dataset"
MAIN_OUTPUT_DIR = r"C:\Users\User\Desktop\b0_RFB_boundary"

# The disease / category folders inside BASE_DATA_PATH.
# Each folder must contain an "Infer_Ori" (images) and "Infer_GT" (masks) subfolder.
DISEASES = [
    "Bacterial Leaf Blight",
    "Bacterial Leaf Streak",
    "Blast",
    "Brown Spot",
    "DownyMildew",
    "Hispa",
    "Tungro",
]

# Model config — must match train.py exactly
ENCODER_NAME = 'timm-efficientnet-b0'
NUM_CLASSES  = 1          # binary segmentation
INPUT_SHAPE  = [640, 480] # [Height, Width]  — resize applied inside PaddyBinaryDataset
BATCH_SIZE   = 1
USE_RFB      = True       # Set to True if trained with RFB

# ═════════════════════════════════════════════════════════════════════════════


# ── SCSE attention patcher (carried over from testing.py) ────────────────────
class SCSEModule(nn.Module):
    def __init__(self, in_channels, mip):
        super().__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, mip, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mip, in_channels, 1),
            nn.Sigmoid(),
        )
        self.sSE = nn.Sequential(nn.Conv2d(in_channels, 1, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)


def patch_model_attention(model, state_dict):
    scse_prefixes = set()
    for k in state_dict.keys():
        if '.cSE.' in k or '.sSE.' in k:
            prefix = k.split('.cSE.')[0].split('.sSE.')[0]
            scse_prefixes.add(prefix)

    if not scse_prefixes:
        return

    logging.info(f"Detected SCSE attention in state_dict. Patching {len(scse_prefixes)} modules...")

    for prefix in scse_prefixes:
        parts = prefix.split('.')
        parent = model
        try:
            for part in parts[:-1]:
                if part.isdigit():
                    parent = parent[int(part)]
                elif isinstance(parent, nn.ModuleDict):
                    parent = parent[part]
                else:
                    parent = getattr(parent, part)
            attr_name = parts[-1]

            sse_weight_key = f"{prefix}.sSE.0.weight"
            cse_weight_key = f"{prefix}.cSE.1.weight"

            if sse_weight_key in state_dict and cse_weight_key in state_dict:
                in_channels = state_dict[sse_weight_key].shape[1]
                mip         = state_dict[cse_weight_key].shape[0]
                scse        = SCSEModule(in_channels, mip)

                if attr_name.isdigit():
                    parent[int(attr_name)] = scse
                else:
                    setattr(parent, attr_name, scse)
        except Exception as e:
            logging.warning(f"Could not patch {prefix}: {e}")
# ─────────────────────────────────────────────────────────────────────────────


def calculate_complexity(model, input_shape, device):
    try:
        from thop import profile
        dummy = torch.randn(1, 3, *input_shape).to(device)
        flops, params = profile(model, inputs=(dummy,), verbose=False)
        return params, flops
    except ImportError:
        logging.warning("'thop' not installed — Params/FLOPs will not be computed. "
                        "Run: pip install thop")
        return 0, 0
    except Exception as e:
        logging.error(f"Complexity calculation failed: {e}")
        return 0, 0


def calculate_metrics(pred_mask, true_mask):
    """Identical metric set to testing.py."""
    pred = pred_mask.detach().cpu().numpy().flatten()
    true = true_mask.detach().cpu().numpy().flatten()
    pred_bin = (pred > 0.5).astype(np.uint8)
    true_bin = (true > 0.5).astype(np.uint8)

    tp = np.sum((pred_bin == 1) & (true_bin == 1))
    tn = np.sum((pred_bin == 0) & (true_bin == 0))
    fp = np.sum((pred_bin == 1) & (true_bin == 0))
    fn = np.sum((pred_bin == 0) & (true_bin == 1))

    eps = 1e-7
    accuracy  = (tp + tn) / (tp + tn + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall    = tp / (tp + fn + eps)
    f1        = 2 * (precision * recall) / (precision + recall + eps)
    iou       = tp / (tp + fp + fn + eps)
    dice      = (2 * tp) / (2 * tp + fp + fn + eps)

    return {
        "Dice": dice, "IoU": iou, "Precision": precision,
        "Recall": recall, "Accuracy": accuracy, "F1_Score": f1,
    }


def save_visual_result(image_tensor, true_mask_tensor, pred_mask_tensor,
                       filename, dice_score, output_dir):
    img_np = image_tensor.permute(1, 2, 0).cpu().numpy()
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-7)
    img_np = (img_np * 255).astype(np.uint8)

    true_bin = (true_mask_tensor.squeeze().cpu().numpy() > 0.5).astype(np.uint8)
    pred_bin = (pred_mask_tensor.squeeze().cpu().numpy() > 0.5).astype(np.uint8)

    fig, ax = plt.subplots(3, 1, figsize=(6, 18))
    ax[0].imshow(img_np);   ax[0].set_title(f"Original: {filename}"); ax[0].axis("off")
    ax[1].imshow(true_bin, cmap='gray'); ax[1].set_title("Ground Truth");    ax[1].axis("off")
    ax[2].imshow(pred_bin, cmap='gray'); ax[2].set_title(f"Pred (Dice: {dice_score:.2f})"); ax[2].axis("off")

    plt.tight_layout()
    save_path = output_dir / f"{Path(filename).stem}_eval.png"
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def run_prediction_on_disease(disease_name, net, device, params, flops):
    """
    Runs inference on one disease folder.
    Folder layout expected (same as testing.py):
        <BASE_DATA_PATH>/<disease_name>/images/   ← images
        <BASE_DATA_PATH>/<disease_name>/masks/    ← ground-truth masks
    """
    img_dir  = os.path.join(BASE_DATA_PATH, disease_name, "images") 
    mask_dir = os.path.join(BASE_DATA_PATH, disease_name, "masks") 

    if not os.path.exists(img_dir) or not os.path.exists(mask_dir):
        logging.warning(f"Skipping '{disease_name}': path not found.")
        return None

    disease_output_dir = Path(MAIN_OUTPUT_DIR) / disease_name
    img_output_dir     = disease_output_dir / "predictions"
    disease_output_dir.mkdir(parents=True, exist_ok=True)
    img_output_dir.mkdir(parents=True, exist_ok=True)

    # Evaluates the full 640x480 resolution (is_train=False)
    dataset = PaddyBinaryDataset(img_dir, mask_dir, is_train=False)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    
    # Get the original image filenames from the dataset
    image_filenames = dataset.image_files

    results = []
    image_idx = 0  # Track which image we're processing

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Predicting {disease_name}"):
            images     = batch['image'].to(device, dtype=torch.float32)
            true_masks = batch['mask'].to(device,  dtype=torch.float32)

            # Binary prediction — sigmoid + threshold, matching train.py val loop
            outputs = net(images)
            probs   = torch.sigmoid(outputs)
            pred_masks = (probs > 0.5).float()

            # Collect per-image metrics
            for i in range(images.shape[0]):
                # Retrieve filename from the dataset's image list (preserves original name)
                img_name = image_filenames[image_idx]
                image_idx += 1

                metrics = calculate_metrics(pred_masks[i], true_masks[i])
                metrics['Filename'] = img_name
                results.append(metrics)

                save_visual_result(
                    images[i], true_masks[i], pred_masks[i],
                    img_name, metrics['Dice'], img_output_dir
                )

    if not results:
        return None

    metric_cols = ['Dice', 'IoU', 'Precision', 'Recall', 'Accuracy', 'F1_Score']
    df    = pd.DataFrame(results)
    means = df[metric_cols].mean().to_dict()

    summary_df = pd.DataFrame(
        [{'Metric': k, 'Value': v} for k, v in means.items()] +
        [{'Metric': 'Params', 'Value': params}, {'Metric': 'FLOPs', 'Value': flops}]
    )

    excel_path = disease_output_dir / f'{disease_name}_metrics.xlsx'
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        summary_df.to_excel(writer, sheet_name='Summary',  index=False)
        df[['Filename'] + metric_cols].to_excel(writer, sheet_name='Detailed', index=False)

    logging.info(
        f"[{disease_name}] Dice={means['Dice']:.4f}  IoU={means['IoU']:.4f}  "
        f"F1={means['F1_Score']:.4f}  → saved to {excel_path}"
    )
    return means


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    try:
        if USE_RFB:
            net = EfficientUNetPlusPlusWithRFB(
                encoder_name=ENCODER_NAME,
                encoder_weights=None,
                in_channels=3,
                classes=NUM_CLASSES,
            )
        else:
            net = smp.EfficientUnetPlusPlus(
                encoder_name=ENCODER_NAME,
                encoder_weights=None,   # weights come from checkpoint
                in_channels=3,
                classes=NUM_CLASSES,
            )

        state_dict     = torch.load(MODEL_PATH, map_location=device, weights_only=True)
        new_state_dict = {k[7:] if k.startswith('module.') else k: v
                          for k, v in state_dict.items()}

        patch_model_attention(net, new_state_dict)
        net.load_state_dict(new_state_dict)
        net.to(device).eval()
        logging.info("Model loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        sys.exit(1)

    params, flops = calculate_complexity(net, INPUT_SHAPE, device)

    all_results = []

    for disease in DISEASES:
        means = run_prediction_on_disease(disease, net, device, params, flops)
        if means:
            means['Disease'] = disease
            all_results.append(means)

    if all_results:
        overall_df = pd.DataFrame(all_results)
        cols       = ['Disease'] + [c for c in overall_df.columns if c != 'Disease']
        overall_df = overall_df[cols]

        mean_row             = overall_df.mean(numeric_only=True).to_dict()
        mean_row['Disease']  = 'OVERALL_MEAN'
        overall_df           = pd.concat(
            [overall_df, pd.DataFrame([mean_row])], ignore_index=True
        )

        out_path = Path(MAIN_OUTPUT_DIR) / 'calculated_mean.xlsx'
        overall_df.to_excel(out_path, index=False)
        logging.info(f"Overall means saved to {out_path}")

    print("\n--- All Predictions Completed ---")
