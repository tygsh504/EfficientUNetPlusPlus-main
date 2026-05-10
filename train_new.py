# train.py
import argparse
import logging
import os
import sys

import torch

# Monkey-patch for timm 0.3.2 compatibility with PyTorch 2.0+
if not hasattr(torch, '_six'):
    import types, collections.abc
    torch._six = types.ModuleType('torch._six')
    torch._six.container_abcs = collections.abc
    sys.modules['torch._six'] = torch._six

import torch.nn as nn
from torch import optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

import pandas as pd
import matplotlib.pyplot as plt
import segmentation_models_pytorch.segmentation_models_pytorch as smp
import numpy as np

# Import your custom dataset
from utils.dataset import PaddyBinaryDataset

# Import ASPP model
from model_with_aspp import EfficientUNetPlusPlusWithASPP

# Import RFB model
from model_with_rfb import EfficientUNetPlusPlusWithRFB

# Import DenseASPP model
from model_with_denseaspp import EfficientUNetPlusPlusWithDenseASPP

# ============ ADVANCED LOSS FUNCTIONS FOR PADDY FIELD SEGMENTATION ============
class FocalLovaszLoss(nn.Module):
    """
    Combined Focal Loss + Lovasz Loss for robust segmentation with noisy backgrounds
    """
    def __init__(self, focal_weight=0.5, lovasz_weight=0.5, class_weight=None):
        super(FocalLovaszLoss, self).__init__()
        self.focal_weight = focal_weight
        self.lovasz_weight = lovasz_weight
        self.focal_loss = smp.losses.FocalLoss(mode='binary')
        self.lovasz_loss = smp.losses.LovaszLoss(mode='binary')
        self.class_weight = class_weight
    
    def forward(self, pred, target):
        focal = self.focal_loss(pred, target)
        lovasz = self.lovasz_loss(pred, target)
        return self.focal_weight * focal + self.lovasz_weight * lovasz


class WeightedDiceLoss(nn.Module):
    """
    Weighted Dice Loss with smooth edges for multiple leaves detection
    """
    def __init__(self, pos_weight=1.0, smooth=1.0):
        super(WeightedDiceLoss, self).__init__()
        self.pos_weight = pos_weight  # Weight for positive class (diseased leaf)
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        
        # Weighted intersection
        intersection = (pred * target * self.pos_weight).sum()
        
        # Weighted dice
        dice = (2.0 * intersection + self.smooth) / (
            (pred * self.pos_weight).sum() + target.sum() + self.smooth
        )
        return 1.0 - dice


class BoundaryAwareLoss(nn.Module):
    """
    Loss function that emphasizes boundaries between leaves and background
    Useful for multiple leaves in noisy background
    """
    def __init__(self, boundary_weight=2.0):
        super(BoundaryAwareLoss, self).__init__()
        self.boundary_weight = boundary_weight
        self.dice_loss = smp.losses.DiceLoss(mode='binary')
    
    def forward(self, pred, target):
        dice = self.dice_loss(pred, target)
        
        # Compute edges/boundaries using simple sobel-like operation
        # Emphasize gradients at boundaries
        pred_sigmoid = torch.sigmoid(pred)
        target_float = target.float()
        
        # Compute spatial gradients separately for height and width directions
        # Vertical gradient (height direction)
        grad_pred_v = torch.abs(pred_sigmoid[:, :, 1:, :] - pred_sigmoid[:, :, :-1, :])
        grad_target_v = torch.abs(target_float[:, :, 1:, :] - target_float[:, :, :-1, :])
        
        # Horizontal gradient (width direction) 
        grad_pred_h = torch.abs(pred_sigmoid[:, :, :, 1:] - pred_sigmoid[:, :, :, :-1])
        grad_target_h = torch.abs(target_float[:, :, :, 1:] - target_float[:, :, :, :-1])
        
        # Boundary loss: combine both directions
        # Pad gradients to same size as input for comparison
        boundary_loss_v = torch.mean(torch.abs(grad_pred_v - grad_target_v))
        boundary_loss_h = torch.mean(torch.abs(grad_pred_h - grad_target_h))
        boundary_loss = (boundary_loss_v + boundary_loss_h) / 2.0
        
        return dice + self.boundary_weight * boundary_loss


# ============ LEARNING RATE SCHEDULER WITH WARMUP ============
class WarmupScheduler(object):
    """
    Learning rate scheduler with warmup phase
    """
    def __init__(self, optimizer, warmup_epochs=5, total_epochs=150, base_lr=0.001):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.current_epoch = 0
    
    def step(self, epoch):
        self.current_epoch = epoch
        if epoch < self.warmup_epochs:
            # Linear warmup
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            # Cosine annealing after warmup
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.base_lr * 0.5 * (1.0 + np.cos(np.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr


def train_net(net, device, training_set, validation_set, dir_checkpoint,
              epochs=150, batch_size=4, lr=0.001, save_cp=True, accumulation_steps=2,
              loss_type='combined', use_warmup=True):

    # Uses the cleanly instantiated dataset from main()
    train_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}_{loss_type}')
    global_step = 0

    logging.info(f'''Starting binary training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {len(training_set)}
        Validation size: {len(validation_set)}
        Device:          {device.type}
        Loss Type:       {loss_type}
        Warmup:          {use_warmup}
    ''')

    optimizer = optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    
    if use_warmup:
        warmup_scheduler = WarmupScheduler(optimizer, warmup_epochs=5, total_epochs=epochs, base_lr=lr)
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)

    if loss_type == 'combined':
        # Focal + Lovasz + Dice for best performance with noisy backgrounds
        criterion_focal_lovasz = FocalLovaszLoss(focal_weight=0.4, lovasz_weight=0.3)
        criterion_weighted_dice = WeightedDiceLoss(pos_weight=1.5, smooth=1.0)
    elif loss_type == 'boundary_aware':
        # Boundary-aware loss emphasizes multiple leaf edges
        criterion_boundary = BoundaryAwareLoss(boundary_weight=2.0)
    else:
        # Default to Focal + Dice
        criterion_focal = smp.losses.FocalLoss(mode='binary')
        criterion_dice = smp.losses.DiceLoss(mode='binary')

    # Initialize history tracking
    history = {
        'epoch': [],
        'learning_rate': [],
        'train_loss': [],
        'val_loss': [],
        'train_dice': [],
        'val_dice': []
    }

    best_val_dice = 0.0
    use_amp = device.type == 'cuda'
    scaler = None
    if use_amp:
        scaler = torch.cuda.amp.GradScaler()

    for epoch in range(epochs):
        net.train()
        epoch_loss = 0
        train_dice_score = 0
        
        with tqdm(total=len(training_set), desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            optimizer.zero_grad(set_to_none=True)
            for batch_idx, batch in enumerate(train_loader):
                imgs = batch['image'].to(device=device, dtype=torch.float32)
                true_masks = batch['mask'].to(device=device, dtype=torch.float32)
                
                with torch.amp.autocast(device.type, enabled=use_amp):
                    masks_pred = net(imgs)
                    
                    if loss_type == 'combined':
                        loss = criterion_focal_lovasz(masks_pred, true_masks) + criterion_weighted_dice(masks_pred, true_masks)
                    elif loss_type == 'boundary_aware':
                        loss = criterion_boundary(masks_pred, true_masks)
                    else:
                        loss = criterion_focal(masks_pred, true_masks) + criterion_dice(masks_pred, true_masks)
                    
                    loss = loss / accumulation_steps
                    
                epoch_loss += loss.item() * accumulation_steps

                # Calculate Train Dice
                with torch.no_grad():
                    probs = torch.sigmoid(masks_pred)
                    preds = (probs > 0.5).float()
                    intersection = (preds * true_masks).sum()
                    dice = (2. * intersection) / (preds.sum() + true_masks.sum() + 1e-8)
                    train_dice_score += dice.item()

                if use_amp:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                if ((batch_idx + 1) % accumulation_steps == 0) or (batch_idx + 1 == len(train_loader)):
                    if use_amp:
                        scaler.unscale_(optimizer) # Unscale before clipping
                        nn.utils.clip_grad_value_(net.parameters(), 0.1)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        nn.utils.clip_grad_value_(net.parameters(), 0.1)
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                writer.add_scalar('Loss/train', loss.item() * accumulation_steps, global_step)
                pbar.set_postfix(**{'loss (batch)': loss.item() * accumulation_steps, 'dice': dice.item()})
                pbar.update(imgs.shape[0])
                global_step += 1
                
        avg_train_loss = epoch_loss / len(train_loader)
        avg_train_dice = train_dice_score / len(train_loader)

        if use_warmup:
            warmup_scheduler.step(epoch)

        # --- Validation Phase ---
        net.eval()
        val_loss = 0
        val_dice_score = 0
        
        with torch.no_grad():
            with tqdm(total=len(validation_set), desc='Validation', unit='img', leave=False) as pbar_val:
                for batch in val_loader:
                    imgs = batch['image'].to(device=device, dtype=torch.float32)
                    true_masks = batch['mask'].to(device=device, dtype=torch.float32)
                    
                    with torch.amp.autocast(device.type, enabled=use_amp):
                        masks_pred = net(imgs)
                        
                        if loss_type == 'combined':
                            loss = criterion_focal_lovasz(masks_pred, true_masks) + criterion_weighted_dice(masks_pred, true_masks)
                        elif loss_type == 'boundary_aware':
                            loss = criterion_boundary(masks_pred, true_masks)
                        else:
                            loss = criterion_focal(masks_pred, true_masks) + criterion_dice(masks_pred, true_masks)
                        
                        val_loss += loss.item()

                    # Calculate Dice Coefficient manually for tracking
                    probs = torch.sigmoid(masks_pred)
                    preds = (probs > 0.5).float()
                    
                    intersection = (preds * true_masks).sum()
                    dice = (2. * intersection) / (preds.sum() + true_masks.sum() + 1e-8)
                    val_dice_score += dice.item()
                    pbar_val.update(imgs.shape[0])

        avg_val_loss = val_loss / len(val_loader)
        avg_val_dice = val_dice_score / len(val_loader)

        if not use_warmup:
            scheduler.step(avg_val_loss)

        # --- Logging and History ---
        current_lr = optimizer.param_groups[0]['lr']
        history['epoch'].append(epoch + 1)
        history['learning_rate'].append(current_lr)
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_dice'].append(avg_train_dice)
        history['val_dice'].append(avg_val_dice)

        logging.info(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Train Dice: {avg_train_dice:.4f} | Val Dice: {avg_val_dice:.4f} | LR: {current_lr:.6f}")

        # --- Checkpoint Saving ---
        if save_cp:
            os.makedirs(dir_checkpoint, exist_ok=True)
            # Save latest
            torch.save(net.state_dict(), os.path.join(dir_checkpoint, 'CP_last.pth'))
            
            # Save best based on Validation Dice
            if avg_val_dice > best_val_dice:
                best_val_dice = avg_val_dice
                torch.save(net.state_dict(), os.path.join(dir_checkpoint, 'CP_best.pth'))
                logging.info(f'New best checkpoint saved! Val Dice: {best_val_dice:.4f}')
            
    writer.close()

    # --- Post-Training Metrics Generation ---
    df = pd.DataFrame(history)
    excel_path = os.path.join(dir_checkpoint, 'training_metrics.xlsx')
    df.to_excel(excel_path, index=False)
    logging.info(f'Metrics saved to {excel_path}')

    try:
        plt.figure(figsize=(18, 5))
        
        plt.subplot(1, 3, 1)
        plt.plot(history['epoch'], history['train_loss'], label='Train Loss', color='blue')
        plt.plot(history['epoch'], history['val_loss'], label='Val Loss', color='orange', linestyle='--')
        plt.title('Loss over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 3, 2)
        plt.plot(history['epoch'], history['learning_rate'], label='Learning Rate', color='green')
        plt.title('Learning Rate')
        plt.xlabel('Epoch')
        plt.ylabel('LR')
        plt.yscale('log')
        plt.grid(True)

        plt.subplot(1, 3, 3)
        plt.plot(history['epoch'], history['train_dice'], label='Train Dice', color='blue')
        plt.plot(history['epoch'], history['val_dice'], label='Val Dice', color='red', linestyle='--')
        plt.title('Dice Coefficient')
        plt.xlabel('Epoch')
        plt.ylabel('Dice Coeff')
        plt.legend()
        plt.grid(True)

        plot_path = os.path.join(dir_checkpoint, 'training_plot.jpg')
        plt.savefig(plot_path)
        plt.close()
        logging.info(f'Training plot saved to {plot_path}')
    except Exception as e:
        logging.error(f"Failed to save plots: {e}")

def get_args():
    parser = argparse.ArgumentParser(description='EfficientUNet++ Binary Train Script')
    parser.add_argument('-ti', '--training-images-dir', type=str, required=True)
    parser.add_argument('-tm', '--training-masks-dir', type=str, required=True)
    parser.add_argument('-vi', '--validation-images-dir', type=str, required=True)
    parser.add_argument('-vm', '--validation-masks-dir', type=str, required=True)
    parser.add_argument('-enc', '--encoder', type=str, default='timm-efficientnet-b0')
    parser.add_argument('-e', '--epochs', type=int, default=150)
    parser.add_argument('-b', '--batch-size', type=int, default=4)
    parser.add_argument('-l', '--learning-rate', type=float, default=0.001)
    parser.add_argument('-a', '--accumulation-steps', type=int, default=2)
    parser.add_argument('-c', '--dir_checkpoint', type=str, default='checkpoints/')
    
    parser.add_argument('--loss-type', type=str, default='focal_dice', 
                        choices=['combined', 'boundary_aware', 'focal_dice'],
                        help='Loss function type: combined (Focal+Lovasz+Dice), boundary_aware (for edges), focal_dice (default)')
    parser.add_argument('--no-warmup', action='store_true', 
                        help='Disable learning rate warmup scheduler')
    # ========== Bottleneck option ==========
    parser.add_argument('--use', type=str, choices=['rfb', 'aspp', 'denseaspp'], default=None,
                        help='Enable RFB, ASPP, or DenseASPP module at bottleneck for multi-scale context')
    parser.add_argument('--aspp-rates', type=int, nargs='+', default=[6, 12, 18],
                        help='Atrous convolution rates for ASPP (default: 6 12 18)')
    parser.add_argument('--denseaspp-rates', type=int, nargs='+', default=[3, 6, 12, 18],
                        help='Atrous convolution rates for DenseASPP (default: 3 6 12 18)')
    return parser.parse_args()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    if args.use == 'aspp':
        logging.info(f"Creating EfficientUNetPlusPlus WITH ASPP (rates: {args.aspp_rates})")
        net = EfficientUNetPlusPlusWithASPP(
            encoder_name=args.encoder, 
            encoder_weights="imagenet", 
            in_channels=3, 
            classes=1,
            aspp_rates=args.aspp_rates
        )
    elif args.use == 'rfb':
        logging.info("Creating EfficientUNetPlusPlus WITH RFB")
        net = EfficientUNetPlusPlusWithRFB(
            encoder_name=args.encoder, 
            encoder_weights="imagenet", 
            in_channels=3, 
            classes=1
        )
    elif args.use == 'denseaspp':
        logging.info(f"Creating EfficientUNetPlusPlus WITH DenseASPP (rates: {args.denseaspp_rates})")
        net = EfficientUNetPlusPlusWithDenseASPP(
            encoder_name=args.encoder, 
            encoder_weights="imagenet", 
            in_channels=3, 
            classes=1,
            denseaspp_rates=args.denseaspp_rates
        )
    else:
        logging.info("Creating standard EfficientUNetPlusPlus (without Bottleneck)")
        net = smp.EfficientUnetPlusPlus(
            encoder_name=args.encoder, 
            encoder_weights="imagenet", 
            in_channels=3, 
            classes=1 
        )

    net.to(device=device)

    # Instantiate datasets using the updated crop policy API
    training_set = PaddyBinaryDataset(args.training_images_dir, args.training_masks_dir, is_train=True, patch_size=256)
    validation_set = PaddyBinaryDataset(args.validation_images_dir, args.validation_masks_dir, is_train=False)

    try:
        train_net(net=net, device=device, training_set=training_set, validation_set=validation_set,
                  dir_checkpoint=args.dir_checkpoint, epochs=args.epochs, batch_size=args.batch_size,
                  lr=args.learning_rate, accumulation_steps=args.accumulation_steps,
                  loss_type=args.loss_type, use_warmup=not args.no_warmup)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        sys.exit(0)