"""Grid search for optimal cls_threshold and pixel_threshold on validation set."""

import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast

import config
from dataset import build_dataloaders
from model import build_model
from utils import compute_dice


def load_best_model(device):
    model = build_model()
    ckpt_path = os.path.join(config.OUTPUT_DIR, 'best_model.pth')
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    if 'ema_state_dict' in ckpt:
        model.load_state_dict(ckpt['ema_state_dict'])
    else:
        model.load_state_dict(ckpt['model_state_dict'])
    print(f"Loaded best model from epoch {ckpt.get('epoch', '?')} (dice={ckpt.get('best_dice', '?')})")
    model = model.to(device)
    model.eval()
    return model


@torch.no_grad()
def collect_predictions(model, loader, device):
    """Run model on val set, collect raw probs, cls_probs, and masks."""
    all_probs = []
    all_cls_probs = []
    all_masks = []

    for batch in loader:
        images = batch['image'].to(device, non_blocking=True)
        masks = batch['mask']

        with autocast('cuda'):
            outputs = model(images)

        logits = outputs['logits']
        cls_logits = outputs['cls_logits']

        probs = torch.sigmoid(logits)
        probs = F.interpolate(probs, size=config.MASK_SIZE, mode='bilinear', align_corners=False)
        masks_resized = F.interpolate(masks, size=config.MASK_SIZE, mode='nearest')

        cls_probs = torch.sigmoid(cls_logits)

        all_probs.append(probs.cpu().numpy())
        all_cls_probs.append(cls_probs.cpu().numpy())
        all_masks.append(masks_resized.numpy())

    all_probs = np.concatenate(all_probs, axis=0)       # (N, 4, 350, 525)
    all_cls_probs = np.concatenate(all_cls_probs, axis=0)  # (N, 4)
    all_masks = np.concatenate(all_masks, axis=0)        # (N, 4, 350, 525)

    return all_probs, all_cls_probs, all_masks


def compute_dice_np(preds, targets, eps=1e-6):
    """Compute mean dice on numpy arrays (N, C, H, W)."""
    N, C, H, W = preds.shape
    preds = preds.reshape(N * C, -1)
    targets = targets.reshape(N * C, -1)
    intersection = (preds * targets).sum(axis=1)
    sum_p = preds.sum(axis=1)
    sum_t = targets.sum(axis=1)
    dice = (2 * intersection) / (sum_p + sum_t + eps)
    empty = (sum_p + sum_t) == 0
    dice[empty] = 1.0
    return dice.mean()


def grid_search(all_probs, all_cls_probs, all_masks):
    """Grid search over cls_threshold and pixel_threshold."""
    cls_thresholds = np.arange(0.20, 0.85, 0.05)
    pixel_thresholds = np.arange(0.20, 0.65, 0.05)

    print(f"\nGrid search: {len(cls_thresholds)} cls x {len(pixel_thresholds)} pixel = "
          f"{len(cls_thresholds) * len(pixel_thresholds)} combinations\n")

    best_dice = 0.0
    best_cls_th = 0.0
    best_pixel_th = 0.0
    results = []

    for cls_th in cls_thresholds:
        for pixel_th in pixel_thresholds:
            # Apply thresholds
            preds = (all_probs > pixel_th).astype(np.float32)
            cls_mask = (all_cls_probs > cls_th).astype(np.float32)  # (N, 4)
            preds = preds * cls_mask[:, :, None, None]

            dice = compute_dice_np(preds, all_masks)
            results.append((cls_th, pixel_th, dice))

            if dice > best_dice:
                best_dice = dice
                best_cls_th = cls_th
                best_pixel_th = pixel_th

    # Print top 10 results
    results.sort(key=lambda x: -x[2])
    print(f"{'cls_th':>8} {'pixel_th':>10} {'dice':>8}")
    print("-" * 30)
    for cls_th, pixel_th, dice in results[:10]:
        marker = " <-- best" if dice == best_dice else ""
        print(f"{cls_th:>8.2f} {pixel_th:>10.2f} {dice:>8.4f}{marker}")

    # Also print heatmap-style table
    print("\nDice heatmap (cls_th \\ pixel_th):")
    header = "cls\\pix |" + "".join(f" {p:.2f}  " for p in pixel_thresholds)
    print(header)
    print("-" * len(header))
    for cls_th in cls_thresholds:
        row = f" {cls_th:.2f}  |"
        for pixel_th in pixel_thresholds:
            d = next(r[2] for r in results if abs(r[0] - cls_th) < 0.001 and abs(r[1] - pixel_th) < 0.001)
            row += f" {d:.3f} "
        print(row)

    print(f"\nBest: cls_threshold={best_cls_th:.2f}, pixel_threshold={best_pixel_th:.2f}, dice={best_dice:.4f}")
    return best_cls_th, best_pixel_th, best_dice


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model and val data
    _, val_loader = build_dataloaders(config.IDX_FOLD)
    print(f"Val set: {len(val_loader.dataset)} images")

    model = load_best_model(device)

    # Collect predictions
    print("Running inference on validation set...")
    all_probs, all_cls_probs, all_masks = collect_predictions(model, val_loader, device)
    print(f"Collected: probs {all_probs.shape}, cls_probs {all_cls_probs.shape}, masks {all_masks.shape}")

    # Grid search
    best_cls_th, best_pixel_th, best_dice = grid_search(all_probs, all_cls_probs, all_masks)

    # Compare with current defaults
    print(f"\nCurrent defaults: cls={config.CLS_THRESHOLD}, pixel={config.PIXEL_THRESHOLD}")
    preds_default = (all_probs > config.PIXEL_THRESHOLD).astype(np.float32)
    cls_mask_default = (all_cls_probs > config.CLS_THRESHOLD).astype(np.float32)
    preds_default = preds_default * cls_mask_default[:, :, None, None]
    dice_default = compute_dice_np(preds_default, all_masks)
    print(f"Default dice: {dice_default:.4f}")
    print(f"Best dice:    {best_dice:.4f} (improvement: +{best_dice - dice_default:.4f})")


if __name__ == '__main__':
    main()
