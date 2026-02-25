"""5-fold threshold search with 3^5 checkpoint combinations.

Usage:
  Step 1 (GPU): python threshold_search_5fold.py --collect
  Step 2 (CPU): python threshold_search_5fold.py --search
"""

import argparse
import itertools
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast

import config
from dataset import build_dataloaders
from model import build_model


def load_model_from_ckpt(ckpt_path, device):
    model = build_model()
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    if 'ema_state_dict' in ckpt:
        model.load_state_dict(ckpt['ema_state_dict'])
    else:
        model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)
    model.eval()
    epoch = ckpt.get('epoch', '?')
    dice = ckpt.get('best_dice', '?')
    return model, epoch, dice


@torch.no_grad()
def collect_val_predictions(model, loader, device):
    """Run model on val set, return probs, cls_probs, masks."""
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

        all_probs.append(probs.cpu().half().numpy())
        all_cls_probs.append(cls_probs.cpu().half().numpy())
        all_masks.append(masks_resized.numpy().astype(np.uint8))

    return (np.concatenate(all_probs, axis=0),
            np.concatenate(all_cls_probs, axis=0),
            np.concatenate(all_masks, axis=0))


def collect_all(device):
    """Phase 1: Collect predictions from all 15 models, save to disk."""
    save_dir = os.path.join(config.OUTPUT_DIR, 'oof_predictions')
    os.makedirs(save_dir, exist_ok=True)

    for fold in range(config.NUM_FOLDS):
        fold_dir = os.path.join(config.OUTPUT_DIR, f'fold_{fold}')
        _, val_loader = build_dataloaders(fold)
        print(f"\n{'='*50}")
        print(f"Fold {fold}: {len(val_loader.dataset)} val images")

        masks_path = os.path.join(save_dir, f'masks_fold{fold}.npy')

        for ckpt_idx in range(1, 4):
            ckpt_path = os.path.join(fold_dir, f'best_model_{ckpt_idx}.pth')
            if not os.path.exists(ckpt_path):
                print(f"  Skipping fold {fold} ckpt {ckpt_idx}: {ckpt_path} not found")
                continue

            probs_path = os.path.join(save_dir, f'probs_fold{fold}_ckpt{ckpt_idx}.npy')
            cls_path = os.path.join(save_dir, f'cls_fold{fold}_ckpt{ckpt_idx}.npy')

            if os.path.exists(probs_path) and os.path.exists(cls_path):
                print(f"  Fold {fold} ckpt {ckpt_idx}: already exists, skipping")
                continue

            t0 = time.time()
            model, epoch, dice = load_model_from_ckpt(ckpt_path, device)
            print(f"  Fold {fold} ckpt {ckpt_idx}: epoch={epoch}, dice={dice}")

            probs, cls_probs, masks = collect_val_predictions(model, val_loader, device)
            np.save(probs_path, probs)
            np.save(cls_path, cls_probs)
            if not os.path.exists(masks_path):
                np.save(masks_path, masks)

            del model
            torch.cuda.empty_cache()
            print(f"    Saved: probs {probs.shape}, cls {cls_probs.shape} ({time.time()-t0:.0f}s)")

    print("\nAll predictions collected.")


def precompute_stats(probs, masks, pixel_thresholds):
    """Reduce (N, 4, 350, 525) to (N, 4) stats per pixel_threshold.

    Returns:
        stats: dict[pix_idx] -> (sum_pred, intersection), each (N, 4)
        sum_target: (N, 4), independent of threshold
    """
    probs = probs.astype(np.float32)
    masks_flat = masks.reshape(masks.shape[0], masks.shape[1], -1).astype(np.float32)
    sum_target = masks_flat.sum(axis=2)  # (N, 4)

    stats = {}
    for pi, pix_th in enumerate(pixel_thresholds):
        binary_flat = (probs > pix_th).reshape(probs.shape[0], probs.shape[1], -1).astype(np.float32)
        sum_pred = binary_flat.sum(axis=2)  # (N, 4)
        intersection = (binary_flat * masks_flat).sum(axis=2)  # (N, 4)
        stats[pi] = (sum_pred, intersection)

    return stats, sum_target


def search_all():
    """Phase 2: Load predictions, run 3^5 x threshold grid search."""
    save_dir = os.path.join(config.OUTPUT_DIR, 'oof_predictions')

    cls_thresholds = np.arange(0.20, 0.85, 0.05)
    pixel_thresholds = np.arange(0.20, 0.65, 0.05)

    # Pre-compute (N,4) stats for each fold x checkpoint x pixel_threshold
    # This is the expensive step, but only 5 folds x 3 ckpts x 9 pixel_thresholds = 135 large-array ops
    print("Loading predictions and pre-computing stats...")
    stats_cache = {}  # [fold][ckpt_idx] = (stats_dict, sum_target)
    cls_cache = {}    # [fold][ckpt_idx] = cls_probs (N, 4)

    for fold in range(config.NUM_FOLDS):
        masks = np.load(os.path.join(save_dir, f'masks_fold{fold}.npy'))
        stats_cache[fold] = {}
        cls_cache[fold] = {}

        for ckpt_idx in range(1, 4):
            probs_path = os.path.join(save_dir, f'probs_fold{fold}_ckpt{ckpt_idx}.npy')
            cls_path = os.path.join(save_dir, f'cls_fold{fold}_ckpt{ckpt_idx}.npy')
            if not os.path.exists(probs_path):
                continue

            t0 = time.time()
            probs = np.load(probs_path)
            cls_probs = np.load(cls_path).astype(np.float32)

            stats, sum_target = precompute_stats(probs, masks, pixel_thresholds)
            stats_cache[fold][ckpt_idx] = (stats, sum_target)
            cls_cache[fold][ckpt_idx] = cls_probs

            del probs
            print(f"  Fold {fold} ckpt {ckpt_idx}: {masks.shape[0]} images ({time.time()-t0:.1f}s)")

        del masks

    # Enumerate all 3^5 combinations
    ckpt_options = []
    for fold in range(config.NUM_FOLDS):
        ckpt_options.append(sorted(stats_cache[fold].keys()))
    combinations = list(itertools.product(*ckpt_options))

    num_th = len(cls_thresholds) * len(pixel_thresholds)
    total_evals = len(combinations) * num_th
    print(f"\nSearching {len(combinations)} combos x {num_th} threshold pairs = {total_evals} evaluations...")

    best_dice = 0.0
    best_combo = None
    best_cls_th = 0.0
    best_pix_th = 0.0
    eps = 1e-6

    t0 = time.time()
    for combo_idx, combo in enumerate(combinations):
        for pi, pix_th in enumerate(pixel_thresholds):
            # Gather (N, 4) stats for this combo and pixel_threshold
            all_sum_pred = []
            all_intersection = []
            all_sum_target = []
            all_cls_probs = []

            for fold in range(config.NUM_FOLDS):
                ckpt_idx = combo[fold]
                stats, sum_target = stats_cache[fold][ckpt_idx]
                sum_pred, intersection = stats[pi]
                all_sum_pred.append(sum_pred)
                all_intersection.append(intersection)
                all_sum_target.append(sum_target)
                all_cls_probs.append(cls_cache[fold][ckpt_idx])

            # Concatenate across folds: (total_N, 4), small arrays
            cat_sum_pred = np.concatenate(all_sum_pred, axis=0)
            cat_intersection = np.concatenate(all_intersection, axis=0)
            cat_sum_target = np.concatenate(all_sum_target, axis=0)
            cat_cls_probs = np.concatenate(all_cls_probs, axis=0)

            for ci, cls_th in enumerate(cls_thresholds):
                cls_mask = (cat_cls_probs > cls_th).astype(np.float32)
                masked_sum_pred = cat_sum_pred * cls_mask
                masked_intersection = cat_intersection * cls_mask

                dice_arr = (2 * masked_intersection) / (masked_sum_pred + cat_sum_target + eps)
                empty = (masked_sum_pred + cat_sum_target) == 0
                dice_arr[empty] = 1.0

                mean_dice = dice_arr.mean()
                if mean_dice > best_dice:
                    best_dice = mean_dice
                    best_combo = combo
                    best_cls_th = cls_th
                    best_pix_th = pix_th

        if (combo_idx + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"  {combo_idx+1}/{len(combinations)} combos done ({elapsed:.0f}s)")

    elapsed = time.time() - t0
    print(f"\nSearch complete in {elapsed:.0f}s")
    print(f"\n{'='*60}")
    print("BEST RESULT:")
    combo_str = ", ".join(f"fold{f}_ckpt{c}" for f, c in enumerate(best_combo))
    print(f"  Checkpoints: [{combo_str}]")
    print(f"  cls_threshold={best_cls_th:.2f}, pixel_threshold={best_pix_th:.2f}")
    print(f"  OOF Dice: {best_dice:.4f}")
    ckpt_arg = ",".join(str(c) for c in best_combo)
    print(f"\n  Inference command:")
    print(f"  python inference.py --ensemble --ckpts {ckpt_arg} "
          f"--cls-threshold {best_cls_th:.2f} --pixel-threshold {best_pix_th:.2f}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--collect', action='store_true', help='Phase 1: collect predictions (requires GPU)')
    parser.add_argument('--search', action='store_true', help='Phase 2: grid search (CPU only)')
    args = parser.parse_args()

    if not args.collect and not args.search:
        print("Usage: python threshold_search_5fold.py --collect  (then)  --search")
        return

    if args.collect:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        collect_all(device)

    if args.search:
        search_all()


if __name__ == '__main__':
    main()
