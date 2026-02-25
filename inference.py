"""Inference + submission CSV generation for cloud segmentation.

Supports single model and 5-fold ensemble (averaging logits).

Usage:
  Single model:  python inference.py --checkpoint outputs/fold_0/best_model_1.pth
  5-fold ensemble: python inference.py --ensemble --ckpts 1,1,1,1,1
"""

import os
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.amp import autocast
import tqdm

import config
from dataset import build_test_dataloader
from model import build_model
from utils import mask2rle


def load_model(checkpoint_path, device):
    model = build_model()
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if 'ema_state_dict' in ckpt:
        model.load_state_dict(ckpt['ema_state_dict'])
        print(f"Loaded EMA weights from {checkpoint_path} (epoch {ckpt.get('epoch', '?')})")
    else:
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"Loaded model weights from {checkpoint_path}")
    model = model.to(device)
    model.eval()
    return model


def tta_forward(model, images):
    """TTA: original + hflip + vflip + hvflip. Returns averaged logits and cls_logits."""
    out0 = model(images)
    logits_0 = out0['logits']
    cls_0 = out0['cls_logits']

    images_hflip = torch.flip(images, dims=[3])
    out1 = model(images_hflip)
    logits_1 = torch.flip(out1['logits'], dims=[3])
    cls_1 = out1['cls_logits']

    images_vflip = torch.flip(images, dims=[2])
    out2 = model(images_vflip)
    logits_2 = torch.flip(out2['logits'], dims=[2])
    cls_2 = out2['cls_logits']

    images_hvflip = torch.flip(images, dims=[2, 3])
    out3 = model(images_hvflip)
    logits_3 = torch.flip(out3['logits'], dims=[2, 3])
    cls_3 = out3['cls_logits']

    logits = (logits_0 + logits_1 + logits_2 + logits_3) / 4.0
    cls_logits = (cls_0 + cls_1 + cls_2 + cls_3) / 4.0
    return logits, cls_logits


@torch.no_grad()
def run_inference_single(model, loader, device, use_tta=True):
    """Single model inference."""
    predictions = {}
    for batch in tqdm.tqdm(loader, desc="Inference"):
        images = batch['image'].to(device, non_blocking=True)
        image_ids = batch['image_id']

        with autocast('cuda'):
            if use_tta:
                logits, cls_logits = tta_forward(model, images)
            else:
                outputs = model(images)
                logits, cls_logits = outputs['logits'], outputs['cls_logits']

        probs = torch.sigmoid(logits)
        probs = F.interpolate(probs, size=config.MASK_SIZE, mode='bilinear', align_corners=False)
        probs = probs.cpu().numpy()
        cls_probs = torch.sigmoid(cls_logits).cpu().numpy()

        for j, img_id in enumerate(image_ids):
            predictions[img_id] = {'probs': probs[j], 'cls_probs': cls_probs[j]}

    return predictions


@torch.no_grad()
def run_inference_ensemble(models, loader, device, use_tta=True):
    """5-fold ensemble: average logits across models, then sigmoid."""
    predictions = {}
    num_models = len(models)

    for batch in tqdm.tqdm(loader, desc=f"Ensemble inference ({num_models} models)"):
        images = batch['image'].to(device, non_blocking=True)
        image_ids = batch['image_id']

        # Accumulate logits from all models
        logits_sum = None
        cls_logits_sum = None

        for model in models:
            with autocast('cuda'):
                if use_tta:
                    logits, cls_logits = tta_forward(model, images)
                else:
                    outputs = model(images)
                    logits, cls_logits = outputs['logits'], outputs['cls_logits']

            if logits_sum is None:
                logits_sum = logits
                cls_logits_sum = cls_logits
            else:
                logits_sum = logits_sum + logits
                cls_logits_sum = cls_logits_sum + cls_logits

        # Average logits, then sigmoid
        logits_avg = logits_sum / num_models
        cls_logits_avg = cls_logits_sum / num_models

        probs = torch.sigmoid(logits_avg)
        probs = F.interpolate(probs, size=config.MASK_SIZE, mode='bilinear', align_corners=False)
        probs = probs.cpu().numpy()
        cls_probs = torch.sigmoid(cls_logits_avg).cpu().numpy()

        for j, img_id in enumerate(image_ids):
            predictions[img_id] = {'probs': probs[j], 'cls_probs': cls_probs[j]}

    return predictions


def make_submission(predictions, cls_threshold=config.CLS_THRESHOLD,
                    pixel_threshold=config.PIXEL_THRESHOLD):
    """Convert predictions dict to submission DataFrame."""
    image_labels = []
    rles = []

    for image_id in sorted(predictions.keys()):
        pred = predictions[image_id]
        probs = pred['probs']          # (4, 350, 525)
        cls_probs = pred['cls_probs']  # (4,)

        for c in range(config.NUM_CLASSES):
            prob_c = probs[c]
            cls_present = cls_probs[c] > cls_threshold
            mask = (prob_c > pixel_threshold).astype(np.uint8)
            if not cls_present:
                mask = np.zeros_like(mask)

            rle = mask2rle(mask)
            image_labels.append(f"{image_id}_{config.LABELS[c]}")
            rles.append(rle)

    df = pd.DataFrame({'Image_Label': image_labels, 'EncodedPixels': rles})
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to single checkpoint')
    parser.add_argument('--ensemble', action='store_true',
                        help='Use 5-fold ensemble')
    parser.add_argument('--ckpts', type=str, default='1,1,1,1,1',
                        help='Checkpoint index per fold, e.g. "1,2,1,1,3"')
    parser.add_argument('--output', type=str,
                        default=os.path.join(config.OUTPUT_DIR, 'submission.csv'))
    parser.add_argument('--no-tta', action='store_true')
    parser.add_argument('--cls-threshold', type=float, default=config.CLS_THRESHOLD)
    parser.add_argument('--pixel-threshold', type=float, default=config.PIXEL_THRESHOLD)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Build test loader
    test_loader = build_test_dataloader()
    print(f"Test set: {len(test_loader.dataset)} images")

    if args.ensemble:
        # Load one model per fold
        ckpt_indices = [int(x) for x in args.ckpts.split(',')]
        assert len(ckpt_indices) == config.NUM_FOLDS, \
            f"Need {config.NUM_FOLDS} checkpoint indices, got {len(ckpt_indices)}"

        models = []
        for fold, ckpt_idx in enumerate(ckpt_indices):
            ckpt_path = os.path.join(config.OUTPUT_DIR, f'fold_{fold}', f'best_model_{ckpt_idx}.pth')
            model = load_model(ckpt_path, device)
            models.append(model)

        predictions = run_inference_ensemble(models, test_loader, device, use_tta=not args.no_tta)
    else:
        # Single model
        if args.checkpoint is None:
            args.checkpoint = os.path.join(config.OUTPUT_DIR, 'fold_0', 'best_model_1.pth')
        model = load_model(args.checkpoint, device)
        predictions = run_inference_single(model, test_loader, device, use_tta=not args.no_tta)

    # Generate submission
    df = make_submission(predictions, args.cls_threshold, args.pixel_threshold)

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"\nSubmission saved to {args.output}")
    print(f"Shape: {df.shape}")

    expected_rows = len(test_loader.dataset) * config.NUM_CLASSES
    assert len(df) == expected_rows, f"Expected {expected_rows} rows, got {len(df)}"

    non_empty = (df['EncodedPixels'] != '').sum()
    print(f"Non-empty masks: {non_empty}/{len(df)} ({100*non_empty/len(df):.1f}%)")


if __name__ == '__main__':
    main()
