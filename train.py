"""Training loop for Cloud Segmentation with DINOv3 SAT + FPN."""

import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

import config
from dataset import build_dataloaders
from losses import CloudSegLoss
from model import build_model
from utils import EMA, compute_dice


def build_optimizer(model):
    param_groups = model.get_trainable_parameters()
    return torch.optim.AdamW(param_groups, lr=config.LR, weight_decay=config.WEIGHT_DECAY)


def build_scheduler(optimizer, total_steps):
    def lr_lambda(step):
        if step < config.WARMUP_STEPS:
            return step / max(config.WARMUP_STEPS, 1)
        progress = (step - config.WARMUP_STEPS) / max(total_steps - config.WARMUP_STEPS, 1)
        return 0.5 * (1.0 + np.cos(np.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, ema, device, writer, epoch, global_step):
    model.train()
    running_loss = 0.0
    running_bce = 0.0
    running_dice = 0.0
    running_cls = 0.0
    num_batches = 0

    optimizer.zero_grad()

    for i, batch in enumerate(loader):
        images = batch['image'].to(device, non_blocking=True)
        masks = batch['mask'].to(device, non_blocking=True)

        with autocast('cuda'):
            outputs = model(images)
            loss_dict = criterion(outputs, masks)
            loss = loss_dict['loss'] / config.ACCUMULATION_STEPS

        scaler.scale(loss).backward()

        if (i + 1) % config.ACCUMULATION_STEPS == 0:
            scaler.unscale_(optimizer)
            all_params = [p for g in model.get_trainable_parameters() for p in g['params']]
            torch.nn.utils.clip_grad_norm_(all_params, 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
            ema.update(model)

        running_loss += loss_dict['loss'].item()
        running_bce += loss_dict['bce_loss'].item()
        running_dice += loss_dict['dice_loss'].item()
        running_cls += loss_dict['cls_loss'].item()
        num_batches += 1
        global_step += 1

        if num_batches % 50 == 0:
            avg_loss = running_loss / num_batches
            lr = optimizer.param_groups[0]['lr']
            print(f"  [Epoch {epoch}] step {num_batches}/{len(loader)} "
                  f"loss={avg_loss:.4f} lr={lr:.2e}")
            writer.add_scalar('train/loss', avg_loss, global_step)
            writer.add_scalar('train/bce_loss', running_bce / num_batches, global_step)
            writer.add_scalar('train/dice_loss', running_dice / num_batches, global_step)
            writer.add_scalar('train/cls_loss', running_cls / num_batches, global_step)
            writer.add_scalar('train/lr', lr, global_step)

    avg_loss = running_loss / max(num_batches, 1)
    return avg_loss, global_step


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_masks = []

    for batch in loader:
        images = batch['image'].to(device, non_blocking=True)
        masks = batch['mask'].to(device, non_blocking=True)

        with autocast('cuda'):
            outputs = model(images)
            loss_dict = criterion(outputs, masks)

        running_loss += loss_dict['loss'].item()

        logits = outputs['logits']
        cls_logits = outputs['cls_logits']

        # Resize logits to mask submission size for metric computation
        probs = torch.sigmoid(logits)
        probs_resized = F.interpolate(probs, size=config.MASK_SIZE, mode='bilinear', align_corners=False)
        masks_resized = F.interpolate(masks, size=config.MASK_SIZE, mode='nearest')

        preds_binary = (probs_resized > config.PIXEL_THRESHOLD).float()

        # Use classification head for thresholding
        cls_probs = torch.sigmoid(cls_logits)  # (B, C)
        cls_mask = (cls_probs > config.CLS_THRESHOLD).float().unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        preds_binary = preds_binary * cls_mask

        all_preds.append(preds_binary.cpu())
        all_masks.append(masks_resized.cpu())

    all_preds = torch.cat(all_preds, dim=0)
    all_masks = torch.cat(all_masks, dim=0)

    avg_loss = running_loss / max(len(loader), 1)
    dice_score = compute_dice(all_preds, all_masks)

    return avg_loss, dice_score


class TopKCheckpoint:
    """Keep top-K checkpoints by dice score."""

    def __init__(self, output_dir, k=3):
        self.output_dir = output_dir
        self.k = k
        self.top_k = []  # list of (dice, epoch, path)

    def update(self, dice, epoch, model_state, ema_state, optimizer_state):
        ckpt_data = {
            'epoch': epoch,
            'model_state_dict': model_state,
            'ema_state_dict': ema_state,
            'optimizer_state_dict': optimizer_state,
            'best_dice': dice,
        }

        if len(self.top_k) < self.k:
            path = os.path.join(self.output_dir, f'best_model_{len(self.top_k) + 1}.pth')
            torch.save(ckpt_data, path)
            self.top_k.append((dice, epoch, path))
            self.top_k.sort(key=lambda x: -x[0])
            print(f"  -> Saved checkpoint (dice={dice:.4f}, rank {self._rank(dice)}/{self.k})")
            return True
        elif dice > self.top_k[-1][0]:
            # Replace worst
            _, _, old_path = self.top_k.pop()
            torch.save(ckpt_data, old_path)
            self.top_k.append((dice, epoch, old_path))
            self.top_k.sort(key=lambda x: -x[0])
            # Rename files to match rank order
            self._reorder_files()
            print(f"  -> Saved checkpoint (dice={dice:.4f}, rank {self._rank(dice)}/{self.k})")
            return True
        return False

    def _rank(self, dice):
        for i, (d, _, _) in enumerate(self.top_k):
            if abs(d - dice) < 1e-8:
                return i + 1
        return self.k

    def _reorder_files(self):
        """Rename checkpoint files to match rank order."""
        # First rename to temp names to avoid conflicts
        temp_map = {}
        for i, (dice, epoch, old_path) in enumerate(self.top_k):
            temp_path = os.path.join(self.output_dir, f'_tmp_best_{i}.pth')
            if os.path.exists(old_path):
                os.rename(old_path, temp_path)
            temp_map[i] = temp_path

        # Then rename to final names
        new_top_k = []
        for i, (dice, epoch, _) in enumerate(self.top_k):
            new_path = os.path.join(self.output_dir, f'best_model_{i + 1}.pth')
            if os.path.exists(temp_map[i]):
                os.rename(temp_map[i], new_path)
            new_top_k.append((dice, epoch, new_path))
        self.top_k = new_top_k

    def best_dice(self):
        return self.top_k[0][0] if self.top_k else 0.0

    def summary(self):
        lines = []
        for i, (dice, epoch, path) in enumerate(self.top_k):
            lines.append(f"  #{i+1}: dice={dice:.4f} (epoch {epoch}) -> {os.path.basename(path)}")
        return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, default=0, help='Fold index (0-4)')
    args = parser.parse_args()

    fold = args.fold
    output_dir = os.path.join(config.OUTPUT_DIR, f'fold_{fold}')
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Fold: {fold}")

    # Build data
    train_loader, val_loader = build_dataloaders(fold)
    print(f"Train: {len(train_loader.dataset)} images, Val: {len(val_loader.dataset)} images")

    # Build model
    model = build_model()
    model = model.to(device)

    # Build training components
    criterion = CloudSegLoss(
        bce_weight=config.BCE_WEIGHT, dice_weight=config.DICE_WEIGHT,
        cls_weight=config.CLS_WEIGHT, ignore_negative=config.IGNORE_NEGATIVE,
    )
    optimizer = build_optimizer(model)
    total_steps = (len(train_loader) // config.ACCUMULATION_STEPS) * config.NUM_EPOCHS
    scheduler = build_scheduler(optimizer, total_steps)
    scaler = GradScaler('cuda')
    ema = EMA(model, decay=config.EMA_DECAY)

    writer = SummaryWriter(log_dir=os.path.join(output_dir, 'logs'))
    topk_ckpt = TopKCheckpoint(output_dir, k=3)

    global_step = 0

    for epoch in range(1, config.NUM_EPOCHS + 1):
        t0 = time.time()
        print(f"\n{'='*60}")
        print(f"[Fold {fold}] Epoch {epoch}/{config.NUM_EPOCHS}")
        print(f"{'='*60}")

        # Train
        train_loss, global_step = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, scaler, ema, device,
            writer, epoch, global_step,
        )

        # Validate with EMA model
        val_loss, val_dice = validate(ema.module(), val_loader, criterion, device)

        elapsed = time.time() - t0
        print(f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
              f"val_dice={val_dice:.4f} time={elapsed:.0f}s")

        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/dice', val_dice, epoch)

        # Save top-K checkpoints
        topk_ckpt.update(
            val_dice, epoch,
            model.state_dict(), ema.state_dict(), optimizer.state_dict(),
        )

    writer.close()
    print(f"\n[Fold {fold}] Training complete.")
    print(f"Top-3 checkpoints:\n{topk_ckpt.summary()}")


if __name__ == '__main__':
    main()
