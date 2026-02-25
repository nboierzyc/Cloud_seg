import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── RLE encode / decode ──────────────────────────────────────────────

def rle2mask(height, width, encoded):
    """Decode RLE string to binary mask of shape (height, width)."""
    if isinstance(encoded, float) or encoded == '':
        return np.zeros((height, width), dtype=np.uint8)
    s = encoded.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0::2], s[1::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(height * width, dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape((width, height)).T


def mask2rle(img):
    """Encode binary mask (H, W) to RLE string."""
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


# ── Dice metric ──────────────────────────────────────────────────────

def compute_dice(pred, target, eps=1e-6):
    """Compute per-channel Dice score.

    Args:
        pred: (B, C, H, W) binary predictions (0/1 numpy or tensor)
        target: (B, C, H, W) binary ground-truth
    Returns:
        dict with 'mean_dice' and per-class dice
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()

    B, C, H, W = pred.shape
    pred = pred.reshape(B * C, H * W)
    target = target.reshape(B * C, H * W)

    intersection = np.sum(pred * target, axis=1)
    sum_p = np.sum(pred, axis=1)
    sum_t = np.sum(target, axis=1)

    dice = (2 * intersection) / (sum_p + sum_t + eps)

    # For empty GT and empty prediction, dice = 1
    empty = (sum_p + sum_t) == 0
    dice[empty] = 1.0

    mean_dice = float(np.mean(dice))
    return mean_dice


# ── Dice Loss ─────────────────────────────────────────────────────────

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        Args:
            logits: (B, C, H, W) raw logits
            targets: (B, C, H, W) binary targets
        """
        probs = torch.sigmoid(logits)
        B, C, H, W = probs.shape
        probs = probs.view(B * C, -1)
        targets = targets.view(B * C, -1).float()

        intersection = (probs * targets).sum(dim=1)
        union = probs.sum(dim=1) + targets.sum(dim=1)

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


# ── EMA ───────────────────────────────────────────────────────────────

class EMA:
    """Exponential Moving Average of model parameters."""

    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for s_param, m_param in zip(self.shadow.parameters(), model.parameters()):
            s_param.data.mul_(self.decay).add_(m_param.data, alpha=1.0 - self.decay)
        for s_buf, m_buf in zip(self.shadow.buffers(), model.buffers()):
            s_buf.copy_(m_buf)

    def state_dict(self):
        return self.shadow.state_dict()

    def load_state_dict(self, state_dict):
        self.shadow.load_state_dict(state_dict)

    def module(self):
        return self.shadow
