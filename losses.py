import torch
import torch.nn as nn
import torch.nn.functional as F


class CloudSegLoss(nn.Module):
    """Combined BCE + Dice + Classification loss with ignore_negative support.

    Following 1st place solution:
    - ignore_negative: only compute seg loss on classes that have clouds in GT
    - cls_weight: weight for classification (cloud present/absent) loss
    """

    def __init__(self, bce_weight=0.75, dice_weight=0.25, cls_weight=0.1,
                 ignore_negative=True, smooth=1.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.cls_weight = cls_weight
        self.ignore_negative = ignore_negative
        self.smooth = smooth
        self.cls_bce = nn.BCEWithLogitsLoss()

    def forward(self, outputs, targets):
        """
        Args:
            outputs: dict with 'logits' (B, C, H, W) and 'cls_logits' (B, C)
            targets: (B, C, H, W) binary masks
        Returns:
            dict with loss components
        """
        logits = outputs['logits']
        cls_logits = outputs['cls_logits']
        B, C, H, W = targets.shape

        # Classification labels: does this class have any cloud pixels?
        cls_labels = (targets.view(B, C, -1).sum(dim=2) > 0).float()  # (B, C)
        loss_cls = self.cls_bce(cls_logits, cls_labels)

        # Per-class BCE loss
        bce_per_pixel = F.binary_cross_entropy_with_logits(
            logits, targets.float(), reduction='none'
        )  # (B, C, H, W)
        bce_per_class = bce_per_pixel.mean(dim=[2, 3])  # (B, C)

        # Per-class Dice loss
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(B, C, -1)
        targets_flat = targets.view(B, C, -1).float()
        intersection = (probs_flat * targets_flat).sum(dim=2)  # (B, C)
        union = probs_flat.sum(dim=2) + targets_flat.sum(dim=2)  # (B, C)
        dice_per_class = 1.0 - (2.0 * intersection + self.smooth) / (union + self.smooth)  # (B, C)

        if self.ignore_negative:
            # Only average over classes that have clouds (positive classes)
            mask = cls_labels.detach()  # (B, C)
            num_pos = mask.sum().clamp(min=1)
            bce_loss = (bce_per_class * mask).sum() / num_pos
            dice_loss = (dice_per_class * mask).sum() / num_pos
        else:
            bce_loss = bce_per_class.mean()
            dice_loss = dice_per_class.mean()

        loss_seg = self.bce_weight * bce_loss + self.dice_weight * dice_loss
        loss = loss_seg + self.cls_weight * loss_cls

        return {
            'loss': loss,
            'bce_loss': bce_loss,
            'dice_loss': dice_loss,
            'cls_loss': loss_cls,
        }
