"""DINOv3 SAT ViT-L + FPN Adapter + Segmentation Head with Classification Head."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

import config


# ── DINOv3 Backbone ──────────────────────────────────────────────────

class DINOv3Backbone(nn.Module):
    """Extract multi-layer features from DINOv3 SAT ViT-L, with partial unfreezing."""

    def __init__(self, layers=None, unfreeze_last_n=config.BACKBONE_UNFREEZE_LAYERS):
        super().__init__()
        self.layers = layers or config.DINO_LAYERS  # 1-indexed: [6, 12, 18, 24]
        self.model = AutoModel.from_pretrained(config.DINO_MODEL)
        self.patch_size = 16
        self.hidden_size = 1024  # DINOv3 ViT-L hidden dim
        self.num_prefix_tokens = 1 + 4  # 1 CLS + 4 register tokens
        self.unfreeze_last_n = unfreeze_last_n

        # Freeze all parameters first
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze last N transformer blocks + final norm
        num_blocks = len(self.model.layer)  # 24
        self.freeze_until = num_blocks - unfreeze_last_n  # 18
        for i in range(self.freeze_until, num_blocks):
            for param in self.model.layer[i].parameters():
                param.requires_grad = True
        for param in self.model.norm.parameters():
            param.requires_grad = True

    def train(self, mode=True):
        super().train(mode)
        # Keep frozen parts in eval mode
        self.model.embeddings.eval()
        for i in range(self.freeze_until):
            self.model.layer[i].eval()
        return self

    def get_unfrozen_parameters(self):
        """Return parameters of unfrozen layers (for separate lr group)."""
        params = []
        for i in range(self.freeze_until, len(self.model.layer)):
            params.extend(self.model.layer[i].parameters())
        params.extend(self.model.norm.parameters())
        return params

    def forward(self, x):
        B, _, H, W = x.shape
        h = H // self.patch_size
        w = W // self.patch_size

        outputs = self.model(pixel_values=x, output_hidden_states=True)
        hidden_states = outputs.hidden_states

        features = []
        for layer_idx in self.layers:  # [6, 12, 18, 24]
            hs = hidden_states[layer_idx]  # (B, seq_len, C)
            hs = hs[:, self.num_prefix_tokens:, :]  # remove CLS + registers
            hs = hs.reshape(B, h, w, -1).permute(0, 3, 1, 2)  # (B, C, h, w)
            features.append(hs)

        return features


# ── Feature Adapter (FPN) ────────────────────────────────────────────

class FeatureAdapter(nn.Module):
    """Convert multi-layer ViT features to multi-scale feature maps with FPN."""

    def __init__(self, in_dim=config.DINO_FEATURE_DIM, out_dim=config.DECODER_DIM):
        super().__init__()
        # Lateral connections: project each layer to out_dim
        self.laterals = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_dim, out_dim, 1),
                nn.GroupNorm(32, out_dim),
            )
            for _ in range(4)
        ])

        # Generate multi-scale from base resolution
        # DINOv3 patch_size=16, IMG_SIZE=(384,576) -> feature map = 24x36

        # Scale 0: upsample 2x (24x36 -> 48x72)
        self.scale_0 = nn.Sequential(
            nn.ConvTranspose2d(out_dim, out_dim, kernel_size=2, stride=2),
            nn.GroupNorm(32, out_dim),
            nn.GELU(),
        )
        # Scale 1: base resolution (24x36)
        self.scale_1 = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, 3, padding=1),
            nn.GroupNorm(32, out_dim),
            nn.GELU(),
        )
        # Scale 2: downsample 2x (24x36 -> 12x18)
        self.scale_2 = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, 3, stride=2, padding=1),
            nn.GroupNorm(32, out_dim),
            nn.GELU(),
        )
        # Scale 3: downsample 4x (24x36 -> 6x9)
        self.scale_3 = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, 3, stride=2, padding=1),
            nn.GroupNorm(32, out_dim),
            nn.GELU(),
            nn.Conv2d(out_dim, out_dim, 3, stride=2, padding=1),
            nn.GroupNorm(32, out_dim),
            nn.GELU(),
        )

    def forward(self, features):
        # Apply lateral projections
        laterals = [lat(f) for lat, f in zip(self.laterals, features)]

        # FPN top-down pathway
        for i in range(len(laterals) - 2, -1, -1):
            laterals[i] = laterals[i] + F.interpolate(
                laterals[i + 1], size=laterals[i].shape[2:], mode='bilinear', align_corners=False
            )

        # Generate multi-scale outputs
        ms_features = [
            self.scale_0(laterals[0]),  # highest resolution
            self.scale_1(laterals[1]),  # base resolution
            self.scale_2(laterals[2]),  # 2x downsampled
            self.scale_3(laterals[3]),  # 4x downsampled
        ]

        return ms_features


# ── FPN Segmentation Head + Classification Head ─────────────────────

class FPNSegHead(nn.Module):
    """Simple per-pixel segmentation head + classification head on merged FPN features."""

    def __init__(self, in_dim=config.DECODER_DIM, num_classes=config.NUM_CLASSES, dropout=0.2):
        super().__init__()
        self.in_dim = in_dim

        # Segmentation head on merged multi-scale features
        self.seg_head = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, 3, padding=1),
            nn.GroupNorm(32, in_dim),
            nn.GELU(),
            nn.Conv2d(in_dim, in_dim, 3, padding=1),
            nn.GroupNorm(32, in_dim),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(in_dim, num_classes, 1),
        )

        # Classification head (cloud present/absent per class)
        # Uses attention-weighted pooling like 1st place solution
        cls_feat = in_dim // 4
        self.cls_head = nn.Sequential(
            nn.Linear(in_dim, cls_feat),
            nn.LayerNorm(cls_feat),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(cls_feat, num_classes),
        )

    def forward(self, ms_features):
        # Merge all FPN scales by upsampling to highest resolution and summing
        target_size = ms_features[0].shape[2:]
        merged = ms_features[0]
        for f in ms_features[1:]:
            merged = merged + F.interpolate(f, size=target_size, mode='bilinear', align_corners=False)

        # Segmentation
        logits = self.seg_head(merged)  # (B, num_classes, H, W)

        # Classification with attention-weighted pooling (1st place approach)
        # Use seg predictions to focus pooling on cloud regions
        p = torch.sigmoid(logits).detach()
        p_max = p.max(dim=1, keepdim=True)[0]  # (B, 1, H, W) - max across classes
        weighted = merged * p_max  # weight features by cloud probability
        pooled = F.adaptive_avg_pool2d(weighted, 1).squeeze(-1).squeeze(-1)  # (B, dim)
        cls_logits = self.cls_head(pooled)  # (B, num_classes)

        return {'logits': logits, 'cls_logits': cls_logits}


# ── Full Model ───────────────────────────────────────────────────────

class CloudSegModel(nn.Module):
    """DINOv3 SAT + FPN Adapter + Segmentation/Classification Head."""

    def __init__(self):
        super().__init__()
        self.backbone = DINOv3Backbone()
        self.adapter = FeatureAdapter()
        self.head = FPNSegHead()

    def forward(self, x):
        features = self.backbone(x)
        ms_features = self.adapter(features)
        outputs = self.head(ms_features)

        # Upsample logits to input resolution
        outputs['logits'] = F.interpolate(
            outputs['logits'], size=x.shape[2:], mode='bilinear', align_corners=False
        )
        return outputs

    def get_trainable_parameters(self):
        """Return param groups: backbone (low lr) + adapter/head (normal lr)."""
        return [
            {'params': self.backbone.get_unfrozen_parameters(), 'lr': config.BACKBONE_LR},
            {'params': list(self.adapter.parameters()) + list(self.head.parameters())},
        ]


def build_model():
    model = CloudSegModel()
    backbone_trainable = sum(p.numel() for p in model.backbone.get_unfrozen_parameters())
    head_trainable = sum(p.numel() for p in model.adapter.parameters()) + \
                     sum(p.numel() for p in model.head.parameters())
    total = sum(p.numel() for p in model.parameters())
    print(f"Model: {total / 1e6:.1f}M total params, "
          f"backbone unfrozen: {backbone_trainable / 1e6:.1f}M (lr={config.BACKBONE_LR}), "
          f"head: {head_trainable / 1e6:.1f}M (lr={config.LR})")
    return model
