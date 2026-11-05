"""A compact CNN for chart images.

Input : [B, C, S, S] float in 0..1 (C = number of drawn lines)
Output: * one 3-way logit vector per horizon (down / flat / up)
        * a regression vector with the projected MA path (pct change per day)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class ChartCNN(nn.Module):
    def __init__(self, in_channels: int, n_horizons: int, path_horizon: int,
                 n_classes: int = 3, width: int = 16, dropout: float = 0.3):
        super().__init__()
        self.n_horizons = n_horizons
        self.n_classes = n_classes
        self.features = nn.Sequential(
            _block(in_channels, width),        # S/2
            _block(width, width * 2),          # S/4
            _block(width * 2, width * 4),      # S/8
            _block(width * 4, width * 4),      # S/16
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.trunk = nn.Sequential(
            nn.Linear(width * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.cls_heads = nn.ModuleList([nn.Linear(256, n_classes) for _ in range(n_horizons)])
        self.path_head = nn.Linear(256, path_horizon)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.trunk(self.features(x))
        logits = torch.stack([h(z) for h in self.cls_heads], dim=1)  # [B, H, 3]
        path = self.path_head(z)                                      # [B, P]
        return logits, path


def multitask_loss(logits: torch.Tensor, path: torch.Tensor,
                   labels: torch.Tensor, path_target: torch.Tensor,
                   class_w: torch.Tensor | None = None,
                   path_weight: float = 20.0) -> tuple[torch.Tensor, dict[str, float]]:
    """Cross-entropy per horizon + scaled MSE on the path (returns are small,
    so the MSE is multiplied to matter)."""
    cls_losses = []
    for h in range(logits.shape[1]):
        w = class_w[h] if class_w is not None else None
        cls_losses.append(F.cross_entropy(logits[:, h], labels[:, h], weight=w))
    cls_loss = torch.stack(cls_losses).mean()
    path_loss = F.mse_loss(path, path_target) * path_weight
    total = cls_loss + path_loss
    return total, {"cls": cls_loss.item(), "path": path_loss.item()}
