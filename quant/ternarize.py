"""
Ternary quantization (TWN-style) utilities and layers adapted for Sfiral topology.
Implements layer-wise ternarization of weights to {-alpha, 0, +alpha}
with threshold Delta = t * E(|W|) and Straight-Through Estimator (STE).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class TernaryConfig:
    t: float = 0.7  # threshold factor
    per_channel: bool = False
    channel_dim: int = 0
    enable: bool = True

def _compute_delta_alpha(w: torch.Tensor, cfg: TernaryConfig) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    abs_w = w.abs()
    if cfg.per_channel:
        reduce_dims = [d for d in range(w.dim()) if d != cfg.channel_dim]
        mean_abs = abs_w.mean(dim=reduce_dims, keepdim=True)
        delta = cfg.t * mean_abs
        mask = (abs_w > delta).to(w.dtype)
        num = (abs_w * mask).sum(dim=reduce_dims, keepdim=True)
        den = mask.sum(dim=reduce_dims, keepdim=True).clamp(min=1.0)
        alpha = num / den
    else:
        mean_abs = abs_w.mean()
        delta = cfg.t * mean_abs
        mask = (abs_w > delta).to(w.dtype)
        num = (abs_w * mask).sum()
        den = mask.sum().clamp(min=1.0)
        alpha = num / den
    return delta, alpha, mask

def ternarize_weight(w: torch.Tensor, cfg: TernaryConfig) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not cfg.enable:
        return w, torch.tensor(1.0, device=w.device, dtype=w.dtype), torch.tensor(0.0, device=w.device, dtype=w.dtype)

    with torch.no_grad():
        delta, alpha, mask = _compute_delta_alpha(w, cfg)
        sign = torch.sign(w)
        w_tern = alpha * sign * mask

    w_q = w_tern.detach() - w.detach() + w
    return w_q, alpha.detach(), delta.detach()

class TernaryQuantizer(nn.Module):
    def __init__(self, t: float = 0.7, per_channel: bool = False, channel_dim: int = 0, enable: bool = True):
        super().__init__()
        self.cfg = TernaryConfig(t=t, per_channel=per_channel, channel_dim=channel_dim, enable=enable)

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        w_q, _, _ = ternarize_weight(w, self.cfg)
        return w_q

    def set_enable(self, enable: bool) -> None:
        self.cfg.enable = enable

class TernaryLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, t: float = 0.7, per_channel: bool = False):
        super().__init__(in_features, out_features, bias=bias)
        self.quant = TernaryQuantizer(t=t, per_channel=per_channel, channel_dim=0, enable=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_q, _, _ = ternarize_weight(self.weight, self.quant.cfg)
        return F.linear(x, w_q, self.bias)