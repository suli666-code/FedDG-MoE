"""Adapter modules for content/style decoupling and softmax routing."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


_VALID_ADAPTER_MODES = {"full", "extract_style"}


def _normalize_adapter_mode(mode: str) -> str:
    normalized = str(mode).strip().lower()
    if normalized not in _VALID_ADAPTER_MODES:
        raise ValueError(f"Unknown adapter mode: {mode}")
    return normalized


class DecoupledAdapter(nn.Module):
    """Content/style decoupled adapter with externally scheduled routing temperature."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        content_r: int = 16,
        style_r: int = 4,
        adapter_scale: float = 1.0,
        router_temp_init: float = 5.0,
        router_temp_min: float = 1.5,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.content_r = int(content_r)
        self.style_r = int(style_r)
        self.content_scale = float(adapter_scale) / max(1.0, float(self.content_r))
        self.style_scale = float(adapter_scale) / max(1.0, float(self.style_r))
        self.router_temp_init = float(router_temp_init)
        self.router_temp_min = float(router_temp_min)
        self.router_progress: float = 0.0
        self.router_temp_current: float = self.router_temp_init

        self.input_norm = nn.LayerNorm(self.in_features)
        self.adapter_dropout = nn.Dropout(p=0.2)
        self.act = nn.SiLU()

        self.content_down = nn.Linear(self.in_features, self.content_r, bias=False)
        self.content_norm = nn.InstanceNorm1d(self.content_r, affine=True)
        self.content_up = nn.Linear(self.content_r, self.out_features, bias=False)

        self.style_down = nn.Linear(self.in_features * 2, self.style_r, bias=False)
        self.style_up = nn.Linear(self.style_r, self.out_features, bias=False)

        self.router_mlp = nn.Sequential(
            nn.Linear(self.in_features, 64, bias=True),
            nn.SiLU(),
            nn.Linear(64, 2, bias=True),
        )
        self.gamma = nn.Parameter(torch.ones(1))
        self.last_style_stats_raw: torch.Tensor | None = None
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.content_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.content_up.weight)
        nn.init.kaiming_uniform_(self.style_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.style_up.weight)
        nn.init.kaiming_uniform_(self.router_mlp[0].weight, a=math.sqrt(5))
        nn.init.zeros_(self.router_mlp[0].bias)
        nn.init.kaiming_uniform_(self.router_mlp[2].weight, a=math.sqrt(5))
        nn.init.zeros_(self.router_mlp[2].bias)

    def set_router_progress(self, progress: float) -> None:
        progress_clamped = min(1.0, max(0.0, float(progress)))
        self.router_progress = progress_clamped
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress_clamped))
        self.router_temp_current = self.router_temp_min + (
            self.router_temp_init - self.router_temp_min
        ) * cosine

    def _content_out(self, x_norm: torch.Tensor) -> torch.Tensor:
        content_hidden = self.content_down(x_norm)
        if content_hidden.size(1) > 1:
            content_hidden = self.content_norm(content_hidden.transpose(1, 2)).transpose(1, 2)
        content_hidden = self.adapter_dropout(self.act(content_hidden))
        content_out = self.content_up(content_hidden) * self.content_scale
        return content_out

    def _style_statistics(self, x_norm: torch.Tensor) -> torch.Tensor:
        patch_tokens = x_norm[:, 1:, :] if x_norm.size(1) > 1 else x_norm
        mu = torch.mean(patch_tokens, dim=1, keepdim=True)
        std = torch.std(patch_tokens, dim=1, keepdim=True, unbiased=False) + 1e-6
        style_statistics = torch.cat([mu, std], dim=-1)
        self.last_style_stats_raw = style_statistics
        return style_statistics

    def forward(
        self,
        x: torch.Tensor,
        target_inference_mode: str = "full",
    ) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"Adapter expects token input [B, T, C], got {tuple(x.shape)}")
        adapter_mode = _normalize_adapter_mode(target_inference_mode)
        x_norm = self.input_norm(x)

        self.last_style_stats_raw = None
        content_out = self._content_out(x_norm)

        style_statistics = self._style_statistics(x_norm)
        if adapter_mode == "extract_style":
            return torch.zeros(
                (x_norm.size(0), x_norm.size(1), self.out_features),
                device=x_norm.device,
                dtype=x_norm.dtype,
            )

        style_hidden = self.adapter_dropout(self.act(self.style_down(style_statistics)))
        global_style_out = self.style_up(style_hidden) * self.style_scale
        style_out = global_style_out.expand(-1, x_norm.size(1), -1)

        route_logits = self.router_mlp(x_norm[:, 0, :])
        route_weights = torch.softmax(route_logits / self.router_temp_current, dim=-1)
        alpha = route_weights[:, 0:1].unsqueeze(-1)
        beta = route_weights[:, 1:2].unsqueeze(-1)

        out = (alpha * content_out + beta * style_out) * self.gamma
        return out


__all__ = ["DecoupledAdapter"]
