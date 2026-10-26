from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn
from torch.nn import functional as F


def _group_count(channels: int) -> int:
    for groups in (32, 24, 16, 12, 8, 6, 4, 3, 2):
        if channels % groups == 0:
            return groups
    return 1


class Float32GroupNorm(nn.GroupNorm):
    """Keep numerically sensitive normalization in FP32 under autocast."""

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        normalized = F.group_norm(
            value.float(),
            self.num_groups,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return normalized.to(value.dtype)


class SourceAdapter(nn.Module):
    def __init__(self, input_channels: int, width: int):
        super().__init__()
        self.projection = nn.Conv2d(input_channels, width, kernel_size=1)
        self.normalization = Float32GroupNorm(_group_count(width), width)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if not torch.is_autocast_enabled(value.device.type):
            value = value.to(self.projection.weight.dtype)
        return F.gelu(self.normalization(self.projection(value)))


class SpatialResidualBlock(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        groups = _group_count(width)
        self.norm1 = Float32GroupNorm(groups, width)
        self.conv1 = nn.Conv2d(width, width, kernel_size=3, padding=1)
        self.norm2 = Float32GroupNorm(groups, width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, padding=1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = self.conv1(F.gelu(self.norm1(value)))
        residual = self.conv2(F.gelu(self.norm2(residual)))
        return value + residual


class OwnershipDecoder(nn.Module):
    """Small static decoder for mutually exclusive A1/A2/background logits."""

    def __init__(
        self,
        *,
        input_channels: Mapping[str, int],
        width: int = 192,
        residual_blocks: int = 3,
    ):
        super().__init__()
        if not input_channels or any(channels < 1 for channels in input_channels.values()):
            raise ValueError("input channels must define at least one positive-width source")
        if width < 1 or residual_blocks < 0:
            raise ValueError("decoder width must be positive and residual block count non-negative")
        self.source_names = tuple(input_channels)
        self.adapters = nn.ModuleDict(
            {
                name: SourceAdapter(int(input_channels[name]), width)
                for name in self.source_names
            }
        )
        self.source_logits = nn.Parameter(torch.zeros(len(self.source_names)))
        self.blocks = nn.Sequential(*(SpatialResidualBlock(width) for _ in range(residual_blocks)))
        self.output_norm = Float32GroupNorm(_group_count(width), width)
        self.head = nn.Conv2d(width, 3, kernel_size=1)

    def encode_spatial(
        self,
        spatial: Mapping[str, torch.Tensor],
        *,
        output_size: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        if set(spatial) != set(self.source_names):
            raise ValueError(
                f"spatial source set mismatch: expected {sorted(self.source_names)}, got {sorted(spatial)}"
            )
        first = spatial[self.source_names[0]]
        if first.ndim != 4:
            raise ValueError("spatial tensors must have shape [B,C,H,W]")
        if output_size is None:
            output_size = max(
                (tuple(value.shape[-2:]) for value in spatial.values()),
                key=lambda shape: shape[0] * shape[1],
            )
        batch_size = first.shape[0]
        adapted = []
        for name in self.source_names:
            value = spatial[name]
            if value.ndim != 4 or value.shape[0] != batch_size:
                raise ValueError("all spatial sources must be batched [B,C,H,W] with equal batch size")
            value = self.adapters[name](value)
            if tuple(value.shape[-2:]) != tuple(output_size):
                value = F.interpolate(value, size=output_size, mode="bilinear", align_corners=False)
            adapted.append(value)
        source_weights = torch.softmax(self.source_logits.float(), dim=0).to(adapted[0].dtype)
        fused = sum(weight * value for weight, value in zip(source_weights, adapted, strict=True))
        return self.blocks(fused)

    def forward(
        self,
        spatial: Mapping[str, torch.Tensor],
        *,
        output_size: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        fused = self.encode_spatial(spatial, output_size=output_size)
        return self.head(F.gelu(self.output_norm(fused)))


class SemanticOwnershipDecoder(OwnershipDecoder):
    """Query-conditioned decoder with exact A1/A2 swap equivariance."""

    requires_actor_states = True

    def __init__(
        self,
        *,
        input_channels: Mapping[str, int],
        semantic_dim: int = 5120,
        width: int = 192,
        residual_blocks: int = 3,
    ):
        super().__init__(
            input_channels=input_channels,
            width=width,
            residual_blocks=residual_blocks,
        )
        if semantic_dim < 1:
            raise ValueError("semantic dimension must be positive")
        del self.head
        self.semantic_dim = semantic_dim
        self.query_projection = nn.Linear(semantic_dim, width, bias=False)
        self.context_projection = nn.Linear(semantic_dim, 2 * width)
        self.pixel_projection = nn.Conv2d(width, width, kernel_size=1, bias=False)
        self.background_head = nn.Conv2d(width, 1, kernel_size=1)
        self.foreground_head = nn.Conv2d(width, 1, kernel_size=1)
        self.logit_scale = nn.Parameter(torch.tensor(2.0))

    def forward(
        self,
        spatial: Mapping[str, torch.Tensor],
        *,
        actor_states: torch.Tensor,
        output_size: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        if actor_states.ndim != 3 or actor_states.shape[1:] != (2, self.semantic_dim):
            raise ValueError(
                f"actor states must have shape [B,2,{self.semantic_dim}], got {tuple(actor_states.shape)}"
            )
        fused = self.encode_spatial(spatial, output_size=output_size)
        if actor_states.shape[0] != fused.shape[0]:
            raise ValueError("actor states and spatial features must have equal batch size")

        projection_states = actor_states
        if not torch.is_autocast_enabled(actor_states.device.type):
            projection_states = actor_states.to(self.query_projection.weight.dtype)
        pair_context = projection_states.mean(dim=1)
        modulation = self.context_projection(pair_context).to(fused.dtype)
        scale, bias = modulation.chunk(2, dim=-1)
        fused = fused * (1.0 + 0.1 * torch.tanh(scale)[:, :, None, None])
        fused = fused + 0.1 * bias[:, :, None, None]
        fused = F.gelu(self.output_norm(fused))

        actor_residuals = projection_states - pair_context[:, None, :]
        queries = F.normalize(self.query_projection(actor_residuals).float(), dim=-1)
        pixels = F.normalize(self.pixel_projection(fused).float(), dim=1)
        similarity = torch.einsum("bchw,bac->bahw", pixels, queries)
        similarity = self.logit_scale.float().exp().clamp(max=100.0) * similarity
        foreground = self.foreground_head(fused).float()
        actor_logits = foreground[:, None, :, :, :] + similarity[:, :, None, :, :]
        background = self.background_head(fused).float()
        return torch.cat((background, actor_logits[:, :, 0]), dim=1)
