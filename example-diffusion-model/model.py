"""Minimal noise predictor model for a diffusion example."""

from __future__ import annotations

import math

import torch
from torch import nn


class SinusoidalPositionEmbeddings(nn.Module):
    """Create sinusoidal timestep embeddings."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        exponent = -math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * exponent)
        emb = timesteps[:, None].float() * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb


class ConvBlock(nn.Module):
    """A small convolutional block with normalization and activation."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class NoisePredictor(nn.Module):
    """Simple convolutional network that predicts noise given an image and timestep."""

    def __init__(self, img_channels: int = 1, base_channels: int = 32, time_dim: int = 32) -> None:
        super().__init__()
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.conv1 = ConvBlock(img_channels + time_dim, base_channels)
        self.conv2 = ConvBlock(base_channels, base_channels * 2)
        self.conv3 = ConvBlock(base_channels * 2, base_channels)
        self.conv4 = nn.Conv2d(base_channels, img_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        time_emb = self.time_embed(timesteps)
        time_map = time_emb[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        x = torch.cat([x, time_map], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return self.conv4(x)
