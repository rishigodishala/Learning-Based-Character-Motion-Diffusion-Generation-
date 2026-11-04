"""
Transformer-Based Motion Diffusion Model (MDM) for 47-Dimensional H1 Locomotion.
Phase 8: Full-sequence offline diffusion generation on Unitree H1 motion representations.
Input/Output Tensor Shape: (Batch_Size, 196, 47)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict


class TimestepEmbedding(nn.Module):
    """
    Sinusoidal Timestep Embedding for Diffusion Timesteps.
    """
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.SiLU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half_dim = self.embed_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps.float().unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if self.embed_dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return self.mlp(emb)


class MotionDiffusionTransformer(nn.Module):
    """
    Transformer-based Denoising Network predicting noise epsilon given noisy motion x_t and timestep t.
    """
    def __init__(self, feature_dim: int = 47, latent_dim: int = 512, num_layers: int = 8, num_heads: int = 8, dim_feedforward: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.feature_dim = feature_dim
        self.latent_dim = latent_dim

        self.input_projection = nn.Linear(feature_dim, latent_dim)
        self.time_embed = TimestepEmbedding(latent_dim)
        
        # Positional embedding for sequence length 196
        self.pos_embed = nn.Parameter(torch.randn(1, 196, latent_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.skip_proj = nn.Linear(latent_dim, latent_dim)
        self.output_projection = nn.Linear(latent_dim, feature_dim)

    def forward(self, x_t: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_t: Noisy motion sequence of shape (B, 196, 47)
            timesteps: Diffusion timesteps of shape (B,)
        Returns:
            eps_pred: Predicted noise tensor of shape (B, 196, 47)
        """
        B, L, D = x_t.shape
        h_in = self.input_projection(x_t)
        h = h_in + self.pos_embed[:, :L, :]
        t_emb = self.time_embed(timesteps).unsqueeze(1)  # (B, 1, latent_dim)
        h = h + t_emb

        h_out = self.transformer(h)
        h_combined = h_out + self.skip_proj(h_in)
        eps_pred = self.output_projection(h_combined)
        return eps_pred


class GaussianDiffusionManager:
    """
    Manages DDPM forward noise schedule and offline reverse sampling process.
    """
    def __init__(self, num_timesteps: int = 1000, beta_start: float = 0.0001, beta_end: float = 0.02, device: str = "cpu"):
        self.num_timesteps = num_timesteps
        self.device = device

        self.betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward process: Add noise to x_0 at timestep t.
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_a_cum = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus_a_cum = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)

        x_t = sqrt_a_cum * x_0 + sqrt_one_minus_a_cum * noise
        return x_t, noise

    @torch.no_grad()
    def p_sample_offline_trajectory(self, model: nn.Module, batch_size: int = 1, sequence_length: int = 196) -> torch.Tensor:
        """
        Offline Reverse Process: Sample complete 196-frame H1 trajectory from pure Gaussian noise.
        """
        model.eval()
        x_t = torch.randn(batch_size, sequence_length, 47, device=self.device)

        for step in reversed(range(self.num_timesteps)):
            t = torch.full((batch_size,), step, device=self.device, dtype=torch.long)
            eps_pred = model(x_t, t)

            beta = self.betas[step]
            sqrt_alpha = torch.sqrt(self.alphas[step])
            sqrt_one_minus_a_cum = self.sqrt_one_minus_alphas_cumprod[step]

            mean = (1.0 / sqrt_alpha) * (x_t - (beta / sqrt_one_minus_a_cum) * eps_pred)

            if step > 0:
                noise = torch.randn_like(x_t)
                sigma = torch.sqrt(beta)
                x_t = mean + sigma * noise
            else:
                x_t = mean

        return x_t
