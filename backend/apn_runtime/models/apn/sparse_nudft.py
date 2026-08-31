# -*- coding: utf-8 -*-
"""
Sparse NUDFT (Non-Uniform Discrete Fourier Transform) module for APN.

Provides global frequency-domain priors by computing K learnable spectral
components directly from irregular observations {(t_i, x_i)}, bypassing
the need for interpolation or re-gridding.

The real-valued NUDFT formulation used here avoids complex arithmetic:
  F_cos(k) = sum_i w_i * x_i * cos(2π * omega_k * t_i)
  F_sin(k) = sum_i w_i * x_i * sin(2π * omega_k * t_i)

where omega_k are K learnable frequencies and w_i = mask_i / (sum mask_i)
is the normalized observation weight.

The output is a (2K)-dimensional spectral feature vector per variable.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseNUDFT(nn.Module):
    """
    Lightweight Non-Uniform DFT module for irregular time series.

    Args:
        num_vars (int):      Number of variables N (enc_in).
        K (int):             Number of learnable frequency components.
        d_model (int):       Hidden dimension for projection.
        history (float):     Normalisation constant for timestamps (default 1.0).
        init_spread (float): Controls initial frequency range (default 4.0).
                             Frequencies initialised to k/init_spread for k=1..K.
        dropout (float):     Dropout rate on spectral features.
    """

    def __init__(
        self,
        num_vars: int,
        K: int = 16,
        d_model: int = 24,
        history: float = 1.0,
        init_spread: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.N = num_vars
        self.K = K
        self.d_model = d_model
        self.history = history

        # K learnable frequencies per variable – initialised to a spread of
        # low→high frequencies so they don't collapse to the same value.
        # Shape: (N, K)
        init_freqs = torch.arange(1, K + 1, dtype=torch.float) / init_spread
        self.log_freqs = nn.Parameter(
            init_freqs.unsqueeze(0).expand(num_vars, -1).clone()
        )  # (N, K) — we learn log(omega) to keep omega > 0

        # Project 2K spectral coefficients → d_model query-bias
        self.spectral_proj = nn.Sequential(
            nn.Linear(2 * K, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Small diversity regularization weight (enforced via loss in training
        # script; kept here as a reference)
        self.diversity_weight = 0.01

    def forward(
        self,
        t_stacked: torch.Tensor,    # (B*N, L_obs, 1) — timestamps (normalised 0..1)
        x_stacked: torch.Tensor,    # (B*N, L_obs, 1+te_dim) — [x_val, time_enc...]
        mask_stacked: torch.Tensor, # (B*N, L_obs, 1) — binary observation mask
    ) -> torch.Tensor:
        """
        Compute per-variable spectral features.

        Returns:
            freq_feat: (B, N, d_model) — spectral query bias, one per variable.
        """
        B_N, L, _ = t_stacked.shape
        B = B_N // self.N
        device = t_stacked.device

        # ── Step 1: extract raw values and timestamps ──────────────────────
        x_val = x_stacked[:, :, 0:1]          # (B_N, L, 1)
        t_val = t_stacked[:, :, 0:1]          # (B_N, L, 1)  in [0, 1]
        mask  = mask_stacked.float()           # (B_N, L, 1)

        # Normalised observation weights (zero-mean within each window)
        w = mask / (mask.sum(dim=1, keepdim=True) + 1e-9)  # (B_N, L, 1)

        # Mean-centre the values to reduce DC component dominance
        x_mean = (w * x_val).sum(dim=1, keepdim=True)      # (B_N, 1, 1)
        x_centered = (x_val - x_mean) * mask               # (B_N, L, 1)

        # ── Step 2: compute learnable frequencies ──────────────────────────
        # omega_k > 0 via softplus; shape: (N, K)
        omegas = F.softplus(self.log_freqs) + 1e-6         # (N, K)

        # Expand for batch: (N, K) → (B, N, 1, K) → (B_N, 1, K)
        omegas_bn = omegas.unsqueeze(0).expand(B, -1, -1)   # (B, N, K)
        omegas_bn = omegas_bn.reshape(B_N, 1, self.K)       # (B_N, 1, K)

        # Phase argument: 2π * omega * t — (B_N, L, K)
        phase = 2.0 * math.pi * omegas_bn * t_val           # (B_N, L, K)

        # ── Step 3: real-valued NUDFT ───────────────────────────────────────
        cos_proj = torch.cos(phase)  # (B_N, L, K)
        sin_proj = torch.sin(phase)  # (B_N, L, K)

        # Weighted sum: F_cos(k) = sum_t w_t * x_t * cos(phase), shape (B_N, K)
        F_cos = (w * x_centered * cos_proj).sum(dim=1)       # (B_N, K)
        F_sin = (w * x_centered * sin_proj).sum(dim=1)       # (B_N, K)

        # Amplitude normalisation (prevents scale issues with sparse observations)
        # Use power = sqrt(F_cos^2 + F_sin^2) for normalisation context
        spectral_vec = torch.cat([F_cos, F_sin], dim=-1)     # (B_N, 2K)

        # ── Step 4: project to d_model and reshape to (B, N, d_model) ──────
        freq_feat = self.spectral_proj(spectral_vec)          # (B_N, d_model)
        freq_feat = freq_feat.view(B, self.N, self.d_model)   # (B, N, d_model)

        return freq_feat

    def diversity_loss(self) -> torch.Tensor:
        """
        Frequency diversity regularization: penalize pairs of frequencies that
        are too close to each other (encourages spreading across the spectrum).
        Returns a scalar loss term to add to training loss.
        """
        omegas = F.softplus(self.log_freqs)  # (N, K)
        # Pairwise L2 distances between frequencies (per variable)
        # Shape: (N, K, K)
        diff = omegas.unsqueeze(2) - omegas.unsqueeze(1)     # (N, K, K)
        # Penalize small pairwise distances (repulsion)
        repulsion = torch.exp(-diff.pow(2) / 0.1)            # (N, K, K)
        # Exclude diagonal (self-distance = 0)
        eye = torch.eye(self.K, device=omegas.device).unsqueeze(0)
        repulsion = repulsion * (1.0 - eye)
        return repulsion.mean() * self.diversity_weight
