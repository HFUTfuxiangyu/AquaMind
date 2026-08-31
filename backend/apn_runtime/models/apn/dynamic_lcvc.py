import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicLCVC(nn.Module):
    """
    Dynamic Low-rank Cross-Variable Coupling (Dynamic LCVC) Module

    Dynamically modulates the low-rank coupling matrices at each time step / patch
    conditioned on the patch embedding via a lightweight gating network:
        W_t = U @ diag(g(patch_emb_t)) @ V^T

    [ODAG extension]  When odag=True, the per-edge coupling is further scaled by a
    learnable gate conditioned on the co-observation density between every pair of
    variables.  Edges between variable pairs that are rarely observed together are
    suppressed, preventing spurious correlations from propagating at high missing
    rates.  Optionally, only the top-k strongest edges per variable are kept.

    Args:
        num_vars      : Number of variables/channels N
        rank          : Low-rank parameter r
        d_patch       : Hidden patch embedding dimension
        gating_hidden : Hidden dim of the content-gating MLP
        odag          : If True, enable Observation-Density Adaptive Graph gating
        odag_topk     : Keep only the top-k outgoing edges per variable (0 = keep all)
    """

    def __init__(self, num_vars, rank, d_patch, gating_hidden=16,
                 odag=False, odag_topk=0):
        super().__init__()
        self.num_vars  = num_vars
        self.rank      = rank
        self.odag      = odag
        self.odag_topk = odag_topk

        # Low-rank base projections (shared across time / patches)
        self.U = nn.Parameter(torch.randn(num_vars, rank) * 0.02)
        self.V = nn.Parameter(torch.randn(num_vars, rank) * 0.02)

        # Content gating network: produces dynamic rank-modulation factors
        self.gate = nn.Sequential(
            nn.Linear(d_patch, gating_hidden),
            nn.GELU(),
            nn.Linear(gating_hidden, rank),
            nn.Sigmoid(),   # factors ∈ [0, 1]
        )

        # ODAG: a single Linear(1→1) that maps co-obs density → edge gate logit
        # Initialised to zero so that sigmoid(0) = 0.5 at start → neutral scaling
        if self.odag:
            self.odag_gate = nn.Linear(1, 1, bias=True)
            nn.init.zeros_(self.odag_gate.weight)
            nn.init.zeros_(self.odag_gate.bias)

    # ------------------------------------------------------------------
    def _compute_odag_weights(self, obs_density_mat: torch.Tensor) -> torch.Tensor:
        """
        Turn a [B, N, N] co-observation density matrix into a
        [B, N, N] edge weight matrix in (0, 1).

        Optionally zeroes out the bottom-(N - topk) edges per source variable.
        """
        # obs_density_mat: [B, N, N], values ∈ [0, 1]
        # Compute edge gate ∈ (0, 1) for each (b, i, j)
        edge_gate = torch.sigmoid(
            self.odag_gate(obs_density_mat.unsqueeze(-1))  # [B, N, N, 1]
        ).squeeze(-1)   # [B, N, N]

        if self.odag_topk > 0 and self.odag_topk < self.num_vars:
            # Keep the top-k largest gate values per row (per source variable)
            topk_vals, _ = torch.topk(edge_gate, self.odag_topk, dim=-1,
                                      largest=True, sorted=False)
            # k-th largest value per row → threshold
            kth = topk_vals.min(dim=-1, keepdim=True).values  # [B, N, 1]
            sparse_mask = (edge_gate >= kth).float()
            edge_gate = edge_gate * sparse_mask

        return edge_gate   # [B, N, N]

    # ------------------------------------------------------------------
    def forward(self, x, patch_emb, current_epoch=None, obs_density_mat=None):
        """
        Args:
            x               : [B, N, P, D]  input variable patch embeddings
            patch_emb       : [B, N, P, D]  patch embeddings for content gating
            current_epoch   : int or None   current training epoch (warmup control)
            obs_density_mat : [B, N, N] or None
                              Co-observation density matrix.  Entry [b, i, j] is
                              the fraction of look-back time steps where both
                              variable i and j have a valid observation.
                              Pass None to disable ODAG (backward compatible).

        Returns:
            x_coupled : [B, N, P, D]
        """
        B, M, P, D = x.shape   # M == N == num_vars

        # ── 1. Content gate: [B, M, P, rank] ──────────────────────────────
        gate_factor = self.gate(patch_emb)

        # Warmup: constrain gate to [0.4, 0.6] for the first 5 epochs
        if self.training and current_epoch is not None and current_epoch < 5:
            gate_factor = 0.4 + 0.2 * gate_factor

        # Average over variable dim → shared gate per (patch, rank)
        # gate_pooled: [B, P, rank]
        gate_pooled = gate_factor.mean(dim=1)

        # ── 2. Low-rank coupling: W_t = U @ diag(gate_t) @ V^T ────────────
        # Step 2a: project to low-rank subspace  [B, P, rank, D]
        x_low = torch.einsum('bmpd,mr->bprd', x, self.V)

        # Step 2b: dynamic rank modulation        [B, P, rank, D]
        x_low_mod = x_low * gate_pooled.unsqueeze(-1)

        # Step 2c: project back to variable space [B, M, P, D]
        x_coupled = torch.einsum('bprd,mr->bmpd', x_low_mod, self.U)

        # ── 3. ODAG: observation-density adaptive edge gating ──────────────
        if self.odag and obs_density_mat is not None:
            edge_gate = self._compute_odag_weights(obs_density_mat)  # [B, N, N]

            # Scale coupled signal of each target variable by the mean
            # incoming gate weight from all source variables.
            # incoming_strength: [B, N, 1, 1]  (broadcast over P and D)
            incoming_strength = edge_gate.mean(dim=-1).unsqueeze(-1).unsqueeze(-1)
            x_coupled = x_coupled * incoming_strength

        return x_coupled
