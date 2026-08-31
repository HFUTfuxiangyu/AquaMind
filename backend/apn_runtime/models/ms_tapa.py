import torch
import torch.nn as nn
import torch.nn.functional as F

from models.APN import AttentionPatchAggregation

class MSTAPAModule(nn.Module):
    def __init__(self, N, P_coarse, P_fine, S, te_dim, hid_dim, history, dropout_rate=0.1, apn_asym=1, apn_conf=0, iaf=1, apn_vat_tapa=0):
        """
        Multi-Scale Time-Aware Patch Aggregation (MS-TAPA)
        - N: Number of variables
        - P_coarse: Number of patches in coarse scale
        - P_fine: Number of patches in fine scale
        - S: Patch stride (None means auto-calculate)
        - te_dim: Time embedding dimension
        - hid_dim: Hidden dimension
        - history: Historical length (typically 1.0)
        - dropout_rate: Dropout
        - apn_asym: Asymmetry flag
        - apn_conf: Confidence flag
        - iaf: Irregularity-Aware Fusion flag (1=True, 0=False)
        - apn_vat_tapa: Value-Aware Time-Gated Patch Aggregation flag (1=True, 0=False)
        """
        super().__init__()
        self.P_coarse = P_coarse
        self.P_fine = P_fine
        self.iaf = iaf

        assert self.P_fine % self.P_coarse == 0, "P_fine must be perfectly divisible by P_coarse"

        # Independent TAPA modules for each scale
        self.tapa_coarse = AttentionPatchAggregation(
            N=N, P=P_coarse, S=S, te_dim=te_dim, hid_dim=hid_dim,
            history=history, dropout_rate=dropout_rate,
            apn_asym=apn_asym, apn_conf=apn_conf,
            apn_vat_tapa=apn_vat_tapa
        )
        self.tapa_fine = AttentionPatchAggregation(
            N=N, P=P_fine, S=S, te_dim=te_dim, hid_dim=hid_dim,
            history=history, dropout_rate=dropout_rate,
            apn_asym=apn_asym, apn_conf=apn_conf,
            apn_vat_tapa=apn_vat_tapa
        )

        # Fusion Gate
        gate_input_dim = hid_dim * 2
        if self.iaf:
            gate_input_dim += 1 # Added observation density feature
            
        self.fusion_gate = nn.Sequential(
            nn.Linear(gate_input_dim, hid_dim),
            nn.SiLU(), # replaced ReLU with SiLU to fix Dying ReLU
            nn.Linear(hid_dim, 2) # output logits for [fine, coarse]
        )
        
        # Zero-initialize the final linear layer for equal weighting (0.5, 0.5) at start
        nn.init.zeros_(self.fusion_gate[-1].weight)
        nn.init.zeros_(self.fusion_gate[-1].bias)

    def compute_observation_density(self, mask_stacked, num_patches_coarse):
        """
        Calculates observation density (proportion of 1s) for each coarse patch.
        mask_stacked: [B_N, L_obs, 1]
        Returns: [B_N, num_patches_coarse]
        """
        B_N, L_obs, _ = mask_stacked.shape
        patch_size = L_obs // num_patches_coarse
        usable_L = patch_size * num_patches_coarse
        chunks = mask_stacked[:, :usable_L, 0].reshape(B_N, num_patches_coarse, patch_size)
        return chunks.float().mean(dim=-1)

    def forward(self, t_stacked, x_with_te, mask_stacked):
        """
        t_stacked: [B_N, L_obs, 1]
        x_with_te: [B_N, L_obs, 1+te_dim]
        mask_stacked: [B_N, L_obs, 1]
        """
        # Forward through both scales
        h_coarse, conf_coarse, c_coarse = self.tapa_coarse(t_stacked, x_with_te, mask_stacked)
        h_fine, conf_fine, c_fine = self.tapa_fine(t_stacked, x_with_te, mask_stacked)

        # Shapes: h_coarse -> [B_N, P_coarse, D], h_fine -> [B_N, P_fine, D]
        scale_ratio = self.P_fine // self.P_coarse

        # Upsample coarse to fine resolution
        h_coarse_up = h_coarse.repeat_interleave(scale_ratio, dim=1) # [B_N, P_fine, D]
        
        # Calculate gate inputs
        if self.iaf:
            density_coarse = self.compute_observation_density(mask_stacked, self.P_coarse) # [B_N, P_coarse]
            density_up = density_coarse.repeat_interleave(scale_ratio, dim=1).unsqueeze(-1) # [B_N, P_fine, 1]
            gate_input = torch.cat([h_fine, h_coarse_up, density_up], dim=-1) # [B_N, P_fine, D*2 + 1]
        else:
            gate_input = torch.cat([h_fine, h_coarse_up], dim=-1) # [B_N, P_fine, D*2]
            
        fusion_logits = self.fusion_gate(gate_input) # [B_N, P_fine, 2]
        fusion_weights = F.softmax(fusion_logits, dim=-1) # [B_N, P_fine, 2]
        
        # Weighted sum
        w_fine = fusion_weights[..., 0].unsqueeze(-1)    # [B_N, P_fine, 1]
        w_coarse = fusion_weights[..., 1].unsqueeze(-1)  # [B_N, P_fine, 1]
        
        h_fused = w_fine * h_fine + w_coarse * h_coarse_up # [B_N, P_fine, D]
        
        # Return fine-scale conf_scores and center points to maintain API compatibility
        return h_fused, conf_fine, c_fine
