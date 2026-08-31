# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.globals import logger
from utils.ExpConfigs import ExpConfigs

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x


class AttentionPatchAggregation(nn.Module):
    def __init__(self, N, P, S, te_dim, hid_dim, history, dropout_rate=0.1, apn_asym=1, apn_conf=0, apn_vat_tapa=0):
        super().__init__()
        self.N = N
        self.P = P
        self.apn_conf = apn_conf
        self.S = max(history / P, 1e-6) if S is None else S
        self.history = history
        self.hid_dim = hid_dim
        self.te_dim = te_dim
        self.feature_dim = 1 + te_dim
        self.apn_asym = apn_asym
        self.delta_left_params = nn.Parameter(torch.zeros(N, P))
        self.raw_log_width_params = nn.Parameter(torch.full((N, P), math.log(self.S)))
        self.tau_params = nn.Parameter(torch.zeros(N))
        self.alpha_params = nn.Parameter(torch.zeros(N))
        self.gamma_params = nn.Parameter(torch.zeros(N))
        self.projection_layer = nn.Linear(self.feature_dim, self.hid_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hid_dim, hid_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hid_dim * 2, hid_dim)
        )
        self.norm = nn.LayerNorm(hid_dim)
        
        if self.apn_conf >= 2:
            self.conf_mlp = nn.Sequential(
                nn.Linear(3, 16),
                nn.GELU(),
                nn.Linear(16, 1),
                nn.Sigmoid()
            )

        self.apn_vat_tapa = apn_vat_tapa
        if self.apn_vat_tapa:
            self.patch_queries = nn.Parameter(torch.randn(N, P, hid_dim))
            self.key_projection = nn.Linear(self.feature_dim, self.hid_dim)
            self.value_projection = nn.Linear(self.feature_dim, self.hid_dim)

    def forward(self, t_stacked, x_with_te, mask_stacked):
        current_device = t_stacked.device
        B_N, L_obs_pad, _ = t_stacked.shape
        B = B_N // self.N
        patch_centers = torch.linspace(self.S / 2, self.history - self.S / 2, self.P, device=current_device)
        base_left_boundaries = (patch_centers - self.S / 2).unsqueeze(0)
        t_left_n_p = base_left_boundaries + self.delta_left_params
        width_learned_n_p = torch.exp(self.raw_log_width_params) + 1e-6
        t_right_n_p = t_left_n_p + width_learned_n_p
        c_p_n_p = (t_left_n_p + t_right_n_p) / 2.0
        base_tau = F.softplus(self.tau_params).unsqueeze(-1) + 1e-6
        
        if self.apn_asym:
            delta_t_n_p = torch.abs(self.history - c_p_n_p)
            gamma_n = F.softplus(self.gamma_params).unsqueeze(-1) + 1e-6
            alpha_n_p = self.alpha_params.unsqueeze(-1) * torch.exp(-gamma_n * delta_t_n_p)
            taus_left = base_tau * torch.exp(alpha_n_p)
            taus_right = base_tau * torch.exp(-alpha_n_p)
        else:
            taus_left = base_tau.expand(-1, self.P)
            taus_right = base_tau.expand(-1, self.P)
        
        t_left_b_n = t_left_n_p.unsqueeze(0).expand(B, -1, -1).reshape(B_N, self.P).unsqueeze(-1)
        t_right_b_n = t_right_n_p.unsqueeze(0).expand(B, -1, -1).reshape(B_N, self.P).unsqueeze(-1)
        taus_left_b_n = taus_left.unsqueeze(0).expand(B, -1, -1).reshape(B_N, self.P).unsqueeze(-1)
        taus_right_b_n = taus_right.unsqueeze(0).expand(B, -1, -1).reshape(B_N, self.P).unsqueeze(-1)
        t_raw_b_n = t_stacked.transpose(-1, -2)
        weights_raw = torch.sigmoid((t_right_b_n - t_raw_b_n) / taus_right_b_n) * \
                      torch.sigmoid((t_raw_b_n - t_left_b_n) / taus_left_b_n)
        mask_b_n = mask_stacked.transpose(-1, -2)
        temporal_weights = weights_raw * mask_b_n
        sum_weights = temporal_weights.sum(dim=-1, keepdim=True) + 1e-9
        
        conf_scores = None
        if getattr(self, 'apn_conf', 0):
            norm_weights = temporal_weights / sum_weights
            entropy = -torch.sum(norm_weights * torch.log(norm_weights + 1e-9), dim=-1)
            L_obs = temporal_weights.shape[-1]
            max_entropy = math.log(max(L_obs, 2))
            c_base = torch.clamp(1.0 - (entropy / max_entropy), min=0.0)
            epsilon = 0.01
            k = 5.0
            valid_obs_count = (norm_weights > epsilon).float().sum(dim=-1)
            c_dense = c_base * torch.tanh(valid_obs_count / k)
            
            if self.apn_conf == 1:
                conf_scores = c_dense
            elif self.apn_conf >= 2:
                obs_features = torch.stack([
                    c_base,
                    valid_obs_count / k,
                    c_dense
                ], dim=-1)
                conf_scores = self.conf_mlp(obs_features).squeeze(-1)

        if self.apn_vat_tapa:
            # VAT-TAPA: Cross-attention over observation points within the temporal gating window
            # Project x_with_te to Key (K) and Value (V)
            K = self.key_projection(x_with_te)  # (B_N, L_obs, hid_dim)
            V = self.value_projection(x_with_te)  # (B_N, L_obs, hid_dim)
            
            # self.patch_queries: (N, P, hid_dim) -> (B, N, P, hid_dim) -> (B_N, P, hid_dim)
            Q = self.patch_queries.unsqueeze(0).repeat(B, 1, 1, 1).view(B_N, self.P, self.hid_dim)
            
            # Compute raw attention scores (B_N, P, L_obs)
            attn_scores = torch.bmm(Q, K.transpose(-1, -2)) * (self.hid_dim ** -0.5)
            
            # Apply soft window gating mask (log bias)
            attn_scores = attn_scores + torch.log(temporal_weights + 1e-8)
            
            # Softmax over timeline dimension
            attn_weights = F.softmax(attn_scores, dim=-1)
            
            # Aggregate values
            h_patches_proj = torch.bmm(attn_weights, V)  # (B_N, P, hid_dim)
        else:
            # Standard TAPA (content-independent pooling)
            weighted_features_sum = torch.bmm(temporal_weights, x_with_te)
            h_patches_avg = weighted_features_sum / sum_weights
            h_patches_proj = self.projection_layer(h_patches_avg)
            
        h_patches = self.norm(h_patches_proj + self.ffn(h_patches_proj))
        return h_patches, conf_scores, c_p_n_p


class IMTS_SubModel(nn.Module):
    def __init__(self, configs):
        super(IMTS_SubModel, self).__init__()
        self.configs = configs
        self.hid_dim = configs.d_model

        self.te_dim = configs.apn_te_dim
        self.N = configs.enc_in
        self.P = configs.apn_npatch
        self.n_layer = configs.apn_nlayer
        self.attn_heads = configs.apn_attn_heads

        self.dropout_rate = configs.dropout
        self.batch_size = None

        self.te_scale = nn.Linear(1, 1)
        self.te_periodic = nn.Linear(1, self.te_dim - 1)

        self.apn_conf = getattr(configs, 'apn_conf', 0)
        self.apn_ms_tapa = getattr(configs, 'apn_ms_tapa', 0)
        apn_vat_tapa = getattr(configs, 'apn_vat_tapa', 0)
        if self.apn_ms_tapa:
            from models.ms_tapa import MSTAPAModule
            # Override self.P with P_fine so subsequent layers use the correct patch count
            self.P = getattr(configs, 'apn_ms_tapa_fine', 16)
            self.patching = MSTAPAModule(
                N=self.N,
                P_coarse=getattr(configs, 'apn_ms_tapa_coarse', 8),
                P_fine=self.P,
                S=None,
                te_dim=self.te_dim,
                hid_dim=self.hid_dim,
                history=1.0,
                dropout_rate=self.dropout_rate,
                apn_asym=getattr(configs, 'apn_asym', 1),
                apn_conf=self.apn_conf,
                iaf=getattr(configs, 'apn_ms_tapa_iaf', 1),
                apn_vat_tapa=apn_vat_tapa
            )
        else:
            self.patching = AttentionPatchAggregation(
                N=self.N,
                P=self.P,
                S=None,
                te_dim=self.te_dim,
                hid_dim=self.hid_dim,
                history=1.0,
                dropout_rate=self.dropout_rate,
                apn_asym=getattr(configs, 'apn_asym', 1),
                apn_conf=self.apn_conf,
                apn_vat_tapa=apn_vat_tapa
            )

        self.conf_tau = nn.Parameter(torch.tensor(0.0))
        if not self.apn_conf:
            self.conf_tau.requires_grad = False

        self.patch_pos_enc = PositionalEncoding(self.hid_dim, max_len=self.P)
        self.use_ctrope = getattr(configs, 'use_ctrope', 0)
        if self.use_ctrope:
            from models.apn.ctrope import CTRoPE
            self.ctrope = CTRoPE(
                d_model=self.hid_dim,
                omega_base_init=getattr(configs, 'ctrope_omega_init', 100.0),
                learnable=bool(getattr(configs, 'ctrope_learnable', 1))
            )
        self.var_queries = nn.Parameter(torch.randn(1, self.N, 1, self.hid_dim))
        self.aggregation_norm = nn.LayerNorm(self.hid_dim)

        # ── LCVC: Lightweight Cross-Variable Coupling ──────────────────────────
        self.lcvc_mode = getattr(configs, 'lcvc_mode', 'static')
        self.apn_lcvc = getattr(configs, 'apn_lcvc', 0)
        self.lcvc_rank = getattr(configs, 'apn_lcvc_rank', 4)
        if self.apn_lcvc:
            if self.lcvc_mode == 'dynamic':
                from models.apn.dynamic_lcvc import DynamicLCVC
                self.lcvc = DynamicLCVC(
                    num_vars=self.N,
                    rank=self.lcvc_rank,
                    d_patch=self.hid_dim,
                    gating_hidden=getattr(configs, 'lcvc_gating_hidden', 16),
                    odag=bool(getattr(configs, 'lcvc_odag', 0)),
                    odag_topk=getattr(configs, 'lcvc_odag_topk', 0),
                )
            else:
                # Low-rank coupling matrices: W = U @ V^T ∈ R^(N×N), params = 2*N*r
                self.lcvc_U = nn.Parameter(torch.randn(self.N, self.lcvc_rank) * 0.01)
                self.lcvc_V = nn.Parameter(torch.randn(self.N, self.lcvc_rank) * 0.01)
            # Residual gate γ: small init ensures training stability (≈ original at start)
            self.lcvc_gamma = nn.Parameter(torch.tensor(0.1))
            self.lcvc_norm = nn.LayerNorm(self.hid_dim)

        # ── Direction 1: SparseNUDFT – Global Frequency Prior ──────────────────
        # Fills the low-pass filter blind spot of TAPA by injecting K learnable
        # spectral components from the raw irregular observations.
        self.apn_nudft = getattr(configs, 'apn_nudft', 0)
        self.nudft_k   = getattr(configs, 'apn_nudft_k', 16)
        if self.apn_nudft:
            from models.apn.sparse_nudft import SparseNUDFT
            self.nudft = SparseNUDFT(
                num_vars=self.N,
                K=self.nudft_k,
                d_model=self.hid_dim,
                history=1.0,
                dropout=self.dropout_rate,
            )
            # Gate parameter: controls how much spectral bias is blended in
            # (initialised near 0 so training starts from the original model)
            self.nudft_gate = nn.Parameter(torch.tensor(0.1))

        # ── Direction 2: Δt-aware Decoder – Gap-sensitive decoding ────────────
        # The time gap Δt = t_pred - t_last_obs encodes prediction difficulty;
        # injecting it into the decoder lets the MLP adapt to growing uncertainty.
        self.apn_dt_decoder = getattr(configs, 'apn_dt_decoder', 0)
        self.dt_emb_dim = getattr(configs, 'apn_dt_emb_dim', 8)
        if self.apn_dt_decoder:
            # log(1 + Δt) embedding: handles wide range of gaps gracefully
            self.dt_embed = nn.Sequential(
                nn.Linear(1, self.dt_emb_dim),
                nn.GELU(),
            )

        # ── CDPH: Confidence-Driven Probabilistic Head ─────────────────────────
        self.apn_prob = getattr(configs, 'apn_prob', 0)
        # Compute effective decoder input dimension
        _dec_in_dim = self.hid_dim + self.te_dim
        if self.apn_dt_decoder:
            _dec_in_dim += self.dt_emb_dim
        if self.apn_prob:
            # Shared trunk + independent mean/log-var heads
            self.decoder_shared = nn.Sequential(
                nn.Linear(_dec_in_dim, self.hid_dim * 2),
                nn.ReLU(inplace=True),
                nn.Dropout(self.dropout_rate),
            )
            self.decoder_mean    = nn.Linear(self.hid_dim * 2, 1)
            # Outputs log σ² directly (unconstrained); avoids Softplus instability
            self.decoder_log_var = nn.Linear(self.hid_dim * 2, 1)
            # Learnable coupling strength between confidence and predicted variance
            self.cdph_eta = nn.Parameter(torch.tensor(1.0))
        else:
            # Original point-prediction decoder (unchanged)
            self.decoder = nn.Sequential(
                nn.Linear(_dec_in_dim, self.hid_dim * 2),
                nn.ReLU(inplace=True),
                nn.Dropout(self.dropout_rate),
                nn.Linear(self.hid_dim * 2, 1)
            )

    def LearnableTE(self, tt):
        out1 = self.te_scale(tt)
        out2 = torch.sin(self.te_periodic(tt))
        return torch.cat([out1, out2], -1)

    @staticmethod
    def _compute_obs_density_mat(mask_stacked: torch.Tensor, B: int, N: int) -> torch.Tensor:
        """
        Compute co-observation density matrix [B, N, N].
        Entry [b, i, j] = fraction of look-back time steps where both
        variable i and variable j carry a valid observation (mask == 1).

        mask_stacked: [B*N, L_obs, 1]  (binary, 1 = observed)
        """
        # Reshape to [B, N, L_obs]
        mask_bn = mask_stacked.squeeze(-1).view(B, N, -1).float()  # [B, N, L]
        L = mask_bn.shape[-1]
        # co_obs[b, i, j] = Σ_t mask[b,i,t] * mask[b,j,t]  / L
        # Efficient via batch matmul: [B, N, L] @ [B, L, N] → [B, N, N]
        co_obs = torch.bmm(mask_bn, mask_bn.transpose(1, 2)) / (L + 1e-9)  # [B, N, N]
        return co_obs

    def IMTS_Model_Logic(self, x_with_te, mask_stacked, time_features_stacked, current_epoch=None):
        B = self.batch_size
        N_vars = self.N
        h_patches_stacked, conf_scores_stacked, c_p_n_p = self.patching(time_features_stacked, x_with_te, mask_stacked)
        
        if self.use_ctrope:
            h_patches_updated = h_patches_stacked.view(B, N_vars, self.P, self.hid_dim)
        else:
            h_patches_stacked_pe = self.patch_pos_enc(h_patches_stacked)
            h_patches_updated = h_patches_stacked_pe.view(B, N_vars, self.P, self.hid_dim)

        # Compute per-variable average confidence c_bar: (B, N)
        # Used by both LCVC (conf-modulated form) and CDPH (variance coupling)
        c_bar = None
        if self.apn_conf and conf_scores_stacked is not None:
            c_bar = conf_scores_stacked.view(B, N_vars, self.P).mean(dim=-1)  # (B, N)

        # ── LCVC: per-Patch cross-variable coupling (Method B) ─────────────────
        if self.apn_lcvc:
            if self.lcvc_mode == 'dynamic':
                # Compute co-observation density matrix for ODAG (only if needed)
                obs_density_mat = None
                if getattr(self.lcvc, 'odag', False):
                    obs_density_mat = self._compute_obs_density_mat(
                        mask_stacked, B, N_vars
                    )  # [B, N, N]
                # Dynamic LCVC forward
                h_coupled = self.lcvc(
                    h_patches_updated,
                    patch_emb=h_patches_updated,
                    current_epoch=current_epoch,
                    obs_density_mat=obs_density_mat,
                )
            else:
                # Low-rank coupling matrix W = U @ V^T, shape (N, N)
                W = torch.matmul(self.lcvc_U, self.lcvc_V.t())  # (N, N)

                if self.apn_lcvc >= 2 and c_bar is not None:
                    # Confidence-modulated form:
                    # W_conf[b, i, j] = c_bar[b, i] * W[i, j]
                    # → (B, N, N): only high-confidence variables "broadcast" cross-variable info
                    W_conf = c_bar.unsqueeze(-1) * W.unsqueeze(0)  # (B, N, N)
                    # Per-patch coupling: for each patch p independently
                    # h_patches_updated: (B, N, P, D)
                    # output[b, m, p, d] = Σ_n W_conf[b,m,n] * h[b,n,p,d]
                    h_coupled = torch.einsum('bmn,bnpd->bmpd', W_conf, h_patches_updated)
                else:
                    # Basic form: same coupling for all batch items
                    # output[b, m, p, d] = Σ_n W[m,n] * h[b,n,p,d]
                    h_coupled = torch.einsum('mn,bnpd->bmpd', W, h_patches_updated)

            # Residual add with learnable gate and layer norm
            h_patches_updated = h_patches_updated + \
                self.lcvc_gamma * self.lcvc_norm(h_coupled)

        # ── Direction 1: SparseNUDFT frequency-domain bias ────────────────────
        # Compute per-variable spectral features from raw observations and inject
        # as an additive bias into the query aggregation scores.
        freq_bias = None
        if self.apn_nudft:
            # freq_feat: (B, N, d_model)
            freq_feat = self.nudft(time_features_stacked, x_with_te, mask_stacked)
            # Project to (B, N, 1, P) – used as a patch-level attention bias
            # by broadcasting through the var_query dimension
            freq_bias = freq_feat  # kept as (B, N, d_model) for now

        # ── Original query aggregation (modified for CT-RoPE + NUDFT) ─────────
        if self.use_ctrope:
            t_q = torch.ones(1, N_vars, 1, device=self.var_queries.device)
            q_rot = self.ctrope(self.var_queries, t_q)
            
            t_k = c_p_n_p.unsqueeze(0).to(h_patches_updated.device)
            k_rot = self.ctrope(h_patches_updated, t_k)
            
            attn_scores = torch.matmul(q_rot, k_rot.transpose(-1, -2)) * (self.hid_dim ** -0.5)
        else:
            # Standard query: var_queries (1, N, 1, D) @ patches (B, N, D, P)
            if self.apn_nudft and freq_bias is not None:
                # Blend spectral features into queries with learnable gate
                # freq_bias: (B, N, D) → (B, N, 1, D)
                freq_bias_expanded = freq_bias.unsqueeze(2)  # (B, N, 1, D)
                q_enhanced = self.var_queries + self.nudft_gate * freq_bias_expanded
                attn_scores = torch.matmul(q_enhanced, h_patches_updated.transpose(-1, -2)) * (self.hid_dim ** -0.5)
            else:
                attn_scores = torch.matmul(self.var_queries, h_patches_updated.transpose(-1, -2)) * (self.hid_dim ** -0.5)
        
        if self.apn_conf and conf_scores_stacked is not None:
            conf_scores = conf_scores_stacked.view(B, N_vars, 1, self.P)
            attn_scores = attn_scores + torch.log(conf_scores + 1e-8) - F.softplus(self.conf_tau)

        attn_weights = F.softmax(attn_scores, dim=-1)
        h_final = torch.matmul(attn_weights, h_patches_updated)
        h_final = h_final.squeeze(-2)
        h_final = self.aggregation_norm(h_final)

        # Return c_bar explicitly to avoid storing as instance attr (multi-GPU safe)
        return h_final, c_bar

    def encode_to_latent(self, x: torch.Tensor, x_mark: torch.Tensor, x_mask: torch.Tensor, current_epoch=None):
        B, L_obs, N_vars_from_X = x.shape
        self.batch_size = B

        time_features = x_mark[:, :, [0]]

        X_stacked = x.permute(0, 2, 1).reshape(B * N_vars_from_X, L_obs, 1)
        mask_stacked = x_mask.permute(0, 2, 1).reshape(B * N_vars_from_X, L_obs, 1)

        time_features_stacked = time_features.repeat(1, 1, N_vars_from_X).permute(0, 2, 1).reshape(
            B * N_vars_from_X, L_obs, 1)

        te_his = self.LearnableTE(time_features_stacked)
        X_with_te = torch.cat([X_stacked, te_his], dim=-1)

        # ── Direction 2: compute per-variable last observation time ─────────
        # last_obs_time[b, n] = the normalised timestamp of the last valid
        # observation in the history window for variable n of sample b.
        # Used by predict_from_latent to compute Δt for each prediction step.
        last_obs_time = None
        if self.apn_dt_decoder:
            # mask_stacked: (B*N, L, 1) → reshape to (B, N, L)
            mask_bn = mask_stacked.squeeze(-1).view(B, N_vars_from_X, L_obs)
            t_bn = time_features_stacked.squeeze(-1).view(B, N_vars_from_X, L_obs)
            # For each (b, n), find the last timestep where mask == 1.
            # If no observation, default to 0 (beginning of window).
            masked_t = t_bn * mask_bn  # (B, N, L) — zeroed out where masked
            last_obs_time = masked_t.max(dim=-1).values  # (B, N)

        # h_final: (B, N, hid_dim)   c_bar: (B, N) or None
        h_final, c_bar = self.IMTS_Model_Logic(X_with_te, mask_stacked, time_features_stacked, current_epoch=current_epoch)
        return h_final, c_bar, last_obs_time

    def predict_from_latent(self, h_final: torch.Tensor, c_bar: torch.Tensor,
                            y_mark: torch.Tensor, last_obs_time: torch.Tensor = None):
        B = h_final.shape[0]
        N_vars_from_X = self.N

        # ── Decoder ────────────────────────────────────────────────────────────
        time_steps_to_predict = y_mark[:, :, [0]]
        L_pred = time_steps_to_predict.shape[1]
        h_expanded = h_final.unsqueeze(dim=-2).repeat(1, 1, L_pred, 1)           # (B, N, L_pred, D)
        time_steps_to_predict_exp = time_steps_to_predict.view(B, 1, L_pred, 1).repeat(
            1, N_vars_from_X, 1, 1)
        te_pred = self.LearnableTE(time_steps_to_predict_exp)

        # ── Direction 2: Δt-aware Decoder ─────────────────────────────────────
        # Δt[b, n, p] = t_pred[b, p] - t_last_obs[b, n]  (clipped to ≥ 0)
        # Encodes "how far into the future are we predicting for each variable".
        decoder_input = torch.cat([h_expanded, te_pred], dim=-1)                 # (B, N, L_pred, D+te)
        if self.apn_dt_decoder and last_obs_time is not None:
            # t_pred: (B, L_pred) | last_obs_time: (B, N)
            t_pred_flat = time_steps_to_predict.squeeze(-1)                       # (B, L_pred)
            # broadcast: (B, 1, L_pred) - (B, N, 1) = (B, N, L_pred)
            delta_t = (t_pred_flat.unsqueeze(1) - last_obs_time.unsqueeze(-1)).clamp(min=0.0)
            # log-compress: handles huge gaps without exploding gradients
            delta_t_log = torch.log1p(delta_t).unsqueeze(-1)                     # (B, N, L_pred, 1)
            dt_emb = self.dt_embed(delta_t_log)                                  # (B, N, L_pred, dt_emb_dim)
            decoder_input = torch.cat([decoder_input, dt_emb], dim=-1)           # (B, N, L_pred, D+te+dt)

        if self.apn_prob:
            # ── CDPH: Dual-head probabilistic decoding ─────────────────────────
            shared_feat   = self.decoder_shared(decoder_input)                   # (B, N, L_pred, D*2)
            mean_out      = self.decoder_mean(shared_feat)                        # (B, N, L_pred, 1)
            log_var_base  = self.decoder_log_var(shared_feat)                    # (B, N, L_pred, 1)

            if c_bar is not None:
                # Confidence → variance coupling (in log-σ² space, numerically stable):
                # Low confidence → high variance (larger log_var)
                # c_bar: (B, N) → (B, N, 1, 1)
                conf_bias = -self.cdph_eta * torch.log(c_bar.unsqueeze(-1).unsqueeze(-1) + 1e-8)
                log_var_out = log_var_base + conf_bias
            else:
                log_var_out = log_var_base

            outputs_mean    = mean_out.squeeze(-1).permute(0, 2, 1)              # (B, L_pred, N)
            outputs_log_var = log_var_out.squeeze(-1).permute(0, 2, 1)           # (B, L_pred, N)
            return outputs_mean, outputs_log_var
        else:
            # ── Original point-prediction decoding ─────────────────────────────
            outputs_raw = self.decoder(decoder_input)
            outputs = outputs_raw.squeeze(-1).permute(0, 2, 1)                   # (B, L_pred, N)
            return outputs

    def forward(self, x: torch.Tensor, x_mark: torch.Tensor, x_mask: torch.Tensor,
                y_mark: torch.Tensor, current_epoch=None):
        h_final, c_bar, last_obs_time = self.encode_to_latent(x, x_mark, x_mask, current_epoch=current_epoch)
        return self.predict_from_latent(h_final, c_bar, y_mark, last_obs_time=last_obs_time)


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.apn_prob = getattr(configs, 'apn_prob', 0)

        self.model = IMTS_SubModel(configs)

    def encode_to_latent(self, x: torch.Tensor, x_mark: torch.Tensor, x_mask: torch.Tensor, current_epoch=None):
        return self.model.encode_to_latent(x, x_mark, x_mask, current_epoch=current_epoch)

    def predict_from_latent(self, z: torch.Tensor, c_bar: torch.Tensor, y_mark: torch.Tensor = None, **kwargs) -> dict:
        if y_mark is None:
            y_mark = kwargs.get('y_mark')
        y = kwargs.get('y')
        y_mask = kwargs.get('y_mask')
        last_obs_time = kwargs.get('last_obs_time', None)
        f_dim = -1 if self.configs.features == 'MS' else 0

        if self.apn_prob:
            predictions_mean, predictions_log_var = self.model.predict_from_latent(
                z, c_bar, y_mark, last_obs_time=last_obs_time)
            return {
                "pred":         predictions_mean[:, :, f_dim:],
                "pred_log_var": predictions_log_var[:, :, f_dim:],
                "true":         y[:, :, f_dim:] if y is not None else None,
                "mask":         y_mask[:, :, f_dim:] if y_mask is not None else None,
            }
        else:
            predictions = self.model.predict_from_latent(z, c_bar, y_mark, last_obs_time=last_obs_time)
            return {
                "pred": predictions[:, :, f_dim:],
                "true": y[:, :, f_dim:] if y is not None else None,
                "mask": y_mask[:, :, f_dim:] if y_mask is not None else None,
            }

    def forward(self, x: torch.Tensor, x_mark: torch.Tensor, x_mask: torch.Tensor, **kwargs) -> dict:
        current_epoch = kwargs.pop('current_epoch', None)
        h_final, c_bar, last_obs_time = self.encode_to_latent(x, x_mark, x_mask, current_epoch=current_epoch)
        return self.predict_from_latent(h_final, c_bar, last_obs_time=last_obs_time, **kwargs)