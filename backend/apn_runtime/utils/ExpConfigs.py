from dataclasses import dataclass

@dataclass
class ExpConfigs:
    '''
    dataclass for argparse typo check, making life easier

    Make sure to update this dataclass after adding new args in argparse
    '''
    # basic config
    task_name: str
    is_training: int
    model_id: str
    model_name: str
    checkpoints: str
    ablation_name: str

    # dataset & data loader
    dataset_name: str
    dataset_root_path: str
    dataset_file_name: str
    features: str
    target_variable_name: str
    target_variable_index: int
    freq: str
    collate_fn: str
    augmentation_ratio: int
    missing_rate: float
    train_val_loader_shuffle: int
    train_val_loader_drop_last: int
    test_inference_time: int

    # forecasting task
    seq_len: int
    label_len: int
    pred_len: int

    # classification task
    n_classes: int

    # GPU
    use_gpu: int
    gpu_id: int
    use_multi_gpu: int
    gpu_ids: str

    # training
    wandb: int
    sweep: int
    val_interval: int
    num_workers: int
    itr: int
    train_epochs: int
    batch_size: int
    patience: int
    learning_rate: float
    loss: str
    lr_scheduler: str
    pretrained_checkpoint_root_path: str
    pretrained_checkpoint_file_name: str
    n_train_stages: str
    retain_graph: int

    # testing
    checkpoints_test: str
    test_all: int
    test_flop: int
    test_train_time: int
    test_gpu_memory: int
    test_zero_shot: int
    test_dataset_statistics: int
    save_arrays: int
    load_checkpoints_test: int

    # model configs
    # common
    patch_len: int
    patch_stride: int
    revin: int
    revin_affine: int
    kernel_size: int
    individual: int
    channel_independence: int
    scale_factor: int
    top_k: int
    embed_type: int
    enc_in: int
    dec_in: int
    c_out: int
    d_model: int
    d_timesteps: int
    n_heads: int
    n_layers: int
    e_layers: int
    d_layers: int
    hidden_layers: int
    d_ff: int
    moving_avg: int
    factor: int
    dropout: float
    embed: str
    activation: str
    output_attention: int
    node_dim: int
    # PatchTST
    patchtst_fc_dropout: float
    patchtst_head_dropout: float
    patchtst_padding_patch: str
    patchtst_subtract_last: int
    patchtst_decomposition: int
    # Mamba
    mamba_d_conv: int
    mamba_expand: int
    # Latent ODE
    latent_ode_units: int
    latent_ode_gen_layers: int
    latent_ode_rec_layers: int
    latent_ode_z0_encoder: str
    latent_ode_rec_dims: int
    latent_ode_gru_units: int
    latent_ode_classif: int
    latent_ode_linear_classif: int
    # CRU
    cru_num_basis: int
    cru_bandwidth: int
    cru_ts: float
    # NeuralFlows
    neuralflows_flow_model: str
    neuralflows_flow_layers: int
    neuralflows_latents: int
    neuralflows_time_net: str
    neuralflows_time_hidden_dim: int
    # PrimeNet
    primenet_pooling: str
    # mTAN
    mtan_num_ref_points: int
    mtan_alpha: float
    # TimeMixer
    timemixer_decomp_method: str
    timemixer_use_norm: int
    timemixer_down_sampling_layers: int
    timemixer_down_sampling_method: str
    # Nonstationary Transformer
    nonstationarytransformer_p_hidden_dims: list
    nonstationarytransformer_p_hidden_layers: int
    # Informer
    informer_distil: int
    # tPatchGNN
    tpatchgnn_te_dim: int
    # SPECTRON
    spectron_num_kernels: int
    spectron_d_max: float
    spectron_patch_len: int  # <-- 新增
    spectron_patch_stride: int  # <-- 新增
    spectron_num_intra_layers: int
    spectron_kernel_chunk_size: int
    spectron_num_last_patches: int
    # TAC-Mixer
    tac_patch_num: int
    tac_mixer_hidden_dim_p: int
    tac_mixer_hidden_dim_c: int
    tac_decoder_context_k: int
    # ASTGI
    astgi_k_neighbors: int
    astgi_prop_layers: int
    astgi_channel_dim: int
    astgi_time_dim: int
    astgi_mlp_ratio: float
    astgi_channel_dist_weight: float
    # APN
    apn_te_dim: int
    apn_npatch: int
    apn_patch_size: float
    apn_nlayer: int
    apn_attn_heads: int
    apn_asym: int = 1
    apn_conf: int = 0
    apn_multires: int = 0
    apn_contrast: int = 0
    apn_lcvc: int = 0       # 0=off, 1=basic low-rank coupling, 2=confidence-modulated coupling
    apn_lcvc_rank: int = 4  # rank r of the low-rank coupling matrix U,V ∈ R^(N×r)
    apn_prob: int = 0       # 0=point prediction, 1=dual-head probabilistic prediction (mean+log_var)
    apn_ms_tapa: int = 0
    apn_ms_tapa_coarse: int = 8
    apn_ms_tapa_fine: int = 16
    apn_ms_tapa_iaf: int = 1
    apn_vat_tapa: int = 0
    # Direction 1: SparseNUDFT – learnable frequency-domain prior
    apn_nudft: int = 0       # 0=off, 1=enable Sparse NUDFT spectral bias
    apn_nudft_k: int = 16    # number of learnable frequency components K
    # Direction 2: Δt-aware Decoder – gap-sensitive decoding
    apn_dt_decoder: int = 0  # 0=off, 1=enable Δt-aware decoder
    apn_dt_emb_dim: int = 8  # embedding dimension for log-Δt features
    # CT-RoPE
    use_ctrope: int = 0
    ctrope_omega_init: float = 100.0
    ctrope_learnable: int = 1
    # Glocal-IB
    use_glocal_ib: int = 0
    glocal_temperature: float = 0.1
    glocal_lambda_align: float = 0.1
    glocal_extra_mr_min: float = 0.3
    glocal_extra_mr_max: float = 0.8
    detach_full_branch: int = 0
    grad_clip_max_norm: float = 1.0
    # Task A Logging
    grad_cos_sim_interval: int = 50
    grad_csv_output: str = "results/c12_diagnostic/grad_metrics.csv"
    # Task C Dynamic LCVC
    lcvc_mode: str = "static"
    lcvc_gating_hidden: int = 16
    lcvc_odag: int = 0          # 0=off, 1=enable ODAG density-aware edge gating
    lcvc_odag_topk: int = 0     # 0=keep all edges; >0 = keep top-k per variable
    # Used to be compatible with ipython. Never used
    f: int = 1

    # args not presented in argparse
    seq_len_max_irr: int = None # maximum number of observations along time dimension of x, set in irregular time series datasets
    pred_len_max_irr: int = None # maximum number of observations along time dimension of y, set in irregular time series datasets
    patch_len_max_irr: int = None # maximum number of observations along time dimension in a patch of x, set in irregular time series datasets
    subfolder_train: str = "" # timestamp of training in format %Y_%m%d_%H%M
    itr_i: int = 0 # current training iteration. [0, itr-1]
    seed: int = None
