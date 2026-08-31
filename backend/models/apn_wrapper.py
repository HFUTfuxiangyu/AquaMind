import importlib
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)


class ModelNotLoadedError(Exception):
    pass


class APNModelWrapper:
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.device = torch.device(settings.apn_device if torch.cuda.is_available() else "cpu")
        self.model_loaded = False
        self.model_config = None
        self.model_path = None
        self.checkpoint_metadata: Dict[str, Any] = {}

        if model_path:
            self.load_model(model_path)

    def load_model(self, model_path: str) -> bool:
        try:
            model_file = Path(model_path)
            if not model_file.exists():
                logger.warning(f"APN weight file does not exist: {model_path}")
                return False

            Model = self._import_apn_model()
            self.model_config = self._create_config()
            self.model = Model(self.model_config)

            checkpoint = torch.load(model_file, map_location=self.device)
            if isinstance(checkpoint, dict):
                self.checkpoint_metadata = {
                    key: checkpoint[key]
                    for key in ("normalization", "feature_columns", "training", "metrics")
                    if key in checkpoint
                }
                checkpoint_features = checkpoint.get("feature_columns")
                if checkpoint_features and list(checkpoint_features) != list(settings.feature_columns):
                    raise ValueError(
                        f"Checkpoint features {checkpoint_features} do not match configured features {settings.feature_columns}"
                    )
            state_dict = self._extract_state_dict(checkpoint)
            load_result = self.model.load_state_dict(state_dict, strict=False)
            if load_result.missing_keys:
                raise ValueError(f"APN checkpoint missing keys: {load_result.missing_keys[:10]}")
            if load_result.unexpected_keys:
                raise ValueError(f"APN checkpoint unexpected keys: {load_result.unexpected_keys[:10]}")

            self.model.to(self.device)
            self.model.eval()
            self.model_loaded = True
            self.model_path = str(model_file)
            logger.info(f"APN weights loaded: {model_path}")
            return True

        except Exception as exc:
            logger.error(f"APN model load failed: {exc}", exc_info=True)
            self.model = None
            self.model_loaded = False
            return False

    def _import_apn_model(self):
        apn_path = Path(__file__).resolve().parent.parent / "apn_runtime"
        if not apn_path.exists():
            raise ImportError(f"Bundled APN runtime not found: {apn_path}")

        old_path = list(sys.path)
        old_modules = {
            name: module
            for name, module in list(sys.modules.items())
            if name == "models"
            or name.startswith("models.")
            or name == "utils"
            or name.startswith("utils.")
        }
        for name in old_modules:
            sys.modules.pop(name, None)

        try:
            sys.path.insert(0, str(apn_path))
            module = importlib.import_module("models.APN")
            return module.Model
        finally:
            sys.path = old_path
            for name in list(sys.modules):
                if (
                    name == "models"
                    or name.startswith("models.")
                    or name == "utils"
                    or name.startswith("utils.")
                ):
                    sys.modules.pop(name, None)
            sys.modules.update(old_modules)

    def _extract_state_dict(self, checkpoint: Any) -> Dict[str, torch.Tensor]:
        if isinstance(checkpoint, dict):
            for key in ("state_dict", "model_state_dict", "model", "module"):
                value = checkpoint.get(key)
                if isinstance(value, dict):
                    checkpoint = value
                    break

        if not isinstance(checkpoint, dict):
            raise ValueError("Unsupported APN checkpoint format")

        state_dict = {}
        for key, value in checkpoint.items():
            if not isinstance(value, torch.Tensor):
                continue
            clean_key = key[len("module.") :] if key.startswith("module.") else key
            state_dict[clean_key] = value.contiguous()

        if not state_dict:
            raise ValueError("No tensor weights found in APN checkpoint")
        return state_dict

    def _create_config(self) -> Any:
        class SimpleConfig:
            def __init__(self):
                self.task_name = "short_term_forecast"
                self.model_name = "APN"
                self.is_training = 0

                self.seq_len = settings.max_sequence_length
                self.pred_len = settings.prediction_horizon
                self.enc_in = len(settings.feature_columns)
                self.dec_in = len(settings.feature_columns)
                self.c_out = len(settings.feature_columns)
                self.features = "M"
                self.pred_len_max_irr = None
                self.seq_len_max_irr = None
                self.patch_len_max_irr = None

                self.d_model = 24
                self.apn_npatch = 20
                self.apn_patch_size = 0.05
                self.apn_nlayer = 1
                self.apn_attn_heads = 1
                self.apn_te_dim = 8
                self.apn_asym = 1
                self.apn_conf = 0
                self.apn_multires = 0
                self.apn_contrast = 0
                self.apn_prob = 0
                self.apn_lcvc = 0
                self.apn_lcvc_rank = 4
                self.apn_ms_tapa = 0
                self.apn_ms_tapa_coarse = 8
                self.apn_ms_tapa_fine = 16
                self.apn_ms_tapa_iaf = 1
                self.apn_vat_tapa = 0
                self.apn_nudft = 0
                self.apn_nudft_k = 16
                self.apn_dt_decoder = 0
                self.apn_dt_emb_dim = 8
                self.use_ctrope = 0
                self.ctrope_omega_init = 100.0
                self.ctrope_learnable = 1
                self.lcvc_mode = "static"
                self.lcvc_gating_hidden = 16
                self.lcvc_odag = 0
                self.lcvc_odag_topk = 0
                self.dropout = 0.1

        return SimpleConfig()

    def predict(self, data: Dict[str, Any], prediction_horizon: Optional[int] = None) -> Dict[str, Any]:
        if not self.model_loaded:
            raise ModelNotLoadedError("APN model is not loaded")

        try:
            processed = self._preprocess_data(data)
            horizon = prediction_horizon or self.model_config.pred_len
            y_mark = self._create_future_mark(processed["x"], processed["x_mark"], horizon)

            with torch.no_grad():
                output = self.model(
                    x=processed["x"],
                    x_mark=processed["x_mark"],
                    x_mask=processed["x_mask"],
                    y_mark=y_mark,
                )

            return self._postprocess_predictions(output)

        except Exception as exc:
            logger.error(f"APN inference failed: {exc}", exc_info=True)
            raise

    def _preprocess_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        processed = {}
        for key, value in data.items():
            processed[key] = value.to(self.device) if isinstance(value, torch.Tensor) else value

        for key in ("x", "x_mark", "x_mask"):
            value = processed.get(key)
            if not isinstance(value, torch.Tensor):
                raise ValueError(f"Missing tensor input: {key}")
            if len(value.shape) == 2:
                processed[key] = value.unsqueeze(0)
            elif len(value.shape) != 3:
                raise ValueError(f"Invalid {key} shape: {value.shape}")

        return processed

    def _create_future_mark(self, x: torch.Tensor, x_mark: torch.Tensor, horizon: int) -> torch.Tensor:
        batch_size = x.shape[0]
        seq_len = max(x.shape[1], 1)
        last_time = x_mark[:, -1:, [0]]
        steps = torch.arange(1, horizon + 1, dtype=x.dtype, device=x.device).view(1, horizon, 1)
        return last_time + steps.repeat(batch_size, 1, 1) / float(seq_len)

    def _postprocess_predictions(self, predictions: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        result: Dict[str, Any] = {}

        pred_tensor = predictions.get("pred")
        if pred_tensor is not None:
            pred_array = pred_tensor[0].detach().cpu().numpy() if len(pred_tensor.shape) == 3 else pred_tensor.detach().cpu().numpy()
            result["predictions"] = pred_array.tolist()

        true_tensor = predictions.get("true")
        if true_tensor is not None:
            true_array = true_tensor[0].detach().cpu().numpy() if len(true_tensor.shape) == 3 else true_tensor.detach().cpu().numpy()
            result["true_values"] = true_array.tolist()

        result["model_info"] = {
            "model_name": "APN",
            "model_version": "1.0.0",
            "device": str(self.device),
            "weight_path": self.model_path,
            "input_features": settings.feature_columns,
            "metrics": self.checkpoint_metadata.get("metrics"),
        }
        return result

    def is_loaded(self) -> bool:
        return self.model_loaded

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "loaded": self.model_loaded,
            "device": str(self.device),
            "weight_path": self.model_path,
            "config": self.model_config.__dict__ if self.model_config else None,
            "feature_columns": settings.feature_columns,
            "training": self.checkpoint_metadata.get("training"),
            "metrics": self.checkpoint_metadata.get("metrics"),
        }

    def get_normalization(self) -> Optional[Dict[str, Any]]:
        return self.checkpoint_metadata.get("normalization")
