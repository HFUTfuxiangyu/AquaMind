"""Train and evaluate the AquaMind APN water-quality forecasting model."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
sys.path.insert(0, str(BACKEND))

from models.apn_wrapper import APNModelWrapper  # noqa: E402


FEATURES = ["turbidity", "ph", "chlorine", "cod", "ammonia"]


class WindowDataset(Dataset):
    def __init__(self, values: np.ndarray, seq_len: int, pred_len: int, stride: int = 1):
        self.values = torch.tensor(values, dtype=torch.float32)
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.starts = list(range(0, len(values) - seq_len - pred_len + 1, stride))
        total = seq_len + pred_len
        self.time = torch.arange(total, dtype=torch.float32).unsqueeze(-1) / float(seq_len)

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, index):
        start = self.starts[index]
        split = start + self.seq_len
        end = split + self.pred_len
        x = self.values[start:split]
        y = self.values[split:end]
        return x, torch.ones_like(x), self.time[: self.seq_len], self.time[self.seq_len :], y


def evaluate(model, loader, device, min_vals, ranges):
    model.eval()
    squared = 0.0
    absolute = 0.0
    count = 0
    feature_abs = np.zeros(len(FEATURES), dtype=np.float64)
    feature_sq = np.zeros(len(FEATURES), dtype=np.float64)
    with torch.no_grad():
        for x, x_mask, x_mark, y_mark, y in loader:
            x, x_mask, x_mark, y_mark, y = [v.to(device) for v in (x, x_mask, x_mark, y_mark, y)]
            pred = model(x=x, x_mask=x_mask, x_mark=x_mark, y_mark=y_mark)["pred"]
            pred_real = pred * ranges + min_vals
            y_real = y * ranges + min_vals
            error = pred_real - y_real
            squared += error.square().sum().item()
            absolute += error.abs().sum().item()
            count += error.numel()
            feature_abs += error.abs().sum(dim=(0, 1)).cpu().numpy()
            feature_sq += error.square().sum(dim=(0, 1)).cpu().numpy()
    points_per_feature = count / len(FEATURES)
    return {
        "mae": absolute / count,
        "rmse": (squared / count) ** 0.5,
        "per_feature": {
            name: {
                "mae": float(feature_abs[i] / points_per_feature),
                "rmse": float((feature_sq[i] / points_per_feature) ** 0.5),
            }
            for i, name in enumerate(FEATURES)
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=BACKEND / "static" / "model_weights" / "apn_water_model.pth")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--train-stride", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260809)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))

    df = pd.read_csv(args.data, encoding="utf-8-sig")
    missing = [column for column in ["timestamp", *FEATURES, "split"] if column not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if df[FEATURES].isna().any().any():
        raise ValueError("Training data contains missing feature values")

    train_raw = df.loc[df["split"] == "train", FEATURES].to_numpy(dtype=np.float32)
    val_raw = df.loc[df["split"] == "validation", FEATURES].to_numpy(dtype=np.float32)
    test_raw = df.loc[df["split"] == "test", FEATURES].to_numpy(dtype=np.float32)
    min_np = train_raw.min(axis=0)
    max_np = train_raw.max(axis=0)
    range_np = max_np - min_np
    range_np[range_np == 0] = 1.0

    wrapper = APNModelWrapper()
    config = wrapper._create_config()
    Model = wrapper._import_apn_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model(config).to(device)

    normalize = lambda values: (values - min_np) / range_np
    train_ds = WindowDataset(normalize(train_raw), config.seq_len, config.pred_len, args.train_stride)
    val_ds = WindowDataset(normalize(val_raw), config.seq_len, config.pred_len)
    test_ds = WindowDataset(normalize(test_raw), config.seq_len, config.pred_len)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss()
    min_t = torch.tensor(min_np, dtype=torch.float32, device=device).view(1, 1, -1)
    range_t = torch.tensor(range_np, dtype=torch.float32, device=device).view(1, 1, -1)
    best_loss = float("inf")
    best_state = None
    stale = 0
    history = []
    started = time.time()

    print(f"device={device} train_windows={len(train_ds)} val_windows={len(val_ds)} test_windows={len(test_ds)}", flush=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        samples = 0
        for x, x_mask, x_mark, y_mark, y in train_loader:
            x, x_mask, x_mark, y_mark, y = [v.to(device) for v in (x, x_mask, x_mark, y_mark, y)]
            optimizer.zero_grad(set_to_none=True)
            pred = model(x=x, x_mask=x_mask, x_mark=x_mark, y_mark=y_mark)["pred"]
            loss = criterion(pred, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * x.size(0)
            samples += x.size(0)

        val_metrics = evaluate(model, val_loader, device, min_t, range_t)
        train_loss = total_loss / samples
        history.append({"epoch": epoch, "train_mse_normalized": train_loss, "val_mae": val_metrics["mae"], "val_rmse": val_metrics["rmse"]})
        print(f"epoch={epoch:02d} train_mse={train_loss:.6f} val_mae={val_metrics['mae']:.5f} val_rmse={val_metrics['rmse']:.5f}", flush=True)
        if val_metrics["rmse"] < best_loss - 1e-6:
            best_loss = val_metrics["rmse"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                print(f"early_stopping epoch={epoch}", flush=True)
                break

    model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader, device, min_t, range_t)
    training_info = {
        "data_file": args.data.name,
        "synthetic_data": True,
        "seq_len": config.seq_len,
        "pred_len": config.pred_len,
        "train_windows": len(train_ds),
        "validation_windows": len(val_ds),
        "test_windows": len(test_ds),
        "epochs_completed": len(history),
        "seed": args.seed,
        "duration_seconds": round(time.time() - started, 2),
    }
    checkpoint = {
        "model_state_dict": best_state,
        "feature_columns": FEATURES,
        "normalization": {"min": min_np.tolist(), "max": max_np.tolist(), "range": range_np.tolist()},
        "training": training_info,
        "metrics": {"best_validation_rmse": best_loss, "test": test_metrics},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    report_path = args.output.with_suffix(".metrics.json")
    report_path.write_text(json.dumps({"training": training_info, "history": history, "metrics": checkpoint["metrics"]}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"checkpoint={args.output}", flush=True)
    print(f"metrics={report_path}", flush=True)
    print(json.dumps(test_metrics, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
