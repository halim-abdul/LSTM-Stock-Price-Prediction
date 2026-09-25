from __future__ import annotations

import copy
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .config import TrainConfig


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    requested = requested.lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    return torch.device(requested)


def _loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
    )


def fit_model(
    model: nn.Module,
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    config: TrainConfig,
    checkpoint_path: str | Path | None = None,
) -> tuple[pd.DataFrame, torch.device]:
    set_seed(config.seed)
    device = resolve_device(config.device)
    model.to(device)
    train_loader = _loader(train_x, train_y, config.batch_size, True)
    val_loader = _loader(val_x, val_y, config.batch_size, False)

    criterion = nn.HuberLoss(delta=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=config.lr_factor, patience=config.lr_patience, min_lr=config.min_learning_rate
    )

    best_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    stalled = 0
    rows = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        train_sum = train_count = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            optimizer.step()
            train_sum += float(loss.item()) * batch_x.size(0)
            train_count += batch_x.size(0)

        model.eval()
        val_sum = val_count = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                loss = criterion(model(batch_x), batch_y)
                val_sum += float(loss.item()) * batch_x.size(0)
                val_count += batch_x.size(0)

        train_loss = train_sum / max(train_count, 1)
        val_loss = val_sum / max(val_count, 1)
        scheduler.step(val_loss)
        rows.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        })

        if val_loss < best_loss - 1e-8:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            stalled = 0
        else:
            stalled += 1
        if stalled >= config.patience:
            break

    model.load_state_dict(best_state)

    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state_dict": model.state_dict(), "best_validation_loss": best_loss}, checkpoint_path)

    return pd.DataFrame(rows), device
