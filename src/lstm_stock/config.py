from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    ticker: str = "AAPL"
    start: str = "2015-01-01"
    end: str | None = None
    interval: str = "1d"
    date_column: str = "Date"
    target: str = "Close"
    features: list[str] = field(default_factory=lambda: ["Close"])
    train_ratio: float = 0.70
    val_ratio: float = 0.15


@dataclass
class ModelConfig:
    sequence_length: int = 60
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.20


@dataclass
class TrainConfig:
    batch_size: int = 64
    epochs: int = 60
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 10
    lr_patience: int = 4
    lr_factor: float = 0.5
    min_learning_rate: float = 1e-6
    gradient_clip_norm: float = 1.0
    seed: int = 42
    device: str = "auto"


@dataclass
class ForecastConfig:
    horizon: int = 30


@dataclass
class OutputConfig:
    artifact_dir: str = "artifacts"


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    forecast: ForecastConfig = field(default_factory=ForecastConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _section(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Configuration section '{key}' must be a mapping.")
    return value


def validate_config(config: ExperimentConfig) -> None:
    if not config.data.features:
        raise ValueError("At least one input feature is required.")
    if config.data.target not in config.data.features:
        raise ValueError("The target must be included in features for recursive forecasting.")
    if not 0.0 < config.data.train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1.")
    if not 0.0 < config.data.val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1.")
    if config.data.train_ratio + config.data.val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must leave a non-empty test partition.")
    if config.model.sequence_length < 2:
        raise ValueError("sequence_length must be at least 2.")
    if config.model.hidden_size < 1 or config.model.num_layers < 1:
        raise ValueError("hidden_size and num_layers must be positive.")
    if config.model.num_layers == 1 and config.model.dropout != 0:
        config.model.dropout = 0.0
    if config.train.epochs < 1 or config.train.batch_size < 1:
        raise ValueError("epochs and batch_size must be positive.")
    if config.forecast.horizon < 0:
        raise ValueError("forecast horizon cannot be negative.")


def load_config(path: str | Path) -> ExperimentConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    config = ExperimentConfig(
        data=DataConfig(**_section(raw, "data")),
        model=ModelConfig(**_section(raw, "model")),
        train=TrainConfig(**_section(raw, "train")),
        forecast=ForecastConfig(**_section(raw, "forecast")),
        output=OutputConfig(**_section(raw, "output")),
    )
    validate_config(config)
    return config
