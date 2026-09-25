from __future__ import annotations

import torch
from torch import nn


class LSTMRegressor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2) -> None:
        super().__init__()
        recurrent_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=recurrent_dropout,
            batch_first=True,
        )
        projection = max(hidden_size // 2, 1)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, projection),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(projection, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sequence_output, _ = self.lstm(x)
        return self.head(sequence_output[:, -1, :])


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
