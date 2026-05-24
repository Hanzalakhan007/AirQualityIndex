"""PyTorch AQI forecasting model helpers."""
from __future__ import annotations

from io import BytesIO
from typing import Any

import numpy as np
import torch
from torch import nn


class AQIPyTorchRegressor(nn.Module):
    """Compact feed-forward regressor for 4-day AQI forecasting."""

    def __init__(self, input_dim: int, output_dim: int = 4) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(64, output_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


def build_pytorch_model(input_dim: int) -> AQIPyTorchRegressor:
    return AQIPyTorchRegressor(input_dim=input_dim)


def predict_pytorch_model(model: nn.Module, values: Any) -> np.ndarray:
    """Run batched inference and return a NumPy array."""
    model.eval()
    with torch.no_grad():
        tensor = torch.as_tensor(np.asarray(values), dtype=torch.float32)
        predictions = model(tensor).detach().cpu().numpy()
    return predictions


def serialize_pytorch_checkpoint(model: nn.Module, input_dim: int, feature_names: list[str]) -> bytes:
    """Serialize model weights plus metadata for registry/local fallback loading."""
    payload = {
        "state_dict": model.state_dict(),
        "input_dim": int(input_dim),
        "feature_names": list(feature_names),
    }
    buffer = BytesIO()
    torch.save(payload, buffer)
    return buffer.getvalue()


def save_pytorch_checkpoint(path: str, model: nn.Module, input_dim: int, feature_names: list[str]) -> None:
    payload = {
        "state_dict": model.state_dict(),
        "input_dim": int(input_dim),
        "feature_names": list(feature_names),
    }
    torch.save(payload, path)


def load_pytorch_checkpoint(source: Any) -> tuple[nn.Module, dict[str, Any]]:
    """Restore a model from a path or in-memory byte stream."""
    checkpoint = torch.load(source, map_location="cpu")
    input_dim = int(checkpoint["input_dim"])
    model = build_pytorch_model(input_dim)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint
