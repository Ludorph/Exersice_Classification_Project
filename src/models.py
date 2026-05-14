from __future__ import annotations

import numpy as np
import torch
from torch import nn

from data_utils import MEDIAPIPE_LANDMARKS


class LSTMClassifier(nn.Module):
    def __init__(
        self,
        input_size: int,
        num_classes: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=True,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1])


def mediapipe_edges() -> list[tuple[int, int]]:
    idx = {name: i for i, name in enumerate(MEDIAPIPE_LANDMARKS)}
    pairs = [
        ("left_shoulder", "right_shoulder"),
        ("left_shoulder", "left_elbow"),
        ("left_elbow", "left_wrist"),
        ("right_shoulder", "right_elbow"),
        ("right_elbow", "right_wrist"),
        ("left_shoulder", "left_hip"),
        ("right_shoulder", "right_hip"),
        ("left_hip", "right_hip"),
        ("left_hip", "left_knee"),
        ("left_knee", "left_ankle"),
        ("right_hip", "right_knee"),
        ("right_knee", "right_ankle"),
        ("left_ankle", "left_heel"),
        ("left_heel", "left_foot_index"),
        ("right_ankle", "right_heel"),
        ("right_heel", "right_foot_index"),
        ("left_wrist", "left_index_1"),
        ("left_wrist", "left_thumb_2"),
        ("right_wrist", "right_index_1"),
        ("right_wrist", "right_thumb_2"),
        ("nose", "left_eye"),
        ("nose", "right_eye"),
        ("left_eye", "left_ear"),
        ("right_eye", "right_ear"),
    ]
    return [(idx[a], idx[b]) for a, b in pairs]


def normalized_adjacency(num_nodes: int = 33) -> torch.Tensor:
    adj = np.eye(num_nodes, dtype=np.float32)
    for i, j in mediapipe_edges():
        adj[i, j] = 1.0
        adj[j, i] = 1.0
    degree = adj.sum(axis=1)
    degree_inv_sqrt = np.diag(np.power(degree, -0.5))
    norm = degree_inv_sqrt @ adj @ degree_inv_sqrt
    return torch.tensor(norm, dtype=torch.float32)


class STGCNBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, adjacency: torch.Tensor, dropout: float = 0.25) -> None:
        super().__init__()
        self.register_buffer("adjacency", adjacency)
        self.spatial = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.temporal = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=(9, 1), padding=(4, 0)),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout),
        )
        self.residual = (
            nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual(x)
        x = torch.einsum("nctv,vw->nctw", x, self.adjacency)
        x = self.spatial(x)
        x = self.temporal(x)
        return self.relu(x + residual)


class STGCNClassifier(nn.Module):
    def __init__(self, num_classes: int, num_nodes: int = 33, in_channels: int = 3) -> None:
        super().__init__()
        adjacency = normalized_adjacency(num_nodes)
        self.net = nn.Sequential(
            STGCNBlock(in_channels, 64, adjacency),
            STGCNBlock(64, 128, adjacency),
            STGCNBlock(128, 128, adjacency),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)

