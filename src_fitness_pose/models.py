from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .constants import JOINTS, SKELETON_EDGES


def normalized_adjacency() -> torch.Tensor:
    joint_index = {name: index for index, name in enumerate(JOINTS)}
    adjacency = np.eye(len(JOINTS), dtype=np.float32)
    for left, right in SKELETON_EDGES:
        i, j = joint_index[left], joint_index[right]
        adjacency[i, j] = 1.0
        adjacency[j, i] = 1.0
    degree = adjacency.sum(axis=1)
    inverse_sqrt = np.diag(np.power(degree, -0.5))
    return torch.tensor(inverse_sqrt @ adjacency @ inverse_sqrt, dtype=torch.float32)


class GraphConvBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, dropout: float) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)
        self.residual = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        residual = self.residual(x)
        x = torch.einsum("vw,nwf->nvf", adjacency, x)
        x = self.linear(x)
        return torch.relu(self.norm(self.dropout(x)) + residual)


class GNNClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.register_buffer("adjacency", normalized_adjacency())
        dimensions = [input_dim] + [hidden_dim] * num_layers
        self.layers = nn.ModuleList(
            GraphConvBlock(dimensions[index], dimensions[index + 1], dropout)
            for index in range(num_layers)
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, self.adjacency)
        return self.head(x.mean(dim=1))


class JointTransformerClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.joint_embedding = nn.Parameter(torch.zeros(1, len(JOINTS), d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )
        nn.init.normal_(self.joint_embedding, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_projection(x) + self.joint_embedding
        x = self.encoder(x)
        return self.head(x.mean(dim=1))


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

