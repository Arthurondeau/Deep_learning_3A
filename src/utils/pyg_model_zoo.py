"""
pyg models
"""
from typing import Any, Callable, Dict, List, Mapping

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import Tensor, nn
from torch.nn import LayerNorm, ReLU



class TransformerEncoder(nn.Module):
    """Encoder with transformer layers"""

    def __init__(
        self, in_dim: int, hidden_dim: int, n_layers: int,nhead: int,n_classes :int, dropout: float
    ):
        super().__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=in_dim, nhead=nhead)
        self.model = nn.TransformerEncoder(self.encoder_layer, num_layers=n_layers)
        self.classifier = nn.Linear(in_dim, n_classes)

    def forward(self, data: Tensor) -> Tensor:
        """forward"""
        embedding = self.model(data)
        digits = self.classifier(embedding)

        return digits



