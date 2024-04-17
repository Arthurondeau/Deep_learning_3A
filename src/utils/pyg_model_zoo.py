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



# Define the combined model
class TransformerAdv(nn.Module):
    def __init__(
        self, in_dim: int, channels: int, hidden_dim: int, n_layers: int,nhead: int,n_classes :int, dropout: float
    ):
        super().__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=in_dim, nhead=nhead)
        self.model = nn.TransformerEncoder(self.encoder_layer, num_layers=n_layers)
        self.sleep_stage_classifier = nn.Sequential(nn.Linear(in_dim * channels,hidden_dim),nn.Linear(hidden_dim, n_classes))
        self.subject_classifier = nn.Sequential(nn.Linear(in_dim*channels,hidden_dim),nn.Linear(hidden_dim, 10))

    def forward(self, x):
        x = self.model(x)
        x = torch.reshape(x,(x.size()[0],-1))
        sleep_stage_output = self.sleep_stage_classifier(x)
        subject_output = self.subject_classifier(x)
        return sleep_stage_output, subject_output