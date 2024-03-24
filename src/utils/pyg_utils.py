"""pyg utils"""
from typing import Any, Tuple

import numpy as np
import torch
import torch_geometric
from torch import nn
from torch_geometric.data import Data
from torch_geometric.data.data import BaseData


def embed_from_full_model(
    model: torch_geometric.nn.DeepGraphInfomax,
    data: BaseData,
    model_type: str,
) -> np.ndarray:
    """embeddings from full model (like dgi with encoder+prediction head)

    Args:
        model (nn.Module)
        data (torch_geometric.data.Data)

    Returns:
        np.array embedded features
    """
    if model_type == "dgi":
        return embed_from_encoder(model.encoder, data)
    else:
        with torch.no_grad():
            embeds = model.encoder(data)
            embeds = embeds.numpy()
        return embeds


def embed_from_encoder(model: nn.Module, data: BaseData) -> np.ndarray:
    """embedding using only an encoder"""
    model.eval()
    with torch.no_grad():
        model.cpu()
        data.cpu()
        embeds = model.forward(data)
        embeds = embeds.numpy()
    return embeds


def features_permutation_corruption(
    data: Data,
) -> Data:
    """randomly assign feature vectors to nodes as per DGI paper"""
    data_copy = data.clone()
    features = data_copy.x
    data_copy.x = features[torch.randperm(features.size(0))]
    return data_copy


def identity_corruption(
    features: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """identity"""
    return features, edge_index, edge_weight


def sigmoid_summary(tensor: torch.Tensor, *_: Any, **__: Any) -> torch.Tensor:
    """Avg the tensor along the cells dimensions
    tensor: NxD, output: 1*D
    """
    return torch.sigmoid(tensor.mean(dim=0))
