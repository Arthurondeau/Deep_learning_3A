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



class CNNClassifier(nn.Module):
    """CNN Baseline Model"""

    def __init__(self, channels:int , in_dim:int, n_classes:int, num_filters:int =64, kernel_size:int =3, pool_size:int =2):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=channels, out_channels=num_filters, kernel_size=kernel_size, padding=kernel_size // 2)
        self.conv2 = nn.Conv1d(in_channels=num_filters, out_channels=num_filters*2, kernel_size=kernel_size, padding=kernel_size // 2)
        self.maxpool = nn.MaxPool1d(kernel_size=pool_size)
        self.batchnorm1 = nn.BatchNorm1d(num_filters)
        self.batchnorm2 = nn.BatchNorm1d(num_filters*2)
        
        # Calculate output size after max pooling
        self.fc_input_size = num_filters*2 * (in_dim // (pool_size**2))
        self.fc = nn.Linear(self.fc_input_size, n_classes)

    def forward(self, x):
        x = torch.relu(self.batchnorm1(self.conv1(x)))
        x = self.maxpool(x)
        x = torch.relu(self.batchnorm2(self.conv2(x)))
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x



class RNNClassifier(nn.Module):
    """RNN with LSTM or GRU"""

    def __init__(self, in_dim:int, hidden_size:int, num_layers:int, num_classes:int, rnn_type:str='LSTM'):
        super(RNNClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(in_dim, hidden_size, num_layers, batch_first=True)
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(in_dim, hidden_size, num_layers, batch_first=True)        
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.rnn(x, (h0, c0)) if isinstance(self.rnn, nn.LSTM) else self.rnn(x, h0)
        out = self.fc(out[:, -1, :])  # Take the output of the last time step
        return out
    

class BiRNNClassifier(nn.Module):
    """Bi directional LSTM or GRU"""

    def __init__(self, in_dim, hidden_size, num_layers, num_classes, rnn_type='LSTM'):
        super(BiRNNClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(in_dim, hidden_size, num_layers, batch_first=True, bidirectional=True)
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(in_dim, hidden_size, num_layers, batch_first=True, bidirectional=True)
        
        self.fc = nn.Linear(hidden_size*2, num_classes)  # Multiply by 2 for bidirectional

    def forward(self, x):
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(x.device)  # Multiply by 2 for bidirectional
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(x.device)  # Multiply by 2 for bidirectional
        
        out, _ = self.rnn(x, (h0, c0)) if isinstance(self.rnn, nn.LSTM) else self.rnn(x, h0)
        out = self.fc(out[:, -1, :])  # Take the output of the last time step
        return out



class StackedBiRNNClassifier(nn.Module):
    """stack Bi directional LSTM or GRU"""
    def __init__(self, in_dim, hidden_size, num_layers, num_classes, rnn_type='LSTM'):
        super(StackedBiRNNClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        if rnn_type == 'LSTM':
            self.rnn_layers = nn.ModuleList([nn.LSTM(in_dim, hidden_size, batch_first=True, bidirectional=True)])
            self.rnn_layers.extend([nn.LSTM(hidden_size * 2, hidden_size, batch_first=True, bidirectional=True) for _ in range(num_layers - 1)])
        elif rnn_type == 'GRU':
            self.rnn_layers = nn.ModuleList([nn.GRU(in_dim, hidden_size, batch_first=True, bidirectional=True)])
            self.rnn_layers.extend([nn.GRU(hidden_size * 2, hidden_size, batch_first=True, bidirectional=True) for _ in range(num_layers - 1)])
        
        self.fc = nn.Linear(hidden_size*2, num_classes)  # Multiply by 2 for bidirectional

    def forward(self, x):
        for layer in self.rnn_layers:
            out, _ = layer(x)
            x = out
        
        out = self.fc(out[:, -1, :])  # Take the output of the last time step
        return out


class TransformerEncoder(nn.Module):
    """Encoder with transformer layers"""

    def __init__(
        self, in_dim: int, channels: int, hidden_dim: int, n_layers: int,nhead: int,n_classes :int, dropout: float
    ):
        super().__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=in_dim, nhead=nhead)
        self.model = nn.TransformerEncoder(self.encoder_layer, num_layers=n_layers)
        self.classifier = nn.Sequential(nn.Linear(in_dim * channels,hidden_dim),nn.Linear(hidden_dim, n_classes))

    def forward(self, data: Tensor) -> Tensor:
        """forward pass"""
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
        """forward pass"""
        x = self.model(x)
        x = torch.reshape(x,(x.size()[0],-1))
        sleep_stage_output = self.sleep_stage_classifier(x)
        subject_output = self.subject_classifier(x)
        return sleep_stage_output, subject_output
    

