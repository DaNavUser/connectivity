import networkx
import torch_geometric as pyg
import torch_geometric.nn as gnn
import torch
import torch.nn as nn
import torch.nn.functional as F
import os.path as osp

from torch_geometric.data import Data
from torch_geometric.data import Dataset, download_url
from torch_geometric.datasets import GNNBenchmarkDataset
from globals import *


def models_creator(model_name, **kwargs):
    if model_name == 'mpnn':
        model = get_mpnn(**kwargs).to(device)
    elif model_name == 'gat':
        model = get_gat(**kwargs).to(device)
    else:
        raise NotImplementedError(f"model {model_name} isn't supported yet.")
    return model


def get_mpnn(in_channels, hidden_channels, out_channels, num_layers):
    mpnn = gnn.GCN(in_channels=in_channels, hidden_channels=hidden_channels,
                   out_channels=out_channels, num_layers=num_layers)
    return mpnn


def get_gat(in_channels, hidden_channels, out_channels, num_layers):
    """Attending over neighbors"""
    gat = gnn.GAT(in_channels=in_channels, hidden_channels=hidden_channels,
                  out_channels=out_channels, num_layers=num_layers)
    return gat


def graphormer():
    """Global attention to grab global information"""
    # TODO implement
    pass
