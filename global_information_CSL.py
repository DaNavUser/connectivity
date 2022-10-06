import networkx
import torch_geometric as pyg
import torch_geometric.nn as gnn
import torch
import torch.nn as nn
import torch.nn.functional as F
import os.path as osp
import torch.optim as optim
import torch_geometric.transforms as T

from globals import *
from tqdm import tqdm
from models import models_creator
from torch_geometric.data import Data
from torch_geometric.datasets import GNNBenchmarkDataset
from torch_geometric.data import DataLoader, Dataset, download_url

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#
lr = 1e-3
epochs = 250
batch_size = 32
num_nodes = 1312

model_name = 'mlp'  # ['gat', 'mpnn']
in_channels, hidden_channels, out_channels, num_layers = 20, 128, 10, 6
model = models_creator(model_name=model_name, in_channels=in_channels, hidden_channels=hidden_channels,
                       out_channels=out_channels, num_layers=num_layers)
optimizer = optim.Adam(model.parameters(), lr=lr)

print(T.__all__)

criterion = nn.CrossEntropyLoss()
transform = T.Compose([T.ToDevice(device),
                       T.RandomNodeSplit(split='train_rest', num_val=0.1, num_test=.0), ])

dataset = GNNBenchmarkDataset(root='./Iso', name='CSL', split="train", transform=transform)
dl = DataLoader(dataset, batch_size=batch_size, shuffle=True)

random_features = torch.randn(num_nodes, in_channels)  # random features.


def train_epoch(dl, epoch):
    cnt, time = 0, 200
    for data in tqdm(dl):
        batch = data.batch.to(device)
        data.x = torch.randn(data.num_nodes, in_channels)  # random features.
        x, edge_index = data.x.float(), data.edge_index
        x, edge_index = x.to(device), edge_index.to(device)

        pred_ = model(x=x, edge_index=edge_index)
        pred = pred_[data.ptr[:-1]]
        y = data.y

        loss = criterion(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if cnt % time == time - 1:
            print(f'Epoch [{epoch}/ {epochs}] loss: {loss.item():.2f}')
        cnt += 1

    print(f'Epoch [{epoch}/ {epochs}] loss: {loss.item():.2f}')


def trainer():
    for epoch in range(epochs):
        train_epoch(dl, epoch)


print('start training')
trainer()

# After we trained the model let optimize it for specific properties.

# In[ ]:


model.eval()
opt_graph = 1
