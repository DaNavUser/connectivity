import random

import torch

from pathlib import Path
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset, download_url


class ConnectivityDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.n_min, self.n_max = 10, 30
        self.data_path = Path('./data/data_lst.pt')
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [str(self.data_path), ]

    @property
    def processed_file_names(self):
        return ['data_lst_processed.pt']

    def download(self):
        # Download to `self.raw_dir`.
        self.data_creator(n_min=self.n_min, n_max=self.n_max)

    def process(self):
        # Read data into huge `Data` list.
        data_list = torch.load(self.data_path)['data_lst']

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def data_creator(self, n_min, n_max):
        data_lst = []
        for i in range(n_min, n_max):
            graphs = self.graph_creator(i)
            data_lst.extend(graphs)
        Path.mkdir(self.data_path, exist_ok=True)
        torch.save({'data_lst': data_lst}, self.data_path)

    def graph_creator(self, i):
        positive, negative = self.connected_graph(i), self.unconnected_graph(i)
        graphs = [positive, negative]
        return graphs

    def connected_graph(self, i):
        x = torch.ones((1, i), dtype=torch.float)
        in_nodes = list(range(i))
        out_nodes = list(range(1, i)) + [0, ]
        edge_index = torch.tensor([in_nodes,
                                   out_nodes], dtype=torch.long)
        y = torch.ones(1, dtype=torch.float)
        data = Data(x=x, edge_index=edge_index, y=y)
        return data

    def unconnected_graph(self, i):
        j = random.randint(2, i)
        x = torch.ones((1, i), dtype=torch.float)
        in_nodes_1 = list(range(j))
        out_nodes_1 = list(range(1, j)) + [0, ]

        in_nodes_2 = list(range(j, i))
        out_nodes_2 = list(range(j, i)) + [j, ]

        in_nodes = in_nodes_1 + in_nodes_2
        out_nodes = out_nodes_1 + out_nodes_2

        edge_index = torch.tensor([in_nodes,
                                   out_nodes], dtype=torch.long)
        y = torch.ones(0, dtype=torch.float)
        data = Data(x=x, edge_index=edge_index, y=y)
        return data
