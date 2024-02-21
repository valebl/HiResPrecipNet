import pickle
import sys
import numpy as np

import torch
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_convert

from torch_geometric.data import Data, HeteroData

import torch
import numpy as np
from typing import Sequence, Union
from torch_geometric.data import Data, Batch
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.utils import degree

import torch_geometric.transforms as T
transform = T.AddLaplacianEigenvectorPE(k=2)

Graph = Union[HeteroData, None]
Targets = Sequence[Union[np.ndarray, None]]
DoubleCl = Union[bool, None]
Additional_Features = Sequence[torch.tensor]


class Dataset_Graph(Dataset):

    def __init__(
        self,
        graph: Graph,
        targets: Targets,
        **kwargs: Additional_Features
    ):
        self.graph = graph
        self.targets = targets
        self.additional_feature_keys = []
        for key, value in kwargs.items():
            setattr(self, key, value)
            self.additional_feature_keys.append(key)
        self._check_temporal_consistency()
        #self._add_node_degree()


    def __len__(self):
        #return len(self.features)
        return self.graph['low'].x.shape[1] - 24
        
    def _check_temporal_consistency(self):
        if self.targets is not None:
            assert (self.graph['low'].x.shape[1] - 24) == self.targets.shape[1], "Temporal dimension inconsistency."

    def _set_snapshot_count(self):
        self.snapshot_count = len(self)
    
    def _get_high_nodes_degree(self, snapshot):
        node_degree = (degree(snapshot['high','within','high'].edge_index[0], snapshot['high'].num_nodes) / 8).unsqueeze(-1)
        return node_degree

    def _get_features(self, time_index: int):
        time_index_x = time_index + 24
        #x_low = self.graph['low'].x[:,time_index-24:time_index+1,:]
        x_low = self.graph['low'].x[:,time_index_x-24:time_index_x+1:6,:]
        x_low = x_low.flatten(start_dim=1, end_dim=-1)
        return x_low
    
    def _get_target(self, time_index: int):
        return self.targets[:,time_index]

    def _get_train_mask(self, target: torch.tensor):
        return ~torch.isnan(target)
    
    def _get_additional_feature(self, time_index: int, feature_key: str):
        feature = getattr(self, feature_key)[:,time_index]
        return feature
    
    def _get_additional_features(self, time_index: int):
        additional_features = {
            key: self._get_additional_feature(time_index, key)
            for key in self.additional_feature_keys
        }
        return additional_features
    
    def __getitem__(self, time_index: int):
        x_low = self._get_features(time_index)
        y = self._get_target(time_index) if self.targets is not None else None
        train_mask = self._get_train_mask(y) if y is not None else None

        additional_features = self._get_additional_features(time_index)

        snapshot = HeteroData()

        for key, value in additional_features.items():
            if value.shape[0] == self.graph['high'].x.shape[0]:
                snapshot['high'][key] = value
            elif value.shape[0] == self.graph['low'].x.shape[0]:
                snapshot['high'][key] = value
       
        snapshot['high'].y = y
        snapshot['high'].train_mask = train_mask
        snapshot.num_nodes = self.graph.num_nodes
        snapshot['high'].num_nodes = self.graph['high'].num_nodes
        snapshot['low'].num_nodes = self.graph['low'].num_nodes
        snapshot.t = time_index
        
        snapshot['low', 'within', 'low'].edge_index = self.graph['low', 'within', 'low'].edge_index
        snapshot['high', 'within', 'high'].edge_index = self.graph['high', 'within', 'high'].edge_index
        snapshot['low', 'to', 'high'].edge_index = self.graph['low', 'to', 'high'].edge_index

        snapshot['low'].x = x_low 
        #snapshot['high'].x_empty = self.graph['high'].x
        snapshot['high'].x = torch.zeros((snapshot['high'].num_nodes,1))
        snapshot['high'].z_std = self.graph['high'].z_std

        snapshot['high'].lon = self.graph['high'].lon
        snapshot['high'].lat = self.graph['high'].lat
        snapshot['low'].lon = self.graph['low'].lon
        snapshot['low'].lat = self.graph['low'].lat

        #snapshot['high'].laplacian_eigenvector_pe = self.graph['high'].laplacian_eigenvector_pe     
        snapshot['high'].deg = self._get_high_nodes_degree(snapshot)

        return snapshot

        node_idx = torch.randint(high=snapshot['high'].num_nodes, size=(1,)).item()
        num_hops = torch.randint(low=2, high=5, size=(1,)).item()

        subset_low, _, _, _ = k_hop_subgraph(node_idx=node_idx, num_hops=1,
                edge_index=snapshot['low', 'to', 'high'].edge_index,
                relabel_nodes=False)
        
        subset_high, _, _, _ = k_hop_subgraph(node_idx=node_idx, num_hops=2,
                edge_index=snapshot['high', 'within', 'high'].edge_index,
                relabel_nodes=False)

        print(subset_low)
        print(subset_high)
        
        
        subset_dict = {
            'low': subset_low,
            'high': subset_high
        }

        s = snapshot.subgraph(subset_dict=subset_dict)
        s['node_idx'] = node_idx
        
        return s
    

class Dataset_Graph_t(Dataset_Graph):

    def _get_features(self, time_index: int):
        time_index_x = time_index + 24
        x_low = self.graph['low'].x[:,time_index_x,:]
        x_low = x_low.flatten(start_dim=1, end_dim=-1)
        return x_low

class Dataset_Graph_subgraphs(Dataset_Graph):

    def __getitem__(self, time_index: int):
            x_low = self._get_features(time_index)
            y = self._get_target(time_index) if self.targets is not None else None
            train_mask = self._get_train_mask(y) if y is not None else None

            additional_features = self._get_additional_features(time_index)

            snapshot = HeteroData()

            for key, value in additional_features.items():
                if value.shape[0] == self.graph['high'].x.shape[0]:
                    snapshot['high'][key] = value
                elif value.shape[0] == self.graph['low'].x.shape[0]:
                    snapshot['high'][key] = value
        
            snapshot['high'].y = y
            # snapshot['high'].train_mask = train_mask
            snapshot.num_nodes = self.graph.num_nodes
            snapshot['high'].num_nodes = self.graph['high'].num_nodes
            snapshot['low'].num_nodes = self.graph['low'].num_nodes
            snapshot.t = time_index
            
            snapshot['low', 'within', 'low'].edge_index = self.graph['low', 'within', 'low'].edge_index
            snapshot['high', 'within', 'high'].edge_index = self.graph['high', 'within', 'high'].edge_index
            snapshot['low', 'to', 'high'].edge_index = self.graph['low', 'to', 'high'].edge_index

            snapshot['low'].x = x_low
            # snapshot['high'].x_empty = self.graph['high'].x
            snapshot['high'].x = torch.zeros((snapshot['high'].num_nodes,1))
            snapshot['high'].z_std = self.graph['high'].z_std

            snapshot['high'].lon = self.graph['high'].lon
            snapshot['high'].lat = self.graph['high'].lat
            snapshot['low'].lon = self.graph['low'].lon
            snapshot['low'].lat = self.graph['low'].lat

            # snapshot['high'].laplacian_eigenvector_pe = self.graph['high'].laplacian_eigenvector_pe     

            # Mask the subgraph to consider only nodes with non-nan target
            train_nodes = torch.arange(snapshot['high'].num_nodes)[train_mask]
            subset_dict = {'high': train_nodes}
            snapshot = snapshot.subgraph(subset_dict)
            snapshot['high'].deg = self._get_high_nodes_degree(snapshot)

            # Add laplacian positional encodings
            #graph_high = Data(edge_index=snapshot['high', 'within', 'high'].edge_index, num_nodes=snapshot['high'].num_nodes)
            #graph_high = transform(graph_high)
            #snapshot['high'].laplacian_eigenvector_pe = graph_high.laplacian_eigenvector_pe

            return snapshot
    
class Dataset_Graph_subgraphs_t(Dataset_Graph_subgraphs):

    def _get_features(self, time_index: int):
        time_index_x = time_index + 24
        x_low = self.graph['low'].x[:,time_index_x,:]
        x_low = x_low.flatten(start_dim=1, end_dim=-1)
        return x_low


class Iterable_Graph(object):

    def __init__(self, dataset_graph, shuffle):
        self.dataset_graph = dataset_graph
        self.shuffle = shuffle
        if self.shuffle:
            self.sampling_vector = torch.randperm(len(self))
        else:
            self.sampling_vector = torch.arange(0, len(self))

    def __len__(self):
        return len(self.dataset_graph)
    
    def __next__(self):
        if self.t < len(self):
            self.idx = self.sampling_vector[self.t].item()
            self.t = self.t + 1
            return self.idx
        else:
            self.t = 0
            self.idx = 0
            raise StopIteration

    def __iter__(self):
        self.t = 0
        self.idx = 0
        return self

def custom_collate_fn_graph(batch_list):
    return Batch.from_data_list(batch_list)
