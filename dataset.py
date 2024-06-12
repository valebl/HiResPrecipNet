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
        return self.graph['low'].x.shape[1]
        
    def _check_temporal_consistency(self):
        if self.targets is not None:
            assert (self.graph['low'].x.shape[1]) == self.targets.shape[1], "Temporal dimension inconsistency."

    def _set_snapshot_count(self):
        self.snapshot_count = len(self)
    
    def _get_high_nodes_degree(self, snapshot):
        node_degree = (degree(snapshot['high','within','high'].edge_index[0], snapshot['high'].num_nodes) / 8).unsqueeze(-1)
        return node_degree

    def _get_features(self, time_index: int):
        x_low = self.graph['low'].x[:,time_index-24:time_index+1,:]
        # x_low = self.graph['low'].x[:,time_index-24:time_index+1:6,:] # var,time,level ?? nodes, time, var, level
        #x_low = self.graph['low'].x[:,time_index,:]
        #x_low = x_low.flatten(start_dim=1, end_dim=-1)
        x_low = x_low.flatten(start_dim=2, end_dim=-1)
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
       
        #snapshot['high'].mask_high = self.graph["high"].mask_high
        snapshot['high'].y = y
        snapshot['high'].train_mask = train_mask
        snapshot.num_nodes = self.graph.num_nodes
        snapshot['high'].num_nodes = self.graph['high'].num_nodes
        snapshot['low'].num_nodes = self.graph['low'].num_nodes
        snapshot.t = time_index
        
        snapshot['low', 'within', 'low'].edge_index = self.graph['low', 'within', 'low'].edge_index
        snapshot['high', 'within', 'high'].edge_index = self.graph['high', 'within', 'high'].edge_index

        snapshot['low', 'to', 'high'].edge_index = self.graph['low', 'to', 'high'].edge_index

#        snapshot['low_9x', 'within', 'low_9x'].edge_index = self.graph['low_9x', 'within', 'low_9x'].edge_index
#        snapshot['low_25x', 'within', 'low_25x'].edge_index = self.graph['low_25x', 'within', 'low_25x'].edge_index
#        snapshot['low', 'to', 'low_9x'].edge_index = self.graph['low', 'to', 'low_9x'].edge_index
#        snapshot['low_9x', 'to', 'low_25x'].edge_index = self.graph['low_9x', 'to', 'low_25x'].edge_index
#        snapshot['low_25x', 'to', 'high'].edge_index = self.graph['low_25x', 'to', 'high'].edge_index
#        snapshot['low_9x'].x = self.graph['low_9x'].x 
#        snapshot['low_25x'].x = self.graph['low_25x'].x 

        snapshot['low'].x = x_low 
        #snapshot['high'].x_empty = self.graph['high'].x
        # snapshot['high'].x = self.graph['high'].x #torch.zeros((snapshot['high'].num_nodes,1))
        snapshot['high'].x = torch.zeros((snapshot['high'].num_nodes,1))
        snapshot['high'].z_std = self.graph['high'].z_std

        snapshot['high'].lon = self.graph['high'].lon
        snapshot['high'].lat = self.graph['high'].lat
        snapshot['low'].lon = self.graph['low'].lon
        snapshot['low'].lat = self.graph['low'].lat

        #snapshot['high'].laplacian_eigenvector_pe = self.graph['high'].laplacian_eigenvector_pe     
        #snapshot['high'].deg = self._get_high_nodes_degree(snapshot)

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


class Dataset_Hierarchical_Graph(Dataset):

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
        #self._check_temporal_consistency()
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

    def _get_features(self, time_index: int, level):
        time_index_x = time_index + 24
        #x_low = self.graph['low'].x[:,time_index-24:time_index+1,:]
        x_low = self.graph['low'].x[:,time_index_x-24:time_index_x+1:6,:,level].squeeze()  # num_nodes, time, vars, levels
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

        y = self._get_target(time_index) if self.targets is not None else None
        train_mask = self._get_train_mask(y) if y is not None else None

        additional_features = self._get_additional_features(time_index)

        snapshot = HeteroData()
       
        snapshot['low_200'].num_nodes = self.graph['low'].num_nodes
        snapshot['low_500'].num_nodes = self.graph['low'].num_nodes
        snapshot['low_700'].num_nodes = self.graph['low'].num_nodes
        snapshot['low_850'].num_nodes = self.graph['low'].num_nodes
        snapshot['low_1000'].num_nodes = self.graph['low'].num_nodes

        # Horizontal bidirectional edges        
        snapshot['low_200', 'horizontal', 'low_200'].edge_index = self.graph['low', 'horizontal', 'low'].edge_index.clone()
        snapshot['low_500', 'horizontal', 'low_500'].edge_index = self.graph['low', 'horizontal', 'low'].edge_index.clone()
        snapshot['low_700', 'horizontal', 'low_700'].edge_index =self.graph['low', 'horizontal', 'low'].edge_index.clone()
        snapshot['low_850', 'horizontal', 'low_850'].edge_index = self.graph['low', 'horizontal', 'low'].edge_index.clone()
        snapshot['low_1000', 'horizontal', 'low_1000'].edge_index = self.graph['low', 'horizontal', 'low'].edge_index.clone()

        # # Vertical bidirectional edges
        # snapshot['low_200', 'vertical', 'low_500'].edge_index = self.graph['low', 'vertical', 'low'].edge_index.clone()
        # snapshot['low_500', 'vertical', 'low_700'].edge_index =  self.graph['low', 'vertical', 'low'].edge_index.clone()
        # snapshot['low_700', 'vertical', 'low_850'].edge_index =  self.graph['low', 'vertical', 'low'].edge_index.clone()
        # snapshot['low_850', 'vertical', 'low_1000'].edge_index =  self.graph['low', 'vertical', 'low'].edge_index.clone()

        # Vertical unidirectional edges (top-to-down)
        snapshot['low_200', 'to', 'low_500'].edge_index = self.graph['low', 'to', 'low'].edge_index.clone()
        snapshot['low_500', 'to', 'low_700'].edge_index =  self.graph['low', 'to', 'low'].edge_index.clone()
        snapshot['low_700', 'to', 'low_850'].edge_index =  self.graph['low', 'to', 'low'].edge_index.clone()
        snapshot['low_850', 'to', 'low_1000'].edge_index =  self.graph['low', 'to', 'low'].edge_index.clone()

        snapshot['high', 'within', 'high'].edge_index = self.graph['high', 'within', 'high'].edge_index
        snapshot['low_1000', 'to', 'high'].edge_index = self.graph['low', 'to', 'high'].edge_index

        snapshot['low_200'].x = self._get_features(time_index, level=0)
        snapshot['low_500'].x = self._get_features(time_index, level=1)
        snapshot['low_700'].x = self._get_features(time_index, level=2)
        snapshot['low_850'].x = self._get_features(time_index, level=3)
        snapshot['low_1000'].x = self._get_features(time_index, level=4)

        for key, value in additional_features.items():
            if value.shape[0] == self.graph['high'].x.shape[0]:
                snapshot['high'][key] = value
                # elif value.shape[0] == self.graph['low'].x.shape[0]:
                #     snapshot['high'][key] = value

        snapshot.t = time_index
        snapshot['high'].y = y
        snapshot['high'].train_mask = train_mask
        snapshot['high'].num_nodes = self.graph['high'].num_nodes
        snapshot['high'].x = torch.zeros((snapshot['high'].num_nodes,1))
        snapshot['high'].z_std = self.graph['high'].z_std

        snapshot.num_nodes = snapshot['high'].num_nodes + snapshot['low'].num_noodes * 5

        #snapshot['high'].laplacian_eigenvector_pe = self.graph['high'].laplacian_eigenvector_pe     
        #snapshot['high'].deg = self._get_high_nodes_degree(snapshot)

        return snapshot
    
    
class Dataset_Graph_CNN_GNN(Dataset):

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
        return self.graph['low'].x.shape[-1]
        
    def _check_temporal_consistency(self):
        if self.targets is not None:
            assert (self.graph['low'].x.shape[-1]) == self.targets.shape[1], "Temporal dimension inconsistency."

    def _set_snapshot_count(self):
        self.snapshot_count = len(self)
    
    def _get_high_nodes_degree(self, snapshot):
        node_degree = (degree(snapshot['high','within','high'].edge_index[0], snapshot['high'].num_nodes) / 8).unsqueeze(-1)
        return node_degree

    def _get_features(self, time_index: int):
        time_index_x = time_index
        #x_low = self.graph['low'].x[:,time_index-24:time_index+1,:]
        x_low = self.graph['low'].x[:,:,:,time_index_x-24:time_index_x+1:6] # low_num_nodes,vars=5,lev=5,time=5
        #x_low = x_low.flatten(start_dim=1, end_dim=-1)
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

        ######
        # snapshot['low_upscaled'].num_nodes = self.graph['low_upscaled'].lon.shape[0]
        snapshot.t = time_index
        
        snapshot['low', 'within', 'low'].edge_index = self.graph['low', 'within', 'low'].edge_index
        # snapshot['low_upscaled', 'within', 'low_upscaled'].edge_index = self.graph['low_upscaled', 'within', 'low_upscaled'].edge_index
        # snapshot['high', 'within', 'high'].edge_index = self.graph['high', 'within', 'high'].edge_index
        # snapshot['low', 'to', 'low_upscaled'].edge_index = self.graph['low', 'to', 'low_upscaled'].edge_index
        # snapshot['low_upscaled', 'to', 'high'].edge_index = self.graph['low_upscaled', 'to', 'high'].edge_index

        # snapshot['low'].x = x_low 
        # snapshot['low_upscaled'].x = self.graph['low_upscaled'].x
        # snapshot['high'].x = self.graph['high'].x
        # snapshot['high'].z_std = self.graph['high'].z_std

        # snapshot['high'].lon = self.graph['high'].lon
        # snapshot['high'].lat = self.graph['high'].lat
        # snapshot['low'].lon = self.graph['low'].lon
        # snapshot['low'].lat = self.graph['low'].lat
        ######

        snapshot['high', 'within', 'high'].edge_index = self.graph['high', 'within', 'high'].edge_index

        snapshot['low', 'to', 'high'].edge_index = self.graph['low', 'to', 'high'].edge_index

        # snapshot['low_9x', 'within', 'low_9x'].edge_index = self.graph['low_9x', 'within', 'low_9x'].edge_index
        # snapshot['low_25x', 'within', 'low_25x'].edge_index = self.graph['low_25x', 'within', 'low_25x'].edge_index
        # snapshot['low', 'to', 'low_9x'].edge_index = self.graph['low', 'to', 'low_9x'].edge_index
        # snapshot['low_9x', 'to', 'low_25x'].edge_index = self.graph['low_9x', 'to', 'low_25x'].edge_index
        snapshot['low_25x', 'to', 'high'].edge_index = self.graph['low_25x', 'to', 'high'].edge_index
        # snapshot['low_9x'].x = self.graph['low_9x'].x 
        snapshot['low_25x'].x = self.graph['low_25x'].x 

        snapshot['low'].x = x_low 
        #snapshot['high'].x_empty = self.graph['high'].x
        snapshot['high'].x = self.graph['high'].x #torch.zeros((snapshot['high'].num_nodes,1))
        snapshot['high'].z_std = self.graph['high'].z_std

        snapshot['high'].lon = self.graph['high'].lon
        snapshot['high'].lat = self.graph['high'].lat
        snapshot['low'].lon = self.graph['low'].lon
        snapshot['low'].lat = self.graph['low'].lat

        #snapshot['high'].laplacian_eigenvector_pe = self.graph['high'].laplacian_eigenvector_pe     
        #snapshot['high'].deg = self._get_high_nodes_degree(snapshot)

        return snapshot
    

class Dataset_Graph_CNN_GNN_new(Dataset):

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
        return self.graph['low'].x.shape[2]
        
    def _check_temporal_consistency(self):
        if self.targets is not None:
            assert (self.graph['low'].x.shape[2]) == self.targets.shape[1], f"Temporal dimension inconsistency, input has shape {self.graph['low'].x.shape}, target has shape {self.targets.shape}."

    def _set_snapshot_count(self):
        self.snapshot_count = len(self)
    
    def _get_high_nodes_degree(self, snapshot):
        node_degree = (degree(snapshot['high','within','high'].edge_index[0], snapshot['high'].num_nodes) / 8).unsqueeze(-1)
        return node_degree

    def _get_features(self, time_index: int):
        x_low = self.graph['low'].x[:,:,time_index-24:time_index+1,:] # nodes,var,time,level
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

        snapshot['high', 'within', 'high'].edge_index = self.graph['high', 'within', 'high'].edge_index
        snapshot['low', 'to', 'high'].edge_index = self.graph['low', 'to', 'high'].edge_index

        snapshot['low'].x = x_low 
        snapshot['high'].x = self.graph['high'].x
        snapshot['high'].z_std = self.graph['high'].z_std

        snapshot['high'].lon = self.graph['high'].lon
        snapshot['high'].lat = self.graph['high'].lat
        snapshot['low'].lon = self.graph['low'].lon
        snapshot['low'].lat = self.graph['low'].lat

        snapshot['low', 'to', 'low_9x'].edge_index = self.graph['low', 'to', 'low_9x'].edge_index
        snapshot['low_9x', 'to', 'high'].edge_index = self.graph['low_9x', 'to', 'high'].edge_index
        snapshot['low_9x'].x = self.graph['low_9x'].x

        return snapshot
    

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

class Dataset_StaticGraphTemporalSignal(Dataset):

    r"""A data iterator object to contain a static graph with a dynamically
    changing constant time difference temporal feature set (multiple signals).
    The node labels (target) are also temporal. The iterator returns a single
    constant time difference temporal snapshot for a time period (e.g. day or week).
    This single temporal snapshot is a Pytorch Geometric Data object. Between two
    temporal snapshots the features and optionally passed attributes might change.
    However, the underlying graph is the same.

    Args:
        edge_index (Numpy array): Index tensor of edges.
        edge_weight (Numpy array): Edge weight tensor.
        features (Sequence of Numpy arrays): Sequence of node feature tensors.
        targets (Sequence of Numpy arrays): Sequence of node label (target) tensors.
        **kwargs (optional Sequence of Numpy arrays): Sequence of additional attributes.
    """


    def __init__(
        self,
        graph: Graph,
        targets: Targets,
        **kwargs: Additional_Features
    ):
        self.graph = graph
        # self.graph['low'].x = torch.permute(self.graph['low'].x, (3,4,0,1,2)).flatten(end_dim=1).flatten(start_dim=2, end_dim=3) # (720, 140256 25)
        self.targets = targets
        self.additional_feature_keys = []
        for key, value in kwargs.items():
            setattr(self, key, value)
            self.additional_feature_keys.append(key)
        self._check_temporal_consistency()
        #self._add_node_degree()
        self._set_snapshot_count()

    def __len__(self):
        #return len(self.features)
        return self.graph['low'].x.shape[1]
        
    def _check_temporal_consistency(self):
        if self.targets is not None:
            assert (self.graph['low'].x.shape[1]) == self.targets.shape[1], "Temporal dimension inconsistency."

    def _set_snapshot_count(self):
        self.snapshot_count = len(self)
    
    def _get_high_nodes_degree(self, snapshot):
        node_degree = (degree(snapshot['high','within','high'].edge_index[0], snapshot['high'].num_nodes) / 8).unsqueeze(-1)
        return node_degree

    def _get_features(self, time_index: int):
        x_low = self.graph['low'].x[:,time_index-24:time_index+1,:] # num_nodes, time, var*level
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
        snapshot['low', 'to', 'high'].edge_index = self.graph['low', 'to', 'high'].edge_index
        snapshot['high', 'within', 'high'].edge_index = self.graph['high', 'within', 'high'].edge_index
        
        snapshot['low'].x = x_low 
        snapshot['high'].x = self.graph['high'].z_std
        snapshot['high'].z_std = self.graph['high'].z_std

        snapshot['high'].lon = self.graph['high'].lon
        snapshot['high'].lat = self.graph['high'].lat
        snapshot['low'].lon = self.graph['low'].lon
        snapshot['low'].lat = self.graph['low'].lat

        # snapshot['low_25x', 'to', 'high'].edge_index = self.graph['low_25x', 'to', 'high'].edge_index
        # snapshot['low_25x'].x = self.graph['low_25x'].x 

        return snapshot


class Iterable_Graph(object):

    def __init__(self, dataset_graph, shuffle):
        self.dataset_graph = dataset_graph
        self.shuffle = shuffle
        if self.shuffle:
            self.sampling_vector = torch.randperm(len(self)-24) + 24 # from 24 to len
        else:
            self.sampling_vector = torch.arange(24, len(self))

    def __len__(self):
        return len(self.dataset_graph)
    
    def __next__(self):
        if self.t < len(self)-24:
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
        if self.shuffle:
            self.sampling_vector = torch.randperm(len(self)-24) + 24 # from 24 to len
        return self

def custom_collate_fn_graph(batch_list):
    return Batch.from_data_list(batch_list)



#### SUB-PIXEL CONVOLUTION for super-resolution ####

class Dataset_Graph_subpixel(Dataset):

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
        return self.graph['low'].x.shape[0]
        
    def _check_temporal_consistency(self):
        if self.targets is not None:
            assert (self.graph['low'].x.shape[0]) == self.targets.shape[1], "Temporal dimension inconsistency."

    def _set_snapshot_count(self):
        self.snapshot_count = len(self)
    
    def _get_high_nodes_degree(self, snapshot):
        node_degree = (degree(snapshot['high','within','high'].edge_index[0], snapshot['high'].num_nodes) / 8).unsqueeze(-1)
        return node_degree

    def _get_features(self, time_index: int):
        time_index_x = time_index
        x_low = self.graph['low'].x[time_index_x-24:time_index_x+1:6, :, :, :, :] # T, V, L, LAT, LON
        x_low = x_low.flatten(start_dim=0, end_dim=1)
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
        snapshot['low_9x'].num_nodes = self.graph['low_9x'].num_nodes
        
        snapshot['high', 'within', 'high'].edge_index = self.graph['high', 'within', 'high'].edge_index
        snapshot['low', 'to', 'low_9x'].edge_index = self.graph['low', 'to', 'low_9x'].edge_index
        snapshot['low_9x', 'to', 'high'].edge_index = self.graph['low_9x', 'to', 'high'].edge_index

        snapshot['low'].x = x_low 
        snapshot['high'].x = self.graph['high'].x #torch.zeros((snapshot['high'].num_nodes,1))
        snapshot['high'].z_std = self.graph['high'].z_std

        snapshot['high'].lon = self.graph['high'].lon
        snapshot['high'].lat = self.graph['high'].lat
        snapshot['low'].lon = self.graph['low'].lon
        snapshot['low'].lat = self.graph['low'].lat

        snapshot.t = time_index

        return snapshot

def custom_collate_fn_graph_subpixel(batch_list):
    return batch_list
