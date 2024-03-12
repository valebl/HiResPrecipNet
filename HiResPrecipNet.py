import torch.nn as nn
import torch_geometric.nn as geometric_nn
from torch_geometric.nn import GATv2Conv, HeteroConv
from torch_geometric.nn import global_mean_pool
import numpy as np
import torch

class HiResPrecipNet(nn.Module):
    
    def __init__(self, low_in=5*5*5, high_in=1, low2high_out=64, high_out=64):
        super(HiResPrecipNet, self).__init__()

        self.downscaler = GATv2Conv((low_in, high_in), out_channels=low2high_out, dropout=0.0, heads=1, aggr='mean', add_self_loops=False, bias=True)
        
        self.processor = geometric_nn.Sequential('x, edge_index', [
            (geometric_nn.BatchNorm(low2high_out+1), 'x -> x'),
            (GATv2Conv(in_channels=low2high_out+1, out_channels=high_out, heads=2, dropout=0.2, aggr='mean', add_self_loops=True, bias=True), 'x, edge_index -> x'),
            (geometric_nn.BatchNorm(high_out*2), 'x -> x'), 
            nn.ReLU(),
            (GATv2Conv(in_channels=high_out*2, out_channels=high_out, heads=2, dropout=0.2, aggr='mean', add_self_loops=True, bias=True),'x, edge_index -> x'),
            (geometric_nn.BatchNorm(high_out*2), 'x -> x'),
            nn.ReLU(),
            (GATv2Conv(in_channels=high_out*2, out_channels=high_out, heads=2, dropout=0.2, aggr='mean', add_self_loops=True, bias=True),'x, edge_index -> x'),
            (geometric_nn.BatchNorm(high_out*2), 'x -> x'),
            nn.ReLU(),
            (GATv2Conv(in_channels=high_out*2, out_channels=high_out, heads=2, dropout=0.2, aggr='mean', add_self_loops=True, bias=True),'x, edge_index -> x'),
            (geometric_nn.BatchNorm(high_out*2), 'x -> x'),
            nn.ReLU(),
            (GATv2Conv(in_channels=high_out*2, out_channels=high_out, heads=1, dropout=0.0, aggr='mean', add_self_loops=True, bias=True), 'x, edge_index -> x'),
            nn.ReLU(),
            ])
    
        self.predictor = nn.Sequential(
            nn.Linear(high_out, high_out),
            nn.ReLU(),
            nn.Linear(high_out, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
            )

    def forward(self, data):        
        encod_low2high  = self.downscaler((data.x_dict['low'], data['high'].x), data.edge_index_dict[('low','to','high')])
        encod_low2high  = torch.concatenate((data['high'].z_std, encod_low2high ),dim=-1)
        encod_high = self.processor(encod_low2high , data.edge_index_dict[('high','within','high')])
        y_pred = self.predictor(encod_high)
        return y_pred
    

class Hierarchical_HiResPrecipNet(nn.Module):
    
    def __init__(self, low_in=5*5, high_in=1, low2high_out=64, high_out=64):
        super(Hierarchical_HiResPrecipNet, self).__init__()

        self.horizontal = geometric_nn.Sequential('x, edge_index', [
            (GATv2Conv(in_channels=5*5, out_channels=5*5, heads=1, dropout=0.2, aggr='mean', add_self_loops=False, bias=True), 'x, edge_index -> x'),
            nn.ReLU(),
            (GATv2Conv(in_channels=5*5, out_channels=5*5, heads=1, dropout=0.2, aggr='mean', add_self_loops=False, bias=True), 'x, edge_index -> x'),
            nn.ReLU(),
            (GATv2Conv(in_channels=5*5, out_channels=5*5, heads=1, dropout=0.2, aggr='mean', add_self_loops=False, bias=True), 'x, edge_index -> x'),
            nn.ReLU(),
            ])

        self.down_vertical = geometric_nn.Sequential('x, edge_index', [
            (GATv2Conv(in_channels=(5*5,5*5), out_channels=5*5, heads=1, dropout=0.2, aggr='mean', add_self_loops=False, bias=True), 'x, edge_index -> x'),
            nn.ReLU(),
            ])

        self.downscaler = GATv2Conv((low_in, high_in), out_channels=low2high_out, dropout=0.0, heads=1, aggr='mean', add_self_loops=False, bias=True)
        
        self.processor = geometric_nn.Sequential('x, edge_index', [
            (GATv2Conv(in_channels=low2high_out+1, out_channels=high_out, heads=2, dropout=0.2, aggr='mean', add_self_loops=True, bias=True), 'x, edge_index -> x'),
            (geometric_nn.BatchNorm(high_out*2), 'x -> x'), 
            nn.ReLU(),
            (GATv2Conv(in_channels=high_out*2, out_channels=high_out, heads=2, dropout=0.2, aggr='mean', add_self_loops=True, bias=True),'x, edge_index -> x'),
            (geometric_nn.BatchNorm(high_out*2), 'x -> x'),
            nn.ReLU(),
            (GATv2Conv(in_channels=high_out*2, out_channels=high_out, heads=1, dropout=0.2, aggr='mean', add_self_loops=True, bias=True),'x, edge_index -> x'),
            (geometric_nn.BatchNorm(high_out), 'x -> x'),
            nn.ReLU(),
            ])
    
        self.predictor = nn.Sequential(
            nn.Linear(high_out, high_out),
            nn.ReLU(),
            nn.Linear(high_out, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
            )

    def forward(self, data):   

        # Horizontal among the same pressure level
        data['low_200'].x = self.horizontal(data['low_200'].x, data.edge_index_dict['low_200', 'horizontal', 'low_200'])
        data['low_500'].x = self.horizontal(data['low_500'].x, data.edge_index_dict['low_500', 'horizontal', 'low_500'])
        data['low_700'].x = self.horizontal(data['low_700'].x, data.edge_index_dict['low_700', 'horizontal', 'low_700'])
        data['low_850'].x = self.horizontal(data['low_850'].x, data.edge_index_dict['low_850', 'horizontal', 'low_850'])
        data['low_1000'].x = self.horizontal(data['low_1000'].x, data.edge_index_dict['low_1000', 'horizontal', 'low_1000'])

        # Last vertical pass only top-down
        data['low_500'].x = self.down_vertical((data['low_200'].x, data['low_500'].x), data.edge_index_dict['low_200', 'to', 'low_500'])
        data['low_700'].x = self.down_vertical((data['low_500'].x, data['low_700'].x), data.edge_index_dict['low_500', 'to', 'low_700'])
        data['low_850'].x = self.down_vertical((data['low_700'].x, data['low_850'].x), data.edge_index_dict['low_700', 'to', 'low_850'])
        data['low_1000'].x = self.down_vertical((data['low_850'].x, data['low_1000'].x), data.edge_index_dict['low_850', 'to', 'low_1000'])

        encod_low2high  = self.downscaler((data['low_1000'].x, data['high'].x), data.edge_index_dict[('low_1000','to','high')])
        encod_low2high  = torch.concatenate((data['high'].z_std, encod_low2high ),dim=-1)
        encod_high = self.processor(encod_low2high , data.edge_index_dict[('high','within','high')])
        y_pred = self.predictor(encod_high)
        return y_pred


#------------------------#
#----- CNN-GNN mdel -----#
#------------------------#
    
class HiResPrecipNet_CNN_GNN(nn.Module):
    
    def __init__(self, low_in=5*5*5, high_in=1, low2high_out=64, high_out=64):
        super(HiResPrecipNet_CNN_GNN, self).__init__()

        self.node_encoder_cnn = nn.Sequential(
            nn.Conv2d(in_channels=5, out_channels=5, kernel_size=3, padding=1, groups=25),      # (N,Cin,H,W) = (low_num_nodes,5,5,5)
            nn.BatchNorm2d(5),
            nn.ReLu(),
            nn.Conv2d(in_channels=5, out_channels=5, kernel_size=3, padding=1, groups=25),
            nn.BatchNorm2d(5),
            nn.ReLu(),
            nn.Conv2d(in_channels=5, out_channels=5, kernel_size=3, padding=1, groups=25),
            nn.BatchNorm2d(5),
            nn.ReLu(),
            nn.MaxPool2d(kernel_size=2, padding=1, stride=2),                                   # (low_num_nodes,5,3,3)
            nn.Flatten()                                                                        # (low_num_nodes,45)
        )

        self.processor_low = geometric_nn.Sequential('x, edge_index', [
            (GATv2Conv(in_channels=45, out_channels=45, heads=1, dropout=0.2, aggr='mean', add_self_loops=False, bias=True), 'x, edge_index -> x'),
            nn.ReLU(),
            (GATv2Conv(in_channels=45, out_channels=45, heads=1, dropout=0.2, aggr='mean', add_self_loops=False, bias=True), 'x, edge_index -> x'),
            nn.ReLU(),
            (GATv2Conv(in_channels=45, out_channels=45, heads=1, dropout=0.2, aggr='mean', add_self_loops=False, bias=True), 'x, edge_index -> x'),
            nn.ReLU(),
            ])

        self.node_upscaler_cnn = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=5, kernel_size=3, padding=1),                  # (N,Cin,L) = (low_num_nodes,1,45)
            nn.BatchNorm1d(5),
            nn.ReLu(),
            nn.Conv2d(in_channels=5, out_channels=9, kernel_size=3, padding=1),
            nn.BatchNorm1d(9),
            nn.ReLu(),
            nn.Conv1d(in_channels=9, out_channels=9, kernel_size=3, padding=1),
            nn.BatchNorm1d(9),
            nn.ReLu(),
            nn.MaxPool1d(kernel_size=2, padding=2, stride=2),                                   # (low_num_nodes,9,22)
        )

        self.downscaler = GATv2Conv((low_in, high_in), out_channels=low2high_out, dropout=0.0, heads=1, aggr='mean', add_self_loops=False, bias=True)
        
        self.processor = geometric_nn.Sequential('x, edge_index', [
            (geometric_nn.BatchNorm(low2high_out+1), 'x -> x'),
            (GATv2Conv(in_channels=low2high_out+1, out_channels=high_out, heads=2, dropout=0.2, aggr='mean', add_self_loops=True, bias=True), 'x, edge_index -> x'),
            (geometric_nn.BatchNorm(high_out*2), 'x -> x'), 
            nn.ReLU(),
            (GATv2Conv(in_channels=high_out*2, out_channels=high_out, heads=2, dropout=0.2, aggr='mean', add_self_loops=True, bias=True),'x, edge_index -> x'),
            (geometric_nn.BatchNorm(high_out*2), 'x -> x'),
            nn.ReLU(),
            (GATv2Conv(in_channels=high_out*2, out_channels=high_out, heads=2, dropout=0.2, aggr='mean', add_self_loops=True, bias=True),'x, edge_index -> x'),
            (geometric_nn.BatchNorm(high_out*2), 'x -> x'),
            nn.ReLU(),
            (GATv2Conv(in_channels=high_out*2, out_channels=high_out, heads=2, dropout=0.2, aggr='mean', add_self_loops=True, bias=True),'x, edge_index -> x'),
            (geometric_nn.BatchNorm(high_out*2), 'x -> x'),
            nn.ReLU(),
            (GATv2Conv(in_channels=high_out*2, out_channels=high_out, heads=1, dropout=0.0, aggr='mean', add_self_loops=True, bias=True), 'x, edge_index -> x'),
            nn.ReLU(),
            ])
    
        self.predictor = nn.Sequential(
            nn.Linear(high_out, high_out),
            nn.ReLU(),
            nn.Linear(high_out, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
            )

    def forward(self, data):        
        encod_low2high  = self.downscaler((data.x_dict['low'], data['high'].x), data.edge_index_dict[('low','to','high')])
        encod_low2high  = torch.concatenate((data['high'].z_std, encod_low2high ),dim=-1)
        encod_high = self.processor(encod_low2high , data.edge_index_dict[('high','within','high')])
        y_pred = self.predictor(encod_high)
        return y_pred
