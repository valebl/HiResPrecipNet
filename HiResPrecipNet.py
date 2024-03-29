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
    
    def __init__(self, high_in=1, low_up_in=23, low_up_hidden=32, high_hidden=32, upscaled_dim=25,
                 kernel=3, nvars=5, ntimes=5, nlevs=5, high_attr_dim=1):
        super(HiResPrecipNet_CNN_GNN, self).__init__()

        self.nvars = nvars
        self.ntimes = ntimes
        self.nlevs = nlevs

        self.node_encoder_cnn = nn.Sequential(
            nn.Conv2d(in_channels=nvars, out_channels=nvars, kernel_size=kernel, padding=1, groups=nvars),      # (N,Cin,H,W) = (low_num_nodes,vars=5,lev=5,time=5)
            nn.BatchNorm2d(nvars),
            nn.ReLU(),
            nn.Conv2d(in_channels=nvars, out_channels=nvars, kernel_size=kernel, padding=1, groups=nvars),
            nn.BatchNorm2d(nvars),
            nn.ReLU(),
            nn.Conv2d(in_channels=nvars, out_channels=nvars, kernel_size=kernel, padding=1, groups=nvars),
            nn.BatchNorm2d(nvars),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), padding=1, stride=2),                                   # (low_num_nodes,5,3,3)
            nn.Flatten()                                                                                  # (low_num_nodes,45)
        )

        # self.processor_low = geometric_nn.Sequential('x, edge_index', [
        #     (GATv2Conv(in_channels=node_encod_dim, out_channels=node_encod_dim, heads=1, dropout=0.2, aggr='mean', add_self_loops=False, bias=True), 'x, edge_index -> x'),
        #     nn.ReLU(),
        #     (GATv2Conv(in_channels=node_encod_dim, out_channels=node_encod_dim, heads=1, dropout=0.2, aggr='mean', add_self_loops=False, bias=True), 'x, edge_index -> x'),
        #     nn.ReLU(),
        #     (GATv2Conv(in_channels=node_encod_dim, out_channels=node_encod_dim, heads=1, dropout=0.2, aggr='mean', add_self_loops=False, bias=True), 'x, edge_index -> x'),
        #     nn.ReLU(),
        #     ])

        self.node_upscaler_cnn = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=5, kernel_size=3, padding=1),                  # (N,Cin,L) = (low_num_nodes,1,45)
            nn.BatchNorm1d(5),
            nn.ReLU(),
            nn.Conv1d(in_channels=5, out_channels=upscaled_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(upscaled_dim),
            nn.ReLU(),
            nn.Conv1d(in_channels=upscaled_dim, out_channels=upscaled_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(upscaled_dim),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, padding=1, stride=2),                                   # (low_num_nodes,9,22)
        )

        self.processor_upscaled = geometric_nn.Sequential('x, edge_index', [
            (GATv2Conv(in_channels=low_up_in, out_channels=low_up_hidden, heads=2, dropout=0.2, aggr='mean', add_self_loops=False, bias=True), 'x, edge_index -> x'),
            (geometric_nn.BatchNorm(low_up_hidden*2), 'x -> x'), 
            nn.ReLU(),
            (GATv2Conv(in_channels=low_up_hidden*2, out_channels=low_up_hidden, heads=2, dropout=0.2, aggr='mean', add_self_loops=False, bias=True),'x, edge_index -> x'),
            (geometric_nn.BatchNorm(low_up_hidden*2), 'x -> x'),
            nn.ReLU(),
            (GATv2Conv(in_channels=low_up_hidden*2, out_channels=low_up_hidden, heads=2, dropout=0.2, aggr='mean', add_self_loops=False, bias=True),'x, edge_index -> x'),
            (geometric_nn.BatchNorm(low_up_hidden*2), 'x -> x'),
            nn.ReLU(),
            (GATv2Conv(in_channels=low_up_hidden*2, out_channels=low_up_hidden, heads=2, dropout=0.2, aggr='mean', add_self_loops=False, bias=True),'x, edge_index -> x'),
            (geometric_nn.BatchNorm(low_up_hidden*2), 'x -> x'),
            nn.ReLU(),
            (GATv2Conv(in_channels=low_up_hidden*2, out_channels=low_up_hidden, heads=1, dropout=0.0, aggr='mean', add_self_loops=False, bias=True), 'x, edge_index -> x'),
            nn.ReLU()
            ])

        self.downscaler_upscaled2high = GATv2Conv((low_up_hidden, high_in), out_channels=high_hidden, dropout=0.0, heads=1, aggr='mean', add_self_loops=False, bias=True)
        
        self.processor_high = geometric_nn.Sequential('x, edge_index', [
            (geometric_nn.BatchNorm(high_hidden+high_attr_dim), 'x -> x'),
            (GATv2Conv(in_channels=high_hidden+high_attr_dim, out_channels=high_hidden, heads=2, dropout=0.2, aggr='mean', add_self_loops=True, bias=True), 'x, edge_index -> x'),
            (geometric_nn.BatchNorm(high_hidden*2), 'x -> x'), 
            nn.ReLU(),
            (GATv2Conv(in_channels=high_hidden*2, out_channels=high_hidden, heads=2, dropout=0.2, aggr='mean', add_self_loops=True, bias=True),'x, edge_index -> x'),
            (geometric_nn.BatchNorm(high_hidden*2), 'x -> x'),
            nn.ReLU(),
            (GATv2Conv(in_channels=high_hidden*2, out_channels=high_hidden, heads=2, dropout=0.2, aggr='mean', add_self_loops=True, bias=True),'x, edge_index -> x'),
            (geometric_nn.BatchNorm(high_hidden*2), 'x -> x'),
            nn.ReLU(),
            (GATv2Conv(in_channels=high_hidden*2, out_channels=high_hidden, heads=2, dropout=0.2, aggr='mean', add_self_loops=True, bias=True),'x, edge_index -> x'),
            (geometric_nn.BatchNorm(high_hidden*2), 'x -> x'),
            nn.ReLU(),
            (GATv2Conv(in_channels=high_hidden*2, out_channels=high_hidden, heads=1, dropout=0.0, aggr='mean', add_self_loops=True, bias=True), 'x, edge_index -> x'),
            nn.ReLU()
            ])
    
        self.predictor = nn.Sequential(
            nn.Linear(high_hidden, high_hidden),
            nn.ReLU(),
            nn.Linear(high_hidden, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
            )

    def forward(self, data): 
        encod_cnn = self.node_encoder_cnn(data.x_dict['low'])
        #encod_low = self.processor_low(encod_cnn, data.edge_index_dict[('low','within','low')]).squeeze()
        encod_low_upscaled = self.node_upscaler_cnn(encod_cnn.unsqueeze(1)).flatten(end_dim=1)
        encod_low_upscaled = self.processor_upscaled(encod_low_upscaled, data.edge_index_dict[('low_upscaled','within','low_upscaled')]).squeeze()
        encod_upscaled2high  = self.downscaler_upscaled2high((encod_low_upscaled, data['high'].z_std), data.edge_index_dict[('low_upscaled','to','high')])
        # encod_low2high  = self.downscaler((encod_low, data['high'].x), data.edge_index_dict[('low','to','high')])
        encod_upscaled2high  = torch.concatenate((data['high'].z_std, encod_upscaled2high ),dim=-1)
        encod_high = self.processor_high(encod_upscaled2high, data.edge_index_dict[('high','within','high')])
        y_pred = self.predictor(encod_high)
        return y_pred
    

class HiResPrecipNet_up_CNN_GNN(nn.Module):
    
    def __init__(self, up_in=1, high_in=1, low_hidden=64, low_up_hidden=32, high_hidden=32,
                 kernel=3, nvars=5, ntimes=5, nlevs=5, node_encod_dim=45, high_attr_dim=1):
        super(HiResPrecipNet_up_CNN_GNN, self).__init__()

        self.nvars = nvars
        self.ntimes = ntimes
        self.nlevs = nlevs

        self.node_encoder_cnn = nn.Sequential(
            nn.Conv2d(in_channels=nvars, out_channels=nvars, kernel_size=kernel, padding=1, groups=nvars),      # (N,Cin,H,W) = (low_num_nodes,vars=5,lev=5,time=5)
            nn.BatchNorm2d(nvars),
            nn.ReLU(),
            nn.Conv2d(in_channels=nvars, out_channels=nvars, kernel_size=kernel, padding=1, groups=nvars),
            nn.BatchNorm2d(nvars),
            nn.ReLU(),
            nn.Conv2d(in_channels=nvars, out_channels=nvars, kernel_size=kernel, padding=1, groups=nvars),
            nn.BatchNorm2d(nvars),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, padding=1, stride=2),                                   # (low_num_nodes,5,3,3)
            nn.Flatten()                                                                        # (low_num_nodes,45)
        )

        self.processor_low = geometric_nn.Sequential('x, edge_index', [
            (GATv2Conv(in_channels=node_encod_dim, out_channels=low_hidden, heads=2, dropout=0.2, aggr='mean', add_self_loops=False, bias=True), 'x, edge_index -> x'),
            (geometric_nn.BatchNorm(low_hidden*2), 'x -> x'), 
            nn.ReLU(),
            (GATv2Conv(in_channels=low_hidden*2, out_channels=low_hidden, heads=2, dropout=0.2, aggr='mean', add_self_loops=False, bias=True),'x, edge_index -> x'),
            (geometric_nn.BatchNorm(low_hidden*2), 'x -> x'),
            nn.ReLU(),
            (GATv2Conv(in_channels=low_hidden*2, out_channels=low_hidden, heads=2, dropout=0.2, aggr='mean', add_self_loops=False, bias=True),'x, edge_index -> x'),
            (geometric_nn.BatchNorm(low_hidden*2), 'x -> x'),
            nn.ReLU(),
            (GATv2Conv(in_channels=low_hidden*2, out_channels=low_hidden, heads=2, dropout=0.2, aggr='mean', add_self_loops=False, bias=True),'x, edge_index -> x'),
            (geometric_nn.BatchNorm(low_hidden*2), 'x -> x'),
            nn.ReLU(),
            (GATv2Conv(in_channels=low_hidden*2, out_channels=low_hidden, heads=1, dropout=0.0, aggr='mean', add_self_loops=False, bias=True), 'x, edge_index -> x'),
            nn.ReLU()
            ])
        
        self.downscaler_low2upscaled = GATv2Conv((low_hidden, up_in), out_channels=low_up_hidden, dropout=0.2, heads=1, aggr='mean', add_self_loops=False, bias=True)

        self.processor_upscaled = geometric_nn.Sequential('x, edge_index', [
            (GATv2Conv(in_channels=low_up_hidden, out_channels=low_up_hidden, heads=2, dropout=0.2, aggr='mean', add_self_loops=False, bias=True), 'x, edge_index -> x'),
            (geometric_nn.BatchNorm(low_up_hidden*2), 'x -> x'), 
            nn.ReLU(),
            (GATv2Conv(in_channels=low_up_hidden*2, out_channels=low_up_hidden, heads=2, dropout=0.2, aggr='mean', add_self_loops=False, bias=True),'x, edge_index -> x'),
            (geometric_nn.BatchNorm(low_up_hidden*2), 'x -> x'),
            nn.ReLU(),
            (GATv2Conv(in_channels=low_up_hidden*2, out_channels=low_up_hidden, heads=2, dropout=0.2, aggr='mean', add_self_loops=False, bias=True),'x, edge_index -> x'),
            (geometric_nn.BatchNorm(low_up_hidden*2), 'x -> x'),
            nn.ReLU(),
            (GATv2Conv(in_channels=low_up_hidden*2, out_channels=low_up_hidden, heads=2, dropout=0.2, aggr='mean', add_self_loops=False, bias=True),'x, edge_index -> x'),
            (geometric_nn.BatchNorm(low_up_hidden*2), 'x -> x'),
            nn.ReLU(),
            (GATv2Conv(in_channels=low_up_hidden*2, out_channels=low_up_hidden, heads=1, dropout=0.0, aggr='mean', add_self_loops=False, bias=True), 'x, edge_index -> x'),
            nn.ReLU()
            ])

        self.downscaler_upscaled2high = GATv2Conv((low_up_hidden, high_in), out_channels=high_hidden, dropout=0.2, heads=1, aggr='mean', add_self_loops=False, bias=True)
        
        self.processor_high = geometric_nn.Sequential('x, edge_index', [
            (GATv2Conv(in_channels=high_hidden+high_attr_dim, out_channels=high_hidden, heads=2, dropout=0.2, aggr='mean', add_self_loops=False, bias=True), 'x, edge_index -> x'),
            (geometric_nn.BatchNorm(high_hidden*2), 'x -> x'), 
            nn.ReLU(),
            (GATv2Conv(in_channels=high_hidden*2, out_channels=high_hidden, heads=2, dropout=0.2, aggr='mean', add_self_loops=False, bias=True),'x, edge_index -> x'),
            (geometric_nn.BatchNorm(high_hidden*2), 'x -> x'),
            nn.ReLU(),
            (GATv2Conv(in_channels=high_hidden*2, out_channels=high_hidden, heads=2, dropout=0.2, aggr='mean', add_self_loops=False, bias=True),'x, edge_index -> x'),
            (geometric_nn.BatchNorm(high_hidden*2), 'x -> x'),
            nn.ReLU(),
            (GATv2Conv(in_channels=high_hidden*2, out_channels=high_hidden, heads=2, dropout=0.2, aggr='mean', add_self_loops=False, bias=True),'x, edge_index -> x'),
            (geometric_nn.BatchNorm(high_hidden*2), 'x -> x'),
            nn.ReLU(),
            (GATv2Conv(in_channels=high_hidden*2, out_channels=high_hidden, heads=1, dropout=0.0, aggr='mean', add_self_loops=False, bias=True), 'x, edge_index -> x'),
            nn.ReLU()
            ])
    
        self.predictor = nn.Sequential(
            nn.Linear(high_hidden, high_hidden),
            nn.ReLU(),
            nn.Linear(high_hidden, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
            )

    def forward(self, data): 
        encod_low = self.node_encoder_cnn(data.x_dict['low'])
        encod_low = self.processor_low(encod_low, data.edge_index_dict[('low','within','low')]).squeeze() # (num_nodes, 90)
        encod_upscaled  = self.downscaler_low2upscaled((encod_low, data['low_upscaled'].x), data.edge_index_dict[('low','to','low_upscaled')])
        encod_upscaled = self.processor_upscaled(encod_upscaled, data.edge_index_dict[('low_upscaled','within','low_upscaled')]).squeeze() # (num_nodes, 90)
        encod_high  = self.downscaler_upscaled2high((encod_upscaled, data['high'].x), data.edge_index_dict[('low_upscaled','to','high')])
        encod_high  = torch.concatenate((data['high'].z_std, encod_high ),dim=-1)
        encod_high = self.processor_high(encod_high, data.edge_index_dict[('high','within','high')])
        y_pred = self.predictor(encod_high)
        return y_pred
    

    
class HiResPrecipNet_CNN_GNN_5h(nn.Module):
    
    def __init__(self, low_in=45, high_in=1, low2high_out=64, high_out=64, upscaled_dim=25, kernel=(3,3), padding=(1,1), nvars=5, node_encod_dim=45, high_attr_dim=1):
        super(HiResPrecipNet_CNN_GNN_5h, self).__init__()

        self.node_encoder_cnn = nn.Sequential(
            nn.Conv2d(in_channels=nvars, out_channels=nvars, kernel_size=kernel, padding=padding, groups=nvars),      # (N,Cin,H,W) = (low_num_nodes,vars=5,lev=5,time=5)
            nn.BatchNorm2d(nvars),
            nn.ReLU(),
            nn.Conv2d(in_channels=nvars, out_channels=nvars, kernel_size=kernel, padding=padding, groups=nvars),
            nn.BatchNorm2d(nvars),
            nn.ReLU(),
            nn.Conv2d(in_channels=nvars, out_channels=nvars, kernel_size=kernel, padding=padding, groups=nvars),
            nn.BatchNorm2d(nvars),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), padding=padding, stride=2),                                   # (low_num_nodes,5,3,3)
            nn.Flatten()                                                                        # (low_num_nodes,45)
        )

        self.processor_low = geometric_nn.Sequential('x, edge_index', [
            (GATv2Conv(in_channels=node_encod_dim, out_channels=node_encod_dim, heads=1, dropout=0.2, aggr='mean', add_self_loops=False, bias=True), 'x, edge_index -> x'),
            nn.ReLU(),
            (GATv2Conv(in_channels=node_encod_dim, out_channels=node_encod_dim, heads=1, dropout=0.2, aggr='mean', add_self_loops=False, bias=True), 'x, edge_index -> x'),
            nn.ReLU(),
            (GATv2Conv(in_channels=node_encod_dim, out_channels=node_encod_dim, heads=1, dropout=0.2, aggr='mean', add_self_loops=False, bias=True), 'x, edge_index -> x'),
            nn.ReLU(),
            ])

        # self.node_upscaler_cnn = nn.Sequential(
        #     nn.Conv1d(in_channels=1, out_channels=5, kernel_size=3, padding=1),                  # (N,Cin,L) = (low_num_nodes,1,45)
        #     nn.BatchNorm1d(5),
        #     nn.ReLU(),
        #     nn.Conv1d(in_channels=5, out_channels=upscaled_dim, kernel_size=3, padding=1),
        #     nn.BatchNorm1d(upscaled_dim),
        #     nn.ReLU(),
        #     nn.Conv1d(in_channels=upscaled_dim, out_channels=upscaled_dim, kernel_size=3, padding=1),
        #     nn.BatchNorm1d(upscaled_dim),
        #     nn.ReLU(),
        #     nn.MaxPool1d(kernel_size=2, padding=1, stride=2),                                   # (low_num_nodes,9,22)
        # )

        self.downscaler = GATv2Conv((low_in, high_in), out_channels=low2high_out, dropout=0.0, heads=1, aggr='mean', add_self_loops=False, bias=True)
        
        self.processor = geometric_nn.Sequential('x, edge_index', [
            (geometric_nn.BatchNorm(low2high_out+high_attr_dim), 'x -> x'),
            (GATv2Conv(in_channels=low2high_out+high_attr_dim, out_channels=high_out, heads=2, dropout=0.2, aggr='mean', add_self_loops=True, bias=True), 'x, edge_index -> x'),
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
            nn.ReLU()
            ])
    
        self.predictor = nn.Sequential(
            nn.Linear(high_out, high_out),
            nn.ReLU(),
            nn.Linear(high_out, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
            )

    def forward(self, data): 
        encod_cnn = self.node_encoder_cnn(data.x_dict['low'])
        encod_low = self.processor_low(encod_cnn, data.edge_index_dict[('low','within','low')])#.unsqueeze(1)
        # encod_low_upscaled = self.node_upscaler_cnn(encod_low).flatten(end_dim=1)
#        encod_low2high  = self.downscaler((encod_low_upscaled, data['high'].z_std), data.edge_index_dict[('low_upscaled','to','high')])
        encod_low2high  = self.downscaler((encod_low, data['high'].x), data.edge_index_dict[('low','to','high')])
        encod_low2high  = torch.concatenate((data['high'].z_std, encod_low2high ),dim=-1)
        encod_high = self.processor(encod_low2high, data.edge_index_dict[('high','within','high')])
        y_pred = self.predictor(encod_high)
        return y_pred


class HiResPrecipNet_CNN_GNN_24h(nn.Module):
    
    def __init__(self, low_in=135, high_in=1, low2high_out=64, high_out=64, upscaled_dim=25, kernel=(3,3), padding=(1,0), nvars=5, node_encod_dim=135, high_attr_dim=1):
        super(HiResPrecipNet_CNN_GNN_24h, self).__init__()

        self.node_encoder_cnn = nn.Sequential(
            nn.Conv2d(in_channels=nvars, out_channels=nvars, kernel_size=kernel, padding=padding, groups=nvars),      # (N,Cin,H,W) = (low_num_nodes,vars=5,lev=5,time=5)
            nn.BatchNorm2d(nvars),
            nn.ReLU(),
            nn.Conv2d(in_channels=nvars, out_channels=nvars, kernel_size=kernel, padding=padding, groups=nvars),
            nn.BatchNorm2d(nvars),
            nn.ReLU(),
            nn.Conv2d(in_channels=nvars, out_channels=nvars, kernel_size=kernel, padding=padding, groups=nvars),
            nn.BatchNorm2d(nvars),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), padding=padding, stride=2),                                   # (low_num_nodes,5,3,3)
            nn.Flatten()                                                                        # (low_num_nodes,45)
        )

        self.processor_low = geometric_nn.Sequential('x, edge_index', [
            (GATv2Conv(in_channels=node_encod_dim, out_channels=node_encod_dim, heads=1, dropout=0.2, aggr='mean', add_self_loops=False, bias=True), 'x, edge_index -> x'),
            nn.ReLU(),
            (GATv2Conv(in_channels=node_encod_dim, out_channels=node_encod_dim, heads=1, dropout=0.2, aggr='mean', add_self_loops=False, bias=True), 'x, edge_index -> x'),
            nn.ReLU(),
            (GATv2Conv(in_channels=node_encod_dim, out_channels=node_encod_dim, heads=1, dropout=0.2, aggr='mean', add_self_loops=False, bias=True), 'x, edge_index -> x'),
            nn.ReLU(),
            ])

        # self.node_upscaler_cnn = nn.Sequential(
        #     nn.Conv1d(in_channels=1, out_channels=5, kernel_size=3, padding=1),                  # (N,Cin,L) = (low_num_nodes,1,45)
        #     nn.BatchNorm1d(5),
        #     nn.ReLU(),
        #     nn.Conv1d(in_channels=5, out_channels=upscaled_dim, kernel_size=3, padding=1),
        #     nn.BatchNorm1d(upscaled_dim),
        #     nn.ReLU(),
        #     nn.Conv1d(in_channels=upscaled_dim, out_channels=upscaled_dim, kernel_size=3, padding=1),
        #     nn.BatchNorm1d(upscaled_dim),
        #     nn.ReLU(),
        #     nn.MaxPool1d(kernel_size=2, padding=1, stride=2),                                   # (low_num_nodes,9,22)
        # )

        self.downscaler = GATv2Conv((low_in, high_in), out_channels=low2high_out, dropout=0.0, heads=1, aggr='mean', add_self_loops=False, bias=True)
        
        self.processor = geometric_nn.Sequential('x, edge_index', [
            (geometric_nn.BatchNorm(low2high_out+high_attr_dim), 'x -> x'),
            (GATv2Conv(in_channels=low2high_out+high_attr_dim, out_channels=high_out, heads=2, dropout=0.2, aggr='mean', add_self_loops=True, bias=True), 'x, edge_index -> x'),
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
            nn.ReLU()
            ])
    
        self.predictor = nn.Sequential(
            nn.Linear(high_out, high_out),
            nn.ReLU(),
            nn.Linear(high_out, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
            )

    def forward(self, data): 
        encod_cnn = self.node_encoder_cnn(data.x_dict['low'])
        encod_low = self.processor_low(encod_cnn, data.edge_index_dict[('low','within','low')])#.unsqueeze(1)
        # encod_low_upscaled = self.node_upscaler_cnn(encod_low).flatten(end_dim=1)
#        encod_low2high  = self.downscaler((encod_low_upscaled, data['high'].z_std), data.edge_index_dict[('low_upscaled','to','high')])
        encod_low2high  = self.downscaler((encod_low, data['high'].x), data.edge_index_dict[('low','to','high')])
        encod_low2high  = torch.concatenate((data['high'].z_std, encod_low2high ),dim=-1)
        encod_high = self.processor(encod_low2high, data.edge_index_dict[('high','within','high')])
        y_pred = self.predictor(encod_high)
        return y_pred


class HiResPrecipNet_CNN_LSTM_GNN(nn.Module):
    
    def __init__(self, high_in=1, low2high_out=64, high_out=64, upscaled_dim=25, kernel=3, nvars=5, ntimes=25, nlevels=5, node_encod_dim=64, high_attr_dim=1):
        super(HiResPrecipNet_CNN_LSTM_GNN, self).__init__()

        self.ntimes = ntimes
        self.nvars = nvars
        self.nlevels = nlevels

        self.node_encoder_cnn = nn.Sequential(
            nn.Conv1d(in_channels=self.ntimes*self.nvars, out_channels=self.ntimes*self.nvars, kernel_size=kernel, padding=1, groups=self.ntimes*self.nvars),      # (N,Cin,H,W) = (low_num_nodes,vars=5,lev=5,time=5)
            nn.BatchNorm1d(self.ntimes*self.nvars),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.ntimes*self.nvars, out_channels=self.ntimes*self.nvars, kernel_size=kernel, padding=1, groups=self.ntimes*self.nvars),
            nn.BatchNorm1d(self.ntimes*self.nvars),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.ntimes*self.nvars, out_channels=self.ntimes*self.nvars, kernel_size=kernel, padding=0, groups=self.ntimes*self.nvars),
            nn.BatchNorm1d(self.ntimes*self.nvars),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, padding=0, stride=2),                                   # (low_num_nodes,24*5,1)
            #nn.Flatten()                                                                        # (low_num_nodes,24*5)
        )

        self.node_encoder_lstm = nn.LSTM(input_size=self.nlevels, hidden_size=self.nlevels, batch_first=True, num_layers=3)
        
        self.linear = nn.Linear(self.ntimes*self.nvars, node_encod_dim)

        self.processor_low = geometric_nn.Sequential('x, edge_index', [
            (GATv2Conv(in_channels=node_encod_dim, out_channels=node_encod_dim, heads=1, dropout=0.2, aggr='mean', add_self_loops=False, bias=True), 'x, edge_index -> x'),
            nn.ReLU(),
            (GATv2Conv(in_channels=node_encod_dim, out_channels=node_encod_dim, heads=1, dropout=0.2, aggr='mean', add_self_loops=False, bias=True), 'x, edge_index -> x'),
            nn.ReLU(),
            (GATv2Conv(in_channels=node_encod_dim, out_channels=node_encod_dim, heads=1, dropout=0.2, aggr='mean', add_self_loops=False, bias=True), 'x, edge_index -> x'),
            nn.ReLU(),
            ])

        # self.node_upscaler_cnn = nn.Sequential(
        #     nn.Conv1d(in_channels=1, out_channels=5, kernel_size=3, padding=1),                  # (N,Cin,L) = (low_num_nodes,1,45)
        #     nn.BatchNorm1d(5),
        #     nn.ReLU(),
        #     nn.Conv1d(in_channels=5, out_channels=upscaled_dim, kernel_size=3, padding=1),
        #     nn.BatchNorm1d(upscaled_dim),
        #     nn.ReLU(),
        #     nn.Conv1d(in_channels=upscaled_dim, out_channels=upscaled_dim, kernel_size=3, padding=1),
        #     nn.BatchNorm1d(upscaled_dim),
        #     nn.ReLU(),
        #     nn.MaxPool1d(kernel_size=2, padding=1, stride=2),                                   # (low_num_nodes,9,22)
        # )

        self.downscaler = GATv2Conv((node_encod_dim, high_in), out_channels=low2high_out, dropout=0.0, heads=1, aggr='mean', add_self_loops=False, bias=True)
        
        self.processor = geometric_nn.Sequential('x, edge_index', [
            (geometric_nn.BatchNorm(low2high_out+high_attr_dim), 'x -> x'),
            (GATv2Conv(in_channels=low2high_out+high_attr_dim, out_channels=high_out, heads=2, dropout=0.2, aggr='mean', add_self_loops=True, bias=True), 'x, edge_index -> x'),
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
            nn.ReLU()
            ])
    
        self.predictor = nn.Sequential(
            nn.Linear(high_out, high_out),
            nn.ReLU(),
            nn.Linear(high_out, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
            )

    def forward(self, data): # (low_num_nodes,time*vars=25*5,lev=5)
        data.x_dict['low'] = data.x_dict['low'].permute(0,3,1,2) # from (low_num_nodes,vars,lev,time) to (ow_num_nodes,time,vars,lev)
        batch_size = data.x_dict['low'].shape[0]
        encod_low = self.node_encoder_cnn(data.x_dict['low'].view(batch_size, self.ntimes*self.nvars, self.nlevels)).squeeze()
        encod_low = encod_low.reshape(batch_size, self.ntimes, self.nvars)
        encod_low, _ = self.node_encoder_lstm(encod_low) # (num_nodes, 25, 5)
        encod_low = encod_low.reshape(batch_size, self.ntimes * self.nvars)
        encod_cnn_lstm = self.linear(encod_low)
        encod_low = self.processor_low(encod_cnn_lstm, data.edge_index_dict[('low','within','low')])#.unsqueeze(1)
        # encod_low_upscaled = self.node_upscaler_cnn(encod_low).flatten(end_dim=1)
#        encod_low2high  = self.downscaler((encod_low_upscaled, data['high'].z_std), data.edge_index_dict[('low_upscaled','to','high')])
        encod_low2high  = self.downscaler((encod_low, data['high'].x), data.edge_index_dict[('low','to','high')])
        encod_low2high  = torch.concatenate((data['high'].z_std, encod_low2high ),dim=-1)
        encod_high = self.processor(encod_low2high, data.edge_index_dict[('high','within','high')])
        y_pred = self.predictor(encod_high)
        return y_pred
