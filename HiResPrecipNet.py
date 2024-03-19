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
    
    def __init__(self, low_in=23, high_in=1, low2high_out=64, high_out=64, upscaled_dim=25, kernel=(3,3), padding=(1,0), nvars=5, node_encod_dim=135, high_attr_dim=1):
        super(HiResPrecipNet_CNN_GNN, self).__init__()

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
            nn.Flatten()                                                                                  # (low_num_nodes,45)
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

    def forward(self, data): 
        encod_cnn = self.node_encoder_cnn(data.x_dict['low'])
        encod_low = self.processor_low(encod_cnn, data.edge_index_dict[('low','within','low')]).squeeze()
#        encod_low_upscaled = self.node_upscaler_cnn(encod_low.unsqueeze(1)).flatten(end_dim=1)
        # encod_low2high  = self.downscaler((encod_low_upscaled, data['high'].z_std), data.edge_index_dict[('low_upscaled','to','high')])
        encod_low2high  = self.downscaler((encod_low, data['high'].x), data.edge_index_dict[('low','to','high')])
        encod_low2high  = torch.concatenate((data['high'].z_std, encod_low2high ),dim=-1)
        encod_high = self.processor(encod_low2high, data.edge_index_dict[('high','within','high')])
        y_pred = self.predictor(encod_high)
        return y_pred


# class HiResPrecipNet_up_CNN_GNN(nn.Module):
    
#     def __init__(self, low_in=23, high_in=1, low2high_out=64, high_out=64, upscaled_dim=25, kernel=(3,3), padding=(1,1), nvars=5, node_encod_dim=45, high_attr_dim=1):
#         super(HiResPrecipNet_up_CNN_GNN, self).__init__()

#         self.node_upscaler_cnn = nn.Sequential(
#             nn.Conv1d(in_channels=1, out_channels=5, kernel_size=3, padding=1),                  # (N,Cin,L) = (low_num_nodes,1,45)
#             nn.BatchNorm1d(5),
#             nn.ReLU(),
#             nn.Conv1d(in_channels=5, out_channels=upscaled_dim, kernel_size=3, padding=1),
#             nn.BatchNorm1d(upscaled_dim),
#             nn.ReLU(),
#             nn.Conv1d(in_channels=upscaled_dim, out_channels=upscaled_dim, kernel_size=3, padding=1),
#             nn.BatchNorm1d(upscaled_dim),
#             nn.ReLU(),
#             nn.MaxPool1d(kernel_size=2, padding=1, stride=2),                                   # (low_num_nodes,9,22)
#         )

#         self.node_encoder_cnn = nn.Sequential(
#             nn.Conv2d(in_channels=nvars, out_channels=nvars, kernel_size=kernel, padding=padding, groups=nvars),      # (N,Cin,H,W) = (low_num_nodes,vars=5,lev=5,time=5)
#             nn.BatchNorm2d(nvars),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=nvars, out_channels=nvars, kernel_size=kernel, padding=padding, groups=nvars),
#             nn.BatchNorm2d(nvars),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=nvars, out_channels=nvars, kernel_size=kernel, padding=padding, groups=nvars),
#             nn.BatchNorm2d(nvars),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=(2,2), padding=padding, stride=2),                                   # (low_num_nodes,5,3,3)
#             nn.Flatten()                                                                                  # (low_num_nodes,45)
#         )

#         self.processor_low = geometric_nn.Sequential('x, edge_index', [
#             (GATv2Conv(in_channels=node_encod_dim, out_channels=node_encod_dim, heads=1, dropout=0.2, aggr='mean', add_self_loops=False, bias=True), 'x, edge_index -> x'),
#             nn.ReLU(),
#             (GATv2Conv(in_channels=node_encod_dim, out_channels=node_encod_dim, heads=1, dropout=0.2, aggr='mean', add_self_loops=False, bias=True), 'x, edge_index -> x'),
#             nn.ReLU(),
#             (GATv2Conv(in_channels=node_encod_dim, out_channels=node_encod_dim, heads=1, dropout=0.2, aggr='mean', add_self_loops=False, bias=True), 'x, edge_index -> x'),
#             nn.ReLU(),
#             ])

#         self.downscaler = GATv2Conv((low_in, high_in), out_channels=low2high_out, dropout=0.0, heads=1, aggr='mean', add_self_loops=False, bias=True)
        
#         self.processor = geometric_nn.Sequential('x, edge_index', [
#             (geometric_nn.BatchNorm(low2high_out+high_attr_dim), 'x -> x'),
#             (GATv2Conv(in_channels=low2high_out+high_attr_dim, out_channels=high_out, heads=2, dropout=0.2, aggr='mean', add_self_loops=True, bias=True), 'x, edge_index -> x'),
#             (geometric_nn.BatchNorm(high_out*2), 'x -> x'), 
#             nn.ReLU(),
#             (GATv2Conv(in_channels=high_out*2, out_channels=high_out, heads=2, dropout=0.2, aggr='mean', add_self_loops=True, bias=True),'x, edge_index -> x'),
#             (geometric_nn.BatchNorm(high_out*2), 'x -> x'),
#             nn.ReLU(),
#             (GATv2Conv(in_channels=high_out*2, out_channels=high_out, heads=2, dropout=0.2, aggr='mean', add_self_loops=True, bias=True),'x, edge_index -> x'),
#             (geometric_nn.BatchNorm(high_out*2), 'x -> x'),
#             nn.ReLU(),
#             (GATv2Conv(in_channels=high_out*2, out_channels=high_out, heads=2, dropout=0.2, aggr='mean', add_self_loops=True, bias=True),'x, edge_index -> x'),
#             (geometric_nn.BatchNorm(high_out*2), 'x -> x'),
#             nn.ReLU(),
#             (GATv2Conv(in_channels=high_out*2, out_channels=high_out, heads=1, dropout=0.0, aggr='mean', add_self_loops=True, bias=True), 'x, edge_index -> x'),
#             nn.ReLU()
#             ])
    
#         self.predictor = nn.Sequential(
#             nn.Linear(high_out, high_out),
#             nn.ReLU(),
#             nn.Linear(high_out, 32),
#             nn.ReLU(),
#             nn.Linear(32, 1)
#             )

#     def forward(self, data): 
#         encod_cnn = self.node_encoder_cnn(data.x_dict['low'])
#         encod_low = self.processor_low(encod_cnn, data.edge_index_dict[('low','within','low')]).squeeze()
#         encod_low_upscaled = self.node_upscaler_cnn(encod_low.unsqueeze(1)).flatten(end_dim=1)
#         # encod_low2high  = self.downscaler((encod_low_upscaled, data['high'].z_std), data.edge_index_dict[('low_upscaled','to','high')])
#         encod_low2high  = self.downscaler((encod_low_upscaled, data['high'].x), data.edge_index_dict[('encod_low_upscaled','to','high')])
#         encod_low2high  = torch.concatenate((data['high'].z_std, encod_low2high ),dim=-1)
#         encod_high = self.processor(encod_low2high, data.edge_index_dict[('high','within','high')])
#         y_pred = self.predictor(encod_high)
#         return y_pred
    

class HiResPrecipNet_up_CNN_GNN(nn.Module):
    
    def __init__(self, high_in=1, low2high_out=16, high_out=16, low_up_hidden=32, kernel=(3,5), nvars=5, ntimes=25, nlevs=5, node_encod_dim=90, high_attr_dim=0, scale_factor=5):
        super(HiResPrecipNet_up_CNN_GNN, self).__init__()

        self.nvars = nvars
        self.ntimes = ntimes
        self.nlevs = nlevs
        self.scale_factor = scale_factor

        self.sub_pixel_upscaler = nn.Sequential(
            nn.Conv1d(1, 64, 5, 1, 2),                                          # [nfeatures, 1, nnodes] to [nfeatures, 64, nnodes]
            nn.Tanh(),                          
            nn.Conv1d(64, 32, 3, 1, 1),                                         # [nfeatures, 64, nnodes] to [nfeatures, 32, nnodes]
            nn.Tanh(),
            nn.Conv1d(32, self.scale_factor*self.scale_factor, 3, 1, 1),        # [nfeatures, 32, nnodes] to [nfeatures, 1*(r*r), nnodes]
        )

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
            nn.MaxPool2d(kernel_size=2, padding=0, stride=2),                                   # (low_num_nodes,5,3,3)
            nn.Flatten()                                                                        # (low_num_nodes,45)
        )

        self.processor_low = geometric_nn.Sequential('x, edge_index', [
            (GATv2Conv(in_channels=node_encod_dim, out_channels=low_up_hidden, heads=2, dropout=0.2, aggr='mean', add_self_loops=False, bias=True), 'x, edge_index -> x'),
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
        
        self.downscaler = GATv2Conv((low_up_hidden, high_in), out_channels=low2high_out, dropout=0.2, heads=1, aggr='mean', add_self_loops=False, bias=True)
        
        self.processor = geometric_nn.Sequential('x, edge_index', [
            (GATv2Conv(in_channels=low2high_out+high_attr_dim, out_channels=high_out, heads=2, dropout=0.2, aggr='mean', add_self_loops=False, bias=True), 'x, edge_index -> x'),
            (geometric_nn.BatchNorm(high_out*2), 'x -> x'), 
            nn.ReLU(),
            (GATv2Conv(in_channels=high_out*2, out_channels=high_out, heads=2, dropout=0.2, aggr='mean', add_self_loops=False, bias=True),'x, edge_index -> x'),
            (geometric_nn.BatchNorm(high_out*2), 'x -> x'),
            nn.ReLU(),
            (GATv2Conv(in_channels=high_out*2, out_channels=high_out, heads=2, dropout=0.2, aggr='mean', add_self_loops=False, bias=True),'x, edge_index -> x'),
            (geometric_nn.BatchNorm(high_out*2), 'x -> x'),
            nn.ReLU(),
            (GATv2Conv(in_channels=high_out*2, out_channels=high_out, heads=2, dropout=0.2, aggr='mean', add_self_loops=False, bias=True),'x, edge_index -> x'),
            (geometric_nn.BatchNorm(high_out*2), 'x -> x'),
            nn.ReLU(),
            (GATv2Conv(in_channels=high_out*2, out_channels=high_out, heads=1, dropout=0.0, aggr='mean', add_self_loops=False, bias=True), 'x, edge_index -> x'),
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
        nnodes = data.x_dict['low'].shape[0]
        data.x_dict['low'] = data.x_dict['low'].permute(1,2,3,0)
        x_upscaled = self.sub_pixel_upscaler(data.x_dict['low'].view(self.nvars*self.nlevs*self.ntimes, nnodes).unsqueeze(1))
        x_upscaled = x_upscaled.permute(2,1,0)
        x_upscaled = x_upscaled.flatten(end_dim=1).reshape(nnodes*self.scale_factor*self.scale_factor, self.nvars, self.nlevs, self.ntimes)
        encod_low = self.node_encoder_cnn(x_upscaled)
        encod_low = self.processor_low(encod_low, data.edge_index_dict[('low_upscaled','within','low_upscaled')]).squeeze() # (num_nodes, 90)
        encod_low2high  = self.downscaler((encod_low, data['high'].z_std), data.edge_index_dict[('low_upscaled','to','high')])
        #encod_low2high  = torch.concatenate((data['high'].z_std, encod_low2high ),dim=-1)
        encod_high = self.processor(encod_low2high, data.edge_index_dict[('high','within','high')])
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
