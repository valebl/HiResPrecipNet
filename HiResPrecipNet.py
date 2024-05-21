import torch.nn as nn
import torch_geometric.nn as geometric_nn
from torch_geometric.nn import GATv2Conv, HeteroConv, GCNConv
from torch_geometric.nn import global_mean_pool
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric_temporal.nn.recurrent import A3TGCN

############################
### Current stable model ###
############################

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


class HiResPrecipNet_small2(nn.Module):
    
    def __init__(self, low_in=5*5*5, high_in=1, low2high_out=64, high_out=64):
        super(HiResPrecipNet_small2, self).__init__()

        self.downscaler = GATv2Conv((low_in, high_in), out_channels=low2high_out, dropout=0.0, heads=1, aggr='mean', add_self_loops=False, bias=True)
        
        self.processor = geometric_nn.Sequential('x, edge_index', [
            (geometric_nn.BatchNorm(low2high_out+1), 'x -> x'),
            (GATv2Conv(in_channels=low2high_out+1, out_channels=high_out, heads=2, dropout=0.2, aggr='mean', add_self_loops=True, bias=True), 'x, edge_index -> x'),
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
    

class HiResPrecipNet_variance(nn.Module):
    
    def __init__(self, low_in=25*5*5, high_in=1, low2high_out=64, high_out=64):
        super(HiResPrecipNet_variance, self).__init__()

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
        
        self.predictor_theta = nn.Sequential(
            nn.Linear(high_out, high_out),
            nn.ReLU(),
            nn.Linear(high_out, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
            )


    def forward(self, data):        
        encod_low2high  = self.downscaler((data.x_dict['low'], data['high'].x), data.edge_index_dict[('low','to','high')])
        encod_low2high  = torch.concatenate((data['high'].z_std, encod_low2high ),dim=-1)
        encod_high = self.processor(encod_low2high , data.edge_index_dict[('high','within','high')])
        y_pred = self.predictor(encod_high)
        theta_pred = self.predictor_theta(encod_high)
        return y_pred, theta_pred
    

class HiResPrecipNet_gamma(nn.Module):
    
    def __init__(self, low_in=25*5*5, high_in=1, low2high_out=64, high_out=64):
        super(HiResPrecipNet_gamma, self).__init__()

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
    
        self.predictor_alpha = nn.Sequential(
            nn.Linear(high_out, high_out),
            nn.ReLU(),
            nn.Linear(high_out, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Softplus()
            )
        
        self.predictor_beta = nn.Sequential(
            nn.Linear(high_out, high_out),
            nn.ReLU(),
            nn.Linear(high_out, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Softplus()
            )

    def forward(self, data):        
        encod_low2high  = self.downscaler((data.x_dict['low'], data['high'].x), data.edge_index_dict[('low','to','high')])
        encod_low2high  = torch.concatenate((data['high'].z_std, encod_low2high ),dim=-1)
        encod_high = self.processor(encod_low2high , data.edge_index_dict[('high','within','high')])
        alpha_pred = self.predictor_alpha(encod_high)
        beta_pred = self.predictor_beta(encod_high)
        return alpha_pred, beta_pred # alpha, beta
    

class HiResPrecipNet_gauss(nn.Module):
    
    def __init__(self, low_in=5*5*5, high_in=1, low2high_out=64, high_out=64):
        super(HiResPrecipNet_gauss, self).__init__()

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
    
        self.predictor_mu = nn.Sequential(
            nn.Linear(high_out, high_out),
            nn.ReLU(),
            nn.Linear(high_out, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            )
        
        self.predictor_log_sigma = nn.Sequential(
            nn.Linear(high_out, high_out),
            nn.ReLU(),
            nn.Linear(high_out, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            )

    def forward(self, data):        
        encod_low2high  = self.downscaler((data.x_dict['low'], data['high'].x), data.edge_index_dict[('low','to','high')])
        encod_low2high  = torch.concatenate((data['high'].z_std, encod_low2high ),dim=-1)
        encod_high = self.processor(encod_low2high , data.edge_index_dict[('high','within','high')])
        mu_pred = self.predictor_mu(encod_high)
        log_sigma_pred = self.predictor_log_sigma(encod_high)
        return mu_pred, log_sigma_pred   


class HiResPrecipNet_evl(nn.Module):
    
    def __init__(self, low_in=5*5*5, high_in=1, low2high_out=64, high_out=64):
        super(HiResPrecipNet_evl, self).__init__()

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
        
        self.classifier = nn.Sequential(
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
        y_pred_cl = self.classifier(encod_high)
        return y_pred, y_pred_cl
    
    
class HiResPrecipNet2(nn.Module):
    
    def __init__(self, low_in=5*5*5, high_in=1, low2high_out=64, high_out=64):
        super(HiResPrecipNet2, self).__init__()

        self.downscaler = GATv2Conv((low_in, high_in), out_channels=low2high_out, dropout=0.5, heads=1, aggr='mean', add_self_loops=False, bias=True)
        
        self.processor = geometric_nn.Sequential('x, edge_index', [
            (geometric_nn.BatchNorm(low2high_out+1), 'x -> x'),
            (GATv2Conv(in_channels=low2high_out+1, out_channels=high_out, heads=2, dropout=0.5, aggr='mean', add_self_loops=True, bias=True), 'x, edge_index -> x'),
            (geometric_nn.BatchNorm(high_out*2), 'x -> x'), 
            nn.ReLU(),
            (GATv2Conv(in_channels=high_out*2, out_channels=high_out, heads=2, dropout=0.5, aggr='mean', add_self_loops=True, bias=True),'x, edge_index -> x'),
            (geometric_nn.BatchNorm(high_out*2), 'x -> x'),
            nn.ReLU(),
            (GATv2Conv(in_channels=high_out*2, out_channels=high_out, heads=2, dropout=0.5, aggr='mean', add_self_loops=True, bias=True),'x, edge_index -> x'),
            (geometric_nn.BatchNorm(high_out*2), 'x -> x'),
            nn.ReLU(),
            (GATv2Conv(in_channels=high_out*2, out_channels=high_out, heads=2, dropout=0.5, aggr='mean', add_self_loops=True, bias=True),'x, edge_index -> x'),
            (geometric_nn.BatchNorm(high_out*2), 'x -> x'),
            nn.ReLU(),
            (GATv2Conv(in_channels=high_out*2, out_channels=high_out, heads=1, dropout=0.0, aggr='mean', add_self_loops=True, bias=True), 'x, edge_index -> x'),
            nn.ReLU(),
            ])
    
        self.predictor = nn.Sequential(
            nn.Linear(high_out, high_out),
            nn.ReLU(),
            nn.Dropout(p=0.5),
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

    
class HiResPrecipNet_9x_25x(nn.Module):
    
    def __init__(self, low_in=5*5*5, high_in=1, high_out=64):
        super(HiResPrecipNet_9x_25x, self).__init__()

        self.downscaler_low_9x = geometric_nn.Sequential('x, edge_index', [
            (GATv2Conv((low_in, high_in), out_channels=low_in, dropout=0.6, heads=2, aggr='mean', add_self_loops=False, bias=True), 'x, edge_index -> x'),
            nn.ReLU(),
            nn.Linear(low_in*2, low_in),
            nn.ReLU()
        ])

        self.downscaler_9x_25x = geometric_nn.Sequential('x, edge_index', [
            (GATv2Conv((low_in, high_in), out_channels=low_in, dropout=0.6, heads=2, aggr='mean', add_self_loops=False, bias=True), 'x, edge_index -> x'),
            nn.ReLU(),
            nn.Linear(low_in*2, low_in),
            nn.ReLU()
        ])

        self.downscaler_25x_high = geometric_nn.Sequential('x, edge_index', [
            (GATv2Conv((low_in, high_in), out_channels=low_in, dropout=0.6, heads=2, aggr='mean', add_self_loops=False, bias=True), 'x, edge_index -> x'),
            nn.ReLU(),
            nn.Linear(low_in*2, low_in),
            nn.ReLU()
        ])
        
        self.processor = geometric_nn.Sequential('x, edge_index', [
            (GATv2Conv(in_channels=low_in+1, out_channels=high_out, heads=2, dropout=0.6, aggr='mean', add_self_loops=True, bias=True), 'x, edge_index -> x'),
            (geometric_nn.BatchNorm(high_out*2), 'x -> x'), 
            nn.ReLU(),
            (GATv2Conv(in_channels=high_out*2, out_channels=high_out, heads=2, dropout=0.6, aggr='mean', add_self_loops=True, bias=True),'x, edge_index -> x'),
            (geometric_nn.BatchNorm(high_out*2), 'x -> x'),
            nn.ReLU(),
            (GATv2Conv(in_channels=high_out*2, out_channels=high_out, heads=2, dropout=0.6, aggr='mean', add_self_loops=True, bias=True),'x, edge_index -> x'),
            (geometric_nn.BatchNorm(high_out*2), 'x -> x'),
            nn.ReLU(),
            (GATv2Conv(in_channels=high_out*2, out_channels=high_out, heads=2, dropout=0.4, aggr='mean', add_self_loops=True, bias=True),'x, edge_index -> x'),
            (geometric_nn.BatchNorm(high_out*2), 'x -> x'),
            nn.ReLU(),
            (GATv2Conv(in_channels=high_out*2, out_channels=high_out, heads=2, dropout=0.0, aggr='mean', add_self_loops=True, bias=True), 'x, edge_index -> x'),
            nn.ReLU(),
            ])
    
        self.predictor = nn.Sequential(
            nn.Linear(high_out*2, high_out),
            nn.ReLU(),
            nn.Linear(high_out, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
            )

    def forward(self, data):        
        encod  = self.downscaler_low_9x((data.x_dict['low'], data['low_9x'].x), data.edge_index_dict[('low','to','low_9x')])
        encod  = self.downscaler_9x_25x((encod, data['low_25x'].x), data.edge_index_dict[('low_9x','to','low_25x')])
        encod = self.downscaler_25x_high((encod, data['high'].x), data.edge_index_dict[('low_25x','to','high')])
        encod  = torch.concatenate((data['high'].z_std, encod ),dim=-1)
        encod = self.processor(encod , data.edge_index_dict[('high','within','high')])
        y_pred = self.predictor(encod)
        return y_pred    
    
class HiResPrecipNet_9x_25x_CNN(nn.Module):
    
    def __init__(self, encod_dim=45, low_in=64, high_in=1, high_out=64):
        super(HiResPrecipNet_9x_25x_CNN, self).__init__()

        self.node_encoder_cnn = nn.Sequential(
            nn.Conv2d(in_channels=5, out_channels=5, kernel_size=3, padding=1, groups=5),      # (N,Cin,H,W) = (low_num_nodes,5,5,5)
            nn.BatchNorm2d(5),
            nn.ReLU(),
            nn.Conv2d(in_channels=5, out_channels=5, kernel_size=3, padding=1, groups=5),
            nn.BatchNorm2d(5),
            nn.ReLU(),
            nn.Conv2d(in_channels=5, out_channels=5, kernel_size=3, padding=1, groups=5),
            nn.BatchNorm2d(5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, padding=1, stride=2),                                   # (low_num_nodes,5,3,3)
            nn.Flatten()                                                                        # (low_num_nodes,45)
        )

        self.downscaler_low_9x = GATv2Conv((encod_dim, high_in), out_channels=low_in, dropout=0.2, heads=1, aggr='mean', add_self_loops=False, bias=True)

        self.downscaler_9x_25x = GATv2Conv((low_in, high_in), out_channels=low_in, dropout=0.2, heads=1, aggr='mean', add_self_loops=False, bias=True)

        self.downscaler_25x_high = GATv2Conv((low_in, high_in), out_channels=low_in, dropout=0.2, heads=1, aggr='mean', add_self_loops=False, bias=True)
        
        self.processor = geometric_nn.Sequential('x, edge_index', [
            (GATv2Conv(in_channels=low_in+1, out_channels=high_out, heads=2, dropout=0.4, aggr='mean', add_self_loops=True, bias=True), 'x, edge_index -> x'),
            (geometric_nn.BatchNorm(high_out*2), 'x -> x'), 
            nn.ReLU(),
            (GATv2Conv(in_channels=high_out*2, out_channels=high_out, heads=2, dropout=0.4, aggr='mean', add_self_loops=True, bias=True),'x, edge_index -> x'),
            (geometric_nn.BatchNorm(high_out*2), 'x -> x'),
            nn.ReLU(),
            (GATv2Conv(in_channels=high_out*2, out_channels=high_out, heads=2, dropout=0.4, aggr='mean', add_self_loops=True, bias=True),'x, edge_index -> x'),
            (geometric_nn.BatchNorm(high_out*2), 'x -> x'),
            nn.ReLU(),
            (GATv2Conv(in_channels=high_out*2, out_channels=high_out, heads=2, dropout=0.4, aggr='mean', add_self_loops=True, bias=True),'x, edge_index -> x'),
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
        encod = self.node_encoder_cnn(data.x_dict['low'])
        encod  = self.downscaler_low_9x((encod, data['low_9x'].x), data.edge_index_dict[('low','to','low_9x')])
        encod  = self.downscaler_9x_25x((encod, data['low_25x'].x), data.edge_index_dict[('low_9x','to','low_25x')])
        encod = self.downscaler_25x_high((encod, data['high'].x), data.edge_index_dict[('low_25x','to','high')])
        encod  = torch.concatenate((data['high'].z_std, encod ),dim=-1)
        encod = self.processor(encod , data.edge_index_dict[('high','within','high')])
        y_pred = self.predictor(encod)
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
            nn.Conv2d(in_channels=5, out_channels=5, kernel_size=3, padding=1, groups=5),      # (N,Cin,H,W) = (low_num_nodes,5,5,5)
            nn.BatchNorm2d(5),
            nn.ReLu(),
            nn.Conv2d(in_channels=5, out_channels=5, kernel_size=3, padding=1, groups=5),
            nn.BatchNorm2d(5),
            nn.ReLu(),
            nn.Conv2d(in_channels=5, out_channels=5, kernel_size=3, padding=1, groups=5),
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


class HiResPrecipNet_CNN_GRU_GNN(nn.Module):
    
    def __init__(self, high_in=1, low2high_out=64, high_out=64, upscaled_dim=25, kernel=3, nvars=5, ntimes=25, nlevels=5, node_encod_dim=64, high_attr_dim=1):
        super(HiResPrecipNet_CNN_GRU_GNN, self).__init__()

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

        # define the decoder modules
        self.node_encoder_gru = nn.Sequential(
            nn.GRU(input_size=self.nlevels, hidden_size=self.nlevels, batch_first=True, num_layers=3),
        )

        self.dense = nn.Sequential(
            nn.Linear(self.ntimes*self.nvars, node_encod_dim),
            nn.ReLU()
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

    def forward(self, data): # (low_num_nodes,time*vars=25*5,lev=5)
        data.x_dict['low'] = data.x_dict['low'].permute(0,3,1,2) # from (low_num_nodes,vars,lev,time) to (low_num_nodes,time,vars,lev)
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


class HiResPrecipNet_CNN_GNN_new(nn.Module):
    
    def __init__(self, high_in=1, low2high_out=64, high_out=64, upscaled_dim=25, kernel=3, nvars=5, ntimes=25, nlevels=5, node_encod_dim=32, high_attr_dim=1):
        super(HiResPrecipNet_CNN_GNN_new, self).__init__()

        self.ntimes = ntimes
        self.nvars = nvars
        self.nlevels = nlevels

        self.node_encoder_cnn = nn.Sequential(
            nn.Conv3d(1,8, kernel_size=(1,5,3), padding=(0,0,1)),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            nn.Conv3d(8,16, kernel_size=(1,5,3), padding=(0,0,1)),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.Conv3d(16,32, kernel_size=(1,5,3), padding=(0,0,1)),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32,64, kernel_size=(1,5,3), padding=(0,0,1)),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64,32, kernel_size=(1,5,3), padding=0),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1,5,3), padding=0, stride=(1,2,1))                    
        )

        self.dense = nn.Sequential(
            nn.Linear(node_encod_dim*5, node_encod_dim),
            nn.ReLU()
        )

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

    def forward(self, data): # [N,V,L,T] (num_nodes,vars=5,times=25,levels=5)
        encod_low = self.node_encoder_cnn(data.x_dict['low'].unsqueeze(1)) # from [N,1,V,T,L] to [N,8,V,1,1]
        encod_low = self.dense(encod_low.squeeze().flatten(start_dim=1)) # from 32*5=160 to 32
        encod_low2high  = self.downscaler((encod_low, data['high'].x), data.edge_index_dict[('low','to','high')])
        encod_low2high  = torch.concatenate((data['high'].z_std, encod_low2high ),dim=-1)
        encod_high = self.processor(encod_low2high, data.edge_index_dict[('high','within','high')])
        y_pred = self.predictor(encod_high)
        return y_pred


class HiResPrecipNet_CNN_GNN_9x_new(nn.Module):
    
    def __init__(self, high_in=1, low2high_out=64, high_out=64, upscaled_dim=25, kernel=3, nvars=5, ntimes=25, nlevels=5, node_encod_dim=32, high_attr_dim=1):
        super(HiResPrecipNet_CNN_GNN_9x_new, self).__init__()

        self.ntimes = ntimes
        self.nvars = nvars
        self.nlevels = nlevels

        self.node_encoder_cnn_temporal = nn.Sequential(
            nn.Conv3d(1,8, kernel_size=(1,5,1), padding=0),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            nn.Conv3d(8,16, kernel_size=(1,5,1), padding=0),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.Conv3d(16,32, kernel_size=(1,5,1), padding=0),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32,16, kernel_size=(1,5,1), padding=0),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.Conv3d(16,8, kernel_size=(1,5,1), padding=0),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1,5,1), padding=0, stride=(1,2,1))                    
        )

        self.node_encoder_cnn_vertical = nn.Sequential(
            nn.Conv3d(8,16, kernel_size=(1,1,3), padding=(0,0,1)),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.Conv3d(16,node_encod_dim, kernel_size=(1,1,3), padding=0),
            nn.BatchNorm3d(node_encod_dim),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1,1,3), padding=0, stride=(1,1,2))                    
        )

        self.dense = nn.Sequential(
            nn.Linear(node_encod_dim*5, node_encod_dim),
            nn.ReLU()
        )

        self.downscaler_low_9x = geometric_nn.Sequential('x, edge_index', [
            (GATv2Conv((node_encod_dim, high_in), out_channels=node_encod_dim, dropout=0.2, heads=2, aggr='mean', add_self_loops=False, bias=True), 'x, edge_index -> x'),
            nn.ReLU(),
            nn.Linear(node_encod_dim*2, node_encod_dim),
            nn.ReLU()
        ])

        self.downscaler = geometric_nn.Sequential('x, edge_index', [
            (GATv2Conv((node_encod_dim, high_in), out_channels=low2high_out, dropout=0.2, heads=2, aggr='mean', add_self_loops=False, bias=True), 'x, edge_index -> x'),
            nn.ReLU(),
            nn.Linear(low2high_out*2, low2high_out),
            nn.ReLU()
        ])

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

    def forward(self, data): # [N,V,L,T] (num_nodes,vars=5,times=25,levels=5)
        encod_low = self.node_encoder_cnn_temporal(data.x_dict['low'].unsqueeze(1)) # from [N,1,V,T,L] to [N,8,V,1,L]
        encod_low = self.node_encoder_cnn_vertical(encod_low) # from [N,8,V,1,L] to [N,32,V,1,1]
        encod_low = self.dense(encod_low.squeeze().flatten(start_dim=1)) # from 32*5=160 to 32
        encod_low  = self.downscaler_low_9x((encod_low, data['low_9x'].x), data.edge_index_dict[('low','to','low_9x')])
        encod_low2high  = self.downscaler((encod_low, data['high'].x), data.edge_index_dict[('low_9x','to','high')])
        encod_low2high  = torch.concatenate((data['high'].z_std, encod_low2high ),dim=-1)
        encod_high = self.processor(encod_low2high, data.edge_index_dict[('high','within','high')])
        y_pred = self.predictor(encod_high)
        return y_pred


class HiResPrecipNet_CNN_GNN_9x_up_new(nn.Module):
    
    def __init__(self, high_in=1, low2high_out=64, high_out=64, up_factor=3, kernel=3, nvars=5, ntimes=25, nlevels=5, node_encod_dim=32, high_attr_dim=1):
        super(HiResPrecipNet_CNN_GNN_9x_up_new, self).__init__()

        self.ntimes = ntimes
        self.nvars = nvars
        self.nlevels = nlevels

        self.node_encoder_cnn_temporal = nn.Sequential(
            nn.Conv3d(1,8, kernel_size=(1,5,1), padding=0),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            nn.Conv3d(8,16, kernel_size=(1,5,1), padding=0),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.Conv3d(16,32, kernel_size=(1,5,1), padding=0),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32,16, kernel_size=(1,5,1), padding=0),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.Conv3d(16,8, kernel_size=(1,5,1), padding=0),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1,5,1), padding=0, stride=(1,2,1))                    
        )

        self.node_encoder_cnn_vertical = nn.Sequential(
            nn.Conv3d(8,16, kernel_size=(1,1,3), padding=(0,0,1)),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.Conv3d(16,node_encod_dim, kernel_size=(1,1,3), padding=0),
            nn.BatchNorm3d(node_encod_dim),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1,1,3), padding=0, stride=(1,1,2))                    
        )

        self.dense = nn.Sequential(
            nn.Linear(node_encod_dim*5, node_encod_dim),
            nn.ReLU()
        )

        self.node_upscaler_cnn = nn.Sequential(
            nn.Conv2d(in_channels=node_encod_dim, out_channels=node_encod_dim*up_factor**2, kernel_size=3, padding=1), # from (B,32,1,1) to (B,32*3**2,1,1)
            nn.Tanh(),
            nn.PixelShuffle(3), # from (B,32*3**2,1,1) to (B, 32, 3, 3)
            nn.Conv2d(in_channels=node_encod_dim, out_channels=64, kernel_size=3, padding=1),                  # (N,Cin,L) = (low_num_nodes,1,45)
            nn.Tanh(),
            nn.Conv2d(in_channels=64, out_channels=node_encod_dim, kernel_size=3, padding=1),
            nn.Tanh(),
        )

        # self.downscaler_low_9x = geometric_nn.Sequential('x, edge_index', [
        #     (GATv2Conv((node_encod_dim, high_in), out_channels=node_encod_dim, dropout=0.2, heads=2, aggr='mean', add_self_loops=False, bias=True), 'x, edge_index -> x'),
        #     nn.ReLU(),
        #     nn.Linear(node_encod_dim*2, node_encod_dim),
        #     nn.ReLU()
        # ])

        self.downscaler = geometric_nn.Sequential('x, edge_index', [
            (GATv2Conv((node_encod_dim, high_in), out_channels=low2high_out, dropout=0.2, heads=2, aggr='mean', add_self_loops=False, bias=True), 'x, edge_index -> x'),
            nn.ReLU(),
            nn.Linear(low2high_out*2, low2high_out),
            nn.ReLU()
        ])

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

    def forward(self, data): # [N,V,L,T] (num_nodes,vars=5,times=25,levels=5)
        encod_low = self.node_encoder_cnn_temporal(data.x_dict['low'].unsqueeze(1)) # from [N,1,V,T,L] to [N,8,V,1,L]
        encod_low = self.node_encoder_cnn_vertical(encod_low) # from [N,8,V,1,L] to [N,32,V,1,1]
        encod_low = self.dense(encod_low.squeeze().flatten(start_dim=1)) # from 32*5=160 to 32
        # encod_low  = self.downscaler_low_9x((encod_low, data['low_9x'].x), data.edge_index_dict[('low','to','low_9x')])
        encod_low = self.node_upscaler_cnn(encod_low.unsqueeze(-1).unsqueeze(-1)).permute(0,2,3,1).flatten(end_dim=2) # from (B, 32, 3, 3) to (B*3*3, 32)
        encod_low2high  = self.downscaler((encod_low, data['high'].x), data.edge_index_dict[('low_9x','to','high')])
        encod_low2high  = torch.concatenate((data['high'].z_std, encod_low2high ),dim=-1)
        encod_high = self.processor(encod_low2high, data.edge_index_dict[('high','within','high')])
        y_pred = self.predictor(encod_high)
        return y_pred


################################
### Separate variables model ###
#############################ù##

class HiResPrecipNet_subpixel(nn.Module):
    
    def __init__(self, low_in=5*5, high_in=1, low2high_out=5*5, high_out=64, c_in=5, r=3):
        super(HiResPrecipNet_subpixel, self).__init__()

        self.node_upscaler_cnn = nn.Sequential(
            nn.Conv2d(c_in,64,5,1,2),
            nn.Tanh(),
            nn.Conv2d(64,32,3,1,1),
            nn.Tanh(),
            nn.Conv2d(32,c_in*(r*r),3,1,1),
            nn.PixelShuffle(r),
            nn.Sigmoid()
        )

        self.downscaler = GATv2Conv((5*5*5, high_in), out_channels=low2high_out, dropout=0.0, heads=1, aggr='mean', add_self_loops=False, bias=True)
        
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
        for g in data:
            x_upscaled = self.node_upscaler_cnn(g['low'].x)
            x_upscaled = x_upscaled.reshape(5,5,x_upscaled.shape[1],x_upscaled.shape[2],x_upscaled.shape[3])
            g['low_9x'].x = torch.permute(x_upscaled, (3,4,0,1,2)).flatten(end_dim=1).flatten(start_dim=1)
        data = Batch.from_data_list(data)
        encod_low2high  = self.downscaler((data.x_dict['low_9x'], data['high'].x), data.edge_index_dict[('low_9x','to','high')])
        encod_low2high  = torch.concatenate((data['high'].z_std, encod_low2high ),dim=-1)
        encod_high = self.processor(encod_low2high , data.edge_index_dict[('high','within','high')])
        y_pred = self.predictor(encod_high)
        return y_pred


class HiResPrecipNet_temporal(nn.Module):
    
    def __init__(self, low_in=5*5, high_in=1, low2high_out=16, high_out=16, periods=25):
        super(HiResPrecipNet_temporal, self).__init__()

        self.tgnn = geometric_nn.Sequential('x, edge_index', [
            (A3TGCN(in_channels=low_in, out_channels=low_in, periods=periods),  'x, edge_index -> x'),
            ])
        
        self.downscaler = GATv2Conv((low_in, high_in), out_channels=low2high_out, dropout=0.2, heads=4, aggr='mean', concat=False, add_self_loops=False, bias=True)
        
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
            nn.BatchNorm1d(high_out),
            nn.Linear(high_out, 8),
            nn.ReLU(),
            nn.BatchNorm1d(8),
            nn.Linear(8, 1)
            )

    def forward(self, data):
        encod_low = self.tgnn(data['low'].x, data['low', 'within', 'low'].edge_index)
        encod_low2high  = self.downscaler((encod_low, data['high'].x), data.edge_index_dict[('low','to','high')])
        encod_low2high  = torch.concatenate((data['high'].z_std, encod_low2high ),dim=-1)
        encod_high = self.processor(encod_low2high , data.edge_index_dict[('high','within','high')])
        y_pred = self.predictor(encod_high)
        return y_pred

class HiResPrecipNet_temporal_5(nn.Module):
    
    def __init__(self, low_in=5*5, high_in=1, low2high_out=16, high_out=16, periods=25):
        super(HiResPrecipNet_temporal, self).__init__()

        self.tgnn_q = geometric_nn.Sequential('x, edge_index', [
            (A3TGCN(in_channels=low_in, out_channels=low_in, periods=periods),  'x, edge_index -> x'),
            ])
        
        self.tgnn_t = geometric_nn.Sequential('x, edge_index', [
            (A3TGCN(in_channels=low_in, out_channels=low_in, periods=periods),  'x, edge_index -> x'),
            ])
        
        self.tgnn_u = geometric_nn.Sequential('x, edge_index', [
            (A3TGCN(in_channels=low_in, out_channels=low_in, periods=periods),  'x, edge_index -> x'),
            ])
        
        self.tgnn_v = geometric_nn.Sequential('x, edge_index', [
            (A3TGCN(in_channels=low_in, out_channels=low_in, periods=periods),  'x, edge_index -> x'),
            ])
        
        self.tgnn_z = geometric_nn.Sequential('x, edge_index', [
            (A3TGCN(in_channels=low_in, out_channels=low_in, periods=periods),  'x, edge_index -> x'),
            ])
        
        self.downscaler = GATv2Conv((low_in, high_in), out_channels=low2high_out, dropout=0.2, heads=4, aggr='mean', concat=False, add_self_loops=False, bias=True)
        
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
            nn.BatchNorm1d(high_out),
            nn.Linear(high_out, 8),
            nn.ReLU(),
            nn.BatchNorm1d(8),
            nn.Linear(8, 1)
            )

    def forward(self, data):
        encod_low_q = self.tgnn(data['low'].x[:,:,0,:], data['low', 'within', 'low'].edge_index)
        encod_low_t = self.tgnn(data['low'].x[:,:,1,:], data['low', 'within', 'low'].edge_index)
        encod_low_u = self.tgnn(data['low'].x[:,:,2,:], data['low', 'within', 'low'].edge_index)
        encod_low_v = self.tgnn(data['low'].x[:,:,3,:], data['low', 'within', 'low'].edge_index)
        encod_low_z = self.tgnn(data['low'].x[:,:,4,:], data['low', 'within', 'low'].edge_index)
        encod_low  = torch.cat((encod_low_q, encod_low_t, encod_low_u, encod_low_v, encod_low_z),dim=-1)
        encod_low2high  = self.downscaler((encod_low, data['high'].x), data.edge_index_dict[('low','to','high')])
        encod_low2high  = torch.cat((data['high'].z_std, encod_low2high ),dim=-1)
        encod_high = self.processor(encod_low2high , data.edge_index_dict[('high','within','high')])
        y_pred = self.predictor(encod_high)
        return y_pred
    

class HiResPrecipNet_superres(nn.Module):
    
    def __init__(self, high_in=1, low2high_out=5*5, high_out=64, c_in=5*5, r=5):
        super(HiResPrecipNet_superres, self).__init__()

        self.pixel_shuffle = nn.PixelShuffle(r)

        self.downscale_q = GCNConv(c_in,c_in*(r*r))

        self.downscale_t = GCNConv(c_in,c_in*(r*r))

        self.downscale_u = GCNConv(c_in,c_in*(r*r))

        self.downscale_v = GCNConv(c_in,c_in*(r*r))

        self.downscale_z = GCNConv(c_in,c_in*(r*r))

        self.to_high = GATv2Conv((5*5*5, high_in), out_channels=low2high_out, dropout=0.4, heads=4, concat=False, aggr='mean', add_self_loops=False, bias=True)
        
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
        # perform downscaling similarly to sub-pixel convolution + pixel shuffle
        data['low'].x = data['low'].x.flatten(start_dim=2)
        encod_low_q = self.pixel_shuffle(self.downscale_q(data['low'].x[:,0,:], data['low', 'within', 'low'].edge_index).unsqueeze(1).unsqueeze(-1).unsqueeze(-1)).flatten(start_dim=3)
        encod_low_t = self.pixel_shuffle(self.downscale_t(data['low'].x[:,1,:], data['low', 'within', 'low'].edge_index).unsqueeze(1).unsqueeze(-1).unsqueeze(-1)).flatten(start_dim=3)
        encod_low_u = self.pixel_shuffle(self.downscale_u(data['low'].x[:,2,:], data['low', 'within', 'low'].edge_index).unsqueeze(1).unsqueeze(-1).unsqueeze(-1)).flatten(start_dim=3)
        encod_low_v = self.pixel_shuffle(self.downscale_v(data['low'].x[:,3,:], data['low', 'within', 'low'].edge_index).unsqueeze(1).unsqueeze(-1).unsqueeze(-1)).flatten(start_dim=3)
        encod_low_z = self.pixel_shuffle(self.downscale_z(data['low'].x[:,4,:], data['low', 'within', 'low'].edge_index).unsqueeze(1).unsqueeze(-1).unsqueeze(-1)).flatten(start_dim=3)
        encod_low_q = torch.permute(encod_low_q, (0,3,1,2)).flatten(end_dim=1).flatten(start_dim=1)
        encod_low_t = torch.permute(encod_low_t, (0,3,1,2)).flatten(end_dim=1).flatten(start_dim=1)
        encod_low_u = torch.permute(encod_low_u, (0,3,1,2)).flatten(end_dim=1).flatten(start_dim=1)
        encod_low_v = torch.permute(encod_low_v, (0,3,1,2)).flatten(end_dim=1).flatten(start_dim=1)
        encod_low_z = torch.permute(encod_low_z, (0,3,1,2)).flatten(end_dim=1).flatten(start_dim=1)
        # concat the downscaled encodings
        encod_low  = torch.cat((encod_low_q, encod_low_t, encod_low_u, encod_low_v, encod_low_z),dim=-1)
        encod_low2high  = self.to_high((encod_low, data['high'].z_std), data.edge_index_dict[('low_25x','to','high')])
        encod_low2high  = torch.concatenate((data['high'].z_std, encod_low2high ),dim=-1)
        encod_high = self.processor(encod_low2high , data.edge_index_dict[('high','within','high')])
        y_pred = self.predictor(encod_high)
        return y_pred


class HiResPrecipNet_temporal_superres(nn.Module):
    
    def __init__(self, high_in=1, low2high_out=5*5, high_out=64, c_in=5*5, r=5, periods=25):
        super(HiResPrecipNet_temporal_superres, self).__init__()

        self.pixel_shuffle = nn.PixelShuffle(r)

        self.tgnn = geometric_nn.Sequential('x, edge_index', [
            (A3TGCN(in_channels=c_in, out_channels=c_in*(r*r), periods=periods),  'x, edge_index -> x'),
            ])

        self.to_high = GATv2Conv((5*5, high_in), out_channels=low2high_out, dropout=0.4, heads=4, concat=False, aggr='mean', add_self_loops=False, bias=True)
        
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
        encod_low = self.tgnn(data['low'].x, data['low', 'within', 'low'].edge_index) # num_nodes, time, var*level
        # perform downscaling similarly to sub-pixel convolution + pixel shuffle
        encod_low = encod_low.flatten(start_dim=1).unsqueeze(-1).unsqueeze(-1)                     # (num_nodes,1*var*level*r*r,1,1)
        encod_low = self.pixel_shuffle(encod_low)                                                  # (num_nodes,1*var*lev,r,r)
        encod_low = torch.permute(encod_low, (0,2,3,1)).flatten(end_dim=2).flatten(start_dim=1)    # (num_nodes*r*r,time*var*lev)
        encod_low2high  = self.to_high((encod_low, data['high'].z_std), data.edge_index_dict[('low_25x','to','high')])
        encod_low2high  = torch.concatenate((data['high'].z_std, encod_low2high ),dim=-1)
        encod_high = self.processor(encod_low2high , data.edge_index_dict[('high','within','high')])
        y_pred = self.predictor(encod_high)
        return y_pred


