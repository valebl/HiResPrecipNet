import torch.nn as nn
import torch_geometric.nn as geometric_nn
from torch_geometric.nn import GATv2Conv, HeteroConv, GCNConv, SAGEConv, GraphConv
from torch_geometric.nn import global_mean_pool
import numpy as np
import torch

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


# class HiResPrecipNet(nn.Module):
    
#     def __init__(self, low_in=5*5*5, high_in=1, low2high_out=64, high_out=64):
#         super(HiResPrecipNet, self).__init__()

#         self.downscaler = GATv2Conv((low_in, high_in), out_channels=low2high_out, dropout=0.0, heads=1, aggr='mean', add_self_loops=False, bias=True)
        
#         self.processor = geometric_nn.Sequential('x, edge_index', [
#             (geometric_nn.BatchNorm(low2high_out+1+6), 'x -> x'),
#             (GATv2Conv(in_channels=low2high_out+1+6, out_channels=high_out, heads=2, dropout=0.2, aggr='mean', add_self_loops=True, bias=True), 'x, edge_index -> x'),
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
#             nn.ReLU(),
#             ])
    
#         self.predictor = nn.Sequential(
#             nn.Linear(high_out, high_out),
#             nn.ReLU(),
#             nn.Linear(high_out, 32),
#             nn.ReLU(),
#             nn.Linear(32, 1)
#             )

#     def forward(self, data):        
#         encod_low2high  = self.downscaler((data.x_dict['low'], data['high'].x), data.edge_index_dict[('low','to','high')])
#         encod_low2high  = torch.concatenate((data['high'].z_std, data['high'].land_std, encod_low2high),dim=-1)
# #        encod_low2high  = torch.concatenate((data['high'].z_std, encod_low2high ),dim=-1)
# #        return encod_low2high
#         encod_high = self.processor(encod_low2high , data.edge_index_dict[('high','within','high')])
#         y_pred = self.predictor(encod_high)
#         return y_pred
        

#### Modello che al momento sembra funzionare meglio, con RNN e land-use
class HiResPrecipNet_TEST(nn.Module):
    
    def __init__(self, encoding_dim=128, seq_l=25, h_in=5*5, h_hid=5*5, n_layers=2, high_in=6+1, low2high_out=64, high_out=64):
        super(HiResPrecipNet_TEST, self).__init__()

        # input shape (N,L,Hin)
        self.rnn = nn.Sequential(
            nn.GRU(h_in, h_hid, n_layers, batch_first=True),
        )

        self.dense = nn.Sequential(
            nn.Linear(h_in*seq_l, encoding_dim),
            nn.ReLU()
        )

        self.downscaler = geometric_nn.Sequential('x, edge_index', [
            (GraphConv((encoding_dim, high_in), out_channels=low2high_out, dropout=0.2, heads=1, aggr='mean', add_self_loops=False, bias=True), 'x, edge_index -> x')
            ])
        
        self.processor = geometric_nn.Sequential('x, edge_index', [
            (geometric_nn.BatchNorm(low2high_out), 'x -> x'),
            (GATv2Conv(in_channels=low2high_out, out_channels=high_out, heads=2, dropout=0.2, aggr='mean', add_self_loops=True, bias=True), 'x, edge_index -> x'),
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
        encod_rnn, _ = self.rnn(data.x_dict['low']) # out, h
        encod_rnn = encod_rnn.flatten(start_dim=1)
        encod_rnn = self.dense(encod_rnn)
        x_zland  = torch.concatenate((data['high'].z_std, data['high'].land_std),dim=-1)
        encod_low2high  = self.downscaler((encod_rnn, x_zland), data["low", "to", "high"].edge_index)
#        return encod_low2high
        encod_high = self.processor(encod_low2high , data.edge_index_dict[('high','within','high')])
        y_pred = self.predictor(encod_high)
        return y_pred


class HiResPrecipNet_TEST_2(nn.Module):
    
    def __init__(self, encoding_dim=128, seq_l=25, h_in=5*5, h_hid=5*5, n_layers=2, high_in=6+1, low2high_out=64, high_out=32):
        super(HiResPrecipNet_TEST_2, self).__init__()

        # input shape (N,L,Hin)
        self.rnn = nn.Sequential(
            nn.GRU(h_in, h_hid, n_layers, batch_first=True),
        )

        self.dense = nn.Sequential(
            nn.Linear(h_in*seq_l, encoding_dim),
            nn.ReLU()
        )

        self.downscaler = geometric_nn.Sequential('x, edge_index', [
            (GraphConv((encoding_dim, high_in), out_channels=low2high_out, dropout=0.2, heads=1, aggr='mean', add_self_loops=False, bias=True), 'x, edge_index -> x')
            ])
        
        self.processor = geometric_nn.Sequential('x, edge_index', [
            (geometric_nn.BatchNorm(low2high_out), 'x -> x'),
            (GATv2Conv(in_channels=low2high_out, out_channels=high_out, heads=2, dropout=0.2, aggr='mean', add_self_loops=True, bias=True), 'x, edge_index -> x'),
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
            nn.Linear(high_out, 1),
            )

    def forward(self, data):   
        encod_rnn, _ = self.rnn(data.x_dict['low']) # out, h
        encod_rnn = encod_rnn.flatten(start_dim=1)
        encod_rnn = self.dense(encod_rnn)
        x_zland  = torch.concatenate((data['high'].z_std, data['high'].land_std),dim=-1)
        encod_low2high  = self.downscaler((encod_rnn, x_zland), data["low", "to", "high"].edge_index)
#        return encod_low2high
        encod_high = self.processor(encod_low2high , data.edge_index_dict[('high','within','high')])
        y_pred = self.predictor(encod_high)
        return y_pred



class HiResPrecipNet_TEST_3(nn.Module):
    
    def __init__(self, encoding_dim=32, seq_l=25, h_in=5*5, h_hid=5*5, n_layers=1, high_in=6+1, low2high_out=16, high_out=16):
        super(HiResPrecipNet_TEST_3, self).__init__()

        # input shape (N,L,Hin)
        self.rnn = nn.Sequential(
            nn.GRU(h_in, h_hid, n_layers, batch_first=True),
        )

        self.dense = nn.Sequential(
            nn.Linear(h_in*seq_l, encoding_dim),
            nn.ReLU(),
            nn.BatchNorm1d(encoding_dim),  # Added batch normalization
            nn.Dropout(p=0.5)  # Added dropout
            )

        self.downscaler = geometric_nn.Sequential('x, edge_index', [
            (GraphConv((encoding_dim, high_in), out_channels=low2high_out, dropout=0.5, heads=1, aggr='mean', add_self_loops=False, bias=True), 'x, edge_index -> x')
            ])
        
        self.processor = geometric_nn.Sequential('x, edge_index', [
            (geometric_nn.BatchNorm(low2high_out), 'x -> x'),
            (GATv2Conv(in_channels=low2high_out, out_channels=high_out, heads=1, dropout=0.5, aggr='mean', add_self_loops=True, bias=True), 'x, edge_index -> x'),
            (geometric_nn.BatchNorm(high_out), 'x -> x'), 
            nn.ReLU(),
            (GATv2Conv(in_channels=high_out, out_channels=high_out, heads=1, dropout=0.5, aggr='mean', add_self_loops=True, bias=True),'x, edge_index -> x'),
            (geometric_nn.BatchNorm(high_out), 'x -> x'),
            nn.ReLU(),
            (GATv2Conv(in_channels=high_out, out_channels=high_out, heads=1, dropout=0.5, aggr='mean', add_self_loops=True, bias=True),'x, edge_index -> x'),
            (geometric_nn.BatchNorm(high_out), 'x -> x'),
            nn.ReLU(),
            (GATv2Conv(in_channels=high_out, out_channels=high_out, heads=1, dropout=0.5, aggr='mean', add_self_loops=True, bias=True),'x, edge_index -> x'),
            (geometric_nn.BatchNorm(high_out), 'x -> x'),
            nn.ReLU(),
            (GATv2Conv(in_channels=high_out, out_channels=high_out, heads=1, dropout=0.5, aggr='mean', add_self_loops=True, bias=True), 'x, edge_index -> x'),
            nn.ReLU(),
            ])
    
        #self.postprocessor = nn.Sequential(
        #    nn.Linear(high_out, high_out),
        #    nn.ReLU(),
        #    nn.Dropout(p=0.5),  # Added dropout
        #    nn.Linear(high_out, high_out // 2),
        #    nn.ReLU()
        #    )

        self.predictor = nn.Sequential(
            nn.Linear(high_out, 1)
            )

    def forward(self, data):   
        encod_rnn, _ = self.rnn(data.x_dict['low']) # out, h
        encod_rnn = encod_rnn.flatten(start_dim=1)
        encod_rnn = self.dense(encod_rnn)
        x_zland  = torch.concatenate((data['high'].z_std, data['high'].land_std),dim=-1)
        encod_low2high  = self.downscaler((encod_rnn, x_zland), data["low", "to", "high"].edge_index)
#        return encod_low2high
        encod_high = self.processor(encod_low2high , data.edge_index_dict[('high','within','high')])
        #encod_high = self.postprocessor(encod_high)
        y_pred = self.predictor(encod_high)
        return y_pred


class HiResPrecipNet_TEST_skip(nn.Module):
    
    def __init__(self, encoding_dim=128, seq_l=25, h_in=5*5, h_hid=5*5, n_layers=2, high_in=6+1, low2high_out=64, high_out=64):
        super(HiResPrecipNet_TEST_skip, self).__init__()

        # input shape (N,L,Hin)
        self.rnn = nn.Sequential(
            nn.GRU(h_in, h_hid, n_layers, batch_first=True),
        )

        self.dense = nn.Sequential(
            nn.Linear(h_in*seq_l, encoding_dim),
            nn.ReLU()
        )

        self.downscaler = geometric_nn.Sequential('x, edge_index', [
            (GraphConv((encoding_dim, high_in), out_channels=low2high_out, dropout=0.2, heads=1, aggr='mean', add_self_loops=False, bias=True), 'x, edge_index -> x')
            ])
        
        self.processor = geometric_nn.Sequential('x, edge_index', [
            (geometric_nn.BatchNorm(low2high_out), 'x -> x'),
            (GATv2Conv(in_channels=low2high_out, out_channels=high_out, heads=2, dropout=0.2, aggr='mean', add_self_loops=True, bias=True), 'x, edge_index -> x'),
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
            nn.Linear(high_out+1, high_out),
            nn.ReLU(),
            nn.Linear(high_out, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
            )

    def forward(self, data):   
        encod_rnn, _ = self.rnn(data.x_dict['low']) # out, h
        encod_rnn = encod_rnn.flatten(start_dim=1)
        encod_rnn = self.dense(encod_rnn)
        x_zland  = torch.concatenate((data['high'].z_std, data['high'].land_std),dim=-1)
        encod_low2high  = self.downscaler((encod_rnn, x_zland), data["low", "to", "high"].edge_index)
#        return encod_low2high
        encod_high = self.processor(encod_low2high , data.edge_index_dict[('high','within','high')])
        encod_high = torch.concatenate((data['high'].z_std, encod_high),dim=-1)
        y_pred = self.predictor(encod_high)
        return y_pred

