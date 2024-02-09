import torch.nn as nn
import torch_geometric.nn as geometric_nn
from torch_geometric.nn import GATv2Conv, GCN, HeteroConv, SAGEConv
import torch


class HiResPrecipNet(nn.Module):
    
    def __init__(self, low_in=5*5*5, high_in=1, low2high_out=256, high_out=64):
        super(HiResPrecipNet, self).__init__()

        self.downscaler = GATv2Conv((low_in, high_in), out_channels=low2high_out, dropout=0.6, heads=1, aggr='mean', add_self_loops=False, bias=True)
        
        self.processor = geometric_nn.Sequential('x, edge_index', [
            (geometric_nn.BatchNorm(low2high_out+1), 'x -> x'),
            (GATv2Conv(in_channels=low2high_out+1, out_channels=high_out, heads=4, dropout=0.6, aggr='mean', add_self_loops=True, bias=True), 'x, edge_index -> x'),
            (geometric_nn.BatchNorm(high_out*4), 'x -> x'), 
            nn.ReLU(),
            (GATv2Conv(in_channels=high_out*4, out_channels=high_out, heads=4, dropout=0.6, aggr='mean', add_self_loops=True, bias=True),'x, edge_index -> x'),
            (geometric_nn.BatchNorm(high_out*4), 'x -> x'),
            nn.ReLU(),
            (GATv2Conv(in_channels=high_out*4, out_channels=high_out, heads=1, dropout=0.0, aggr='mean', add_self_loops=True, bias=True),'x, edge_index -> x'),
            (geometric_nn.BatchNorm(high_out), 'x -> x'),
            nn.ReLU(),
            ])
        
        self.predictor = nn.Sequential(
            nn.Linear(high_out, high_out),
            nn.ReLU(),
            nn.Linear(high_out, 1)
        )
        
        
        #GATv2Conv(in_channels=high_out*4, out_channels=1, heads=1, dropout=0.0, aggr='mean', add_self_loops=True, bias=True)

    def forward(self, data):
        
        encod_low2high = self.downscaler((data.x_dict['low'], data['high'].x), data.edge_index_dict[('low','to','high')])
        encod_low2high = torch.concatenate((data['high'].z_std, encod_low2high),dim=-1)
        encod_high = self.processor(encod_low2high, data.edge_index_dict[('high','within','high')])
        y_pred = self.predictor(encod_high)
        return y_pred

## Experiments
    
class HiResPrecipNet_fl_2_32_smaller(nn.Module):
    
    def __init__(self, low_in=5*5*5, high_in=1, low2high_out=32, high_out=32):
        super(HiResPrecipNet_fl_2_32_smaller, self).__init__()

        self.low2high = GATv2Conv((low_in, high_in), out_channels=low2high_out, dropout=0.0, heads=1, aggr='mean', add_self_loops=False, bias=True)
        
        self.high_net = geometric_nn.Sequential('x, edge_index', [
            (geometric_nn.BatchNorm(low2high_out+1), 'x -> x'),
            (GATv2Conv(in_channels=low2high_out+1, out_channels=high_out, heads=1, dropout=0.4, aggr='mean', add_self_loops=True, bias=True), 'x, edge_index -> x'),
            (geometric_nn.BatchNorm(high_out), 'x -> x'), 
            nn.ReLU(),
            (GATv2Conv(in_channels=high_out, out_channels=high_out, heads=1, dropout=0.4, aggr='mean', add_self_loops=True, bias=True),'x, edge_index -> x'),
            (geometric_nn.BatchNorm(high_out), 'x -> x'),
            nn.ReLU(),
            (GATv2Conv(in_channels=high_out, out_channels=high_out, heads=1, dropout=0.4, aggr='mean', add_self_loops=True, bias=True),'x, edge_index -> x'),
            (geometric_nn.BatchNorm(high_out), 'x -> x'),
            nn.ReLU(),
            nn.Linear(high_out, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
            ])

    def forward(self, data):
        
        encod_high = self.low2high((data.x_dict['low'], data['high'].x), data.edge_index_dict[('low','to','high')])
        encod_high = torch.concatenate((data['high'].z_std, encod_high),dim=-1)
        y_pred = self.high_net(encod_high, data.edge_index_dict[('high','within','high')])
        return y_pred
