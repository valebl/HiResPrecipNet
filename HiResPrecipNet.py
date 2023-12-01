import torch.nn as nn
import torch_geometric.nn as geometric_nn
from torch_geometric.nn import GATv2Conv, GCN, HeteroConv, SAGEConv
import torch

class Low_within_layer(nn.Module):
    
    def __init__(self, in_channels, out_channels, dropout=0.0, heads=1, aggr='mean', model_type='GATv2Conv'):
        super(Low_within_layer, self).__init__()
        if model_type == 'GATv2Conv':
            self.nn = GATv2Conv(in_channels, out_channels, dropout=dropout, heads=heads, aggr=aggr, add_self_loops=False)
        elif model_type == 'GCN':
            self.nn = GCN(in_channels, out_channels)
    
    def forward(self, x, edge_index):
        return self.nn(x, edge_index)


class High_within_layer(nn.Module):
    
    def __init__(self, in_channels, out_channels, dropout=0.0, heads=1, aggr='mean', model_type='GATv2Conv'):
        super(High_within_layer, self).__init__()
        if model_type == 'GATv2Conv':
            self.nn = GATv2Conv(in_channels, out_channels, dropout=dropout, heads=heads, aggr=aggr, add_self_loops=False)
        elif model_type == 'GCN':
            self.nn = GCN(in_channels, out_channels)
    
    def forward(self, x, edge_index):
        return self.nn(x, edge_index)


class Low_to_high_layer(nn.Module):
    
    def __init__(self, in_channels, out_channels, dropout=0.0, heads=1, aggr='mean', model_type='HeteroConv'):
        super(Low_to_high_layer, self).__init__()
        if model_type == 'HeteroConv':
            self.nn = HeteroConv({
                    ('low', 'to', 'high'): GATv2Conv(in_channels, out_channels, dropout=dropout, heads=heads, aggr=aggr, add_self_loops=False),
                    }, aggr='sum')
    
    def forward(self, x, edge_index):
        return self.nn(x, edge_index)


class HiResPrecipNet(nn.Module):
    
    def __init__(self):
        super(HiResPrecipNet, self).__init__()

        self.low2high = GATv2Conv((512,0), out_channels=128, dropout=0, heads=1, aggr='mean', add_self_loops=False, bias=False)

        self.low_net = geometric_nn.Sequential('x, edge_index', [
            (geometric_nn.BatchNorm(25*5*5), 'x -> x'),
            (High_within_layer(in_channels=25*5*5, out_channels=512), 'x, edge_index -> x'),
            (geometric_nn.BatchNorm(512), 'x -> x'), 
            nn.ReLU(),
            (High_within_layer(in_channels=512, out_channels=512),'x, edge_index -> x'),
            (geometric_nn.BatchNorm(512), 'x -> x'),
            nn.ReLU(),
            (High_within_layer(in_channels=512, out_channels=512), 'x, edge_index -> x'),
            ])
        
        self.high_net = geometric_nn.Sequential('x, edge_index', [
            (geometric_nn.BatchNorm(129), 'x -> x'),
            (High_within_layer(in_channels=129, out_channels=128, heads=2, dropout=0.5), 'x, edge_index -> x'),
            (geometric_nn.BatchNorm(256), 'x -> x'), 
            nn.ReLU(),
            (High_within_layer(in_channels=256, out_channels=128),'x, edge_index -> x'),
            (geometric_nn.BatchNorm(128), 'x -> x'),
            nn.ReLU(),
            (High_within_layer(in_channels=128, out_channels=1), 'x, edge_index -> x'),
            ])

    def forward(self, data):
        
        encod_low = self.low_net(data.x_dict['low'], data.edge_index_dict[('low','within','low')])
        encod_high = self.low2high((encod_low, data.x_dict['high']), data.edge_index_dict[('low','to','high')])
        encod_high = torch.concatenate((data['high'].z_std, encod_high),dim=-1)
        y_pred = self.high_net(encod_high, data.edge_index_dict[('high','within','high')])
        return y_pred