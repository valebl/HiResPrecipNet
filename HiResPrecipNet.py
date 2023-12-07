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
    
    def __init__(self, low_in=25*5*5, high_in=0, low_out=512, low2high_out=128, high_out=128):
        super(HiResPrecipNet, self).__init__()

        self.low2high = GATv2Conv((low_out,high_in), out_channels=low2high_out, dropout=0, heads=1, aggr='mean', add_self_loops=False, bias=False)

        self.low_net = geometric_nn.Sequential('x, edge_index', [
            (geometric_nn.BatchNorm(low_in), 'x -> x'),
            (High_within_layer(in_channels=low_in, out_channels=low_out), 'x, edge_index -> x'),
            (geometric_nn.BatchNorm(low_out), 'x -> x'), 
            nn.ReLU(),
            (High_within_layer(in_channels=low_out, out_channels=low_out),'x, edge_index -> x'),
            (geometric_nn.BatchNorm(low_out), 'x -> x'),
            nn.ReLU(),
            (High_within_layer(in_channels=low_out, out_channels=low_out), 'x, edge_index -> x'),
            ])
        
        self.high_net = geometric_nn.Sequential('x, edge_index', [
            (geometric_nn.BatchNorm(low2high_out+1), 'x -> x'),
            (High_within_layer(in_channels=low2high_out+1, out_channels=high_out, heads=2, dropout=0.5), 'x, edge_index -> x'),
            (geometric_nn.BatchNorm(high_out*2), 'x -> x'), 
            nn.ReLU(),
            (High_within_layer(in_channels=high_out*2, out_channels=high_out),'x, edge_index -> x'),
            (geometric_nn.BatchNorm(high_out), 'x -> x'),
            nn.ReLU(),
            (High_within_layer(in_channels=high_out, out_channels=1), 'x, edge_index -> x'),
            ])

    def forward(self, data):
        
        encod_low = self.low_net(data.x_dict['low'], data.edge_index_dict[('low','within','low')])
        encod_high = self.low2high((encod_low, data.x_dict['high']), data.edge_index_dict[('low','to','high')])
        encod_high = torch.concatenate((data['high'].z_std, encod_high),dim=-1)
        y_pred = self.high_net(encod_high, data.edge_index_dict[('high','within','high')])
        return y_pred
    
    # def forward(self, data):
        
    #     encod_low = self.low_net(data.x_dict['low'], data.edge_index_dict[('low','within','low')])
    #     encod_high = self.low2high((encod_low, data['high'].z_std), data.edge_index_dict[('low','to','high')])
    #     #encod_high = torch.concatenate((data['high'].z_std, encod_high),dim=-1)
    #     y_pred = self.high_net(encod_high, data.edge_index_dict[('high','within','high')])
    #     return y_pred

class HiResPrecipNet_wce(HiResPrecipNet):
    def __init__(self):
        super().__init__()
        
        self.high_net = geometric_nn.Sequential('x, edge_index', [
            (geometric_nn.BatchNorm(129), 'x -> x'),
            (High_within_layer(in_channels=129, out_channels=128, heads=2, dropout=0.5), 'x, edge_index -> x'),
            (geometric_nn.BatchNorm(256), 'x -> x'),
            nn.ReLU(),
            (High_within_layer(in_channels=256, out_channels=128),'x, edge_index -> x'),
            (geometric_nn.BatchNorm(128), 'x -> x'),
            nn.ReLU(),
            (High_within_layer(in_channels=128, out_channels=2), 'x, edge_index -> x'),
            ])


class HiResPrecipNet_Combined(nn.Module):

    def __init__(self, low_in=25*5*5, high_in=0, low_out=512, low2high_out=128, high_out=128):
        super(HiResPrecipNet_Combined, self).__init__()

        self.low2high = GATv2Conv((low_out,high_in), out_channels=low2high_out, dropout=0, heads=1, aggr='mean', add_self_loops=False, bias=False)

        self.low_net = geometric_nn.Sequential('x, edge_index', [
            (geometric_nn.BatchNorm(low_in), 'x -> x'),
            (High_within_layer(in_channels=low_in, out_channels=low_out), 'x, edge_index -> x'),
            (geometric_nn.BatchNorm(low_out), 'x -> x'), 
            nn.ReLU(),
            (High_within_layer(in_channels=low_out, out_channels=low_out),'x, edge_index -> x'),
            (geometric_nn.BatchNorm(low_out), 'x -> x'),
            nn.ReLU(),
            (High_within_layer(in_channels=low_out, out_channels=low_out), 'x, edge_index -> x'),
            ])
        
        self.high_net_cl = geometric_nn.Sequential('x, edge_index', [
            (geometric_nn.BatchNorm(low2high_out+1), 'x -> x'),
            (High_within_layer(in_channels=low2high_out+1, out_channels=high_out, heads=2, dropout=0.5), 'x, edge_index -> x'),
            (geometric_nn.BatchNorm(high_out*2), 'x -> x'), 
            nn.ReLU(),
            (High_within_layer(in_channels=high_out*2, out_channels=high_out),'x, edge_index -> x'),
            (geometric_nn.BatchNorm(high_out), 'x -> x'),
            nn.ReLU(),
            (High_within_layer(in_channels=high_out, out_channels=1), 'x, edge_index -> x'),
            ])
        
        self.high_net_reg = geometric_nn.Sequential('x, edge_index', [
            (geometric_nn.BatchNorm(low2high_out+1), 'x -> x'),
            (High_within_layer(in_channels=low2high_out+1, out_channels=high_out, heads=2, dropout=0.5), 'x, edge_index -> x'),
            (geometric_nn.BatchNorm(high_out*2), 'x -> x'), 
            nn.ReLU(),
            (High_within_layer(in_channels=high_out*2, out_channels=high_out),'x, edge_index -> x'),
            (geometric_nn.BatchNorm(high_out), 'x -> x'),
            nn.ReLU(),
            (High_within_layer(in_channels=high_out, out_channels=1), 'x, edge_index -> x'),
            ])

    def forward(self, data):
        
        encod_low = self.low_net(data.x_dict['low'], data.edge_index_dict[('low','within','low')])
        encod_high_cl = self.low2high((encod_low, data.x_dict['high']), data.edge_index_dict[('low','to','high')])
        encod_high_reg = self.low2high((encod_low, data.x_dict['high']), data.edge_index_dict[('low','to','high')])
        encod_high_cl = torch.concatenate((data['high'].z_std, encod_high_cl),dim=-1)
        encod_high_reg = torch.concatenate((data['high'].z_std, encod_high_reg),dim=-1)
        y_pred_cl = self.high_net_cl(encod_high_cl, data.edge_index_dict[('high','within','high')])
        y_pred_reg = self.high_net_reg(encod_high_reg, data.edge_index_dict[('high','within','high')])
        return y_pred_cl, y_pred_reg
