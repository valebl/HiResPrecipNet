import torch.nn as nn
import torch_geometric.nn as geometric_nn
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn import global_mean_pool
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


