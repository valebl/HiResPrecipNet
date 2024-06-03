import time
import sys
import pickle
import torch.nn as nn
import numpy as np

import torch

from datetime import datetime, timedelta, date
from torch_geometric.transforms import ToDevice
#from pytorch_forecasting.metrics.quantile import QuantileLoss

from datetime import datetime, date

import torch
import torch.nn.functional as F

from torch.autograd import Variable

from torchvision.ops import sigmoid_focal_loss

######################################################
#------------------ GENERAL UTILITIES ---------------
######################################################


def write_log(s, args, mode='a'):
    with open(args.output_path + args.log_file, mode) as f:
        f.write(s)


def use_gpu_if_possible():
    return "cuda:0" if torch.cuda.is_available() else "cpu"


######################################################
#--------------- PREPROCESSING UTILITIES -------------
######################################################


def cut_window(lon_min, lon_max, lat_min, lat_max, lon, lat, z, pr, mask_land=None):
    r'''
    Derives a new version of the longitude, latitude and precipitation
    tensors, by only retaining the values inside the specified lon-lat rectangle
    Args:
        lon_min, lon_max, lat_min, lat_max: integers
        lon, lat, z, pr: tensors
    Returns:
        The new tensors with the selected values
    '''

    bool_lon = torch.logical_and(lon >= lon_min, lon <= lon_max)
    bool_lat = torch.logical_and(lat >= lat_min, lat <= lat_max)
    bool_both = torch.logical_and(bool_lon, bool_lat)
    lon_sel = lon[bool_both]
    lat_sel = lat[bool_both]
    z_sel = z[bool_both]
    pr_sel = pr[:,bool_both]
    if mask_land is None:
        return lon_sel, lat_sel, z_sel, pr_sel, None
    else:
        mask_land = mask_land[bool_both]
        return lon_sel, lat_sel, z_sel, pr_sel, mask_land


def retain_valid_nodes(lon, lat, pr, z, mask_land=None):

    if mask_land is None:
        valid_nodes = ~torch.isnan(pr).all(dim=0)
    else:
        valid_nodes = ~torch.isnan(mask_land)
    lon = lon[valid_nodes]
    lat = lat[valid_nodes]
    pr = pr[:,valid_nodes] 
    z = z[valid_nodes]
    return lon, lat, pr, z

def select_nodes(lon_centre, lat_centre, lon, lat, pr, z, cell_idx, cell_idx_array, mask_1_cell_subgraphs,
        lon_lat_z_graph, pr_graph, count_points, progressive_idx, offset=0.25, offset_9=0.25):
    
    r'''
    Creates the single cell data structure, by only retaining the values
    correspondent to the nodes that fall inside the considered cell, which
    is identified by its centre lon and lat values and a specified offset
    Args:
        lon_centre, lat_centre = integers
        lon, lat, pr = tensors
        cell_idx:
        cell_idx_array:
        offset, offset_9: integers
        mask_1_cell_subgraphs, mask_9_cells_subgraphs: lists
    Returns:

    '''
    bool_lon = np.logical_and(lon >= lon_centre, lon <= lon_centre+offset)
    bool_lat = np.logical_and(lat >= lat_centre, lat <= lat_centre+offset)
    bool_both = np.logical_and(bool_lon, bool_lat)
    progressive_idx_end = progressive_idx + bool_both.sum()
    lon_lat_z_graph[0,progressive_idx:progressive_idx_end] = lon[bool_both]
    lon_lat_z_graph[1,progressive_idx:progressive_idx_end] = lat[bool_both]
    lon_lat_z_graph[2,progressive_idx:progressive_idx_end] = z[bool_both]
    pr_graph[:,progressive_idx:progressive_idx_end] = pr[:,bool_both]
    cell_idx_array[progressive_idx:progressive_idx_end] = cell_idx
    bool_both = cell_idx_array == cell_idx
    mask_1_cell_subgraphs[cell_idx, :] = bool_both
    count_points.append([cell_idx, bool_both.sum()])
    flag_valid_example = False
    for i in torch.argwhere(bool_both):
        if np.all(torch.isnan(pr_graph[:,i])):
            cell_idx_array[i] *= -1
        else:
            flag_valid_example = True
    return cell_idx_array, flag_valid_example, mask_1_cell_subgraphs, lon_lat_z_graph, pr_graph, count_points, progressive_idx_end


def derive_edge_indexes_within(lon_radius, lat_radius, lon_n1 ,lat_n1, lon_n2, lat_n2):
    r'''
    Args:
        lon_n1 (torch.tensor): longitudes of all first nodes in the edges
        lat_n1 (torch.tensor): latitudes of all fisrt nodes in the edges
        lon_n2 (torch.tensor): longitudes of all second nodes in the edges
        lat_n2 (torch.tensor): latitudes of all second nodes in the edges
        
    '''

    edge_indexes = []

    lonlat_n1 = torch.concatenate((lon_n1.unsqueeze(-1), lat_n1.unsqueeze(-1)),dim=-1)
    lonlat_n2 = torch.concatenate((lon_n2.unsqueeze(-1), lat_n2.unsqueeze(-1)),dim=-1)

    for ii, xi in enumerate(lonlat_n1):
        
        bool_lon = abs(lon_n2 - xi[0]) < lon_radius
        bool_lat = abs(lat_n2 - xi[1]) < lat_radius
        bool_both = torch.logical_and(bool_lon, bool_lat).bool()
        jj_list = torch.nonzero(bool_both)
        xj_list = lonlat_n2[bool_both]
        for jj, xj in zip(jj_list, xj_list):
            if not torch.equal(xi, xj):
                edge_indexes.append(torch.tensor([ii, jj]))

    edge_indexes = torch.stack(edge_indexes)

    return edge_indexes


def derive_edge_indexes_low2high(lon_n1 ,lat_n1, lon_n2, lat_n2, n_knn=9, undirected=False):
    
    edge_index = []

    lonlat_n1 = torch.concatenate((lon_n1.unsqueeze(-1), lat_n1.unsqueeze(-1)),dim=-1)
    lonlat_n2 = torch.concatenate((lon_n2.unsqueeze(-1), lat_n2.unsqueeze(-1)),dim=-1)

    dist = torch.cdist(lonlat_n2, lonlat_n1, p=2)
    _ , knn = dist.topk(n_knn, largest=False, dim=-1)

    for n_n2 in range(lonlat_n2.shape[0]):
        for n_n1 in knn[n_n2,:]:
            edge_index.append(torch.tensor([n_n1, n_n2]))
            if undirected:
                edge_index.append(torch.tensor([n_n2, n_n1]))

    edge_index = torch.stack(edge_index)

    return edge_index


def date_to_idxs(year_start, month_start, day_start, year_end, month_end, day_end,
                 first_year, first_month=1, first_day=1):
    r'''
    Cmputes the start and end idxs crrespnding to the specified period, with respect to a
    reference date.
    Args:
        year_start (int): year at which period starts
        month_start (int): month at which period starts
        day_start (int): day at which period starts
        year_end (int): year at which period ends
        month_end (int): month at which period ends
        day_end (int): day at which period ends
        first_year (int): reference year to compute the idxs
    Returns:
        The start and end idxs for the period
    '''

    start_idx = (date(int(year_start), int(month_start), int(day_start)) - date(int(first_year), int(first_month), int(first_day))).days * 24
    end_idx = (date(int(year_end), int(month_end), int(day_end)) - date(int(first_year), int(first_month), int(first_day))).days * 24 + 24

    return start_idx, end_idx


######################################################
#------------------- TRAIN UTILITIES -----------------
######################################################


#-----------------------------------------------------
#---------------------- METRICS ----------------------
#-----------------------------------------------------


class AverageMeter(object):
    '''
    a generic class to keep track of performance metrics during training or testing of models
    (from the Deep Learning tutorials of DSSC)
    '''
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy_binary_one(prediction, target):
    prediction_class = torch.where(prediction > 0.0, 1.0, 0.0) 
    correct_items = (prediction_class == target)
    acc = correct_items.sum().item() / prediction.shape[0]  
    return acc


def accuracy_binary_one_classes(prediction, target):
    prediction_class = torch.where(prediction > 0.0, 1.0, 0.0)
    correct_items = prediction_class == target
    correct_items_class0 = correct_items[target==0.0]
    if correct_items_class0.shape[0] > 0:
        acc_class0 = correct_items_class0.sum().item() / correct_items_class0.shape[0]
    else:
        acc_class0 = torch.nan
    correct_items_class1 = correct_items[target==1.0]
    if correct_items_class1.shape[0] > 0:
        acc_class1 = correct_items_class1.sum().item() / correct_items_class1.shape[0]
    else:
        acc_class1 = torch.nan
    return acc_class0, acc_class1


def accuracy_binary_two(prediction, target):
    prediction = torch.nn.functional.softmax(prediction, dim=-1)
    prediction_class = torch.argmax(prediction, dim=-1).squeeze()
    correct_items = (prediction_class == target)
    acc = correct_items.sum().item() / prediction.shape[0]  
    return acc


def accuracy_binary_two_classes(prediction, target):
    prediction = torch.nn.functional.softmax(prediction, dim=-1)
    prediction_class = torch.argmax(prediction, dim=-1).squeeze()
    correct_items = prediction_class == target
    correct_items_class0 = correct_items[target==0.0]
    if correct_items_class0.shape[0] > 0:
        acc_class0 = correct_items_class0.sum().item() / correct_items_class0.shape[0]
    else:
        acc_class0 = torch.nan
    correct_items_class1 = correct_items[target==1.0]
    if correct_items_class1.shape[0] > 0:
        acc_class1 = correct_items_class1.sum().item() / correct_items_class1.shape[0]
    else:
        acc_class1 = torch.nan
    return acc_class0, acc_class1


#-----------------------------------------------------
#--------------- CUSTOM LOSS FUNCTIONS ---------------
#-----------------------------------------------------


def weighted_mse_loss(input_batch, target_batch, weights):
    #return (weights * (input_batch - target_batch) ** 2).sum() / weights.sum()
    return torch.mean(weights * (input_batch - target_batch) ** 2)

def weighted_mse_loss_ASYM(input_batch, target_batch, weights):
    return torch.mean(torch.abs(input_batch - target_batch) + weights**2 * torch.clamp(target_batch - input_batch, min=0))

def MSE_weighted2(y_true, y_pred):
    return torch.mean(torch.exp(2.0 * torch.expm1(y_true)) * (y_pred - y_true)**2)

class modified_mse_quantile_loss():
    def __init__(self, q=0.85, alpha=0.2):
        self.mse_loss = nn.MSELoss()
        self.q = q
        self.alpha = alpha
    
    def __call__(self, prediction_batch, target_batch):
        loss_quantile = torch.mean(torch.max(self.q*(target_batch-prediction_batch), (1-self.q)*(prediction_batch-target_batch)))
        loss_mse = self.mse_loss(prediction_batch, target_batch) 
        return self.alpha * loss_mse + (1-self.alpha) * loss_quantile


class quantized_loss_crossentropy():
    '''
    w:      weight, computed as 1-h, where h is normalized histogram of
            a given input data x considering B bins the normalization is
            obtained by dividing the bins by the maximum bin count observed for x
    bins:   array cntaining the bin number for each of the nodes
    '''
    def __init__(self):
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction='none')
    
    def __call__(self, prediction_batch, target_batch, bins):
        loss_quantized = 0
        bins = bins.int()
        for b in torch.unique(bins):
            mask_b = bins == b
            omega = mask_b.sum()
            loss_quantized += 1/omega * self.cross_entropy_loss(prediction_batch * mask_b, target_batch * mask_b)
        return torch.mean(loss_quantized)


class mse_theta():
    def __init__(self):
        self.mse_loss = nn.MSELoss(reduction='none')

    def __call__(self, prediction_batch, prediction_theta_batch, target_batch):
        loss_mse = self.mse_loss(prediction_batch, target_batch)
        return torch.mean(loss_mse/prediction_theta_batch)
    

class gamma_nll():

    def __call__(self, alpha_batch, beta_batch, target_batch):
        return torch.mean(- torch.mean(alpha_batch*torch.log(beta_batch) - torch.lgamma(alpha_batch) + (alpha_batch-1)*torch.log(target_batch) - beta_batch*target_batch))


class quantized_loss():
    '''
    w:      weight, computed as 1-h, where h is normalized histogram of
            a given input data x considering B bins the normalization is
            obtained by dividing the bins by the maximum bin count observed for x
    bins:   array cntaining the bin number for each of the nodes
    '''
    def __init__(self):
        self.mse_loss = nn.MSELoss(reduction='mean')
        self.alpha = 0.005

        w_class1_100 = torch.tensor([0.0        , 0.23986506, 0.53060533, 0.57122705, 0.61302775,
                            0.67614517, 0.69597921, 0.72391865, 0.74754971, 0.76440881,
                            0.78548478, 0.79851332, 0.81427415, 0.82624273, 0.83837927,
                            0.84972993, 0.85968978, 0.86968089, 0.87895801, 0.88769771,
                            0.89589746, 0.90379577, 0.91112076, 0.91828936, 0.92493525,
                            0.93118279, 0.93700937, 0.94236466, 0.94743576, 0.95224705,
                            0.95677888, 0.9609333 , 0.96476712, 0.96833488, 0.97174368,
                            0.97491694, 0.97773601, 0.98039223, 0.98273215, 0.98490067,
                            0.9868026 , 0.98856462, 0.99003911, 0.99136291, 0.99249696,
                            0.99350548, 0.9943257 , 0.99507764, 0.99574782, 0.9963224 ,
                            0.99681511, 0.99722419, 0.99762647, 0.99794052, 0.9982412 ,
                            0.99846454, 0.99869109, 0.99890058, 0.99906662, 0.99921134,
                            0.99933734, 0.99944613, 0.99954329, 0.99961771, 0.9996794 ,
                            0.99974081, 0.99978494, 0.99982973, 0.99985965, 0.9998868 ,
                            0.99990736, 0.99992468, 0.99993866, 0.99994849, 0.99995898,
                            0.99996711, 0.99997322, 0.99997683, 0.99998012, 0.99998348,
                            0.99998536, 0.99998966, 0.99998952, 0.99999216, 0.99999299,
                            0.99999292, 0.99999461, 0.99999552, 0.99999548, 0.99999675,
                            0.99999664, 0.99999776, 0.9999978 , 0.99999798, 0.99999834,
                            0.99999845, 0.99999895, 0.99999935, 0.99999967, 0.99999982])
        
        w_all_100 = torch.tensor([0.0        , 0.97944841, 0.9839962 , 0.98859393, 0.99225148,
                            0.99295734, 0.99380695, 0.99463827, 0.9949188 , 0.99559521,
                            0.99583359, 0.99617562, 0.9964578 , 0.99672362, 0.99692317,
                            0.99719223, 0.9973556 , 0.99755417, 0.99772204, 0.99788008,
                            0.9980354 , 0.99818819, 0.99832342, 0.99845276, 0.99857503,
                            0.99869292, 0.99880242, 0.99890566, 0.99899866, 0.99908951,
                            0.99917294, 0.99925090, 0.99932501, 0.99939197, 0.99945406,
                            0.99951285, 0.99956817, 0.99961759, 0.99966372, 0.99970436,
                            0.99974246, 0.99977497, 0.99980573, 0.99983101, 0.99985384,
                            0.99987346, 0.99989073, 0.99990470, 0.99991764, 0.99992906,
                            0.99993876, 0.99994707, 0.99995414, 0.99996069, 0.99996612,
                            0.99997103, 0.99997491, 0.99997881, 0.99998217, 0.99998486,
                            0.99998738, 0.99998933, 0.99999123, 0.99999271, 0.99999393,
                            0.99999496, 0.99999600, 0.99999667, 0.99999737, 0.99999781,
                            0.99999828, 0.99999859, 0.99999883, 0.99999906, 0.99999923,
                            0.9999994 , 0.99999949, 0.99999958, 0.99999966, 0.99999969,
                            0.99999974, 0.99999982, 0.99999981, 0.99999986, 0.99999988,
                            0.99999988, 0.99999990, 0.99999992, 0.99999992, 0.99999994,
                            0.99999994, 0.99999996, 0.99999996, 0.99999996, 0.99999997,
                            0.99999997, 0.99999998, 0.99999999, 0.99999999, 1.        ])
        
        self.w = w_all_100
    
    def __call__(self, prediction_batch, target_batch, bins, device):
        # loss_quantized = torch.zeros((bins.shape)).to(device)
        loss_quantized = 0
        self.w = self.w.to(device)
        bins = bins.int()
        loss_mse = self.mse_loss(prediction_batch, target_batch)
        loss_mae = torch.abs(prediction_batch - target_batch)
        for b in torch.unique(bins):
            mask_b = bins == b
            omega = mask_b.sum()
            loss_quantized += torch.mean(1/omega * self.w[b] * loss_mae[mask_b])
            # loss_quantized += torch.mean(1/omega * self.w[b] * loss_mse[mask_b])
        return loss_mse, self.alpha * loss_quantized
    
def threshold_quantile_loss(predictions, ground_truth, q_low=0.01, q_high=0.99, threshold=np.log1p(1)):
    mask_low = ground_truth <= threshold
    loss_low = mask_low * torch.max(q_low*(ground_truth-predictions), (1-q_low)*(predictions-ground_truth))
    loss_high = ~mask_low * torch.max(q_high*(ground_truth-predictions), (1-q_high)*(predictions-ground_truth))
    #loss_mse = nn.functional.mse_loss(predictions, ground_truth)
    return torch.mean(loss_low + loss_high)

def ghm_c_loss(y_pred, y_true, bins=10):
   # Compute gradient norm 
   probs = torch.sigmoid(y_pred)
   gradient_norm = torch.abs(probs - y_true)
   
   # Bin the gradients
   min_val, max_val = gradient_norm.min(), gradient_norm.max()
   bin_width = (max_val - min_val) / bins
   gradient_bins = torch.floor((gradient_norm - min_val) / bin_width).long()
   
  # Calculate gradient density
   bin_counts = torch.bincount(gradient_bins)
   total_count = bin_counts.sum()
   gradient_density = bin_counts / total_count
  
  # Get bin id for each example
   # bin_ids = gradient_bins[torch.arange(len(gradient_norm)), gradient_norm]

  # Compute loss 
   weights = 1 / (gradient_density[gradient_bins] + 1e-6)
   ce_loss = nn.functional.binary_cross_entropy_with_logits(y_pred, y_true)
   weighted_loss = ce_loss * weights
 
   return weighted_loss.mean()


#-----------------------------------------------------
#-------------------- LOAD CHECKPOINT ----------------
#-----------------------------------------------------


def load_checkpoint(model, checkpoint, log_path, log_file, accelerator, net_names, fine_tuning=True, device=None, output=True):
    if accelerator is None or accelerator.is_main_process:
        if output:
            with open(log_path+log_file, 'a') as f:
                f.write("\nLoading parameters.") 
    state_dict = checkpoint["parameters"]
    for name, param in state_dict.items():
        for net_name in net_names:
            if net_name in name:
                if accelerator is None or accelerator.is_main_process:
                    if output:
                        with open(log_path+log_file, 'a') as f:
                            f.write(f"\nLoading parameters '{name}'")
                param = param.data
                if name.startswith("module"):
                    name = name.partition("module.")[2]
                try:
                    model.state_dict()[name].copy_(param)
                except:
                     if accelerator is None or accelerator.is_main_process:
                        if output:
                            with open(log_path+log_file, 'a') as f:
                                f.write(f"\nParam {name} was not loaded..")
    #if not fine_tuning:
    #    for net_name in net_names:
    #        [param.requires_grad_(False) for name, param in model.named_parameters() if net_name in name]
    return model


def check_freezed_layers(model, log_path, log_file, accelerator):
    for name, param in model.named_parameters():
        n_param = param.numel() 
        if accelerator is None or accelerator.is_main_process:
            with open(log_path+log_file, 'a') as f:
                f.write(f"\nLayer {name} requires_grad = {param.requires_grad} and has {n_param} parameters") 


#-----------------------------------------------------
#--------------- TRAIN AND VALIDATION ----------------
#-----------------------------------------------------

class Trainer(object):

    def train_cl(self, model, dataloader_train, dataloader_val, optimizer, loss_fn, lr_scheduler, accelerator, args,
                        epoch_start, alpha=0.9, gamma=2):
        
        if accelerator.is_main_process:
            with open(args.output_path+args.log_file, 'a') as f:
                f.write(f"\nStart training the classifier.")
        for epoch in range(epoch_start, epoch_start+args.epochs):
            model.train()
            if accelerator.is_main_process:
                with open(args.output_path+args.log_file, 'a') as f:
                    f.write(f"\nEpoch {epoch+1} --- learning rate {optimizer.param_groups[0]['lr']:.8f}")

            # Define objects to track meters
            # -> Train
            loss_meter = AverageMeter()
            acc_meter = AverageMeter()
            acc_class0_meter = AverageMeter()
            acc_class1_meter = AverageMeter()
            # -> Validation
            loss_meter_val = AverageMeter()
            acc_meter_val = AverageMeter()
            acc_class0_meter_val = AverageMeter()
            acc_class1_meter_val = AverageMeter()
            start = time.time()

            for graph in dataloader_train:
                train_mask = graph["high"].train_mask
                optimizer.zero_grad()

                y_pred = model(graph).squeeze()[train_mask]
                y = graph['high'].y[train_mask]
                loss = loss_fn(y_pred, y)
#                loss = loss_fn(y_pred, y, alpha, gamma, reduction='mean')
#                w = graph['high'].w[train_mask]
#                loss = loss_fn(y_pred, y, w, accelerator.device)
                accelerator.backward(loss)
                accelerator.clip_grad_norm_(model.parameters(), 5)
                optimizer.step()
                loss_meter.update(val=loss.item(), n=1)    
                acc = accuracy_binary_one(y_pred, y)
                acc_class0, acc_class1 = accuracy_binary_one_classes(y_pred, y)

                acc_meter.update(val=acc, n=1)
                acc_class0_meter.update(val=acc_class0, n=1)
                acc_class1_meter.update(val=acc_class1, n=1)
                accelerator.log({'epoch':epoch, 'accuracy iteration': acc_meter.val, 'loss avg': loss_meter.avg,
                    'accuracy avg': acc_meter.avg, 'accuracy class0 avg': acc_class0_meter.avg, 'accuracy class1 avg': acc_class1_meter.avg})
                
            end = time.time()

            # End of epoch --> write log and save checkpoint
            accelerator.log({'loss epoch': loss_meter.avg, 'accuracy epoch': acc_meter.avg, 'accuracy class0 epoch': acc_class0_meter.avg, 'accuracy class1 epoch': acc_class1_meter.avg})
            if accelerator.is_main_process:
                with open(args.output_path+args.log_file, 'a') as f:
                    f.write(f"\nEpoch {epoch+1} completed in {end - start:.4f} seconds. Loss - total: {loss_meter.sum:.4f} - average: {loss_meter.avg:.10f}; "+
                            f"acc: {acc_meter.avg:.4f}; acc class 0: {acc_class0_meter.avg:.4f}; acc class 1: {acc_class1_meter.avg:.4f}.")
            
            if lr_scheduler is not None and lr_scheduler.get_last_lr()[0] > 0.00001:
                lr_scheduler.step()

            accelerator.save_state(output_dir=args.output_path+f"checkpoint_{epoch}/")
            torch.save({"epoch": epoch}, args.output_path+f"checkpoint_{epoch}/epoch")
            
            model.eval()
            # Perform validation step
            with torch.no_grad():
                for graph in dataloader_val:
                    
                    train_mask = graph["high"].train_mask

                    y_pred = model(graph).squeeze()[train_mask]
                    y = graph['high'].y[train_mask]
                    loss = loss_fn(y_pred, y)
#                    loss = loss_fn(y_pred, y, alpha, gamma, reduction='mean')
#                    w = graph['high'].w[train_mask]
#                    loss = loss_fn(y_pred, y, w)
                    acc = accuracy_binary_one(y_pred, y)
                    acc_class0, acc_class1 = accuracy_binary_one_classes(y_pred, y)   

                    loss_meter_val.update(val=loss.item(), n=1)    
                    acc_meter_val.update(val=acc, n=1)
                    acc_class0_meter_val.update(val=acc_class0, n=1)
                    acc_class1_meter_val.update(val=acc_class1, n=1)
            
            accelerator.log({'validation loss': loss_meter_val.avg, 'validation accuracy': acc_meter_val.avg,
                                'validation accuracy class0': acc_class0_meter_val.avg, 'validation accuracy class1': acc_class1_meter_val.avg})
        return model

    def train_reg(self, model, dataloader_train, dataloader_val, optimizer, loss_fn, lr_scheduler, accelerator, args, epoch_start=0):
        
        #mse_loss = nn.MSELoss()
        if accelerator.is_main_process:
            with open(args.output_path+args.log_file, 'a') as f:
                f.write(f"\nStart training the regressor.")
        for epoch in range(epoch_start, epoch_start+args.epochs):
            model.train()
            if accelerator.is_main_process:
                with open(args.output_path+args.log_file, 'a') as f:
                    f.write(f"\nEpoch {epoch+1} --- learning rate {optimizer.param_groups[0]['lr']:.8f}")
            loss_meter = AverageMeter()
            loss_mse_meter = AverageMeter()
            loss_q_meter = AverageMeter()
            loss_meter_val = AverageMeter()
            loss_mse_meter_val = AverageMeter()
            loss_q_meter_val = AverageMeter()
            
            start = time.time()

            for graph in dataloader_train:
                train_mask = graph['high'].train_mask
                optimizer.zero_grad()
                y_pred = model(graph).squeeze()[train_mask]
                y = graph['high'].y[train_mask]
                #loss = loss_fn(y_pred, y)
                w = graph['high'].w[train_mask]
                # loss = loss_fn(y_pred, y, w, accelerator.device)
                loss_mse, loss_q = loss_fn(y_pred, y, w, accelerator.device)
                loss = loss_mse + loss_q
                accelerator.backward(loss)
                accelerator.clip_grad_norm_(model.parameters(), 5)
                optimizer.step()
                loss_meter.update(val=loss.item(), n=1)    
                accelerator.log({'epoch':epoch, 'loss iteration': loss_meter.val, 'loss avg': loss_meter.avg})
                #loss_rmse = torch.sqrt(mse_loss(torch.expm1(y_pred), torch.expm1(y)))
                loss_mse_meter.update(val=loss_mse.item(), n=1)    
                loss_q_meter.update(val=loss_q.item(), n=1)    
                accelerator.log({'MSE loss iteration': loss_mse_meter.val, 'MSE loss avg': loss_mse_meter.avg, 'Q loss iteration': loss_q_meter.val, 'Q loss avg': loss_q_meter.avg})
                
            end = time.time()
            accelerator.log({'loss epoch': loss_meter.avg})
            accelerator.log({'MSE loss epoch': loss_mse_meter.avg, 'Q loss epoch': loss_q_meter.avg})
            if accelerator.is_main_process:
                with open(args.output_path+args.log_file, 'a') as f:
                    f.write(f"\nEpoch {epoch+1} completed in {end - start:.4f} seconds. Loss - total: {loss_meter.sum:.4f} - average: {loss_meter.avg:.10f}. ")
            
            if lr_scheduler is not None and lr_scheduler.get_last_lr()[0] > 0.00001:
                lr_scheduler.step()
            
            accelerator.save_state(output_dir=args.output_path+f"checkpoint_{epoch}/")
            torch.save({"epoch": epoch}, args.output_path+f"checkpoint_{epoch}/epoch")
            
            model.eval()
            # Perform validation step
            with torch.no_grad():
                for graph in dataloader_val:
                    
                    train_mask = graph["high"].train_mask

                    y_pred = model(graph).squeeze()[train_mask]
                    y = graph['high'].y[train_mask]
                    #loss = loss_fn(y_pred, y)
                    w = graph['high'].w[train_mask]
                    # loss = loss_fn(y_pred, y, w, accelerator.device)
                    loss_mse, loss_q = loss_fn(y_pred, y, w, accelerator.device)
                    loss = loss_mse + loss_q
                    loss_meter_val.update(val=loss.item(), n=1)    
                    #loss_rmse = torch.sqrt(mse_loss(torch.expm1(y_pred), torch.expm1(y)))
                    loss_mse_meter_val.update(val=loss_mse.item(), n=1)    
                    loss_q_meter_val.update(val=loss_q.item(), n=1)    

            accelerator.log({'validation loss': loss_meter_val.avg})
            accelerator.log({'validation MSE loss': loss_mse_meter_val.avg, 'validation Q loss': loss_q_meter_val.avg})
        return model
    
    def train_reg_gamma(self, model, dataloader_train, dataloader_val, optimizer, loss_fn, lr_scheduler, accelerator, args, epoch_start=0):
        
        #mse_loss = nn.MSELoss()
        if accelerator.is_main_process:
            with open(args.output_path+args.log_file, 'a') as f:
                f.write(f"\nStart training the regressor.")
        for epoch in range(epoch_start, epoch_start+args.epochs):
            model.train()
            if accelerator.is_main_process:
                with open(args.output_path+args.log_file, 'a') as f:
                    f.write(f"\nEpoch {epoch+1} --- learning rate {optimizer.param_groups[0]['lr']:.8f}")
            loss_meter = AverageMeter()
            loss_meter_val = AverageMeter()
            
            start = time.time()

            for graph in dataloader_train:
                train_mask = graph['high'].train_mask
                optimizer.zero_grad()
                theta_pred = model(graph)
                alpha_pred = theta_pred[:,0].squeeze()[train_mask]
                beta_pred = theta_pred[:,1].squeeze()[train_mask]
                y = graph['high'].y[train_mask]
                loss = loss_fn(alpha_pred, beta_pred, y)
                accelerator.backward(loss)
                accelerator.clip_grad_norm_(model.parameters(), 5)
                optimizer.step()
                loss_meter.update(val=loss.item(), n=1)    
                accelerator.log({'epoch':epoch, 'loss iteration': loss_meter.val, 'loss avg': loss_meter.avg})
                
            end = time.time()
            accelerator.log({'loss epoch': loss_meter.avg})
            if accelerator.is_main_process:
                with open(args.output_path+args.log_file, 'a') as f:
                    f.write(f"\nEpoch {epoch+1} completed in {end - start:.4f} seconds. Loss - total: {loss_meter.sum:.4f} - average: {loss_meter.avg:.10f}. ")
            
            if lr_scheduler is not None and lr_scheduler.get_last_lr()[0] > 0.00001:
                lr_scheduler.step()
            
            accelerator.save_state(output_dir=args.output_path+f"checkpoint_{epoch}/")
            torch.save({"epoch": epoch}, args.output_path+f"checkpoint_{epoch}/epoch")
            
            model.eval()
            # Perform validation step
            with torch.no_grad():
                for graph in dataloader_val:
                    
                    train_mask = graph["high"].train_mask
                    theta_pred = model(graph)
                    alpha_pred = theta_pred[:,0].squeeze()[train_mask]
                    beta_pred = theta_pred[:,1].squeeze()[train_mask]
                    y = graph['high'].y[train_mask]
                    loss = loss_fn(alpha_pred, beta_pred, y)
                    loss_meter_val.update(val=loss.item(), n=1)    

            accelerator.log({'validation loss': loss_meter_val.avg})
        return model
    
    def train_reg_theta(self, model, dataloader_train, dataloader_val, optimizer, loss_fn, lr_scheduler, accelerator, args, epoch_start=0):
        
        #mse_loss = nn.MSELoss()
        if accelerator.is_main_process:
            with open(args.output_path+args.log_file, 'a') as f:
                f.write(f"\nStart training the regressor.")
        for epoch in range(epoch_start, epoch_start+args.epochs):
            model.train()
            if accelerator.is_main_process:
                with open(args.output_path+args.log_file, 'a') as f:
                    f.write(f"\nEpoch {epoch+1} --- learning rate {optimizer.param_groups[0]['lr']:.8f}")
            loss_meter = AverageMeter()
            loss_meter_val = AverageMeter()
            
            start = time.time()

            for graph in dataloader_train:
                train_mask = graph['high'].train_mask
                optimizer.zero_grad()
                y_pred, theta_pred = model(graph)
                y_pred = y_pred.squeeze()[train_mask]
                theta_pred = theta_pred.squeeze()[train_mask]
                y = graph['high'].y[train_mask]
                loss = loss_fn(y_pred, theta_pred, y)
                accelerator.backward(loss)
                accelerator.clip_grad_norm_(model.parameters(), 5)
                optimizer.step()
                loss_meter.update(val=loss.item(), n=1)    
                accelerator.log({'epoch':epoch, 'loss iteration': loss_meter.val, 'loss avg': loss_meter.avg})
                
            end = time.time()
            accelerator.log({'loss epoch': loss_meter.avg})
            if accelerator.is_main_process:
                with open(args.output_path+args.log_file, 'a') as f:
                    f.write(f"\nEpoch {epoch+1} completed in {end - start:.4f} seconds. Loss - total: {loss_meter.sum:.4f} - average: {loss_meter.avg:.10f}. ")
            
            if lr_scheduler is not None and lr_scheduler.get_last_lr()[0] > 0.00001:
                lr_scheduler.step()
            
            accelerator.save_state(output_dir=args.output_path+f"checkpoint_{epoch}/")
            torch.save({"epoch": epoch}, args.output_path+f"checkpoint_{epoch}/epoch")
            
            model.eval()
            # Perform validation step
            with torch.no_grad():
                for graph in dataloader_val:
                    
                    train_mask = graph["high"].train_mask
                    y_pred, theta_pred = model(graph)
                    y_pred = y_pred.squeeze()[train_mask]
                    theta_pred = theta_pred.squeeze()[train_mask]
                    y = graph['high'].y[train_mask]
                    loss = loss_fn(y_pred, theta_pred, y)
                    loss_meter_val.update(val=loss.item(), n=1)    

            accelerator.log({'validation loss': loss_meter_val.avg})
        return model
    

#-----------------------------------------------------
#----------------------- TEST ------------------------
#-----------------------------------------------------


class Tester(object):

    def test(self, model_cl, model_reg, dataloader,low_high_graph, args, accelerator=None):
        model_cl.eval()
        model_reg.eval()
        step = 0 

        pr_cl = []
        pr_reg = []
        times = []
        with torch.no_grad():    
            for graph in dataloader:

                t = graph.t
                times.append(t)
                
                # Regressor
                y_pred_reg = model_reg(graph)
                pr_reg.append(torch.expm1(y_pred_reg)) 
                #pr_reg.append(y_pred_reg)

                # Classifier
                y_pred_cl = model_cl(graph)
                #-- sigmoid focal loss ->
                pr_cl.append(torch.where(y_pred_cl >= 0.0, 1.0, 0.0))
                
                if step % 100 == 0:
                    if accelerator is None or accelerator.is_main_process:
                        with open(args.output_path+args.log_file, 'a') as f:
                            f.write(f"\nStep {step} done.")
                step += 1 

        pr_cl = torch.stack(pr_cl)
        pr_reg = torch.stack(pr_reg)
        times = torch.stack(times)

        return pr_cl, pr_reg, times
    
    def test_cl(self, model_cl, dataloader,low_high_graph, args, accelerator=None):
        model_cl.eval()
        step = 0 
        pr_cl = []
        times = []
        with torch.no_grad():    
            for graph in dataloader:

                t = graph.t
                times.append(t)

                # Classifier
                y_pred_cl = model_cl(graph)
                #-- sigmoid focal loss ->
                pr_cl.append(torch.where(y_pred_cl >= 0.0, 1.0, 0.0))
                
                if step % 100 == 0:
                    if accelerator is None or accelerator.is_main_process:
                        with open(args.output_path+args.log_file, 'a') as f:
                            f.write(f"\nStep {step} done.")
                step += 1 
        
        pr_cl = torch.stack(pr_cl)
        times = torch.stack(times)

        return pr_cl, times
    
    def test_reg(self, model_reg, dataloader,low_high_graph, args, accelerator=None):
        model_reg.eval()
        step = 0 
        pr_reg = []
        times = []
        with torch.no_grad():    
            for graph in dataloader:

                t = graph.t
                times.append(t)
                
                # Regressor
                y_pred_reg = model_reg(graph)
                pr_reg.append(torch.expm1(y_pred_reg))

                if step % 100 == 0:
                    if accelerator is None or accelerator.is_main_process:
                        with open(args.output_path+args.log_file, 'a') as f:
                            f.write(f"\nStep {step} done.")
                step += 1 

        pr_reg = torch.stack(pr_reg)
        times = torch.stack(times)

        return pr_reg, times
    

#-----------------------------------------------------
#-------------------- VALIDATION ---------------------
#-----------------------------------------------------
    

class Validator(object):

    def validate_cl(self, model, dataloader, loss_fn):

        model.eval()

        loss_meter = AverageMeter()
        acc_meter = AverageMeter()
        acc_class0_meter = AverageMeter()
        acc_class1_meter = AverageMeter()

        with torch.no_grad():    
            for graph in dataloader:

                train_mask = graph["high"].train_mask

                # Classifier
                y_pred = model(graph)[train_mask]
                y = graph['high'].y[train_mask]
                loss = loss_fn(y_pred, y.to(torch.int64))

                acc = accuracy_binary_two(y_pred, y)
                acc_class0, acc_class1 = accuracy_binary_two_classes(y_pred, y)

                loss_meter.update(val=loss.item(), n=1)    
                acc_meter.update(val=acc, n=1)
                acc_class0_meter.update(val=acc_class0, n=1)
                acc_class1_meter.update(val=acc_class1, n=1)

        return loss_meter.avg, acc_meter.avg, acc_class0_meter.avg, acc_class1_meter.avg
    
    def validate_reg(self, model, dataloader, loss_fn):

        model.eval()

        loss_meter = AverageMeter()

        with torch.no_grad():    
            for graph in dataloader:

                train_mask = graph["high"].train_mask

                # Regressor
                y_pred = model(graph)[train_mask]
                y = graph['high'].y[train_mask]
                w = graph['high'].w[train_mask]
                loss = loss_fn(y_pred, y, w)

                loss_meter.update(val=loss.item(), n=1)   

        return loss_meter.avg

