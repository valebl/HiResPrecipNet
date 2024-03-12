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


def cut_window(lon_min, lon_max, lat_min, lat_max, lon, lat, z, pr):
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
    return lon_sel, lat_sel, z_sel, pr_sel


def retain_valid_nodes(lon, lat, pr, z):

    valid_nodes = ~torch.isnan(pr).all(dim=0)
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

def threshold_quantile_loss(predictions, ground_truth, q_low=0.01, q_high=0.99, threshold=np.log1p(1)):
    mask_low = ground_truth <= threshold
    loss_low = mask_low * torch.max(q_low*(ground_truth-predictions), (1-q_low)*(predictions-ground_truth))
    loss_high = ~mask_low * torch.max(q_high*(ground_truth-predictions), (1-q_high)*(predictions-ground_truth))
    #loss_mse = nn.functional.mse_loss(predictions, ground_truth) 
    return torch.mean(loss_low + loss_high)

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
                        epoch_start, alpha=0.75, gamma=2):
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
                loss = loss_fn(y_pred, y, alpha, gamma, reduction='mean')
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
            for graph in dataloader_val:
                
                train_mask = graph["high"].train_mask

                y_pred = model(graph).squeeze()[train_mask]
                y = graph['high'].y[train_mask]
                loss = loss_fn(y_pred, y, alpha, gamma, reduction='mean')
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
        
        mse_loss = nn.MSELoss()
        if accelerator.is_main_process:
            with open(args.output_path+args.log_file, 'a') as f:
                f.write(f"\nStart training the regressor.")
        for epoch in range(epoch_start, epoch_start+args.epochs):
            model.train()
            if accelerator.is_main_process:
                with open(args.output_path+args.log_file, 'a') as f:
                    f.write(f"\nEpoch {epoch+1} --- learning rate {optimizer.param_groups[0]['lr']:.8f}")
            loss_meter = AverageMeter()
            rmse_loss_meter = AverageMeter()
            loss_meter_val = AverageMeter()
            rmse_loss_meter_val = AverageMeter()
            
            start = time.time()

            for graph in dataloader_train:
                train_mask = graph['high'].train_mask
                optimizer.zero_grad()
                y_pred = model(graph).squeeze()[train_mask]
                y = graph['high'].y[train_mask]
                #loss = loss_fn(y_pred, y)
                w = graph['high'].w[train_mask]
                loss = loss_fn(y_pred, y, w)
                accelerator.backward(loss)
                accelerator.clip_grad_norm_(model.parameters(), 5)
                optimizer.step()
                loss_meter.update(val=loss.item(), n=1)    
                accelerator.log({'epoch':epoch, 'loss iteration': loss_meter.val, 'loss avg': loss_meter.avg})
                loss_rmse = torch.sqrt(mse_loss(torch.expm1(y_pred), torch.expm1(y)))
                rmse_loss_meter.update(val=loss_rmse.item(), n=1)    
                accelerator.log({'epoch':epoch, 'RMSE loss iteration': rmse_loss_meter.val, 'RMSE loss avg': rmse_loss_meter.avg})

            end = time.time()
            accelerator.log({'loss epoch': loss_meter.avg})
            accelerator.log({'RMSE loss epoch': rmse_loss_meter.avg})
            if accelerator.is_main_process:
                with open(args.output_path+args.log_file, 'a') as f:
                    f.write(f"\nEpoch {epoch+1} completed in {end - start:.4f} seconds. Loss - total: {loss_meter.sum:.4f} - average: {loss_meter.avg:.10f}. ")
            
            if lr_scheduler is not None and lr_scheduler.get_last_lr()[0] > 0.00001:
                lr_scheduler.step()
            
            accelerator.save_state(output_dir=args.output_path+f"checkpoint_{epoch}/")
            torch.save({"epoch": epoch}, args.output_path+f"checkpoint_{epoch}/epoch")
            
            model.eval()
            # Perform validation step
            for graph in dataloader_val:
                
                train_mask = graph["high"].train_mask

                y_pred = model(graph).squeeze()[train_mask]
                y = graph['high'].y[train_mask]
                #loss = loss_fn(y_pred, y)
                w = graph['high'].w[train_mask]
                loss = loss_fn(y_pred, y, w)
                loss_meter_val.update(val=loss.item(), n=1)    
                loss_rmse = torch.sqrt(mse_loss(torch.expm1(y_pred), torch.expm1(y)))
                rmse_loss_meter_val.update(val=loss_rmse.item(), n=1)    

            accelerator.log({'validation loss': loss_meter_val.avg})
            accelerator.log({'validation RMSE loss': rmse_loss_meter_val.avg})
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

