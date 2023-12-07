import time
import sys
import pickle

import torch

from datetime import datetime, timedelta, date
from torch_geometric.transforms import ToDevice

from datetime import datetime, date


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


#-----------------------------------------------------
#----------------- GENERAL UTILITIES -----------------
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


def use_gpu_if_possible():
    return "cuda:0" if torch.cuda.is_available() else "cpu"

def accuracy_binary_one(prediction, target):
    prediction_class = torch.where(prediction > 0.5, 1.0, 0.0) 
    correct_items = (prediction_class == target)
    acc = correct_items.sum().item() / prediction.shape[0]  
    return acc

def accuracy_binary_one_class1(prediction, target):
    prediction_class = torch.where(prediction > 0.5, 1.0, 0.0)
    correct_items = (prediction_class == target)[target==1.0]
    if correct_items.shape[0] > 0:
        acc_class1 = correct_items.sum().item() / correct_items.shape[0]
        return acc_class1
    else:
        return 0.0

def accuracy_binary_two(prediction, target):
    prediction = torch.nn.functional.softmax(prediction, dim=-1)
    prediction_class = torch.argmax(prediction, dim=-1).squeeze()
    correct_items = (prediction_class == target)
    acc = correct_items.sum().item() / prediction.shape[0]  
    return acc

def accuracy_binary_two_class1(prediction, target):
    prediction = torch.nn.functional.softmax(prediction)
    prediction_class = torch.argmax(prediction, dim=-1).squeeze()
    correct_items = (prediction_class == target)[target==1.0]
    if correct_items.shape[0] > 0:
        acc_class1 = correct_items.sum().item() / correct_items.shape[0]
        return acc_class1
    else:
        return 0.0

def weighted_mse_loss(input_batch, target_batch, weights):
    #return (weights * (input_batch - target_batch) ** 2).sum() / weights.sum()
    return torch.mean(weights * (input_batch - target_batch) ** 2)

def quantile_loss(prediction_batch, target_batch, q=0.35):
    return torch.mean(torch.max(q*(prediction_batch-target_batch), (q-1)*(prediction_batch-target_batch)))

def load_checkpoint(model, checkpoint, log_path, log_file, accelerator, net_names, fine_tuning=True, device=None):
    if accelerator is None or accelerator.is_main_process:
        with open(log_path+log_file, 'a') as f:
            f.write("\nLoading encoder parameters.") 
    state_dict = checkpoint["parameters"]
    for name, param in state_dict.items():
        for net_name in net_names:
            if net_name in name:
                if accelerator is None or accelerator.is_main_process:
                    with open(log_path+log_file, 'a') as f:
                        f.write(f"\nLoading parameters '{name}'")
                param = param.data
                if name.startswith("module"):
                    name = name.partition("module.")[2]
                try:
                    model.state_dict()[name].copy_(param)
                except:
                     if accelerator is None or accelerator.is_main_process:
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
#------------------ TRAIN AND TEST -------------------
#-----------------------------------------------------

class Trainer(object):

    def _train_epoch_cl(self, epoch, model, dataloader, optimizer, loss_fn, accelerator, args, alpha=0.9, gamma=2):
        loss_meter = AverageMeter()
        performance_meter = AverageMeter()
        acc_class1_meter = AverageMeter()
        start = time.time()
        step = 0
        for graph in dataloader:
            if graph["high"].train_mask.sum().item() == 0:
                continue
            optimizer.zero_grad()
            y_pred = model(graph).squeeze()[graph['high'].train_mask]
            y = graph['high'].y[graph['high'].train_mask]
            loss = loss_fn(y_pred, y, alpha, gamma, reduction='mean')
            accelerator.backward(loss)
            torch.nn.utils.clip_grad_norm_(model.parameters(),5)
            optimizer.step()
            loss_meter.update(val=loss.item(), n=1)    
            performance = accuracy_binary_one(y_pred, y)
            acc_class1 = accuracy_binary_one_class1(y_pred, y)
            performance_meter.update(val=performance, n=1)
            acc_class1_meter.update(val=acc_class1, n=1)
            accelerator.log({'epoch':epoch, 'loss iteration': loss_meter.val, 'accuracy iteration': performance_meter.val, 'loss avg': loss_meter.avg,
                'accuracy avg': performance_meter.avg, 'accuracy class1 avg': acc_class1_meter.avg, 'step':step})
            step += 1
            if accelerator.is_main_process:
                if step % 5000 == 0:
                    checkpoint_dict = {
                        "parameters": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch,
                        }
                    torch.save(checkpoint_dict, args.output_path+f"checkpoint_{epoch}_tmp.pth")
        end = time.time()
        accelerator.log({'loss epoch': loss_meter.avg, 'accuracy epoch': performance_meter.avg, 'accuracy class1 epoch': acc_class1_meter.avg})
        if accelerator.is_main_process:
            with open(args.output_path+args.log_file, 'a') as f:
                f.write(f"\nEpoch {epoch+1} completed in {end - start:.4f} seconds. Loss - total: {loss_meter.sum:.4f} - average: {loss_meter.avg:.10f}; "+
                    f"performance: {performance_meter.avg:.4f}.")

    def _train_epoch_reg(self, epoch, model, dataloader, optimizer, loss_fn, accelerator, args):
        loss_meter = AverageMeter()
        start = time.time()
        step = 0 
        for graph in dataloader:
            if graph['high'].train_mask.sum().item() == 0:
                continue
            optimizer.zero_grad()
            y_pred = model(graph).squeeze()[graph['high'].train_mask]
            y = graph['high'].y[graph['high'].train_mask]
            loss = loss_fn(y_pred, y)
            #w = graph['high'].w[graph['high'].train_mask]
            #loss = loss_fn(y_pred, y, w)
            accelerator.backward(loss)
            torch.nn.utils.clip_grad_norm_(model.parameters(),5)
            optimizer.step()
            loss_meter.update(val=loss.item(), n=1)    
            accelerator.log({'epoch':epoch, 'loss iteration': loss_meter.val, 'loss avg': loss_meter.avg, 'step':step})
            step += 1
            if accelerator.is_main_process:
                if step % 5000 == 0:
                    checkpoint_dict = {
                        "parameters": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch,
                        }
                    torch.save(checkpoint_dict, args.output_path+f"checkpoint_{epoch}_tmp.pth")
        end = time.time()
        accelerator.log({'loss epoch': loss_meter.avg})
        if accelerator.is_main_process:
            with open(args.output_path+args.log_file, 'a') as f:
                f.write(f"\nEpoch {epoch+1} completed in {end - start:.4f} seconds. Loss - total: {loss_meter.sum:.4f} - average: {loss_meter.avg:.10f}. ")


    def _train_epoch_combined(self, epoch, model, dataloader, optimizer, loss_fn_cl, loss_fn_reg, accelerator, args, alpha=0.9, gamma=2, theta=0.9):
        loss_meter = AverageMeter()
        start = time.time()
        step = 0 
        for graph in dataloader:
            train_mask = graph['high'].train_mask
            if train_mask.sum().item() == 0:
                continue
            optimizer.zero_grad()
            y_pred_cl, y_pred_reg = model(graph)
            # cl
            y_pred_cl = y_pred_cl.squeeze()[train_mask]
            y_pred_cl = torch.sigmoid(y_pred_cl)
            y_cl = graph['high'].y_cl[train_mask]
            # reg
            y_pred_reg = y_pred_reg.squeeze()[train_mask] * y_pred_cl
            y_reg = graph['high'].y_reg[train_mask]
            w = graph['high'].w[train_mask]
            # combine
            loss = theta * loss_fn_cl(y_pred_cl, y_cl, alpha, gamma, reduction='mean') + (1-theta) * loss_fn_reg(y_pred_reg, y_reg, w)
            accelerator.backward(loss)
            torch.nn.utils.clip_grad_norm_(model.parameters(),5)
            optimizer.step()
            loss_meter.update(val=loss.item(), n=1)    
            accelerator.log({'epoch':epoch, 'loss iteration': loss_meter.val, 'loss avg': loss_meter.avg, 'step':step})
            step += 1
            if accelerator.is_main_process:
                if step % 5000 == 0:
                    checkpoint_dict = {
                        "parameters": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch,
                        }
                    torch.save(checkpoint_dict, args.output_path+f"checkpoint_{epoch}_tmp.pth")
        end = time.time()
        accelerator.log({'loss epoch': loss_meter.avg})
        if accelerator.is_main_process:
            with open(args.output_path+args.log_file, 'a') as f:
                f.write(f"\nEpoch {epoch+1} completed in {end - start:.4f} seconds. Loss - total: {loss_meter.sum:.4f} - average: {loss_meter.avg:.10f}. ")
    
    def train(self, model, dataloader, optimizer, loss_fn_cl, loss_fn_reg, lr_scheduler, accelerator, args, epoch_start=0):
        model.train()
        for epoch in range(epoch_start, epoch_start+args.epochs):
            if accelerator.is_main_process:
                with open(args.output_path+args.log_file, 'a') as f:
                    f.write(f"\nEpoch {epoch+1} --- learning rate {optimizer.param_groups[0]['lr']:.8f}")
            if args.model_type == 'reg':
                self._train_epoch_reg(epoch, model, dataloader, optimizer, loss_fn_reg, accelerator, args)
            elif args.model_type == 'cl':
                self._train_epoch_cl(epoch, model, dataloader, optimizer, loss_fn_cl, accelerator, args)
            elif args.model_type == 'combined':
                self._train_epoch_combined(epoch, model, dataloader, optimizer, loss_fn_cl, loss_fn_reg, accelerator, args)
            
            if lr_scheduler is not None and lr_scheduler.get_last_lr()[0] > 0.000001:
                lr_scheduler.step()
            if accelerator.is_main_process:
                checkpoint_dict = {
                    "parameters": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    }
                torch.save(checkpoint_dict, args.output_path+f"checkpoint_{epoch}.pth")


class Tester(object):

    def test(self, model_cl, model_reg, dataloader,low_high_graph, args, accelerator=None):
        model_cl.eval()
        model_reg.eval()
        step = 0 
        # device = args.device if accelerator is None else accelerator.device
        # to_device = ToDevice(device)
        with torch.no_grad():    
            for graph in dataloader:

                y_pred_reg = model_reg(graph)
                y_pred_reg = torch.expm1(y_pred_reg)
                low_high_graph.pr_reg[:,graph.t] = torch.where(y_pred_reg >= 0.1, y_pred_reg, torch.tensor(0.0, dtype=y_pred_reg.dtype)).cpu()
                #low_high_graph.pr_reg[:,graph.t] = y_pred_reg.unsqueeze(-1).cpu()

                y_pred_cl = model_cl(graph)
                y_pred_cl = torch.sigmoid(y_pred_cl)
                low_high_graph.pr_cl[:,graph.t] = torch.where(y_pred_cl > 0.5, 1.0, 0.0).cpu()
                #low_high_graph.pr_cl[:,graph.t] = y_pred_cl.cpu()
                
                #y_pred_cl = torch.nn.functional.softmax(y_pred_cl)
                #low_high_graph.pr_cl[:,graph.t] = torch.argmax(y_pred_cl, dim=-1).unsqueeze(-1).float().cpu()
                #low_high_graph.pr_cl[:,graph.t] = y_pred_cl.unsqueeze(-1).cpu()
                
                if step % 100 == 0:
                    with open(args.output_path+args.log_file, 'a') as f:
                        f.write(f"\nStep {step} done.")
                step += 1 
        low_high_graph["pr"] = low_high_graph.pr_cl * low_high_graph.pr_reg 
        return

    def test_combined(self, model, dataloader,low_high_graph, args, accelerator=None):
        model.eval()
        step = 0
        with torch.no_grad():    
            for graph in dataloader:
                # cl
                y_pred_cl, y_pred_reg = model(graph)
                y_pred_cl = torch.sigmoid(y_pred_cl)
                low_high_graph.pr_cl[:,graph.t] = torch.where(y_pred_cl > 0.5, 1.0, 0.0).cpu()
                y_pred_reg = torch.expm1(y_pred_reg * y_pred_cl)
                low_high_graph.pr_reg[:,graph.t] = torch.where(y_pred_reg >= 0.1, y_pred_reg, torch.tensor(0.0, dtype=y_pred_reg.dtype)).cpu()
                if step % 100 == 0:
                    with open(args.output_path+args.log_file, 'a') as f:
                        f.write(f"\nStep {step} done.")
                step += 1 
        low_high_graph["pr"] = low_high_graph.pr_cl * low_high_graph.pr_reg 
        return

