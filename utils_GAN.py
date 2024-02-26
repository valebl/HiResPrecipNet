import time
import sys
import pickle
import torch.nn as nn

import torch

from datetime import datetime, timedelta, date
from torch_geometric.transforms import ToDevice
#from pytorch_forecasting.metrics.quantile import QuantileLoss

from datetime import datetime, date

import torch
import torch.nn.functional as F


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
#---------------------- LOSSES -----------------------
#-----------------------------------------------------

def weighted_mse_loss(input_batch, target_batch, weights):
    #return (weights * (input_batch - target_batch) ** 2).sum() / weights.sum()
    return torch.mean(weights * (input_batch - target_batch) ** 2)

def weighted_mse_loss_ASYM(input_batch, target_batch, weights):
    return torch.mean(torch.abs(input_batch - target_batch) + weights**2 * torch.clamp(target_batch - input_batch, min=0))

def MSE_weighted2(y_true, y_pred):
    return torch.mean(torch.exp(2.0 * torch.expm1(y_true)) * (y_pred - y_true)**2)

#class modified_mse_quantile_loss():
#    def __init__(self, q=0.85, alpha=0.2):
#        self.mse_loss = nn.MSELoss()
#        self.q = q
#        self.alpha = alpha
#    
#    def __call__(self, prediction_batch, target_batch):
#        loss_quantile = torch.mean(torch.max(self.q*(prediction_batch-target_batch), (1-self.q)*(prediction_batch-target_batch)))
#        loss_mse = self.mse_loss(prediction_batch, target_batch) 
#        return self.alpha * loss_mse + (1-self.alpha) * loss_quantile
    

class Reconstruction_loss():
    def __init__(self, q=0.85, alpha=0.2):
        self.q = q
        self.alpha = alpha

    def __call__(self, prediction_batch, target_batch):
        loss_quantile = torch.max(self.q*torch.clamp(prediction_batch-target_batch, min=0), (1-self.q)*torch.clamp(prediction_batch-target_batch, min=0))
        loss_mae = torch.abs(prediction_batch, target_batch) 
        return torch.mean(loss_mae + loss_quantile)


#-----------------------------------------------------
#------------------ LOAD PARAMETERS ------------------
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
#------------------ TRAIN AND TEST -------------------
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

                #-- one ->
                y_pred = model(graph).squeeze()[train_mask]
                y = graph['high'].y[train_mask]
                loss = loss_fn(y_pred, y, alpha, gamma, reduction='mean')
                accelerator.backward(loss)
                #torch.nn.utils.clip_grad_norm_(model.parameters(),5)
                accelerator.clip_grad_norm_(model.parameters(), 5)
                optimizer.step()
                loss_meter.update(val=loss.item(), n=1)    
                acc = accuracy_binary_one(y_pred, y)
                acc_class0, acc_class1 = accuracy_binary_one_classes(y_pred, y)
          
                #-- two ->
                # y_pred = model(graph)[train_mask]
                # y = graph['high'].y[train_mask]
                # loss = loss_fn(y_pred, y.to(torch.int64))
                # accelerator.backward(loss)
                # torch.nn.utils.clip_grad_norm_(model.parameters(),5)
                # optimizer.step()
                # loss_meter.update(val=loss.item(), n=1)    
                # acc = accuracy_binary_two(y_pred, y)
                # acc_class0, acc_class1 = accuracy_binary_two_classes(y_pred, y)

                #-- all ->
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
                optimizer.zero_grad()

                #-- one ->
                y_pred = model(graph).squeeze()[train_mask]
                y = graph['high'].y[train_mask]
                loss = loss_fn(y_pred, y, alpha, gamma, reduction='mean')
                acc = accuracy_binary_one(y_pred, y)
                acc_class0, acc_class1 = accuracy_binary_one_classes(y_pred, y)   

                #-- two ->
                # y_pred = model(graph)[train_mask]
                # y = graph['high'].y[train_mask]
                # loss = loss_fn(y_pred, y.to(torch.int64))
                # acc = accuracy_binary_two(y_pred, y)
                # acc_class0, acc_class1 = accuracy_binary_two_classes(y_pred, y)

                #-- all ->
                loss_meter_val.update(val=loss.item(), n=1)    
                acc_meter_val.update(val=acc, n=1)
                acc_class0_meter_val.update(val=acc_class0, n=1)
                acc_class1_meter_val.update(val=acc_class1, n=1)
            
            accelerator.log({'validation loss': loss_meter_val.avg, 'validation accuracy': acc_meter_val.avg,
                                'validation accuracy class0': acc_class0_meter_val.avg, 'validation accuracy class1': acc_class1_meter_val.avg})

            #if accelerator.is_main_process:
            #    checkpoint_dict = {
            #        "parameters": model.module.state_dict(),
            #        "optimizer": optimizer.state_dict(),
            #        "epoch": epoch,
            #        }
            #    torch.save(checkpoint_dict, args.output_path+f"checkpoint_{epoch}.pth")
        return model

    def train_gan(self, model_G, model_D, dataloader_train_real, dataloader_train_fake, dataloader_val_real, dataloader_val_fake, optimizer_G, optimizer_D,
                  loss_fn_G_adv, loss_fn_G_rec, loss_fn_D, lr_scheduler_G, lr_scheduler_D, accelerator, args, epoch_start=0, alpha=1):

        real_label = 1
        fake_label = 0

        if accelerator.is_main_process:
            with open(args.output_path+args.log_file, 'a') as f:
                f.write(f"\nStart training the regressor.")
        for epoch in range(epoch_start, epoch_start+args.epochs):
            model_D.train()
            model_G.train()
            if accelerator.is_main_process:
                with open(args.output_path+args.log_file, 'a') as f:
                    f.write(f"\nEpoch {epoch+1} --- learning rate {optimizer_G.param_groups[0]['lr']:.8f}")
            
            ## Discriminator
            loss_meter_D = AverageMeter()
            loss_meter_D_real = AverageMeter()
            loss_meter_D_fake = AverageMeter()
            ## Generator
            loss_meter_G = AverageMeter()
            loss_meter_G_adv = AverageMeter()
            loss_meter_G_rec = AverageMeter()
            ## Validation
            loss_meter_val_D = AverageMeter()
            loss_meter_val_G = AverageMeter()
            
            start = time.time()

            for graph_real, graph_fake in zip(dataloader_train_real, dataloader_train_fake):
                
                ##----------------------------------------##
                ##--- Part 1 - Train the Discriminator ---##
                ##----------------------------------------##
                
                optimizer_D.zero_grad()
                ## 1a Real examples
                output = model_D(graph_real).squeeze()
                loss_D_real = loss_fn_D(output, (torch.ones(output.shape)*real_label).to(accelerator.device))
                loss_D_real.backward()
                ## 1b Fake examples
                y_graph_fake = graph_fake['high'].y.clone()      # save ground truth to use in reconstruction loss
                graph_fake['high'].y = model_G(graph_fake)      # derive fake graph from Generator
                output = model_D(graph_fake).squeeze()
                loss_D_fake = loss_fn_D(output, (torch.ones(output.shape)*fake_label).to(accelerator.device))
                loss_D_fake.backward(retain_graph=True)
                optimizer_D.step()    
                loss_D = loss_D_real + loss_D_fake  
                
                ##---------------------------------------##
                ##--- Part 2 - Train the Generator ------##
                ##---------------------------------------##
                
                optimizer_G.zero_grad()
                output = model_D(graph_fake).squeeze()
                loss_G_adv = loss_fn_G_adv(output, (torch.ones(output.shape)*real_label).to(accelerator.device))
                loss_G_rec = loss_fn_G_rec(output, y_graph_fake)
                loss_G = loss_G_adv + alpha * loss_G_rec
                loss_G.backward()
                optimizer_G.step()
                
                ##---------------------------------------##
                ##--- Log on wandb ----------------------##
                ##---------------------------------------##
                
                ## Discriminator                
                loss_meter_D.update(val=loss_D.item(), n=1)    
                accelerator.log({'epoch':epoch, 'loss D iteration': loss_meter_D.val, 'loss D avg': loss_meter_D.avg})
                loss_meter_D_real.update(val=loss_D_real.item(), n=1)    
                loss_meter_D_fake.update(val=loss_D_fake.item(), n=1)    

                ## Generator
                loss_meter_G.update(val=loss_G.item(), n=1)    
                accelerator.log({'epoch':epoch, 'loss G iteration': loss_meter_G.val, 'loss G avg': loss_meter_G.avg})
                loss_meter_G_adv.update(val=loss_G_adv.item(), n=1)    
                loss_meter_G_rec.update(val=loss_G_rec.item(), n=1)    

            end = time.time()
            accelerator.log({'loss D epoch': loss_meter_D.avg})
            accelerator.log({'loss D real epoch': loss_meter_D_real.avg})
            accelerator.log({'loss D fake epoch': loss_meter_D_fake.avg})
            accelerator.log({'loss G epoch': loss_meter_G.avg})
            accelerator.log({'loss G adversarial epoch': loss_meter_G_adv.avg})
            accelerator.log({'loss G reconstruction epoch': loss_meter_G_rec.avg})

            if accelerator.is_main_process:
                with open(args.output_path+args.log_file, 'a') as f:
                    f.write(f"\nEpoch {epoch+1} completed in {end - start:.4f} seconds. Loss D - total: {loss_meter_D.sum:.4f} - average: {loss_meter_D.avg:.10f}. " + 
                            f"Loss G - total: {loss_meter_G.sum:.4f} - average: {loss_meter_G.avg:.10f}.")
            
            # if lr_scheduler is not None and lr_scheduler.get_last_lr()[0] > 0.00001:
            #     lr_scheduler.step()
            
            accelerator.save_state(output_dir=args.output_path+f"checkpoint_{epoch}/")
            torch.save({"epoch": epoch}, args.output_path+f"checkpoint_{epoch}/epoch")

            # Perform validation step
            model_D.eval()
            model_G.eval()
            for graph_real, graph_fake in zip(dataloader_val_real, dataloader_val_fake):
                ##--- Part 1 - Discriminator ---##
                output = model_D(graph_real).squeeze()
                loss_D_real = loss_fn_D(output, (torch.ones(output.shape)*real_label).to(accelerator.device))
                y_graph_fake = graph_fake['high'].y.copy()
                graph_fake['high'].y = model_G(graph_fake)
                output = model_D(graph_fake).squeeze()
                loss_D_fake = loss_fn_D(output, (torch.ones(output.shape)*fake_label).to(accelerator.device))
                loss_D = loss_D_real + loss_D_fake
                ##--- Part 2 - Generator ---##
                loss_G_adv = loss_fn_G_adv(output, (torch.ones(output.shape)*real_label).to(accelerator.device))
                loss_G_rec = loss_fn_G_rec(output, y_graph_fake)
                loss_G = loss_G_adv + alpha * loss_G_rec

                loss_meter_val_D.update(val=loss_D.item(), n=1)    

            accelerator.log({'validation loss D epoch': loss_meter_val_D.avg})
            accelerator.log({'validation loss G epoch': loss_meter_val_G.avg})

        return model_G, model_D


class Tester(object):

    def test(self, model_cl, model_reg, dataloader,low_high_graph, args, accelerator=None):
        model_cl.eval()
        model_reg.eval()
        step = 0 
        # device = args.device if accelerator is None else accelerator.device
        # to_device = ToDevice(device)
        pr_cl = []
        pr_reg = []
        times = []
        with torch.no_grad():    
            for graph in dataloader:

                t = graph.t
                times.append(t)
                
                # Regressor
                y_pred_reg = model_reg(graph)
                # low_high_graph.pr_reg[:,t] = torch.expm1(y_pred_reg).cpu()
                pr_reg.append(torch.expm1(y_pred_reg))
#                pr_reg.append(y_pred_reg)

                # Classifier
                y_pred_cl = model_cl(graph)
                #-- (weighted) cross entropy loss ->
                #low_high_graph.pr_cl[:,t] = torch.argmax(torch.nn.functional.softmax(y_pred_cl, dim=-1), dim =-1).unsqueeze(-1).float().cpu()
                #-- <-
                #-- sigmoid focal loss ->
                # low_high_graph.pr_cl[:,t] = torch.where(y_pred_cl >= 0.0, 1.0, 0.0).cpu()
                pr_cl.append(torch.where(y_pred_cl >= 0.0, 1.0, 0.0))
#                pr_cl.append(torch.nn.functional.sigmoid(y_pred_cl))
                #  pr_cl.append(y_pred_cl)
                #-- <-
                
                if step % 100 == 0:
                    if accelerator is None or accelerator.is_main_process:
                        with open(args.output_path+args.log_file, 'a') as f:
                            f.write(f"\nStep {step} done.")
                step += 1 

        #Comined classifier and regressor
        #low_high_graph["pr"] = low_high_graph.pr_cl * low_high_graph.pr_reg 
        #low_high_graph["pr"] = torch.where(y_pred_cl > 0.0, 1.0, 0.0).cpu() * low_high_graph.pr_reg 
                
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
                #-- (weighted) cross entropy loss ->
                #low_high_graph.pr_cl[:,t] = torch.argmax(torch.nn.functional.softmax(y_pred_cl, dim=-1), dim =-1).unsqueeze(-1).float().cpu()
                #-- <-
                #-- sigmoid focal loss ->
                pr_cl.append(torch.where(y_pred_cl >= 0.0, 1.0, 0.0))
                # pr_cl.append(pr_cl)
                #-- <-
                
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
                #loss = loss_fn(y_pred, y)

                loss_meter.update(val=loss.item(), n=1)   

        return loss_meter.avg

