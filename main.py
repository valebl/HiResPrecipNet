from torch_geometric.data import HeteroData
import torch
from torch import nn
import torchvision.ops
import matplotlib.pyplot as plt
import numpy as np
import pickle
from dataset import Dataset_Graph, Dataset_Graph_Combined, Iterable_Graph
import dataset
import time
import argparse
import sys
import os
import dataset

import HiResPrecipNet as models
import utils
from utils import Trainer, date_to_idxs, load_checkpoint

from accelerate import Accelerator

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

#-- paths
parser.add_argument('--input_path', type=str, help='path to input directory')
parser.add_argument('--output_path', type=str, help='path to output directory')
parser.add_argument('--log_file', type=str, default='log.txt', help='log file')

parser.add_argument('--target_file', type=str, default=None)
parser.add_argument('--graph_file', type=str, default=None) 
parser.add_argument('--weights_file', type=str, default=None) 

parser.add_argument('--out_checkpoint_file', type=str, default="checkpoint.pth")
parser.add_argument('--out_loss_file', type=str, default="loss.csv")

parser.add_argument('--use_accelerate',  action='store_true')
parser.add_argument('--no-use_accelerate', dest='use_accelerate', action='store_false')
parser.add_argument('--wandb_project_name', type=str)

#-- training hyperparameters
parser.add_argument('--pct_trainset', type=float, default=1.0, help='percentage of dataset in trainset')
parser.add_argument('--epochs', type=int, default=15, help='number of total training epochs')
parser.add_argument('--batch_size', type=int, default=64, help='batch size (global)')
parser.add_argument('--step_size', type=int, default=10, help='scheduler step size (global)')
parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay (wd)')
parser.add_argument('--fine_tuning',  action='store_true')
parser.add_argument('--no-fine_tuning', dest='fine_tuning', action='store_false')
parser.add_argument('--load_checkpoint',  action='store_true')
parser.add_argument('--no-load_checkpoint', dest='load_checkpoint', action='store_false')

parser.add_argument('--checkpoint_ctd', type=str, help='checkpoint to load to continue')
parser.add_argument('--ctd_training',  action='store_true')
parser.add_argument('--no-ctd_training', dest='ctd_training', action='store_false')

parser.add_argument('--loss_fn', type=str, default="mse_loss")
parser.add_argument('--lon_min', type=float)
parser.add_argument('--lon_max', type=float)
parser.add_argument('--lat_min', type=float)
parser.add_argument('--lat_max', type=float)
parser.add_argument('--interval', type=float, default=0.25)

parser.add_argument('--model_type', type=str)
parser.add_argument('--model_name', type=str, default='HiResPrecipNet')

#-- start and end training dates
parser.add_argument('--train_year_start', type=float)
parser.add_argument('--train_month_start', type=float)
parser.add_argument('--train_day_start', type=float)
parser.add_argument('--train_year_end', type=float)
parser.add_argument('--train_month_end', type=float)
parser.add_argument('--train_day_end', type=float)
parser.add_argument('--first_year', type=float)

if __name__ == '__main__':

    args = parser.parse_args()

    torch.backends.cudnn.benchmark = False

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)


#-----------------------------------------------------
#--------------- WANDB and ACCELERATE ----------------
#-----------------------------------------------------

    if args.use_accelerate is True:
        accelerator = Accelerator(log_with="wandb")
    else:
        accelerator = None
    
    os.environ['WANDB_API_KEY'] = 'b3abf8b44e8d01ae09185d7f9adb518fc44730dd'
    os.environ['WANDB_USERNAME'] = 'valebl'
    os.environ['WANDB_MODE'] = 'offline'
    os.environ['WANDB_CONFIG_DIR']='./wandb/'

    accelerator.init_trackers(
            project_name=args.wandb_project_name
        )

    if accelerator is None or accelerator.is_main_process:
        with open(args.output_path+args.log_file, 'w') as f:
            f.write("Starting the training...")
            f.write(f"Cuda is available: {torch.cuda.is_available()}. There are {torch.cuda.device_count()} available GPUs.")


#-----------------------------------------------------
#--------------- MODEL, LOSS, OPTIMIZER --------------
#-----------------------------------------------------

    Model = getattr(models, args.model_name)
    model = Model()

    # Loss
    if args.loss_fn == 'sigmoid_focal_loss':
        loss_fn = getattr(torchvision.ops, args.loss_fn)
    elif args.loss_fn == 'weighted_cross_entropy_loss':
        loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([0.1, 1]))
    elif args.loss_fn == 'weighted_mse_loss':
        loss_fn = getattr(utils, args.loss_fn)
    elif args.loss_fn == 'quantile_loss':
        loss_fn = getattr(utils, args.loss_fn) 
    elif args.loss_fn == 'poly_loss':
        loss_fn = getattr(utils, "PolyLoss")
        loss_fn = loss_fn()
    else:
        loss_fn = getattr(nn.functional, args.loss_fn) 
    

#-----------------------------------------------------
#-------------- DATASET AND DATALOADER ---------------
#-----------------------------------------------------

    train_start_idx, train_end_idx = date_to_idxs(args.train_year_start, args.train_month_start,
                                                                      args.train_day_start, args.train_year_end, args.train_month_end,
                                                                      args.train_day_end, args.first_year)
    train_start_idx = max(train_start_idx,25)
    
    with open(args.output_path+args.log_file, 'a') as f:
        f.write(f"Train from {int(args.train_day_start)}/{int(args.train_month_start)}/{int(args.train_year_start)} to " +
                f"{int(args.train_day_end)}/{int(args.train_month_end)}/{int(args.train_year_end)}. Idxs from {train_start_idx} to {train_end_idx}.")

    with open(args.input_path+args.graph_file, 'rb') as f:
        low_high_graph = pickle.load(f)

    low_high_graph['low'].x = low_high_graph['low'].x[:,train_start_idx:train_end_idx,:]

    with open(args.input_path+args.target_file, 'rb') as f:
        target_train = pickle.load(f)

    if args.loss_fn == 'weighted_mse_loss':
        with open(args.input_path+args.weights_file, 'rb') as f:
            weights_reg = pickle.load(f)
        with open(args.output_path+args.log_file, 'a') as f:
            f.write("Not using weights in the loss.")
        dataset_graph = Dataset_Graph(targets=target_train[:,train_start_idx:train_end_idx],
            w=weights_reg[:,train_start_idx:train_end_idx], graph=low_high_graph)
    else:
        with open(args.output_path+args.log_file, 'a') as f:
            f.write("Not using weights in the loss.")
        dataset_graph = Dataset_Graph(targets=target_train[:,train_start_idx:train_end_idx],
            graph=low_high_graph)

    custom_collate_fn = getattr(dataset, 'custom_collate_fn_graph')
        
    sampler_graph = Iterable_Graph(dataset_graph=dataset_graph, shuffle=True)
        
    dataloader = torch.utils.data.DataLoader(dataset_graph, batch_size=args.batch_size, num_workers=0,
                    sampler=sampler_graph, collate_fn=custom_collate_fn)

    if accelerator is None or accelerator.is_main_process:
        total_memory, used_memory, free_memory = map(int, os.popen('free -t -m').readlines()[-1].split()[1:])
        with open(args.output_path+args.log_file, 'a') as f:
            f.write(f"\nRAM memory {round((used_memory/total_memory) * 100, 2)} %")


#-----------------------------------------------------
#------------------ LOAD PARAMETERS ------------------
#-----------------------------------------------------

    epoch_start=0
    
    if args.ctd_training:
        if accelerator is None or accelerator.is_main_process:
            with open(args.output_path+args.log_file, 'a') as f:
                f.write("Continuing the training.")
        checkpoint = torch.load(args.checkpoint_ctd)
        model = load_checkpoint(model, checkpoint, args.output_path, args.log_file, None,
            net_names=["low2high.", "low_net.", "high_net."], fine_tuning=True, device=accelerator.device)
        epoch_start = checkpoint["epoch"] + 1

    #-- define the optimizer and trainable parameters
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.ctd_training:
        optimizer.load_state_dict(checkpoint["optimizer"])
    
    #if args.ctd_training:
    #    optimizer.load_state_dict(checkpoint["optimizer"]

    #check_freezed_layers(model, args.output_path, args.log_file, accelerator)

    #total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    #if accelerator is None or accelerator.is_main_process: 
    #    with open(args.output_path+args.log_file, 'a') as f:
    #        f.write(f"\nTotal number of trainable parameters: {total_params}.")

    if accelerator is not None:
        model, optimizer, dataloader, loss_fn = accelerator.prepare(model, optimizer, dataloader, loss_fn)
        if accelerator.is_main_process:
            with open(args.output_path+args.log_file, 'a') as f:
                f.write("Using accelerator to prepare model, optimizer, dataloader and loss_fn...")
    else:
        with open(args.output_path+args.log_file, 'a') as f:
            f.write("Not using accelerator to prepare model, optimizer, dataloader and loss_fn...")
        model = model.cuda()

    if args.ctd_training:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1, last_epoch=epoch_start)
        #lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.5, last_epoch=-1)
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)
#-----------------------------------------------------
#----------------------- TRAIN -----------------------
#-----------------------------------------------------

    if accelerator is None or accelerator.is_main_process:
        with open(args.output_path+args.log_file, 'a') as f:
            f.write(f"\nStarting with pct_trainset={args.pct_trainset}, lr={args.lr}, "+
                f"weight decay = {args.weight_decay} and epochs={args.epochs}." + 
                f"loss: {loss_fn}") 
            if accelerator is None:
                f.write(f"\nModel = {args.model_name}, batch size = {args.batch_size}")
            else:
                f.write(f"\nModel = {args.model_name}, batch size = {args.batch_size*torch.cuda.device_count()}")

    start = time.time()

    trainer = Trainer()
    trainer.train(model, dataloader, optimizer, loss_fn, lr_scheduler, accelerator, args, epoch_start=epoch_start)
        
    end = time.time()

    if accelerator is None or accelerator.is_main_process:
        with open(args.output_path+args.log_file, 'a') as f:
            f.write(f"\nCompleted in {end - start} seconds.")
            f.write(f"\nDONE!")
    
    
    
    

