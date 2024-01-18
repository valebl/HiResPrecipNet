import numpy as np
import pickle
import torch
import argparse
import time
import os
import matplotlib.pyplot as plt
import sys
from torch import nn
import torchvision.ops

from accelerate import Accelerator

import HiResPrecipNet as models
import dataset
from dataset import Dataset_Graph, Iterable_Graph

import utils
from utils import date_to_idxs, load_checkpoint, Tester

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

#-- paths
parser.add_argument('--input_path', type=str, help='path to input directory')
parser.add_argument('--output_path', type=str, help='path to output directory')
parser.add_argument('--log_file', type=str, default='log.txt', help='log file')

parser.add_argument('--checkpoint_path', type=str)
parser.add_argument('--output_file', type=str, default="G_predictions.pkl")

parser.add_argument('--graph_file', type=str, default=None) 
parser.add_argument('--target_file', type=str, default=None) 
parser.add_argument('--model_name', type=str, default=None) 
parser.add_argument('--model_type', type=str, default=None) 
parser.add_argument('--loss_fn', type=str, default=None) 

parser.add_argument('--lon_min', type=float)
parser.add_argument('--lon_max', type=float)
parser.add_argument('--lat_min', type=float)
parser.add_argument('--lat_max', type=float)
parser.add_argument('--interval', type=float, default=0.25)

#-- start and end training dates
parser.add_argument('--test_year_start', type=int)
parser.add_argument('--test_month_start', type=int)
parser.add_argument('--test_day_start', type=int)
parser.add_argument('--test_year_end', type=int)
parser.add_argument('--test_month_end', type=int)
parser.add_argument('--test_day_end', type=int)
parser.add_argument('--first_year', type=int)
parser.add_argument('--last_epoch', type=int)

parser.add_argument('--batch_size', type=int)

parser.add_argument('--use_accelerate',  action='store_true')
parser.add_argument('--no-use_accelerate', dest='use_accelerate', action='store_false')


if __name__ == '__main__':

    args = parser.parse_args()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
        
    if args.use_accelerate is True:
        accelerator = Accelerator()
    else:
        accelerator = None

    Model = getattr(models, args.model_name)
    model = Model()

    # Loss
    if args.loss_fn == 'sigmoid_focal_loss':
        loss_fn = getattr(torchvision.ops, args.loss_fn)
    elif args.loss_fn == 'weighted_cross_entropy_loss':
        loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([0.25, 0.75]))
    elif args.loss_fn == 'weighted_mse_loss':
        loss_fn = getattr(utils, args.loss_fn)
    elif args.loss_fn == 'quantile_loss':
        loss_fn = getattr(utils, args.loss_fn) 
    elif args.loss_fn == 'poly_loss':
        loss_fn = getattr(utils, "PolyLoss")
        loss_fn = loss_fn()
    else:
        loss_fn = getattr(nn.functional, args.loss_fn) 

    if accelerator is None or accelerator.is_main_process:
        with open(args.output_path + args.log_file, 'w') as f:
            f.write(f"Starting!")

    if accelerator is None or accelerator.is_main_process:
        with open(args.output_path+args.log_file, 'w') as f:
            f.write("Starting the training...")
            f.write(f"Cuda is available: {torch.cuda.is_available()}. There are {torch.cuda.device_count()} available GPUs.")

    test_start_idx, test_end_idx = date_to_idxs(args.test_year_start, args.test_month_start,
                                                args.test_day_start, args.test_year_end, args.test_month_end,
                                                args.test_day_end, args.first_year)
    
    if accelerator is None or accelerator.is_main_process:
        with open(args.output_path + args.log_file, 'a') as f:
            f.write(f"\nStarting the validation, from {int(args.test_day_start)}/{int(args.test_month_start)}/{int(args.test_year_start)} to " +
                    f"{int(args.test_day_end)}/{int(args.test_month_end)}/{int(args.test_year_end)}.")

    with open(args.input_path+args.graph_file, 'rb') as f:
        low_high_graph = pickle.load(f)

    with open(args.input_path+args.target_file, 'rb') as f:
        target_train = pickle.load(f)

    # Define input and target
    low_high_graph['low'].x = low_high_graph['low'].x[:,test_start_idx:test_end_idx,:]
    target_train = target_train[:,test_start_idx:test_end_idx]

    # Define a mask to ignore time indexes with all nan values
    mask_all_nan = []
    initial_time_dim = target_train.shape[1]
    for t in range(initial_time_dim):
        nan_sum = target_train[:,t].isnan().sum()
        mask_all_nan.append(nan_sum < target_train.shape[0])
    mask_all_nan = torch.stack(mask_all_nan)

    if accelerator is None or accelerator.is_main_process:
        with open(args.output_path+args.log_file, 'a') as f:
            f.write(f"\nAfter removing all nan time indexes, {mask_all_nan.sum()}" +
                    f" time indexes are considered ({(mask_all_nan.sum() / initial_time_dim * 100):.1f} % of initial ones).")

    low_high_graph['low'].x = low_high_graph['low'].x[:,mask_all_nan]
    target_train = target_train[:,mask_all_nan]

    if args.loss_fn == 'weighted_mse_loss':
        with open(args.input_path+args.weights_file, 'rb') as f:
            weights_reg = pickle.load(f)
        weights_reg = weights_reg[:,test_start_idx:test_end_idx]
        weights_reg = weights_reg[:,mask_all_nan]

        if accelerator is None or accelerator.is_main_process:
            with open(args.output_path+args.log_file, 'a') as f:
                f.write("\nUsing weights in the loss.")

        dataset_graph = Dataset_Graph(targets=target_train,
            w=weights_reg, graph=low_high_graph)
    else:
        if accelerator is None or accelerator.is_main_process:
            with open(args.output_path+args.log_file, 'a') as f:
                f.write("\nNot using weights in the loss.")
        dataset_graph = Dataset_Graph(targets=target_train,
            graph=low_high_graph)

    custom_collate_fn = getattr(dataset, 'custom_collate_fn_graph')
        
    sampler_graph = Iterable_Graph(dataset_graph=dataset_graph, shuffle=True)
        
    dataloader = torch.utils.data.DataLoader(dataset_graph, batch_size=args.batch_size, num_workers=0,
                    sampler=sampler_graph, collate_fn=custom_collate_fn)

    if args.model_type == "reg":
        loss_reg = []
    elif args.model_type == "cl":
        loss_cl = []
        acc = []
        acc_class0 = []
        acc_class1 = []

    start = time.time()
    
    if accelerator is None or accelerator.is_main_process:
        with open(args.output_path + args.log_file, 'a') as f:
            f.write(f"\n{args.model_name}:")   

    for epoch in range(args.last_epoch):

        output = True if epoch == 0 else False

        checkpoint = args.checkpoint_path + f"checkpoint_{epoch}.pth"       

        if accelerator is None or accelerator.is_main_process:
            with open(args.output_path + args.log_file, 'a') as f:
                f.write(f"\n\nEpoch: {epoch}")            
 
        if accelerator is None:
            checkpoint = torch.load(checkpoint, map_location=torch.device('cpu'))
        else:
            checkpoint = torch.load(checkpoint)

        model = load_checkpoint(model, checkpoint, args.output_path, args.log_file, accelerator, 
                net_names=["low2high.", "low_net.", "high_net."], fine_tuning=False, device=accelerator.device, output=output)

        if accelerator is not None:
            model, dataloader, loss_fn = accelerator.prepare(model, dataloader, loss_fn)
        else:
            model = model.cuda()

        tester = Tester()

        if args.model_type == "reg":
            loss_reg_epoch, = tester.validate_reg(model, dataloader, loss_fn)
            loss_reg.append(loss_reg_epoch)

        elif args.model_type == "cl":
            loss_cl_epoch, acc_epoch, acc_class0_epoch, acc_class1_epoch = tester.validate_cl(model, dataloader, loss_fn)
            loss_cl.append(loss_cl_epoch)
            acc.append(acc_epoch)
            acc_class0.append(acc_class0_epoch)
            acc_class1.append(acc_class1_epoch)
    
    end = time.time()

    if args.model_type == "reg":
        with open(args.output_path + "loss_reg.pkl", 'wb') as f:
            pickle.dump(torch.tensor((loss_reg)))

    elif args.model_type == "cl":
        with open(args.output_path + "loss_cl.pkl", 'wb') as f:
            pickle.dump(torch.tensor((loss_cl)))
        
        with open(args.output_path + "acc.pkl", 'wb') as f:
            pickle.dump(torch.tensor((acc)))
        
        with open(args.output_path + "acc_class0.pkl", 'wb') as f:
            pickle.dump(torch.tensor((acc_class0)))

        with open(args.output_path + "acc_class1.pkl", 'wb') as f:
            pickle.dump(torch.tensor((acc_class1)))

    if accelerator is None or accelerator.is_main_process:
        with open(args.output_path + args.log_file, 'a') as f:
            f.write(f"\nDone. Validation concluded in {end-start} seconds.")
            f.write("\nWrite the files.")

    if accelerator is None or accelerator.is_main_process:
        with open(args.output_path + args.output_file, 'wb') as f:
            pickle.dump(low_high_graph, f)

        