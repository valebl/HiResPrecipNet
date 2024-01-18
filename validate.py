import numpy as np
import pickle
import torch
import argparse
import time
import os
import matplotlib.pyplot as plt
import sys
from torch import nn

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

parser.add_argument('--checkpoint_cl_path', type=str)
parser.add_argument('--checkpoint_reg_path', type=str)
parser.add_argument('--checkpoint', type=str)
parser.add_argument('--output_file', type=str, default="G_predictions.pkl")

parser.add_argument('--graph_file', type=str, default=None) 
parser.add_argument('--model_cl', type=str, default=None) 
parser.add_argument('--model_reg', type=str, default=None) 

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

    loss_fn_reg = getattr(utils, "weighted_mse_loss")
    loss_fn_cl = nn.CrossEntropyLoss(weight=torch.tensor([0.25, 0.75]))

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

    with open(args.input_path+"pr_gripho.pkl", 'rb') as f:
        pr_gripho = pickle.load(f)

    with open(args.input_path+args.graph_file, 'rb') as f:
        low_high_graph = pickle.load(f)


    pr_gripho = pr_gripho[:, test_start_idx:test_end_idx]
    pr_reg = torch.ones(pr_gripho.shape) * torch.nan
    pr_cl = torch.ones(pr_gripho.shape) * torch.nan
    pr = torch.ones(pr_gripho.shape) * torch.nan

    low_high_graph['low'].x = low_high_graph['low'].x[:,test_start_idx:test_end_idx,:]    
    low_high_graph.pr_gripho = pr_gripho
    low_high_graph.pr_cl = pr_cl
    low_high_graph.pr_reg = pr_reg


    dataset_graph = Dataset_Graph(targets=None, graph=low_high_graph)
    
    custom_collate_fn = getattr(dataset, 'custom_collate_fn_graph')
        
    sampler_graph = Iterable_Graph(dataset_graph=dataset_graph, shuffle=False)
        
    dataloader = torch.utils.data.DataLoader(dataset_graph, batch_size=args.batch_size, num_workers=0,
                    sampler=sampler_graph, collate_fn=custom_collate_fn)

    Model_cl = getattr(models, args.model_cl)
    Model_reg = getattr(models, args.model_reg)
    model_cl = Model_cl()
    model_reg = Model_reg()

    loss_reg = []
    loss_cl = []
    acc = []
    acc_class0 = []
    acc_class1 = []

    start = time.time()

    for epoch in range(args.last_epoch):

        if epoch == 0:
            output = True
        else:
            output = False

        checkpoint_cl = args.checkpoint_cl_path + f"checkpoint_{epoch}.pth"
        checkpoint_reg = args.checkpoint_reg_path + f"checkpoint_{epoch}.pth"
        

        if accelerator is None or accelerator.is_main_process:
            with open(args.output_path + args.log_file, 'a') as f:
                f.write("\n\nEpoch: {epoch}")            

        if output and accelerator is None or accelerator.is_main_process:
            with open(args.output_path + args.log_file, 'a') as f:
                f.write("\nClassifier:")    
        if accelerator is None:
            checkpoint_cl = torch.load(checkpoint_cl, map_location=torch.device('cpu'))
        else:
            checkpoint_cl = torch.load(checkpoint_cl)
        model_cl = load_checkpoint(model_cl, checkpoint_cl, args.output_path, args.log_file, None, 
                net_names=["low2high.", "low_net.", "high_net."], fine_tuning=False, device=accelerator.device, output=output)
        
        if output and accelerator is None or accelerator.is_main_process:
            with open(args.output_path + args.log_file, 'a') as f:
                f.write("\nRegressor:")
        if accelerator is None:
            checkpoint_reg = torch.load(checkpoint_reg, map_location=torch.device('cpu'))
        else:
            checkpoint_reg = torch.load(checkpoint_reg)
        model_reg = load_checkpoint(model_reg, checkpoint_reg, args.output_path, args.log_file, None,
                net_names=["low2high.", "low_net.", "high_net."], fine_tuning=False, device=accelerator.device, output=output)

        if accelerator is not None:
            model_cl, model_reg, dataloader = accelerator.prepare(model_cl, model_reg, dataloader)
        else:
            model_cl = model_cl.cuda()
            model_reg = model_reg.cuda()

        tester = Tester()

        loss_reg_epoch, loss_cl_epoch, acc_epoch, acc_class0_epoch, acc_class1_epoch = tester.validate(
            model_cl, model_reg, dataloader, loss_fn_reg, loss_fn_cl)
        
        loss_reg.append(loss_reg_epoch.item())
        loss_cl.append(loss_cl_epoch.item())
        acc.append(acc_epoch.item())
        acc_class0.append(acc_class0_epoch.item())
        acc_class1.append(acc_class1_epoch.item())
    
    end = time.time()

    with open(args.output_path + "loss_reg.pkl", 'wb') as f:
        pickle.dump(torch.tensor((loss_reg)))

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

        