import numpy as np
import pickle
import torch
import argparse
import time
import os
import matplotlib.pyplot as plt
import sys

from accelerate import Accelerator

import HiResPrecipNet as models
import dataset
from dataset import Dataset_Graph, Iterable_Graph

from utils import date_to_idxs, load_checkpoint, Tester

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

#-- paths
parser.add_argument('--input_path', type=str, help='path to input directory')
parser.add_argument('--output_path', type=str, help='path to output directory')
parser.add_argument('--log_file', type=str, default='log.txt', help='log file')

parser.add_argument('--checkpoint_cl', type=str)
parser.add_argument('--checkpoint_reg', type=str)
parser.add_argument('--checkpoint', type=str)
parser.add_argument('--output_file', type=str, default="G_predictions.pkl")

parser.add_argument('--graph_file', type=str, default=None) 
parser.add_argument('--model_type', type=str, default=None)
parser.add_argument('--model_cl', type=str, default=None) 
parser.add_argument('--model_reg', type=str, default=None) 
parser.add_argument('--model_combined', type=str, default=None)

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
    
    if args.model_type == "combined":
        Model = getattr(models, args.model_combined)
        model = Model()
        if args.use_accelerate:
            model, dataloader = accelerator.prepare(model, dataloader)
        else:
            model = model.cuda()
        if not args.use_accelerate or accelerator.is_main_process:
            with open(args.output_path + args.log_file, 'a') as f:
                f.write("\Model:")    
        if not args.use_accelerate:
            checkpoint = torch.load(args.checkpoint, map_location=torch.device('cpu'))
        else:
            checkpoint = torch.load(args.checkpoint)
        model = load_checkpoint(model, checkpoint, args.output_path, args.log_file, None, 
                net_names=["low2high.", "low_net.", "high_net_reg.", "high_res_cl."], fine_tuning=False, device=accelerator.device)
    elif args.model_type == "individual":
        Model_cl = getattr(models, args.model_cl)
        Model_reg = getattr(models, args.model_reg)
        model_cl = Model_cl()
        model_reg = Model_reg()
        if args.use_accelerate:
            model_cl, model_reg, dataloader = accelerator.prepare(model_cl, model_reg, dataloader)
        # else:
        #     model_cl = model_cl.cuda()
        #     model_reg = model_reg.cuda()
        if not args.use_accelerate or accelerator.is_main_process:
            with open(args.output_path + args.log_file, 'a') as f:
                f.write("\nClassifier:")    
        if not args.use_accelerate:
            checkpoint_cl = torch.load(args.checkpoint_cl, map_location=torch.device('cpu'))
        else:
            checkpoint_cl = torch.load(args.checkpoint_cl)
        model_cl = load_checkpoint(model_cl, checkpoint_cl, args.output_path, args.log_file, None, 
                net_names=["low2high.", "low_net.", "high_net."], fine_tuning=False, device=accelerator.device)
        if accelerator is None or accelerator.is_main_process:
            with open(args.output_path + args.log_file, 'a') as f:
                f.write("\nRegressor:")
        if not args.use_accelerate:
            checkpoint_reg = torch.load(args.checkpoint_reg, map_location=torch.device('cpu'))
        else:
            checkpoint_reg = torch.load(args.checkpoint_reg)
        model_reg = load_checkpoint(model_reg, checkpoint_reg, args.output_path, args.log_file, None,
                net_names=["low2high.", "low_net.", "high_net."], fine_tuning=False, device=accelerator.device)
    else:
        raise Exception("args.model_type should be either 'combined' or 'individual'")


    if not args.use_accelerate or accelerator.is_main_process:
        with open(args.output_path + args.log_file, 'a') as f:
            f.write(f"\nStarting the test, from {int(args.test_day_start)}/{int(args.test_month_start)}/{int(args.test_year_start)} to " +
                    f"{int(args.test_day_end)}/{int(args.test_month_end)}/{int(args.test_year_end)}.")

    tester = Tester()

    start = time.time()
    if args.model_type == "combined":
        tester.test_combined(model, dataloader, low_high_graph=low_high_graph, args=args)
    elif args.model_type == "individual":   
        tester.test(model_cl, model_reg, dataloader, low_high_graph=low_high_graph, args=args)
    end = time.time()

    if not args.use_accelerate or accelerator.is_main_process:
        with open(args.output_path + args.log_file, 'a') as f:
            f.write(f"\nDone. Testing concluded in {end-start} seconds.")
            f.write("\nWrite the files.")

    if not args.use_accelerate or accelerator.is_main_process:
        with open(args.output_path + args.output_file, 'wb') as f:
            pickle.dump(low_high_graph, f)

    

  
