import numpy as np
import pickle
import torch
import argparse
import time
import os
import matplotlib.pyplot as plt
import sys

from accelerate import Accelerator

from torch_geometric.data import HeteroData

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
parser.add_argument('--dataset_name', type=str, default=None) 
parser.add_argument('--collate_name', type=str, default=None) 

#-- start and end training dates
parser.add_argument('--test_year_start', type=int)
parser.add_argument('--test_month_start', type=int)
parser.add_argument('--test_day_start', type=int)
parser.add_argument('--test_year_end', type=int)
parser.add_argument('--test_month_end', type=int)
parser.add_argument('--test_day_end', type=int)
parser.add_argument('--first_year', type=int)
parser.add_argument('--first_year_input', type=int)

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
        with open(args.output_path+args.log_file, 'w') as f:
            f.write("Starting the training...")
            f.write(f"Cuda is available: {torch.cuda.is_available()}. There are {torch.cuda.device_count()} available GPUs.")

    test_start_idx, test_end_idx = date_to_idxs(args.test_year_start, args.test_month_start,
                                                args.test_day_start, args.test_year_end, args.test_month_end,
                                                args.test_day_end, args.first_year)
    
    test_start_idx_input, test_end_idx_input = date_to_idxs(args.test_year_start, args.test_month_start,
                                                args.test_day_start, args.test_year_end, args.test_month_end,
                                                args.test_day_end, args.first_year_input)
    
    #correction for start idxs
    if test_start_idx >= 24:
        test_start_idx = test_start_idx-24
        test_start_idx_input = test_start_idx_input-24
    else:
        with open(args.output_path+args.log_file, 'a') as f:
            f.write(f"\ntest_start_idx={test_start_idx} < 24, thus testing will start from idx {test_start_idx+24}")

    with open(args.input_path+"pr_target.pkl", 'rb') as f:
        pr_target = pickle.load(f)

    with open(args.input_path+args.graph_file, 'rb') as f:
        low_high_graph = pickle.load(f)

    pr_target = pr_target[:,test_start_idx:test_end_idx]

    if args.dataset_name == "Dataset_Graph_CNN_GNN":
        low_high_graph['low'].x = low_high_graph['low'].x[:,:,:,test_start_idx_input:test_end_idx_input]    
    elif args.dataset_name == "Dataset_Graph_CNN_GNN_new":
        low_high_graph['low'].x = low_high_graph['low'].x[:,:,test_start_idx_input:test_end_idx_input,:] # nodes, var, time, lev
    elif args.dataset_name == "Dataset_Graph_subpixel": # or args.dataset_name == "Dataset_StaticGraphTemporalSignal":
        low_high_graph['low'].x = low_high_graph['low'].x[test_start_idx_input:test_end_idx_input,:,:,:,:]
    else:
        low_high_graph['low'].x = low_high_graph['low'].x[:,test_start_idx_input:test_end_idx_input,:]    

    Dataset_Graph = getattr(dataset, args.dataset_name)
    dataset_graph = Dataset_Graph(targets=None, graph=low_high_graph)
    
    custom_collate_fn = getattr(dataset, args.collate_name)
        
    sampler_graph = Iterable_Graph(dataset_graph=dataset_graph, shuffle=False)
        
    dataloader = torch.utils.data.DataLoader(dataset_graph, batch_size=args.batch_size, num_workers=0,
                    sampler=sampler_graph, collate_fn=custom_collate_fn)
    
    Model_cl = getattr(models, args.model_cl)
    Model_reg = getattr(models, args.model_reg)
    model_cl = Model_cl()
    model_reg = Model_reg()

    if accelerator is None:
        checkpoint_cl = torch.load(args.checkpoint_cl, map_location=torch.device('cpu'))
        checkpoint_reg = torch.load(args.checkpoint_reg, map_location=torch.device('cpu'))
        device = 'cpu'
    else:
        checkpoint_cl = torch.load(args.checkpoint_cl, map_location=accelerator.device)
        checkpoint_reg = torch.load(args.checkpoint_reg, map_location=accelerator.device)
        device = accelerator.device
    
    if accelerator is None or accelerator.is_main_process:
        with open(args.output_path + args.log_file, 'a') as f:
            f.write("\nLoading classifier state dict.")    
    model_cl.load_state_dict(checkpoint_cl)
    
    if accelerator is None or accelerator.is_main_process:
        with open(args.output_path + args.log_file, 'a') as f:
            f.write("\nLoading regressor state dict.")
    model_reg.load_state_dict(checkpoint_reg)

    if accelerator is not None:
        model_cl, model_reg, dataloader = accelerator.prepare(model_cl, model_reg, dataloader)

    if accelerator is None or accelerator.is_main_process:
        with open(args.output_path + args.log_file, 'a') as f:
            f.write(f"\nStarting the test, from {int(args.test_day_start)}/{int(args.test_month_start)}/{int(args.test_year_start)} to " +
                    f"{int(args.test_day_end)}/{int(args.test_month_end)}/{int(args.test_year_end)} (from idx {test_start_idx} to idx {test_end_idx}).")

    tester = Tester()

    start = time.time()

    # pr_cl, pr_reg, times = tester.test_reg_cl(model_reg, dataloader, low_high_graph=low_high_graph, args=args, accelerator=accelerator)
    pr_cl, pr_reg, times = tester.test(model_cl, model_reg, dataloader, low_high_graph=low_high_graph, args=args, accelerator=accelerator)
    #pr_cl, pr_reg, times, encod_cl, encod_reg = tester.test_encod(model_cl, model_reg, dataloader, low_high_graph=low_high_graph, args=args, accelerator=accelerator)
    end = time.time()

    accelerator.wait_for_everyone()

    # Gather the values in *tensor* across all processes and concatenate them on the first dimension. Useful to
    # regroup the predictions from all processes when doing evaluation.

    times = accelerator.gather(times).squeeze()
    times, indices = torch.sort(times)

    pr_cl = accelerator.gather(pr_cl).squeeze().swapaxes(0,-1)[:,indices]
    pr_reg = accelerator.gather(pr_reg).squeeze().swapaxes(0,-1)[:,indices]

    data = HeteroData()
    data.pr_target = pr_target[:,24:].cpu().numpy() # No predictions for the first 24 hours (since we use the 24h before to make a prediction)
    data.pr_cl = pr_cl.cpu().numpy()
    data.pr_reg = pr_reg.cpu().numpy()
    data.pr = pr_reg.cpu().numpy()
    # data.pr = pr_cl.cpu().numpy() * pr_reg.cpu().numpy()
    data.times = times.cpu().numpy()
    data["low"].lat = low_high_graph["low"].lat.cpu().numpy()
    data["low"].lon = low_high_graph["low"].lon.cpu().numpy()
    data["high"].lat = low_high_graph["high"].lat.cpu().numpy()
    data["high"].lon = low_high_graph["high"].lon.cpu().numpy()
    data["high", "within", "high"].edge_index = low_high_graph["high","within","high"].edge_index.cpu().numpy()

    #data.encod_cl = encod_cl.cpu().numpy()
    #data.encod_reg = encod_reg.cpu().numpy()

    if accelerator is None or accelerator.is_main_process:
        with open(args.output_path + args.log_file, 'a') as f:
            f.write(f"\nDone. Testing concluded in {end-start} seconds.")
            f.write("\nWrite the files.")

    if accelerator is None or accelerator.is_main_process:
        with open(args.output_path + args.output_file, 'wb') as f:
            pickle.dump(data, f)

    

  
