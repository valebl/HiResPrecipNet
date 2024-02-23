from torch_geometric.data import HeteroData
import torch
from torch import nn
import torchvision.ops
import matplotlib.pyplot as plt
import numpy as np
import pickle
from dataset_GAN import Iterable_Graph
import dataset_GAN as dataset
import time
import argparse
import sys
import os
import dataset

import HiResPrecipNet as models
import utils_GAN
from utils_GAN import Trainer, Tester, date_to_idxs, load_checkpoint, check_freezed_layers
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
parser.add_argument('--dataset_name', type=str, default='Dataset_Graph')

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
        accelerator = Accelerator(log_with="wandb", step_scheduler_with_optimizer=False, )
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
            f.write(f"Cuda is available: {torch.cuda.is_available()}. There are {torch.cuda.device_count()} available GPUs.")


#-----------------------------------------------------
#--------------- MODEL, LOSS, OPTIMIZER --------------
#-----------------------------------------------------

    Model_G = getattr(models, "HiResPrecipNet_generator")
    Model_D = getattr(models, "HiResPrecipNet_discriminator")
    model_G = Model_G()
    model_D = Model_D()

    # Loss
    loss_fn_D = nn.CrossEntropyLoss()
    loss_fn_G = nn.CrossEntropyLoss()   

#-----------------------------------------------------
#-------------- DATASET AND DATALOADER ---------------
#-----------------------------------------------------

    train_start_idx, train_end_idx = date_to_idxs(args.train_year_start, args.train_month_start,
                                                                      args.train_day_start, args.train_year_end, args.train_month_end,
                                                                      args.train_day_end, args.first_year)
    train_start_idx = max(train_start_idx,24)

    val_start_idx, val_end_idx = date_to_idxs(2007, 1, 1, 2007, 12, 31, 2001)

    with open(args.input_path+args.graph_file, 'rb') as f:
        low_high_graph = pickle.load(f)

    with open(args.input_path+args.target_file, 'rb') as f:
        target_train = pickle.load(f)

    # Define input and target
    low_high_graph['low'].x = low_high_graph['low'].x[:,min(train_start_idx, val_start_idx)-24:max(train_end_idx, val_end_idx),:]
    target_train = target_train[:,min(train_start_idx, val_start_idx):max(train_end_idx, val_end_idx)]


    # Define a mask to ignore time indexes with all nan values
    mask_not_all_nan = [torch.tensor(True) for i in range(24)]
    initial_time_dim = target_train.shape[1]
    for t in range(initial_time_dim):
        nan_sum = target_train[:,t].isnan().sum()
        mask_not_all_nan.append(nan_sum < target_train.shape[0])
    mask_not_all_nan = torch.stack(mask_not_all_nan)

    if accelerator is None or accelerator.is_main_process:
        with open(args.output_path+args.log_file, 'a') as f:
            f.write(f"\nAfter removing all nan time indexes, {mask_not_all_nan.sum()}" +
                    f" time indexes are considered ({(mask_not_all_nan.sum() / initial_time_dim * 100):.1f} % of initial ones).")
            
    # Fix the idxs
    idxs = torch.arange(initial_time_dim) # from 0 to the number of idxs considered

    if train_start_idx > val_end_idx:
        offset = val_start_idx
        train_idxs_list = [*range(train_start_idx - offset, train_end_idx - offset)]
        val_idxs_list = [*range(val_start_idx - offset, val_end_idx - offset)]
    else:
        offset = train_start_idx
        train_idxs_list = [*range(train_start_idx - offset, val_start_idx - offset)] + [*range(val_end_idx - offset,  train_end_idx - offset)]
        val_idxs_list = [*range(val_start_idx - offset, val_end_idx - offset)]

    train_idxs = torch.tensor(train_idxs_list)[mask_not_all_nan[train_idxs_list]]
    val_idxs = torch.tensor(val_idxs_list)[mask_not_all_nan[val_idxs_list]]
    
    if accelerator is None or accelerator.is_main_process:
        with open(args.output_path+args.log_file, 'a') as f:
            f.write(f"\nTrain from {int(args.train_day_start)}/{int(args.train_month_start)}/{int(args.train_year_start)} to " +
                    f"{int(args.train_day_end)}/{int(args.train_month_end)}/{int(args.train_year_end)} with validation 01/01/2007 to 31/12/2007" +
                    f"Idxs from {train_start_idx} to {val_start_idx} and from {val_end_idx} to {train_end_idx}.")

    #low_high_graph['low'].x = low_high_graph['low'].x[:,mask_not_all_nan]
    #target_train = target_train[:,mask_not_all_nan[24:]]

    Dataset_Graph = getattr(dataset, args.dataset_name)
    
    if args.loss_fn == 'weighted_mse_loss' or args.loss_fn == "weighted_mse_loss_ASYM":
        with open(args.input_path+args.weights_file, 'rb') as f:
            weights_reg = pickle.load(f)
        weights_reg = weights_reg[:,min(train_start_idx, val_start_idx):max(train_end_idx, val_end_idx)]
        #weights_reg = weights_reg[:,mask_not_all_nan[24:]]

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
        
    #-- split into trainset and testset
    # generator=torch.Generator().manual_seed(42)
    # len_trainset = int(len(dataset_graph) * args.pct_trainset)
    # len_validationset = len(dataset_graph) - len_trainset
    dataset_graph_train = torch.utils.data.Subset(dataset_graph, train_idxs)
    dataset_graph_val = torch.utils.data.Subset(dataset_graph, val_idxs)

    sampler_graph_train = Iterable_Graph(dataset_graph=dataset_graph_train, shuffle=True)
    sampler_graph_val = Iterable_Graph(dataset_graph=dataset_graph_val, shuffle=True)

    if accelerator is None or accelerator.is_main_process:
        with open(args.output_path+args.log_file, 'a') as f:
            f.write(f'\nTrainset size = {len(dataset_graph_train)}, validationset size = {len(sampler_graph_val)}.')

    dataloader_train_real = torch.utils.data.DataLoader(dataset_graph_train, batch_size=args.batch_size, num_workers=0,
                    sampler=sampler_graph_train, collate_fn=custom_collate_fn)
    
    dataloader_train_fake = torch.utils.data.DataLoader(dataset_graph_train, batch_size=args.batch_size, num_workers=0,
                    sampler=sampler_graph_train, collate_fn=custom_collate_fn)

    dataloader_val_real = torch.utils.data.DataLoader(dataset_graph_val, batch_size=args.batch_size, num_workers=0,
                    sampler=sampler_graph_val, collate_fn=custom_collate_fn)
    
    dataloader_val_fake = torch.utils.data.DataLoader(dataset_graph_val, batch_size=args.batch_size, num_workers=0,
                    sampler=sampler_graph_val, collate_fn=custom_collate_fn)

    if accelerator is None or accelerator.is_main_process:
        total_memory, used_memory, free_memory = map(int, os.popen('free -t -m').readlines()[-1].split()[1:])
        with open(args.output_path+args.log_file, 'a') as f:
            f.write(f"\nRAM memory {round((used_memory/total_memory) * 100, 2)} %")

    optimizer_G = torch.optim.Adam(model_G.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer_D = torch.optim.Adam(model_D.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    lr_scheduler_G = torch.optim.lr_scheduler.StepLR(optimizer_G, step_size=args.step_size, gamma=0.5)
    lr_scheduler_D = torch.optim.lr_scheduler.StepLR(optimizer_D, step_size=args.step_size, gamma=0.5)


#-----------------------------------------------------
#------------------ LOAD PARAMETERS ------------------
#-----------------------------------------------------

    epoch_start=0
    
    if args.ctd_training:
        if accelerator is None or accelerator.is_main_process:
            with open(args.output_path+args.log_file, 'a') as f:
                f.write("\nContinuing the training.")
        accelerator.load_state(args.checkpoint_ctd)
        epoch_start = torch.load(args.checkpoint_ctd+"epoch")["epoch"] + 1
        #checkpoint = torch.load(args.checkpoint_ctd) 
        #model = load_checkpoint(model, checkpoint, args.output_path, args.log_file, None,
        #    net_names=["low2high.", "low_net.", "high_net."], fine_tuning=True, device=accelerator.device)
        #epoch_start = checkpoint["epoch"] + 1

    #-- define the optimizer and trainable parameters
    #if args.ctd_training:
    #    with open(args.output_path+args.log_file, 'a') as f:
    #        f.write("\nLoading optimizer paramaters.")
    #    optimizer.load_state_dict(checkpoint["optimizer"])
    
    #if args.ctd_training:
    #    optimizer.load_state_dict(checkpoint["optimizer"]

    #check_freezed_layers(model, args.output_path, args.log_file, accelerator)

    #total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    #if accelerator is None or accelerator.is_main_process: 
    #    with open(args.output_path+args.log_file, 'a') as f:
    #        f.write(f"\nTotal number of trainable parameters: {total_params}.")

    
    # Freeze the parameters referring to the high-res initial node features, which are all zero
    #model.downscaler.lin_r.weight.requires_grad = False
    
    if accelerator is not None:
        model_D, model_G, optimizer_D, optimizer_G, dataloader_train_real, dataloader_train_fake, dataloader_val_real, dataloader_val_fake, lr_scheduler_G, lr_scheduler_D, loss_fn_G, loss_fn_D = accelerator.prepare(
            model_D, model_G, optimizer_D, optimizer_G, dataloader_train_real, dataloader_train_fake, dataloader_val_real, dataloader_val_fake, lr_scheduler_G, lr_scheduler_D, loss_fn_G, loss_fn_D)
        if accelerator.is_main_process:
            with open(args.output_path+args.log_file, 'a') as f:
                f.write("\nUsing accelerator to prepare model, optimizer, dataloader and loss_fn...")
    # else:
    #     with open(args.output_path+args.log_file, 'a') as f:
    #         f.write("\nNot using accelerator to prepare model, optimizer, dataloader and loss_fn...")
    #     model_G = model.cuda()

    # check_freezed_layers(model, args.output_path, args.log_file, accelerator)

    total_params_D = sum(p.numel() for p in model_D.parameters() if p.requires_grad)
    total_params_G = sum(p.numel() for p in model_G.parameters() if p.requires_grad)
    if accelerator is None or accelerator.is_main_process: 
        with open(args.output_path+args.log_file, 'a') as f:
            f.write(f"\nTotal number of trainable parameters D: {total_params_D}, G: {total_params_D}.")

#-----------------------------------------------------
#----------------------- TRAIN -----------------------
#-----------------------------------------------------

    if accelerator is None or accelerator.is_main_process:
        with open(args.output_path+args.log_file, 'a') as f:
            f.write(f"\nUsing epochs={args.epochs}.")
            if accelerator is None:
                f.write(f"\nModel = {args.model_name}, batch size = {args.batch_size}")
            else:
                f.write(f"\nModel = {args.model_name}, batch size = {args.batch_size*torch.cuda.device_count()}")

    start = time.time()

    trainer = Trainer()
    if args.model_type == "cl":
        model_G, model_D = trainer.train_cl(model_G, model_D, dataloader_train_real, dataloader_train_fake, dataloader_val_real, dataloader_val_fake, optimizer_G, optimizer_D,
                  loss_fn_G, loss_fn_D, lr_scheduler_G, lr_scheduler_D, accelerator, args)
    elif args.model_type == "reg":
        model_G, model_D = trainer.train_reg(model_G, model_D, dataloader_train_real, dataloader_train_fake, dataloader_val_real, dataloader_val_fake, optimizer_G, optimizer_D,
                  loss_fn_G, loss_fn_D, lr_scheduler_G, lr_scheduler_D, accelerator, args,)
        
    end = time.time()

    if accelerator is None or accelerator.is_main_process:
        with open(args.output_path+args.log_file, 'a') as f:
            f.write(f"\nCompleted in {end - start} seconds.")
            f.write(f"\nDONE!")


#-----------------------------------------------------
#------------------------ TEST -----------------------
#-----------------------------------------------------

    # test_start_idx, test_end_idx = date_to_idxs(year_start=2015, month_start=12, day_start=1,
    #                                                 year_end=2016, month_end=12, day_end=31, first_year=2001)

    # with open(args.input_path+args.graph_file, 'rb') as f:
    #     low_high_graph = pickle.load(f)
        
    # predictions = torch.ones((low_high_graph['high'].num_nodes, test_end_idx-test_start_idx)) * torch.nan

    # low_high_graph.predictions = predictions
    # low_high_graph['low'].x = low_high_graph['low'].x[:,test_start_idx-24:test_end_idx,:]    

    # Dataset_Graph_test = getattr(dataset, "Dataset_Graph")

    # dataset_graph = Dataset_Graph_test(targets=None, graph=low_high_graph)
    # custom_collate_fn = getattr(dataset, 'custom_collate_fn_graph')
    # sampler_graph = Iterable_Graph(dataset_graph=dataset_graph, shuffle=False)
            
    # dataloader = torch.utils.data.DataLoader(dataset_graph, batch_size=1, num_workers=0,
    #                 sampler=sampler_graph, collate_fn=custom_collate_fn)

    # if accelerator is None or accelerator.is_main_process:
    #     with open(args.output_path + args.log_file, 'a') as f:
    #         f.write(f"\nStarting the test, from 01/11/2015 to 31/12/2015 (from idx {test_start_idx} to idx {test_end_idx}).")

    # model, dataloader = accelerator.prepare(model, dataloader)

    # tester = Tester()

    # start = time.time()

    # if args.model_type == "cl":
    #     predictions, times = tester.test_cl(model, dataloader, low_high_graph=low_high_graph, args=args, accelerator=accelerator)
    # elif args.model_type == "reg":
    #     predictions, times = tester.test_reg(model, dataloader, low_high_graph=low_high_graph, args=args, accelerator=accelerator)

    # end = time.time()

    # accelerator.wait_for_everyone()

    # times = accelerator.gather(times).squeeze()
    # times, indices = torch.sort(times)

    # predictions = accelerator.gather(predictions).squeeze().swapaxes(0,1)[:,indices]

    # if accelerator is None or accelerator.is_main_process:
    #     with open(args.output_path + args.log_file, 'a') as f:
    #         f.write(f"\nDone. Testing concluded in {end-start} seconds.")
    #         f.write("\nWrite the files.")

    # if accelerator is None or accelerator.is_main_process:
    #     with open(args.output_path + "predictions.pkl", 'wb') as f:
    #         pickle.dump(predictions.cpu(), f)
    
        
        
    

