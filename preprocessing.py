import numpy as np
import xarray as xr
import pickle
import time
import argparse
import sys
import torch
import matplotlib.pyplot as plt
import netCDF4 as nc

from torch_geometric.data import Data, HeteroData
import torch_geometric.transforms as T
transform = T.AddLaplacianEigenvectorPE(k=2)

from utils import write_log, cut_window, retain_valid_nodes, derive_edge_indexes_within, derive_edge_indexes_low2high

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

#-- paths
parser.add_argument('--output_path', type=str)
parser.add_argument('--log_file', type=str)
parser.add_argument('--input_path_gripho', type=str)
parser.add_argument('--input_path_topo', type=str)
parser.add_argument('--gripho_file', type=str)
parser.add_argument('--topo_file', type=str)

#-- lat lon grid values
parser.add_argument('--lon_min', type=float)
parser.add_argument('--lon_max', type=float)
parser.add_argument('--lat_min', type=float)
parser.add_argument('--lat_max', type=float)
parser.add_argument('--lon_grid_radius_high', type=float)
parser.add_argument('--lat_grid_radius_high', type=float)

#-- other
parser.add_argument('--suffix_phase_2', type=str, default='')
parser.add_argument('--load_stats', action='store_true', default=True)
parser.add_argument('--no-load_stats', dest='load_stats', action='store_false')
parser.add_argument('--stats_path', type=str)
parser.add_argument('--stats_file_high', type=str)
parser.add_argument('--means_file_low', type=str, default='means.pkl')
parser.add_argument('--stds_file_low', type=str, default='stds.pkl')
parser.add_argument('--mean_std_over_variable_low', action='store_true')
parser.add_argument('--mean_std_over_variable_and_level_low', dest='mean_std_over_variable_low', action='store_false')
parser.add_argument('--predictors_type', type=str)

#-- era5
parser.add_argument('--input_path_phase_2a', type=str, help='path to input directory')
parser.add_argument('--input_files_prefix_low', type=str, help='prefix for the input files (convenction: {prefix}{parameter}.nc)', default='sliced_')
parser.add_argument('--n_levels_low', type=int, help='number of pressure levels considered', default=5)
    

if __name__ == '__main__':

    ######################################################
    ##--------------- PRELIMINARY STUFF ----------------##
    ######################################################

    args = parser.parse_args()
    
    write_log("\nStart!", args, 'w')

    time_start = time.time()

    ######################################################
    ##-------------------- PHASE 2A --------------------##
    ##---------- PREPROCESSING LOW RES DATA ------------##
    ######################################################

    if args.predictors_type == "era5":
        params = ['q', 't', 'u', 'v', 'z']
    elif args.predictors_type == "regcm":
        params = ['hus', 'ta', 'ua', 'va', 'zg']
    else:
        raise Exception("args.predictors_type should be either era5 or regcm")
    
    n_params = len(params)
 
    #-------------------------#
    # INPUT TENSOR FROM FILES #
    #-------------------------#
    
    write_log('\nStarting the preprocessing of the low resolution data.', args, 'w')

    for p_idx, p in enumerate(params):
        if args.predictors_type == "ERA5":
            write_log(f'\nPreprocessing {args.input_files_prefix_low}{p}.nc ...', args)
            with nc.Dataset(f'{args.input_path_low}{args.input_files_prefix_low}{p}.nc') as ds:
                data = ds[p][:]
                if p_idx == 0: # first parameter being processed -> get dimensions and initialize the input dataset
                    lat_low = ds['latitude'][:]
                    lon_low = ds['longitude'][:]
                    lat_dim = len(lat_low)
                    lon_dim = len(lon_low)
                    time_dim = len(ds['time'])
                    input_ds = np.zeros((time_dim, n_params, args.n_levels_low, lat_dim, lon_dim), dtype=np.float32) # time, variables, levels, lat, lon
            input_ds[:, p_idx,:,:,:] = data

        elif args.predictors_type == "RegCM":
            for l_idx, level in enumerate(['200', '500', '700', '850', '1000']):
                write_log(f'\nPreprocessing {args.input_files_prefix_low}{p}.nc for level {level}...', args)
                with nc.Dataset(f'{args.input_path_low}{args.input_files_prefix_low}{p}.nc') as ds:
                    var_name = f"{p}{level}"
                    data = ds[var_name][:]
                    if p_idx == 0 and l_idx == 0: # first parameter being processed -> get dimensions and initialize the input dataset
                        lat_low = ds['latitude'][:]
                        lon_low = ds['longitude'][:]
                        lat_dim = len(lat_low)
                        lon_dim = len(lon_low)
                        time_dim = len(ds['time'])
                        input_ds = np.zeros((time_dim, n_params, args.n_levels_low, lat_dim, lon_dim), dtype=np.float32) # time, variables, levels, lat, lon
                input_ds[:, p_idx,l_idx,:,:] = data

    lat_low, lon_low = torch.meshgrid(torch.flip(torch.tensor(lat_low),[0]), torch.tensor(lon_low), indexing='ij')

    lat_low = lat_low.flatten()
    lon_low = lon_low.flatten()

    #--------------------------#
    # POST-PROCESSING OF INPUT #
    #--------------------------#
        
    #-- Flip the dataset --#
    input_ds = np.flip(input_ds, 3) # the origin in the input files is in the top left corner, while we use the bottom left corner    

    #-- Standardize the dataset--#
    with open(args.output_path + args.log_file, 'a') as f:
        f.write(f'\nStandardizing the dataset.')
    
    input_ds_standard = np.zeros((input_ds.shape), dtype=np.float32)

    if args.use_precomputed_stats:
        with open(args.stats_path+args.means_file_low, 'rb') as f:
            means = pickle.load(f)
        with open(args.stats_path+args.stds_file_low, 'rb') as f:
            stds = pickle.load(f)

    if not args.mean_std_over_variable_low:
        if not args.use_precomputed_stats:
            means = np.zeros((5))
            stds = np.zeros((5))
            for var in range(5):
                m = np.mean(input_ds[:,var,:,:,:])
                s = np.std(input_ds[:,var,:,:,:])
                input_ds_standard[:,var,:,:,:] = (input_ds[:,var,:,:,:]-m)/s
                means[var] = m
                stds[var] = s
        else:
            for var in range(5):
                input_ds_standard[:,var,:,:,:] = (input_ds[:,var,:,:,:]-means[var])/stds[var]    
    else:
        if not args.use_precomputed_stats:
            means = np.zeros((5,5))
            stds = np.zeros((5,5))
            for var in range(5):
                for lev in range(5):
                    m = np.mean(input_ds[:,var,lev,:,:])
                    s = np.std(input_ds[:,var,lev,:,:])
                    input_ds_standard[:,var,lev,:,:] = (input_ds[:,var,lev,:,:]-m)/s
                    means[var, lev] = m
                    stds[var, lev] = s
        else:
            for var in range(5):
                for lev in range(5):
                    input_ds_standard[:,var,lev,:,:] = (input_ds[:,var,lev,:,:]-means[var, lev])/stds[var, lev]

    if not args.use_precomputed_stats:
        with open(args.output_path + "means.pkl", 'wb') as f:
            pickle.dump(means, f)
        with open(args.output_path + "stds.pkl", 'wb') as f:
            pickle.dump(stds, f)
        
    input_ds = torch.tensor(input_ds)

    input_ds = torch.permute(input_ds, (3,4,0,1,2)) # lat, lon, time, vars, levels
    input_ds = torch.flatten(input_ds, end_dim=1)   # num_nodes, time, vars, levels

    input_ds = torch.flatten(input_ds, start_dim=2, end_dim=-1)

    with open(args.output_path + args.log_file, 'a') as f:
        f.write(f'\nPreprocessing of low resolution data finished.')

    
    ######################################################
    ##-------------------- PHASE 2B --------------------##    
    ##--------- PREPROCESSING HIGH RES DATA ------------##
    ######################################################

    write_log(f"\n\nStarting the preprocessing of low resolution data.", args)

    #-------------------------------#
    # CUT LON, LAT, PR, Z TO WINDOW #
    #-------------------------------#

    gripho = xr.open_dataset(args.input_path_gripho + args.gripho_file)
    topo = xr.open_dataset(args.input_path_topo + args.topo_file)

    lon = torch.tensor(gripho.lon.to_numpy())
    lat = torch.tensor(gripho.lat.to_numpy())
    pr = torch.tensor(gripho.pr.to_numpy())
    z = torch.tensor(topo.z.to_numpy())

    write_log("\nCutting the window...", args)

    #-- Cut gripho and topo to the desired window --#
    lon_high, lat_high, z_high, pr_high = cut_window(args.lon_min, args.lon_max, args.lat_min, args.lat_max, lon, lat, z, pr)

    write_log(f"\nDone! Window is [{lon_high.min()}, {lon_high.max()}] x [{lat_high.min()}, {lat_high.max()}] with {pr_high.shape[1]} nodes.", args)

    write_log(f"\nlon shape {lon_high.shape}, lat shape {lat_high.shape}, pr shape {pr_high.shape}, z shape {z_high.shape}", args)

    #------------------------------------#
    # REMOVE NODES NOT IN LAND TERRITORY #
    #------------------------------------#

    lon_high, lat_high, pr_high, z_high = retain_valid_nodes(lon_high, lat_high, pr_high, z_high)
    pr_high = pr_high.swapaxes(0,1) # (num_nodes, time)

    print(lon_high.shape, lat_high.shape, pr_high.shape, z_high.shape)

    num_nodes_high = pr_high.shape[0]

    write_log(f"\nAfter removing the non land territory nodes, the high resolution graph has {num_nodes_high} nodes.", args)


    #---------------------------------------#
    # CLASSIFICATION AND REGRESSION TARGETS #
    #---------------------------------------#

    threshold = 0.1 # mm

    #-- CLASSIFICATION --#

    pr_sel_cl = torch.where(pr_high >= threshold, 1, 0).float()
    pr_sel_cl[torch.isnan(pr_high)] = torch.nan

    #-- REGRESSION --#

    pr_sel_reg = torch.where(pr_high >= threshold, torch.log1p(pr_high), torch.nan).float()
    pr_sel_reg[torch.isnan(pr_high)] = torch.nan

    weights = [1,2,5,10,20,50]
    weights_thresholds = [0,1,5,10,20,50]

    reg_weights = torch.ones(pr_high.shape, dtype=torch.float32) * weights[0]
    
    for i, w in enumerate(weights):
        thresh = weights_thresholds[i]
        reg_weights[pr_high >= thresh] = w

    reg_weights[torch.isnan(pr_high)] = torch.nan

    #mean_all_weights = np.nanmean(reg_weights)
    #reg_weights[~torch.isnan(pr_high)] = reg_weights[~torch.isnan(pr_high)] / mean_all_weights

    #-- WRITE THE FILES --#

    with open(args.output_path + 'target_train_cl.pkl', 'wb') as f:
        pickle.dump(pr_sel_cl, f)    
     
    with open(args.output_path + 'target_train_reg.pkl', 'wb') as f:
        pickle.dump(pr_sel_reg, f)    

    with open(args.output_path + 'reg_weights.pkl', 'wb') as f:
        pickle.dump(reg_weights, f)    

    with open(args.output_path + 'pr_gripho.pkl', 'wb') as f:
        pickle.dump(pr_high, f)    


    #---------------------------#
    # STANDARDISE LON LAT AND Z #
    #---------------------------#
    
    if args.load_stats:
        with open(args.stats_path + args.stats_file_high, 'rb') as f:
            precomputed_stats = pickle.load(f)
        mean_z = precomputed_stats[0]
        std_z = precomputed_stats[1]
        mode = "precomputed"
    else:
        mean_z = z_high.mean()
        std_z = z_high.std()
        mode = "local"
        stats_z = torch.tensor([mean_z, std_z])
        with open(args.output_path + "stats_z.pkl", 'wb') as f:
            pickle.dump(stats_z, f)

    write_log(f"\nUsing {mode} statistics for z: mean={mean_z}, std={std_z}", args)
    z_high_std = (z_high - mean_z) / std_z


    #-----------------#
    # BUILD THE GRAPH #
    #-----------------#

    low_high_graph = HeteroData()
    high_graph = Data()

    #-- EDGES --#

    edges_high = derive_edge_indexes_within(lon_radius=args.lon_grid_radius_high, lat_radius=args.lat_grid_radius_high,
                                  lon_n1=lon_high, lat_n1=lat_high, lon_n2=lon_high, lat_n2=lat_high)
    
    edges_low2high = derive_edge_indexes_low2high(lon_n1=lon_low, lat_n1=lat_low,
                                  lon_n2=lon_high, lat_n2=lat_high)

    #-- TO GRAPH ATTRIBUTES --#

    low_high_graph['low'].x = input_ds_standard
    low_high_graph['low'].lat = lat_low
    low_high_graph['low'].lon = lon_low

    low_high_graph['high'].x = torch.zeros((num_nodes_high, 0))
    low_high_graph['high'].lat = lat_high
    low_high_graph['high'].lon = lon_high
    low_high_graph['high'].z_std = z_high_std.unsqueeze(-1)

    low_high_graph['high', 'within', 'high'].edge_index = edges_high.swapaxes(0,1)
    low_high_graph['low', 'to', 'high'].edge_index = edges_low2high.swapaxes(0,1)

    #-- LAPLACIAN EIGENVECTORS --#

    high_graph.edge_index = edges_high.swapaxes(0,1)
    high_graph.x = torch.zeros((num_nodes_high, 0))
    high_graph = transform(high_graph)
    low_high_graph['high'].laplacian_eigenvector_pe = high_graph.laplacian_eigenvector_pe

    #-- WRITE THE GRAPH --#

    with open(args.output_path + 'low_high_graph' + args.suffix_phase_2 + '.pkl', 'wb') as f:
        pickle.dump(low_high_graph, f)

    write_log(f"\nIn total, preprocessing took {time.time() - time_start} seconds", args)            


    
