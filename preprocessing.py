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
# import torch_geometric.transforms as T
# transform = T.AddLaplacianEigenvectorPE(k=2)

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
parser.add_argument('--interval', type=float)

#-- start and end training dates
parser.add_argument('--train_year_start', type=float)
parser.add_argument('--train_month_start', type=float)
parser.add_argument('--train_day_start', type=float)
parser.add_argument('--train_year_end', type=float)
parser.add_argument('--train_month_end', type=float)
parser.add_argument('--train_day_end', type=float)
parser.add_argument('--first_year', type=float)

#-- lon/lat radius to identify when two nodes are connected by an edge
#-- this value is hand-calibrated and may be different for other data
parser.add_argument('--lon_grid_radius_high', type=float, default=0.0625)
parser.add_argument('--lat_grid_radius_high', type=float, default=0.05)
parser.add_argument('--lon_grid_radius_low', type=float, default=0.26)
parser.add_argument('--lat_grid_radius_low', type=float, default=0.26)
parser.add_argument('--lon_grid_radius_low2high', type=float, default=0.35)
parser.add_argument('--lat_grid_radius_low2high', type=float, default=0.35)

#-- other
parser.add_argument('--suffix', type=str, default='')
parser.add_argument('--use_precomputed_stats', action='store_true', default=True)
parser.add_argument('--stats_path', type=str)
parser.add_argument('--stats_file_high', type=str)
parser.add_argument('--means_file_low', type=str, default='means.pkl')
parser.add_argument('--stds_file_low', type=str, default='stds.pkl')
parser.add_argument('--mean_std_over_variable_low', action='store_true')
parser.add_argument('--mean_std_over_variable_and_level_low', dest='mean_std_over_variable_low', action='store_false')

#-- era5
parser.add_argument('--input_path_low', type=str, help='path to input directory')
parser.add_argument('--output_path_low', type=str, help='path to output directory')
parser.add_argument('--input_files_prefix_low', type=str, help='prefix for the input files (convenction: {prefix}{parameter}.nc)', default='sliced_')
parser.add_argument('--n_levels_low', type=int, help='number of pressure levels considered', default=5)



def write_log(s, args, mode='a'):
    with open(args.output_path + args.log_file, mode) as f:
        f.write(s)


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


def derive_edge_indexes(lon_radius, lat_radius, lon_n1 ,lat_n1, lon_n2, lat_n2):
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


if __name__ == '__main__':

    ######################################################
    ##                PRELIMINARY STUFF                 ##
    ######################################################

    args = parser.parse_args()
    
    write_log("\nStart!", args, 'w')

    time_start = time.time()

    ######################################################
    ##           PREPROCESSING LOW RES DATA             ##
    ######################################################

    params = ['q', 't', 'u', 'v', 'z']
    n_params = len(params)
 
    #-----------------------------------------------------
    #-------------- INPUT TENSOR FROM FILES --------------
    #-----------------------------------------------------
    
    with open(args.output_path + args.log_file, 'w') as f:
        f.write(f'\nStarting the preprocessing of the low resolution data.')

    for p_idx, p in enumerate(params):
        with open(args.output_path + args.log_file, 'a') as f:
            f.write(f'\nPreprocessing {args.input_files_prefix_low}{p}.nc ...')
        #with xr.open_dataset(f'{args.input_path}/{args.input_files_prefix}{p}.nc') as f:
        #    data = f[p].values
        with nc.Dataset(f'{args.input_path_low}{args.input_files_prefix_low}{p}.nc') as ds:
            data = ds[p][:]
            if p_idx == 0: # first parameter being processed -> get dimensions and initialize the input dataset
                lat_low = ds['latitude'][:]
                lon_low = ds['longitude'][:]
                lat_dim = len(lat_low)
                lon_dim = len(lon_low)
                time_dim = len(ds['time'])
                input_ds = np.zeros((time_dim, n_params, args.n_levels_low, lat_dim, lon_dim), dtype=np.float32) # variables, levels, time, lat, lon
        input_ds[:, p_idx,:,:,:] = data

    lat_low, lon_low = torch.meshgrid(torch.flip(torch.tensor(lat_low),[0]), torch.tensor(lon_low), indexing='ij')

    lat_low = lat_low.flatten()
    lon_low = lon_low.flatten()

    #-----------------------------------------------------
    #-------------- POST-PROCESSING OF INPUT--------------
    #-----------------------------------------------------
    
    # flip the dataset
    input_ds = np.flip(input_ds, 3) # the origin in the input files is in the top left corner, while we use the bottom left corner    

    # standardizing the dataset
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
    
    input_ds_standard = torch.tensor(input_ds_standard)

    input_ds_standard = torch.permute(input_ds_standard, (3,4,0,1,2)) # lat, lon, time, vars, levels
    input_ds_standard = torch.flatten(input_ds_standard, end_dim=1)   # num_nodes, time, vars, levels

    input_ds_standard = torch.flatten(input_ds_standard, start_dim=2, end_dim=-1)

    with open(args.output_path + args.log_file, 'a') as f:
        f.write(f'\nPreprocessing of low resolution data finished.')

    
    ######################################################
    ##          PREPROCESSING HIGH RES DATA             ##
    ######################################################

    write_log(f"\n\nStarting the preprocessing of low resolution data.", args)

    #-----------------------------------------------------
    #----------- CUT LON, LAT, PR, Z TO WINDOW -----------
    #-----------------------------------------------------

    gripho = xr.open_dataset(args.input_path_gripho + args.gripho_file)
    topo = xr.open_dataset(args.input_path_topo + args.topo_file)

    lon = torch.tensor(gripho.lon.to_numpy())
    lat = torch.tensor(gripho.lat.to_numpy())
    pr = torch.tensor(gripho.pr.to_numpy())
    z = torch.tensor(topo.z.to_numpy())

    write_log("\nCutting the window...", args)

    # cut gripho and topo to the desired window
    lon_high, lat_high, z_high, pr_high = cut_window(args.lon_min, args.lon_max, args.lat_min, args.lat_max, lon, lat, z, pr)

    write_log(f"\nDone! Window is [{lon_high.min()}, {lon_high.max()}] x [{lat_high.min()}, {lat_high.max()}] with {pr_high.shape[1]} nodes.", args)

    write_log(f"\nlon shape {lon_high.shape}, lat shape {lat_high.shape}, pr shape {pr_high.shape}, z shape {z_high.shape}", args)

    #-----------------------------------------------------
    #-------- REMOVE NODES NOT IN LAND TERRITORY ---------
    #-----------------------------------------------------

    lon_high, lat_high, pr_high, z_high = retain_valid_nodes(lon_high, lat_high, pr_high, z_high)
    pr_high = pr_high.swapaxes(0,1) # (num_nodes, time)

    print(lon_high.shape, lat_high.shape, pr_high.shape, z_high.shape)

    num_nodes_high = pr_high.shape[0]

    write_log(f"\nAfter removing the non land territory nodes, the high resolution graph has {num_nodes_high} nodes.", args)


    #-------------------------------------------------
    #----- CLASSIFICATION AND REGRESSION TARGETS -----
    #-------------------------------------------------

    threshold = 0.1 # mm

    pr_sel_cl = torch.where(pr_high >= threshold, 1, 0).float()
    pr_sel_cl[torch.isnan(pr_high)] = torch.nan

    pr_sel_reg = torch.where(pr_high >= threshold, torch.log1p(pr_high), torch.nan).float()
    pr_sel_cl[torch.isnan(pr_high)] = torch.nan

    weights = [1,2,5,10,20,50]
    weights_thresholds = [0,1,5,10,20,50]

    reg_weights = torch.ones(pr_high.shape, dtype=torch.float32) * weights[0]
    
    for i, w in enumerate(weights):
        thresh = weights_thresholds[i]
        reg_weights[pr_high >= thresh] = w

    reg_weights[torch.isnan(pr_high)] = torch.nan

    with open(args.output_path + 'target_train_cl.pkl', 'wb') as f:
        pickle.dump(pr_sel_cl, f)    
     
    with open(args.output_path + 'target_train_reg.pkl', 'wb') as f:
        pickle.dump(pr_sel_reg, f)    

    with open(args.output_path + 'reg_weights.pkl', 'wb') as f:
        pickle.dump(reg_weights, f)    

    with open(args.output_path + 'pr_gripho.pkl', 'wb') as f:
        pickle.dump(pr_high, f)    

    #-------------------------------------------------
    #----------- STANDARDISE LON LAT AND Z -----------
    #-------------------------------------------------
    
    if args.use_precomputed_stats:
        with open(args.stats_path + args.stats_file_high, 'rb') as f:
            precomputed_stats = pickle.load(f)
        mean_z = precomputed_stats[0]
        std_z = precomputed_stats[1]
        mode = "precomputed"
    else:
        mean_z = z_high.mean()
        std_z = z_high.std()
        mode = "local"

    write_log(f"\nUsing {mode} statistics for z: mean={mean_z}, std={std_z}", args)
    z_high_std = (z_high - mean_z) / std_z


    ######################################################
    ##                 BUILD THE GRAPH                  ##
    ######################################################

    low_high_graph = HeteroData()

    #-----------------------------------------------------
    #----------------------- EDGES -----------------------
    #-----------------------------------------------------
    
    edges_low = derive_edge_indexes(lon_radius=args.lon_grid_radius_low, lat_radius=args.lat_grid_radius_low,
                                  lon_n1=lon_low, lat_n1=lat_low, lon_n2=lon_low, lat_n2=lat_low)

    edges_high = derive_edge_indexes(lon_radius=args.lon_grid_radius_high, lat_radius=args.lat_grid_radius_high,
                                  lon_n1=lon_high, lat_n1=lat_high, lon_n2=lon_high, lat_n2=lat_high)
    
    edges_low2high = derive_edge_indexes(lon_radius=args.lon_grid_radius_low2high, lat_radius=args.lat_grid_radius_low2high,
                                  lon_n1=lon_low, lat_n1=lat_low, lon_n2=lon_high, lat_n2=lat_high)
    

    #-----------------------------------------------------
    #--------------- TO GRAPH ATTRIBUTES -----------------
    #-----------------------------------------------------

    low_high_graph['low'].x = input_ds_standard
    low_high_graph['low'].lat = lat_low
    low_high_graph['low'].lon = lon_low

    low_high_graph['high'].x = torch.zeros((num_nodes_high, 0))
    low_high_graph['high'].lat = lat_high
    low_high_graph['high'].lon = lon_high
    low_high_graph['high'].z_std = z_high_std.unsqueeze(-1)

    low_high_graph['low', 'within', 'low'].edge_index = edges_low.swapaxes(0,1)
    low_high_graph['high', 'within', 'high'].edge_index = edges_high.swapaxes(0,1)
    low_high_graph['low', 'to', 'high'].edge_index = edges_low2high.swapaxes(0,1)

    with open(args.output_path + 'low_high_graph' + args.suffix + '.pkl', 'wb') as f:
        pickle.dump(low_high_graph, f)

    write_log(f"\nIn total, preprocessing took {time.time() - time_start} seconds", args)            


    