import numpy as np 
import xarray as xr

def apply_bxmask_to_raw_dataset(original_meop_dataset, threshold = 2e-6, vars_to_mask = ['bx','salinity', 'temperature','buoyancy','maskprof']):
    '''  Input original dataset, threshold value for bx, and list of variables to mask"
         Output: masked dataset 
    '''
    masked_dataset = original_meop_dataset.copy()
    mask = np.abs(masked_dataset['bx']) > threshold
    total_points = mask.size
    n_masked = mask.sum().item()
    
    print(f"\nMask condition: |bx| > {threshold}")
    print(f"Total points: {total_points}")
    print(f"Points to mask: {n_masked}  ({100*n_masked/total_points:.2f}%)\n")
    
    for var in vars_to_mask:
        before = np.isfinite(masked_dataset[var]).sum().item()
        masked_dataset[var] = masked_dataset[var].where(~mask)
        after = np.isfinite(masked_dataset[var]).sum().item()

        print(f"{var}: kept {after} / {before} points "
            f"({100*after/before:.2f}% kept)")
        
    return masked_dataset
    

def extend_dataset_to_depth(copy_of_original_meop_dataset: xr.Dataset ,depth_to_extend_to = 700):
    "Input:copy of original dataset, depth to extend too (automatically 700)"
    extended_dataset = copy_of_original_meop_dataset.copy()
    list_of_non_depth_vars = ['sla','eke','ke_neuro','sla_neuro','eke_neuro', 'strain_neuro','zeta_neuro','MLD'] #,'ss','sn','ess']
    extended_dataset_depth_variables_only = extended_dataset.drop_vars(list_of_non_depth_vars) #drop variables without depth components
    data_at_500m = extended_dataset_depth_variables_only.where(copy_of_original_meop_dataset['depth'] == 500,drop = True).squeeze(dim = 'depth')
    original_distance_between_points = np.abs(copy_of_original_meop_dataset.depth[2] - copy_of_original_meop_dataset.depth[1])
    array_of_depths_to_add = np.arange(501,700 + 1,original_distance_between_points)
    duplicate_data_till_depth = []
    for depth in array_of_depths_to_add:
        extended_dataset_depth_variables_at_depth = data_at_500m.assign_coords(depth = depth) 
        duplicate_data_till_depth.append(extended_dataset_depth_variables_at_depth)
    ds_extended = xr.concat([extended_dataset_depth_variables_only] + duplicate_data_till_depth, dim='depth') 
    for var in list_of_non_depth_vars:
        ds_extended[var] = copy_of_original_meop_dataset[var]
    
    return ds_extended
    

def slice_dataset_at_ends(meop_dataset):
    '''Slices data at visually determined points at the start and end of there being full data from top to bottom

        Input: Dataset
        Output: Sliced Dataset
    '''
    ##for unique_names[0], slice(120,-50): ct112-033-14
    ##for unique_names[1], slice(73,-5):ct112-035-14
    ##for unique_names[2], slice(45,-35):ct112-048-14
    ##for unique_names[3], slice(73,-79):ct112-049-14
    ##for unique_names[4], slice(210,-286):ct112-049-14
    ##for unique_names[5], slice(210,-286):NO VALUES
    ##for unique_names[6], slice(50,-555):ct131-048BAT2-15
    ##for unique_names[7], slice(275,-150):ct132-331-16 
    ##for unique_names[8], slice(406,-450):ct139-331BAT-16
    ##for unique_names[9], slice(365,-390):ct139-622-17
    ##for unique_names[10], slice(365,-390):NO VALUES
    ##for unique_names[11], slice(231,-270):ct139-F620-17
    ##for unique_names[12], slice(67,-7):ct158-198-19
    ##for unique_names[13], slice(102,-5):ct158-199-19
    ##for unique_names[14], slice(405,-15):ct158-309-BAT-15
    ##for unique_names[15], slice(184,-473):ct158-310-BAT-15
    ##for unique_names[16], slice(89,-370):ft22-686-18
    ##for unique_names[17], slice(30,-187):ft22-873-18
    ##for unique_names[18], slice(91,-228):ft22-874-18
    ##for unique_names[19], slice(3,-39):ft22-875-18
    ##for unique_names[20], slice(99,-830):ft22-876-18
    ##for unique_names[21], slice(47,-9):ft22-878-18
    ##for unique_names[22], slice(163,-49):ft22-879-18
    ##for unique_names[23], slice(297,-607):ft22-881-18
    ##for unique_names[24], slice(3,-167):ft22-882-18
    
    unique_names = np.unique(meop_dataset.name)
    
    sliced_data = []
    for idx_name in range(len(unique_names)):
        tag = unique_names[idx_name]
        seal_dataset = meop_dataset.where(meop_dataset.name == tag, drop= True)
        distance = np.arange(seal_dataset.sizes['time']) * 1000.00 #distance in meters
        ds_dist = seal_dataset.rename({'time': 'distance'}).assign_coords(distance=distance) 
        ds_dist['depth'] = ds_dist['depth'] #note to self this was negative
 

        start_of_clean = max(ds_dist.buoyancy.isel(depth = -1).where(ds_dist.buoyancy.isel(depth = -1) != 0)< 1500).distance.values
        idx_start = np.where(ds_dist.distance == start_of_clean)[0][0] + 3

        end_idx = [50,5,35,79,286,286,555,150,450,390,390,270,7,5,15,473,370,187,228,39,830,9,49,607,167]
  
        ds_dist_selected = ds_dist.isel(distance = slice(idx_start,- end_idx[idx_name])) ## shorten so that you can compute the w values without the overflow error
        times = seal_dataset['time'].isel(time = slice(idx_start,- end_idx[idx_name]))
        seal_dataset_subset = seal_dataset.isel(time = slice(idx_start,- end_idx[idx_name]))
        sliced_data.append(seal_dataset_subset)
    concatenated_data = xr.concat(sliced_data,dim = 'time')

    return concatenated_data
    
    
def apply_mask_to_seal(ds, seal_sel, distance_start, distance_end, depth, threshold_value, list_of_vars=None):
    ''' Applies a mask to the seal data 
    '''
    if list_of_vars is None:
        list_of_vars = ['buoyancy','f','temperature','salinity','N2','bx','M4','qErtel','inv_Ri','vertical_velocity']

    ds_seal = ds.where(ds.name == seal_sel, drop=True)
    ds_sliced = ds_seal.isel(distance=slice(distance_start, distance_end))
    variable_at_depth = ds_sliced.buoyancy.sel(depth=depth, drop=True)
    new_mask = variable_at_depth > threshold_value
    
    mask_full_depths = xr.DataArray(
        np.tile(new_mask.values, (ds_sliced.depth.size, 1)),
        dims=("depth", "distance"),
        coords={"depth": ds_sliced.depth, "distance": ds_sliced.distance}
    )
    
    masked_data = {var: ds_sliced[var].where(~mask_full_depths) for var in list_of_vars}

    for var in list_of_vars:
        var_data = ds_seal[var].copy()
        var_data[dict(distance=slice(distance_start, distance_end))] = masked_data[var]
        ds_seal[var] = var_data


    seal_mask = ds.name.values == seal_sel
    seal_indices = np.where(seal_mask)[0]
    full_slice_indices = seal_indices[distance_start:distance_end]

    for var in list_of_vars:
        if var in ds.data_vars:
            ds[var].values[:, full_slice_indices] = ds_seal[var].isel(distance=slice(distance_start, distance_end)).values

    return ds
    


# ds_test = xr.open_dataset('/Users/kat/Desktop/LS_organized/data/updated_along_track_strain_raw_seal_data.nc')
# print(ds_test)
# sliced = slice_dataset_at_ends(ds_test)
# print(sliced)