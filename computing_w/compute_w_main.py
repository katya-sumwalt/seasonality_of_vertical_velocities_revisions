import cleanup_seal_data as csd 
import compute_w as cw 
import compute_w_with_depth_varying_strain as cwwdvs
import xarray as xr
import numpy as np
import matplotlib.pyplot as  plt

## CLEAN  UP THE RAW MEOP DATASET
raw_data = xr.open_dataset('/Users/kat/Desktop/Seasonality_of_Vertical_Velocities/raw_data/seal_data.nc') 
masked_dataset = csd.apply_bxmask_to_raw_dataset(raw_data)
extended_masked_dataset = csd.extend_dataset_to_depth(masked_dataset)
sliced_extended_masked_dataset = csd.slice_dataset_at_ends(extended_masked_dataset)


## SET UP THE STRAIN  FILE
type_of_strain = ['H_4000_h_over_H_180','H_4000_h_over_H_574','H_5500_h_over_H_467']
idx_for_strain = 0
strain = xr.open_dataset(f'/Users/kat/Desktop/Seasonality_of_Vertical_Velocities/varying_strain_with_depth/data_strain_times_fd/strain_with_f_d_{type_of_strain[idx_for_strain]}.nc') #need to extend to 700
strain_to_extend = strain.copy()
depths_to_add = np.arange(raw_data.depth.min(),700+1,1) #and extend that value to 500 meters depth
strain_data_extended_to_700m = strain_to_extend.reindex(depth=depths_to_add, method="ffill")#(time: 118121, depth: 686


## SELECT A SUBSET OF THE DATA 
unique_names = np.unique(sliced_extended_masked_dataset.name)
select_seal = unique_names[0]
seal_dataset = sliced_extended_masked_dataset.where(sliced_extended_masked_dataset.name == select_seal, drop= True)
strain_dataset = strain_data_extended_to_700m[f'strain_with_vertical_profile_{type_of_strain[idx_for_strain]}'].where(strain_data_extended_to_700m.name == select_seal, drop= True)
print('made it to line 26')
# seal_dataset.to_netcdf('/Users/kat/Desktop/All_of_Work_With_Lia/Siegelman_Lab_Initial_Submit/code/computing_vertical_vel_OCT15_SUBSET_TEST/subset_seal.nc')
# strain_dataset.to_netcdf(f'/Users/kat/Desktop/All_of_Work_With_Lia/Siegelman_Lab_Initial_Submit/code/computing_vertical_vel_OCT15_SUBSET_TEST/subset_seal_strain_varying_depth_{type_of_strain[1]}.nc')
# seal_data_subset = xr.open_dataset('/Users/kat/Desktop/All_of_Work_With_Lia/Siegelman_Lab_Initial_Submit/code/computing_vertical_vel_OCT15_SUBSET_TEST/subset_seal.nc')
# strain_data_subset = xr.open_dataset(f'/Users/kat/Desktop/All_of_Work_With_Lia/Siegelman_Lab_Initial_Submit/code/computing_vertical_vel_OCT15_SUBSET_TEST/subset_seal_strain_varying_depth_{type_of_strain[1]}.nc')


## MERGE THE DATASETS TOGETHER WITH A NEW DISTANCE COORDINATE
seal_data_subset = seal_dataset
strain_data_subset = strain_dataset
strain_data_subset = strain_data_subset.where(strain_data_subset.time == seal_dataset.time) 
distance = np.arange(seal_data_subset.sizes['time']) * 1000.00  ## gives distance in meters
ds_dist_seal = seal_data_subset.rename({'time': 'distance'}).assign_coords(distance=distance)  
ds_dist_strain = strain_data_subset.rename({'time': 'distance'}).assign_coords(distance=distance)  
merged_depth_varying_strain_and_seal_data = xr.merge([ds_dist_seal,ds_dist_strain]) 

## SUBSET THE SURFACE BOUNDARY CONDITION
w_m_surface = xr.open_dataset('/Users/kat/Desktop/Seasonality_of_Vertical_Velocities/surface_w_m_boundary_condition/w_m_surface_BC.nc')
w_m_surface_boundary_condition = w_m_surface.__xarray_dataarray_variable__.where(w_m_surface.name == select_seal, drop= True).where(w_m_surface.time == seal_dataset.time).isel(time = slice(0,1000)).values 


##  SELECT A CHUNK OF DATA TO COMPUTE THE VERTICAL VELOCITY DIAGNOSTICS FOR VARYING DEPTHS AND INDICES
chunk_to_compute_w = merged_depth_varying_strain_and_seal_data.isel(distance = slice(0,1000)) #depth 686, distance 1000
# chunk_to_compute_w.to_netcdf('/Users/kat/Desktop/All_of_Work_With_Lia/Siegelman_Lab_Initial_Submit/code/computing_vertical_vel_OCT15_SUBSET_TEST/chunk_to_compute_w_TEST.nc')
# w_depth_varying_strain = cwwdvs.compute_w_with_depth_varying_strain(chunk_to_compute_w,f'strain_with_vertical_profile_{type_of_strain[idx_for_strain]}')
# w_depth_varying_strain.to_netcdf(f'/Users/kat/Desktop/All_of_Work_With_Lia/Siegelman_Lab_Initial_Submit/code/computing_vertical_vel_OCT15_SUBSET_TEST/w_depth_varying_strain_TEST_{type_of_strain[idx_for_strain]}.nc')

w_surf = cwwdvs.compute_w_with_depth_varying_strain(chunk_to_compute_w,'strain_neuro',w_m_surface_boundary_condition)
w_surf.to_netcdf(f'/Users/kat/Desktop/All_of_Work_With_Lia/Siegelman_Lab_Initial_Submit/code/computing_vertical_vel_OCT15_SUBSET_TEST/w_depth_varying_strain_TEST_surface_strain_with_WM_surface_BC.nc')




