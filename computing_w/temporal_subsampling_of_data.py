import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

base = "/Users/kat/Desktop/All_of_Work_With_Lia/LS/siegelman-lab-ksumwalt/seal-vertical-velocities/Data/east"

fall   = xr.open_dataset(f"{base}/seal_data_with_vertical_velocity_east_fall_200m_not_removed_4.nc")
summer = xr.open_dataset(f"{base}/seal_data_with_vertical_velocity_east_summer_200m_not_removed_3.nc")
winter = xr.open_dataset(f"{base}/seal_data_with_vertical_velocity_east_winter_200m_not_removed_3.nc")
spring = xr.open_dataset(f"{base}/seal_data_with_vertical_velocity_east_spring_200m_not_removed.nc")

#### GOAL : ###### 
## My Goal here is to obtain even temporal sampling of all variables at all depth

##Step 1: get the size of the masked variable for fall at a depth
##Step 2 : for each_tag use tUse the size of this variable (n_dat_depth) to randomly select len(n_at_depth)indices 

# def subsample_dataarray(index_of_depth_to_subsample,dataset_being_subsampled,dataset_setting_subsample = fall,var = 'vertical_velocity'):
#     ''' This function subsampled a dataset and returns an array for a variable.
#     '''
#     ## select the data of both datasets for a specific depth
#     data_at_depth_setting_subsample_for_depth = dataset_setting_subsample.isel(depth = index_of_depth_to_subsample,drop = True)
#     data_at_depth_getting_subsampled_for_depth = dataset_being_subsampled.isel(depth = index_of_depth_to_subsample,drop = True)

#     ## select the data for a specific variable and apply the mask
#     data_depth_subsampled_and_masked_subsampling1 = data_at_depth_setting_subsample_for_depth[var] * data_at_depth_setting_subsample_for_depth['maskprof']
#     data_depth_subsampled_and_masked_being_subsampled1 = data_at_depth_getting_subsampled_for_depth[var] * data_at_depth_getting_subsampled_for_depth['maskprof']

#     data_depth_subsampled_and_masked_subsampling = data_depth_subsampled_and_masked_subsampling1.dropna(dim = 'distance')
#     data_depth_subsampled_and_masked_being_subsampled = data_depth_subsampled_and_masked_being_subsampled1.dropna(dim = 'distance')

#     ## get the number of points that exists for the fall dataarray masked at that depth
#     count_of_data_setting_subsample_masked_data = data_depth_subsampled_and_masked_subsampling.count(dim = 'distance') #gives 1795 for idx_depth = 300
#     count_of_valid_points_at_depth = count_of_data_setting_subsample_masked_data.values

#     ## randomly generate indexes in the size of the subsampled at depth dataarray
#     rng = np.random.default_rng() 
#     idx_max = len(data_depth_subsampled_and_masked_being_subsampled.time)
#     idx_randomly_generated = rng.choice(idx_max,size = count_of_valid_points_at_depth)

#     ## from the dataset to be subsampled select the depth sleected masked datarray and use the indices to select the data from this array which we will call the subsample 

#     subsampled_data = data_depth_subsampled_and_masked_being_subsampled.isel(distance = idx_randomly_generated)
#     print(f'shape of subsampled data at depth idx {index_of_depth_to_subsample}: \n {np.shape(subsampled_data)}')
#     print(f'shape of fall data at depth idx {index_of_depth_to_subsample}: \n {np.shape(data_depth_subsampled_and_masked_subsampling)}')
    
#     return subsampled_data

# def create_subsampled_ds(data_setting_subsample,dataset_being_subsampled,var = 'vertical_velocity'):
#     subsampled_list = []
#     for depth_idx in range(len(data_setting_subsample.depth)):
#         depth = float(data_setting_subsample.depth[depth_idx])
#         subsampled_data_at_depth = subsample_dataarray(depth_idx,dataset_being_subsampled,data_setting_subsample,'vertical_velocity')
#         subsampled_list.append(subsampled_data_at_depth.assign_coords(depth=depth))


#     subsampled_ds = xr.concat([da.drop_indexes('distance') for da in subsampled_list],
#                             dim='distance', coords='all')
#     return subsampled_ds



def subsample_dataarray(index_of_depth_to_subsample,dataset_being_subsampled,dataset_setting_subsample = fall,var = ['temperature','vertical_velocity','maskprof','time','bx']):  # <-- changed
    ''' This function subsampled a dataset and returns an array for a variable.
    '''
    ## select the data of both datasets for a specific depth
    data_at_depth_setting_subsample_for_depth = dataset_setting_subsample.isel(depth = index_of_depth_to_subsample,drop = True)
    data_at_depth_getting_subsampled_for_depth = dataset_being_subsampled.isel(depth = index_of_depth_to_subsample,drop = True)

    ## select the data for a specific variable and apply the mask
    data_depth_subsampled_and_masked_subsampling1 = data_at_depth_setting_subsample_for_depth[var] * data_at_depth_setting_subsample_for_depth['maskprof']
    data_depth_subsampled_and_masked_being_subsampled1 = data_at_depth_getting_subsampled_for_depth[var] * data_at_depth_getting_subsampled_for_depth['maskprof']

    data_depth_subsampled_and_masked_subsampling = data_depth_subsampled_and_masked_subsampling1.dropna(dim = 'distance')
    data_depth_subsampled_and_masked_being_subsampled = data_depth_subsampled_and_masked_being_subsampled1.dropna(dim = 'distance')

    ## get the number of points that exists for the fall dataarray masked at that depth
    count_of_data_setting_subsample_masked_data = data_depth_subsampled_and_masked_subsampling[var[0]].count(dim = 'distance')  # <-- changed
    count_of_valid_points_at_depth = count_of_data_setting_subsample_masked_data.values

    ## randomly generate indexes in the size of the subsampled at depth dataarray
    rng = np.random.default_rng() 
    idx_max = data_depth_subsampled_and_masked_being_subsampled.sizes['distance']  # <-- changed
    idx_randomly_generated = rng.choice(idx_max,size = count_of_valid_points_at_depth)

    subsampled_data = data_depth_subsampled_and_masked_being_subsampled.isel(distance = idx_randomly_generated)
    print(f'shape of subsampled data at depth idx {index_of_depth_to_subsample}: \n {subsampled_data.sizes}')  # <-- changed
    print(f'shape of fall data at depth idx {index_of_depth_to_subsample}: \n {data_depth_subsampled_and_masked_subsampling.sizes}')  # <-- changed
    
    return subsampled_data

# def create_subsampled_ds(data_setting_subsample,dataset_being_subsampled,var = ['temperature','vertical_velocity','maskprof']):  # <-- changed
#     subsampled_list = []
#     for depth_idx in range(len(data_setting_subsample.depth)):
#         depth = float(data_setting_subsample.depth[depth_idx])
#         subsampled_data_at_depth = subsample_dataarray(depth_idx,dataset_being_subsampled,data_setting_subsample,var)  # <-- changed
#         subsampled_list.append(subsampled_data_at_depth.assign_coords(depth=depth))

#     subsampled_ds = xr.concat([da.drop_indexes('distance') for da in subsampled_list],
#                             dim='distance', coords='all')
#     return subsampled_ds


def create_subsampled_ds(data_setting_subsample,dataset_being_subsampled,var = ['temperature','vertical_velocity','maskprof','time','bx']):
    subsampled_list = []
    for depth_idx in range(len(data_setting_subsample.depth)):                  
        depth = float(data_setting_subsample.depth[depth_idx])
        subsampled_data_at_depth = subsample_dataarray(depth_idx,dataset_being_subsampled,data_setting_subsample,var)
        n = subsampled_data_at_depth.sizes['distance']                               
        subsampled_data_at_depth = (subsampled_data_at_depth.drop_indexes('distance')  
                                    .rename(distance='sample')
                                    .assign_coords(sample=np.arange(n)))
        subsampled_list.append(subsampled_data_at_depth.assign_coords(depth=depth))

    subsampled_ds = xr.concat(subsampled_list, dim='depth', join='outer')              
    subsampled_ds = subsampled_ds.rename(sample='distance')                           
    subsampled_ds = subsampled_ds.assign_coords(depth=data_setting_subsample.depth.values)  
    return subsampled_ds                                                           
summer_subsampled = create_subsampled_ds(fall,summer)
winter_subsampled= create_subsampled_ds(fall,winter)
spring_subsampled = create_subsampled_ds(fall,spring)
print(spring_subsampled)

# summer_subsampled_temperature = create_subsampled_ds(fall,summer,'temperature')
# winter_subsampled_temperature = create_subsampled_ds(fall,winter,'temperature')
# spring_subsampled_temperature = create_subsampled_ds(fall,spring,'temperature')





path_to_save = '/Users/kat/Desktop/All_of_Work_With_Lia/LS/siegelman-lab-ksumwalt/seal-vertical-velocities/Data/east/'
summer_subsampled.to_netcdf(f'{path_to_save}summer_subsample.nc')
winter_subsampled.to_netcdf(f'{path_to_save}winter_subsample.nc')
spring_subsampled.to_netcdf(f'{path_to_save}spring_subsample.nc')






###### TRYING TO DO IT FOR ALL OF THE VARIABLES
# summer_subset_list_all_vars = []
# winter_subset_list_all_vars = []
# spring_subset_list_all_vars = []

# for var in summer.variables:
#     try:
#         summer_subsampled_var = create_subsampled_ds(fall,summer,str(var))
#         winter_subsampled_var = create_subsampled_ds(fall,winter,str(var))
#         spring_subsampled_var = create_subsampled_ds(fall,spring,str(var))
        
#         summer_subset_list_all_vars.append(summer_subsampled_var.assign_var())
#         winter_subset_list_all_vars.append(winter_subsampled_var)
#         spring_subset_list_all_vars.append(summer_subsampled_var)
#     except:
#         print('this data has no depth')
        
# def to_own_dims(da):
#     var = da.name
#     ds = da.to_dataset()
#     rename = {c: f'{c}_{var}' for c in ds.coords}   # distance, lat, lon, name, time, depth
#     return ds.rename(rename)



# summer_subsampled = xr.merge([to_own_dims(da) for da in summer_subset_list_all_vars])
# # winter_subsampled = xr.merge([to_own_dims(da) for da in winter_subset_list_all_vars])
# # spring_subsampled = xr.merge([to_own_dims(da) for da in spring_subset_list_all_vars])

# print(summer_subsampled)
# summer_subsampled = xr.merge([])


# print(summer_subsampled)



# subsampled_list = []
# for depth_idx in range(len(fall.depth)):
#     depth = float(fall.depth[depth_idx])
#     subsampled_data_at_depth = subsample_dataarray(depth_idx)
#     subsampled_list.append(subsampled_data_at_depth.assign_coords(depth=depth))


################################################################################################################################################################################################################################################
################################################################################################################################################################################################################################################
################################################################################################################################################################################################################################################
################################################################################################################################################################################################################################################
################################################################################################################################################################################################################################################
# PLOT
################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################
################################################################################################################################################################################################################################################
################################################################################################################################################################################################################################################
################################################################################################################################################################################################################################################

import matplotlib.ticker as ticker
plt.rcParams['font.family'] = 'Times New Roman' #set fontsize to times new roman
plt.rcParams['mathtext.fontset'] = 'stix'
formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_powerlimits((0, 0))  # always use scientific notation

summer_color = 'tab:blue' 
fall_color   = 'tab:orange'
winter_color = 'tab:green' 
spring_color = 'tab:red'

depth = -1 * summer.depth
ax2 = plt.subplot(211)
counts = winter_subsampled['vertical_velocity'].groupby('depth').count()
ax2.plot(counts, -counts.depth, color=winter_color, linewidth=4, label='winter')

counts = summer_subsampled['vertical_velocity'].groupby('depth').count()
ax2.plot(counts, -counts.depth, color=summer_color, linewidth=2.5, linestyle='--', label='summer')

counts = spring_subsampled['vertical_velocity'].groupby('depth').count()
ax2.plot(counts, -counts.depth, color=spring_color, linewidth=1.5, linestyle=':', label='spring')  # thinnest, top

ax2.plot((fall.vertical_velocity * fall.maskprof).count(dim = 'distance'),depth, color = fall_color)
ax2.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
ax2.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
ax2.set_xscale('log')
ax2.set_xlabel('# of Observations',fontsize = 15)
ax2.set_ylabel('Depth [m]',fontsize = 15)
ax2.set_xlim([1e2,2.5e4]) ## ALSO CUT THE YLIM
ax2.set_ylim([15,500]) ## ALSO CUT THE YLIM
ax2.invert_yaxis()  
ax2.grid(True)


ax2 = plt.subplot(212)

counts = winter_subsampled['vertical_velocity'].groupby('depth').count()
ax2.plot(counts, -counts.depth, color=winter_color, linewidth=4, label='winter')

counts = summer_subsampled['vertical_velocity'].groupby('depth').count()
ax2.plot(counts, -counts.depth, color=summer_color, linewidth=2.5, linestyle='--', label='summer')

counts = spring_subsampled['vertical_velocity'].groupby('depth').count()
ax2.plot(counts, -counts.depth, color=spring_color, linewidth=1.5, linestyle=':', label='spring')  # thinnest, top

ax2.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
ax2.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
ax2.set_xscale('log')
ax2.set_xlabel('# of Observations',fontsize = 15)
ax2.set_ylabel('Depth [m]',fontsize = 15)
ax2.set_xlim([1e2,2.5e4]) ## ALSO CUT THE YLIM
ax2.set_ylim([15,500]) ## ALSO CUT THE YLIM
ax2.invert_yaxis()  
ax2.grid(True)
plt.show()


###