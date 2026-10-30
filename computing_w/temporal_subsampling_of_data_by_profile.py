import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

base = "/Users/kat/Desktop/All_of_Work_With_Lia/LS/siegelman-lab-ksumwalt/seal-vertical-velocities/Data/east"

fall   = xr.open_dataset(f"{base}/seal_data_with_vertical_velocity_east_fall_200m_not_removed_4.nc")
summer = xr.open_dataset(f"{base}/seal_data_with_vertical_velocity_east_summer_200m_not_removed_3.nc")
winter = xr.open_dataset(f"{base}/seal_data_with_vertical_velocity_east_winter_200m_not_removed_3.nc")
spring = xr.open_dataset(f"{base}/seal_data_with_vertical_velocity_east_spring_200m_not_removed.nc")


def subsample_datarray_by_profiles(dataset_being_subsampled,dataset_setting_subsample = fall,var = 'vertical_velocity'):
    '''
    Step (1): Identify how many fall profiles go all the way to 500m () n_500, similarly, identify how many profiles (by looping through the distance) in the dataset to be subsampled go to 500m. From the dataset to be subsampled select
              n_500 profiles. Then go up to the next depth in the fall dataset and identify how many more profiles I need to get (n_499) from the dataset to be subsampled.
    '''
    ## Select The datasets at the maximum depth to start when you call this function
    dataset_being_subsampled_by_profile = dataset_being_subsampled.copy()

    ## valid (masked, not NaN) points for both datasets, shape (depth, distance)
    valid_setting_subsample = (dataset_setting_subsample[var] * dataset_setting_subsample['maskprof']).notnull().transpose('depth','distance').values
    valid_being_subsampled = (dataset_being_subsampled[var] * dataset_being_subsampled['maskprof']).notnull().transpose('depth','distance').values

    n_profiles = dataset_being_subsampled.sizes['distance']
    selected = np.zeros(n_profiles, dtype=bool)        # which profiles have been picked
    cutoff_idx = np.full(n_profiles, -1)                # deepest depth index kept for each picked profile

    rng = np.random.default_rng()

    ## loop from the deepest depth (500m) up to the surface
    for index_of_depth_to_subsample in range(valid_setting_subsample.shape[0]-1, -1, -1):

        ## get the number of profiles that exist for fall at that depth (n_500, n_499, ...)
        count_of_valid_points_at_depth = int(valid_setting_subsample[index_of_depth_to_subsample].sum())

        ## how many of the already picked profiles have data at this depth
        n_already_selected = int((selected & valid_being_subsampled[index_of_depth_to_subsample]).sum())
        n_needed = count_of_valid_points_at_depth - n_already_selected
        if n_needed <= 0:
            continue

        ## candidates: profiles with data at this depth that were not picked yet
        candidates = np.where(valid_being_subsampled[index_of_depth_to_subsample] & ~selected)[0]
        if len(candidates) < n_needed:
            print(f'depth idx {index_of_depth_to_subsample}: need {n_needed} more profiles, only {len(candidates)} available')
            n_needed = len(candidates)

        ## randomly pick the extra profiles (no repeats)
        idx_randomly_generated = rng.choice(candidates, size = n_needed, replace = False)
        selected[idx_randomly_generated] = True
        cutoff_idx[idx_randomly_generated] = index_of_depth_to_subsample

    idx_selected = np.where(selected)[0]
    subsampled_data = dataset_being_subsampled_by_profile.isel(distance = idx_selected) #this grabs the whole profiles

    ## cut each profile below the depth it was picked at, so the count at every depth matches fall
    keep = xr.DataArray(np.arange(valid_setting_subsample.shape[0])[:, None] <= cutoff_idx[idx_selected][None, :],
                        dims = ('depth','distance'))
    for v in subsampled_data.data_vars:
        if 'depth' in subsampled_data[v].dims and 'distance' in subsampled_data[v].dims:
            subsampled_data[v] = subsampled_data[v].where(keep)

    print(f'number of profiles selected: {len(idx_selected)} out of {n_profiles}')

    return subsampled_data



path_to_save = '/Users/kat/Desktop/All_of_Work_With_Lia/LS/siegelman-lab-ksumwalt/seal-vertical-velocities/Data/east/'


## subsample by whole profiles (matches fall counts at every depth) and save
# summer_subsampled_by_profile = subsample_datarray_by_profiles(summer, fall, 'vertical_velocity')
# winter_subsampled_by_profile = subsample_datarray_by_profiles(winter, fall, 'vertical_velocity')
# spring_subsampled_by_profile = subsample_datarray_by_profiles(spring, fall, 'vertical_velocity')
# print(spring_subsampled_by_profile)

# summer_subsampled_by_profile.to_netcdf(f'{path_to_save}summer_subsample_by_profile.nc')
# winter_subsampled_by_profile.to_netcdf(f'{path_to_save}winter_subsample_by_profile.nc')
# spring_subsampled_by_profile.to_netcdf(f'{path_to_save}spring_subsample_by_profile.nc')


summer_subsampled = xr.open_dataset(f'{path_to_save}summer_subsample_by_profile.nc')
winter_subsampled = xr.open_dataset(f'{path_to_save}winter_subsample_by_profile.nc')
spring_subsampled = xr.open_dataset(f'{path_to_save}spring_subsample_by_profile.nc')

print(winter_subsampled)


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
# ax2.set_ylim([15,500]) ## ALSO CUT THE YLIM
ax2.invert_yaxis()  
ax2.grid(True)




# BOTTOM: subsample by whole profiles vs fall
ax2 = plt.subplot(212)
ax2.plot((fall.vertical_velocity * fall.maskprof).count(dim = 'distance'), depth, color = fall_color, linewidth=6, label='fall')  # thickest, bottom
 
counts = (winter_subsampled.vertical_velocity * winter_subsampled.maskprof).count(dim = 'distance')
ax2.plot(counts, np.abs(counts.depth), color=winter_color, linewidth=4, label='winter')
 
counts = (summer_subsampled.vertical_velocity * summer_subsampled.maskprof).count(dim = 'distance')
ax2.plot(counts, np.abs(counts.depth), color=summer_color, linewidth=2.5, linestyle='--', label='summer')
 
counts = (spring_subsampled.vertical_velocity * spring_subsampled.maskprof).count(dim = 'distance')
ax2.plot(counts, np.abs(counts.depth), color=spring_color, linewidth=1.5, linestyle=':', label='spring')  # thinnest, top
 
ax2.set_title('Subsampled by profile', fontsize = 15)
ax2.set_xscale('log')
ax2.set_xlabel('# of Observations',fontsize = 15)
ax2.set_ylabel('Depth [m]',fontsize = 15)
ax2.set_xlim([1e2,2.5e4]) ## ALSO CUT THE YLIM
ax2.set_ylim([15,500]) ## ALSO CUT THE YLIM
ax2.invert_yaxis()  
ax2.grid(True)
ax2.legend()
plt.tight_layout()
plt.show()