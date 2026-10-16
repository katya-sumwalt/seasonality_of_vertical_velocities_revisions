
''' This script contains a function to calculate the confidence interval of the kurtosis by 
    1. Select a 1000 km segement 
    2. Compute the kurtosis over that segment for each depth
    3. Calculate the CI over that segement for each depth
    4. Compute the mean CI over that entire season CI(depth)
'''

import xarray as xr 
import matplotlib.pyplot as plt  
import lmoments3 as lm #kurtosis
import numpy as np
import warnings
warnings.filterwarnings("ignore", message="invalid value encountered in scalar divide")
# Signal / statistics
from scipy.ndimage import uniform_filter1d
from scipy.signal import welch
from scipy.stats import (
    linregress, sem, t, chi2, gaussian_kde
)
import lmoments3 as lm
# Skill metrics
# !pip install xskillscore
import xskillscore as xs
# Dask / parallel
import dask
from dask import delayed, compute
from dask.distributed import Client, as_completed
from dask.diagnostics import ProgressBar


def mean_ci95_depth_distance(da, sample_dim="chunk"):
    """
    Compute mean(depth) and 95% CI(depth) for the mean using SEM + t critical value.
    da dims: (depth, distance)
    """
    mu = da.mean(dim=sample_dim, skipna=True)
    sd = da.std(dim=sample_dim, skipna=True)
    n  = da.count(dim=sample_dim)

    sem = sd / np.sqrt(n)

    df = xr.where(n > 1, n - 1, np.nan)
    tcrit = xr.apply_ufunc(
        lambda x: t.ppf(0.975, x),
        df,
        vectorize=True,
        dask="allowed",
        output_dtypes=[float],
    )

    ci95 = tcrit * sem
    return mu, ci95


def calculate_kurtosis(row):
    row = row[np.isfinite(row)]
    return lm.lmom_ratios(row, nmom=4)[3]

def calculate_kurtosis_of_segment(subset_of_dataset, variable):
    ''' Input: subset_of_dataset, likley 500km or 1000 km chunk
               variable- variable to compute the 95 percent confidnence interval and the kurtosis over, default = 'vertical_velocity'
        Output: 
        
    '''
    variable_data_for_subset = subset_of_dataset[variable] * subset_of_dataset.maskprof #data_(depth)
    L_kurtosis = np.apply_along_axis(calculate_kurtosis, axis=1, arr=np.asarray(variable_data_for_subset))
    return L_kurtosis 


 

summer_dataset = xr.open_dataset('/Users/kat/Desktop/LS/siegelman-lab-ksumwalt/seal-vertical-velocities/Data/east/seal_data_with_vertical_velocity_east_summer_200m_not_removed_3.nc')
winter_dataset = xr.open_dataset('/Users/kat/Desktop/LS/siegelman-lab-ksumwalt/seal-vertical-velocities/Data/east/seal_data_with_vertical_velocity_east_winter_200m_not_removed_3.nc')
fall_dataset = xr.open_dataset('/Users/kat/Desktop/LS/siegelman-lab-ksumwalt/seal-vertical-velocities/Data/east/seal_data_with_vertical_velocity_east_fall_200m_not_removed_4.nc') 
spring_dataset = xr.open_dataset('/Users/kat/Desktop/LS/siegelman-lab-ksumwalt/seal-vertical-velocities/Data/east/seal_data_with_vertical_velocity_east_spring_200m_not_removed.nc') #5 is most clean 



def calculate_kurtosis_for_seal(tag = 'ct112-033-14',dataset = summer_dataset,variable = 'vertical_velocity'):

    dataset_subset_by_tag = dataset.where(dataset.name == tag, drop = True) #select the seal tag dataset
    number_of_chunks = (dataset_subset_by_tag.distance[-1].values - dataset_subset_by_tag.distance[0].values)/ 1000 // 1500 # get the number of chunks
    list_of_start_idx = np.arange(0,(number_of_chunks)*1000,1500) #create a list of start indices
    end_idx_value = int((dataset_subset_by_tag.distance[-1].values - dataset_subset_by_tag.distance[0].values)/1000) #get end index value
    start_end_value_idx = int(end_idx_value - 1500) #get start of the end index value
    #subset each data setgement by distance for the tag
    segments_of_data = [dataset_subset_by_tag.isel(distance = slice(int(start_idx),int((start_idx + 1000)))) for start_idx in list_of_start_idx]
    # print(len(segments_of_data))
    #calculate the L-Kurtosis per segment
    L_kurtosis_per_segement = [calculate_kurtosis_of_segment(seg,variable) for seg in segments_of_data]
    #creat an end segement and get the L- Kurtosis
    end_segment = dataset_subset_by_tag.isel(distance = slice(start_end_value_idx,end_idx_value))
    # print(np.shape(end_segment))
    L_kurtosis_for_end_segement = calculate_kurtosis_of_segment(end_segment,variable)
    L_kurtosis_per_segement.append(L_kurtosis_for_end_segement)

    return L_kurtosis_per_segement 


######### TEST OF CALCULATE KURTOSIS FOR SEAL FUNCTION  ################
# summer_dataset_subset =  summer_dataset.where(summer_dataset.name == 'ct112-033-14', drop = True)
# print(summer_dataset_subset)
# kurtosis = calculate_kurtosis_of_segment(summer_dataset_subset)
# print(kurtosis)
# L_kurtosis_per_segement = calculate_kurtosis_for_seal() #this works
# print(np.shape(L_kurtosis_per_segement))


def calculate_ci_kurtosis_for_season(dataset,variable):
    name_list = np.unique(dataset.name)
    mean_kurtosis = []
    for seal_tag in name_list: 
        try: 
            kurtosis_per_tag = np.array(calculate_kurtosis_for_seal(seal_tag,dataset,variable)) 
            # print(np.shape(kurtosis_per_tag))
        except: 
            print(f'did not work for {seal_tag}')
        
        # mean_kurtosis_for_tag = kurtosis_per_tag.mean(axis = 0)
        mean_kurtosis.append(kurtosis_per_tag)
        
    mean_kurtosis_seasin = np.concatenate(mean_kurtosis,axis = 0)

    ds_kurt = xr.Dataset(
        data_vars= dict(kurtosis = (['chunk','depth'],mean_kurtosis_seasin)),
        coords = dict(depth = ('depth',summer_dataset.depth.data))
    )

    ci = mean_ci95_depth_distance(ds_kurt)[1]

    
    return ci


# ds_kurt_ci = calculate_ci_kurtosis_for_season(fall_dataset)
# print('fall')
# print(ds_kurt_ci)
# print('summer')
# ds_kurt_ci = calculate_ci_kurtosis_for_season(summer_dataset)
# print(ds_kurt_ci)
# # print(ds_kurt_ci)


# ds_winter_kurtosis,ci_kurtosis_winter = calculate_kurtosis_for_season(winter_dataset,'winter')
# ds_fall_kurtosis,ci_kurtosis_fall = calculate_kurtosis_for_season(fall_dataset,'fall')
# ds_spring_kurtosis,ci_kurtosis_spring = calculate_kurtosis_for_season(spring_dataset,'spring')
# print(summer_dataset['temperature'].where(summer_dataset.name == 'ct131-035BAT2-15',drop = True))
    
# summer_l_kurtosis = [calculate_kurtosis_for_seal(tag = tag_name,dataset = summer_dataset) for tag_name in np.unique(summer_dataset.name)]

# winter_l_kurtosis = [calculate_kurtosis_for_seal(tag = tag_name,dataset = winter_dataset) for tag_name in np.unique(summer_dataset.name)]
# fall_l_kurtosis = [calculate_kurtosis_for_seal(tag = tag_name,dataset = summer_dataset) for tag_name in np.unique(summer_dataset.name)]
# spring_l_kurtosis = [calculate_kurtosis_for_seal(tag = tag_name,dataset = summer_dataset) for tag_name in np.unique(summer_dataset.name)]