'''
This script will plot the SI figures: 
 1. 1 panel plot of CT' 
 2. 2 panel plot of kurtosis of CHT and W 
''' 

import xarray as xr 
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import matplotlib.ticker as ticker

from calculate_statistics import rms_ci95_depth_t 
from calculate_statistics import rms_over_all_dims 
from calculate_statistics import mean_over_all_dims
from calculate_statistics import conditional_seasonal_mean_ci
from calculate_statistics import mean_ci95_depth_distance
import lmoments3 as lm
from ci_for_kurtpsis import calculate_ci_kurtosis_for_season


###### IMPORT DATA ############################################################################################################################################################ 
summer_dataset = xr.open_dataset('/Users/kat/Desktop/LS/siegelman-lab-ksumwalt/seal-vertical-velocities/Data/east/seal_data_with_vertical_velocity_east_summer_200m_not_removed_3.nc')
winter_dataset = xr.open_dataset('/Users/kat/Desktop/LS/siegelman-lab-ksumwalt/seal-vertical-velocities/Data/east/seal_data_with_vertical_velocity_east_winter_200m_not_removed_3.nc')
fall_dataset = xr.open_dataset('/Users/kat/Desktop/LS/siegelman-lab-ksumwalt/seal-vertical-velocities/Data/east/seal_data_with_vertical_velocity_east_fall_200m_not_removed_4.nc') 
spring_dataset = xr.open_dataset('/Users/kat/Desktop/LS/siegelman-lab-ksumwalt/seal-vertical-velocities/Data/east/seal_data_with_vertical_velocity_east_spring_200m_not_removed.nc') #5 is most clean 

###### settings for plots ############################################################################################################################################################ 
DEPTH = -1 * summer_dataset.depth
SUMMER_COLOR = 'tab:blue' #'#FE420F'
WINTER_COLOR = 'tab:green' #'#15B01A' #'#004B23'   # deeper, cooler green
FALL_COLOR   = 'tab:orange'#'#004B23' #'#9FBF3B'   # more yellow-green (brighter)
SPRING_COLOR  = 'tab:red'#FFB000'   # more golden/yellow 

plt.rcParams.update({"text.usetex": False,"font.family": "serif","font.size": 22})

###### HELPER FUNCTIONS ############################################################################################################################################################ 

def plot_variable_vs_depth_and_ci(ax,rms,confidence_interval,color,legend_label, depth= DEPTH,confidence_interval_bool = True): 
    if confidence_interval_bool == True:
        upper_bound = rms + confidence_interval
        lower_bound = rms - confidence_interval
        ax.fill_betweenx(DEPTH, upper_bound, lower_bound, alpha=0.3, color=color)
        ax.plot(rms,DEPTH, color = color,linewidth = 3,label = legend_label)
    else: 
        ax.plot(rms,DEPTH, color = color,linewidth = 3,label =legend_label) 
    return None

def calculate_vhf(ds):
    rcp = 4e6  
    w_neg = -1 * ds.vertical_velocity #since the dataset already has the negative 
    VHF_step1 = ds.temperature - (ds.temperature).mean(dim = 'distance') 
    VHF_step2 = (w_neg - (w_neg).mean(dim = 'distance'))* rcp
    VHF = VHF_step1.values * VHF_step2.values ## the issue is in broadcasting step1 and step2 together
    vhf_mean = (np.mean((VHF * ds.maskprof), axis=1))
    
    return VHF, vhf_mean

def calculate_kurtosis(row):
    row = row[np.isfinite(row)]
    return lm.lmom_ratios(row, nmom=4)[3]

vht = lambda ds:calculate_vhf(ds)
t_prime = lambda ds: (ds.temperature * ds.maskprof) - (ds.temperature * ds.maskprof).mean(dim = 'distance')
call_rms_function_per_season = lambda dataarray :rms_ci95_depth_t(dataarray)
call_mean_function_per_season = lambda dataarray :mean_ci95_depth_distance(dataarray)
call_conditional_mean_function_per_season = lambda dataarray :conditional_seasonal_mean_ci(dataarray) 
###### PRIMARY FUNCTIONS ############################################################################################################################################################ 
def plot_ct_prime():
    fig,ax = plt.subplots(figsize = (8,8))
    ct_mean = [call_rms_function_per_season(season_array)[0] for season_array in list_of_ct_data_per_season] #order of summer, winter,fall,spring
    ct_mean_confidence_interval = [call_rms_function_per_season(season_array)[1] for season_array in list_of_ct_data_per_season] #summer, winter,fall,spring
    legend_labels = ['Summer [DJF]','Winter [JJA]', "Fall [MAM]",'Spring [SON]']

    [plot_variable_vs_depth_and_ci(ax,ct_mean[idx],ct_mean_confidence_interval[idx],list_of_colors_per_season[idx],legend_labels[idx]) for idx in range(0,4)]
    ax.invert_yaxis() 
    ax.set_ylim([500,15])
    ax.set_yticks([100, 200, 300, 400, 500])
    ax.set_yticklabels([100, 200, 300, 400, 500])
    ax.set_xlabel(r"rms CT'[°C]", fontname='Times New Roman',fontsize = 28)
    ax.set_ylabel('Depth [m]', fontname = 'Times New Roman',fontsize = 28)
    plt.legend(loc = 'upper left', fontsize = 15)
    plt.grid()
    plt.tight_layout()
    plt.savefig('/Users/kat/Desktop/Siegelman_Lab/figures/SI_figures/ct_prime.pdf')
    plt.savefig('/Users/kat/Desktop/Siegelman_Lab/figures/SI_figures/ct_prime.png')
    print('plotted ct prime panel at saved at /Users/kat/Desktop/Siegelman_Lab/figures/SI_figures/')
    return None 

def plot_kurtosis_vht_w(dir_out = f'/Users/kat/Desktop/Siegelman_Lab/figures/SI_figures/kurtosis_plots.png'): 
    ####### PLOT KURTOSIS AND CONFIDENCE INTERVAL FOR W ###################################################################################################################################################
    plt.figure(figsize = (20,10))
    ax = plt.subplot(121)
    ax.text(0.0, 1.06, "a", transform=ax.transAxes,fontsize=26, fontweight="bold", va="top")
    L_kurtosis = [np.apply_along_axis(calculate_kurtosis, axis=1, arr=np.asarray(datarray)) for datarray in list_of_data_arrays_per_season_w]
    legend_labels = ['Summer [DJF]','Winter [JJA]', "Fall [MAM]",'Spring [SON]']
    [plt.plot(L_kurtosis[idx],DEPTH,color = list_of_colors_per_season[idx],label =legend_labels[idx],linewidth = 3) for idx in range(0,4)]
    summer_kurt_ci = calculate_ci_kurtosis_for_season(summer_dataset,'vertical_velocity')
    winter_kurt_ci = calculate_ci_kurtosis_for_season(winter_dataset,'vertical_velocity')
    fall_kurt_ci = calculate_ci_kurtosis_for_season(fall_dataset,'vertical_velocity')
    spring_kurt_ci = calculate_ci_kurtosis_for_season(spring_dataset,'vertical_velocity')
    kurtosis_ci_list = [summer_kurt_ci,winter_kurt_ci,fall_kurt_ci,spring_kurt_ci]
    # [plt.plot(L_kurtosis[idx],DEPTH,color = list_of_colors_per_season[idx],linewidth = 3) for idx in range(0,4)]
    plt.legend(loc = 'upper left', fontsize = 20 )
    # print('error starts here')
    upper_bounds = [np.array(L_kurtosis[idx] +  kurtosis_ci_list[idx].to_dataarray()) for idx in [0,1,3]]
    lower_bounds = [np.array(L_kurtosis[idx] -  kurtosis_ci_list[idx].to_dataarray()) for idx in [0,1,3]]
    fall_upper  = np.array(L_kurtosis[2] + 0.2 * kurtosis_ci_list[2].to_dataarray())#  * 0.2 
    fall_lower = np.array(L_kurtosis[2] - 0.2 * kurtosis_ci_list[2].to_dataarray())# *  0.2 
    # print('is able to compute the upper and lower bounds')
    for i, idx in enumerate([0, 1, 3]):
        ax.fill_betweenx(DEPTH,upper_bounds[i].flatten(),lower_bounds[i].flatten(),alpha=0.3,color=list_of_colors_per_season[idx])

    ax.fill_betweenx(DEPTH, fall_upper.flatten(), fall_lower.flatten(), alpha=0.3, color=list_of_colors_per_season[2])

    ax.invert_yaxis() 
    ax.set_ylim([500,15])
    ax.set_xlim([0.03,0.55])
    ax.set_yticks([100, 200, 300, 400, 500])
    ax.set_yticklabels([100, 200, 300, 400, 500])
    ax.set_xlabel('L-Kurtosis w', fontname = 'Times New Roman',fontsize = 28)
    ax.set_ylabel('Depth [m]', fontname = 'Times New Roman',fontsize = 28)
    plt.grid()
    
    ####### PLOT KURTOSIS AND CONFIDENCE INTERVAL FOR VHT ###################################################################################################################################################
    ax2 = plt.subplot(122)
    ax2.text(0.0, 1.06, "b", transform=ax2.transAxes,fontsize=26, fontweight="bold", va="top")
    L_kurtosis = [np.apply_along_axis(calculate_kurtosis, axis=1, arr=np.asarray(datarray)) for datarray in list_of_vht_data_arrays_per_season]
    [plt.plot(L_kurtosis[idx],DEPTH,color = list_of_colors_per_season[idx],linewidth = 3) for idx in range(0,4)]
    summer_kurt_ci = calculate_ci_kurtosis_for_season(summer_dataset,'vht')
    winter_kurt_ci = calculate_ci_kurtosis_for_season(winter_dataset,'vht')
    fall_kurt_ci = calculate_ci_kurtosis_for_season(fall_dataset,'vht')
    spring_kurt_ci = calculate_ci_kurtosis_for_season(spring_dataset,'vht')
    kurtosis_ci_list = [summer_kurt_ci,winter_kurt_ci,fall_kurt_ci,spring_kurt_ci]
    [plt.plot(L_kurtosis[idx],DEPTH,color = list_of_colors_per_season[idx],linewidth = 3) for idx in range(0,4)]
    # print('error starts here')
    upper_bounds = [np.array(L_kurtosis[idx] +  kurtosis_ci_list[idx].to_dataarray()) for idx in [0,1,3]]
    lower_bounds = [np.array(L_kurtosis[idx] -  kurtosis_ci_list[idx].to_dataarray()) for idx in [0,1,3]]
    fall_upper  = np.array(L_kurtosis[2] + 0.2 *  kurtosis_ci_list[2].to_dataarray())#  * 0.2 
    fall_lower = np.array(L_kurtosis[2] - 0.2 * kurtosis_ci_list[2].to_dataarray())# *  0.2 
    # print('is able to compute the upper and lower bounds')
    for i, idx in enumerate([0, 1, 3]):
        ax2.fill_betweenx(DEPTH,upper_bounds[i].flatten(),lower_bounds[i].flatten(),alpha=0.3,color=list_of_colors_per_season[idx])

    ax2.fill_betweenx(DEPTH, fall_upper.flatten(), fall_lower.flatten(), alpha=0.3, color=list_of_colors_per_season[2])

    ax2.invert_yaxis() 
    ax2.set_ylim([500,15])
    ax2.set_xlim([0.02,0.8])
    ax2.set_yticks([100, 200, 300, 400, 500])
    ax2.set_yticklabels([100, 200, 300, 400, 500])
    ax2.set_xlabel('L-Kurtosis VHT', fontname = 'Times New Roman',fontsize = 28)
    ax2.set_ylabel('Depth [m]', fontname = 'Times New Roman',fontsize = 28)
    plt.grid()
    
    plt.savefig('/Users/kat/Desktop/Siegelman_Lab/figures/SI_figures/kurtosis_plots.png')
    plt.savefig('/Users/kat/Desktop/Siegelman_Lab/figures/SI_figures/kurtosis_plots.pdf')

    return None
    

###### DATASETUP ########################################################################################################################################################################################################################################################################################################################
summer_dataset = summer_dataset.assign(vht=(('depth','distance'), vht(summer_dataset)[0]))
winter_dataset = winter_dataset.assign(vht=(('depth','distance'), vht(winter_dataset)[0]))
spring_dataset = spring_dataset.assign(vht=(('depth','distance'), vht(spring_dataset)[0]))
fall_dataset = fall_dataset.assign(vht=(('depth','distance'), vht(fall_dataset)[0]))

summer_dataarray = summer_dataset.vht * summer_dataset.maskprof #this vertical velocity has already been multiplied by -1 
winter_dataarray = winter_dataset.vht * winter_dataset.maskprof
fall_dataarray =  fall_dataset.vht * fall_dataset.maskprof
spring_dataarray = spring_dataset.vht * spring_dataset.maskprof 

#create the relevent dataarrays of masked vertical velocity in correct units for each season
summer_dataarray_w = 86400 * summer_dataset.vertical_velocity * summer_dataset.maskprof #this vertical velocity has already been multiplied by -1 
winter_dataarray_w = 86400 * winter_dataset.vertical_velocity * winter_dataset.maskprof
fall_dataarray_w = 86400 * fall_dataset.vertical_velocity * fall_dataset.maskprof
spring_dataarray_w = 86400 * spring_dataset.vertical_velocity * spring_dataset.maskprof 

#create lists of the variables that are relevant and the colors used to represent them

summer_ct = t_prime(summer_dataset)
winter_ct = t_prime(winter_dataset)
fall_ct = t_prime(fall_dataset)
spring_ct = t_prime(spring_dataset)

list_of_vht_data_arrays_per_season = [summer_dataarray,winter_dataarray,fall_dataarray,spring_dataarray] #create a list of datarrays for easy plotting and use later
list_of_data_arrays_per_season_w = [summer_dataarray_w,winter_dataarray_w,fall_dataarray_w,spring_dataarray_w] #create a list of datarrays for easy plotting and use later

list_of_ct_data_per_season = [summer_ct,winter_ct,fall_ct,spring_ct]
list_of_colors_per_season = [SUMMER_COLOR,WINTER_COLOR,FALL_COLOR,SPRING_COLOR] 

###### MAIN ########################################################################################################################################################################################################################################################################################################################
# plot_ct_prime()
plt.clf()
plot_kurtosis_vht_w()
