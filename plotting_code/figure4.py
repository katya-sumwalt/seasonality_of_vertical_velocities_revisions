''' 
This plot will a 2x2 panel plot containing the VHT pdf, the VHT rms, the mean VHT and the conditional mean VHT
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
from ci_for_kurtpsis import calculate_ci_kurtosis_for_season
# from figure3 import plot_variable_vs_depth_and_ci

summer_dataset = xr.open_dataset('/Users/kat/Desktop/LS/siegelman-lab-ksumwalt/seal-vertical-velocities/Data/east/seal_data_with_vertical_velocity_east_summer_200m_not_removed_3.nc')
winter_dataset = xr.open_dataset('/Users/kat/Desktop/LS/siegelman-lab-ksumwalt/seal-vertical-velocities/Data/east/seal_data_with_vertical_velocity_east_winter_200m_not_removed_3.nc')
fall_dataset = xr.open_dataset('/Users/kat/Desktop/LS/siegelman-lab-ksumwalt/seal-vertical-velocities/Data/east/seal_data_with_vertical_velocity_east_fall_200m_not_removed_4.nc') 
spring_dataset = xr.open_dataset('/Users/kat/Desktop/LS/siegelman-lab-ksumwalt/seal-vertical-velocities/Data/east/seal_data_with_vertical_velocity_east_spring_200m_not_removed.nc') #5 is most clean 

DEPTH = -1 * summer_dataset.depth
SUMMER_COLOR = 'tab:blue' #'#FE420F'
WINTER_COLOR = 'tab:green' #'#15B01A' #'#004B23'   # deeper, cooler green
FALL_COLOR   = 'tab:orange'#'#004B23' #'#9FBF3B'   # more yellow-green (brighter)
SPRING_COLOR  = 'tab:red'#FFB000'   # more golden/yellow 

plt.rcParams.update({
    "text.usetex": False,          # enable full LaTeX rendering
    "font.family": "serif",       # use a serif font
    "font.size": 22
})

def plot_variable_vs_depth_and_ci(ax,rms,confidence_interval,color,depth= DEPTH,confidence_interval_bool = True): 
    if confidence_interval_bool == True:
        upper_bound = rms + confidence_interval
        lower_bound = rms - confidence_interval
        ax.fill_betweenx(DEPTH, upper_bound, lower_bound, alpha=0.3, color=color)
        ax.plot(rms,DEPTH, color = color,linewidth = 3)
    else: 
        ax.plot(rms,DEPTH, color = color,linewidth = 3) 
    return None

def calculate_vhf(ds):
    rcp = 4e6  
    w_neg = -1 * ds.vertical_velocity #since the dataset already has the negative 
    VHF_step1 = ds.temperature - (ds.temperature).mean(dim = 'distance') 
    VHF_step2 = (w_neg - (w_neg).mean(dim = 'distance'))* rcp
    VHF = VHF_step1.values * VHF_step2.values ## the issue is in broadcasting step1 and step2 together
    vhf_mean = (np.mean((VHF * ds.maskprof), axis=1))
    
    return VHF, vhf_mean

vht = lambda ds:calculate_vhf(ds)
t_prime = lambda ds: (ds.temperature * ds.maskprof) - (ds.temperature * ds.maskprof).mean(dim = 'distance')

call_rms_function_per_season = lambda dataarray :rms_ci95_depth_t(dataarray)
call_mean_function_per_season = lambda dataarray :mean_ci95_depth_distance(dataarray)
call_conditional_mean_function_per_season = lambda dataarray :conditional_seasonal_mean_ci(dataarray)




summer_dataset = summer_dataset.assign(vht=(('depth','distance'), vht(summer_dataset)[0]))
winter_dataset = winter_dataset.assign(vht=(('depth','distance'), vht(winter_dataset)[0]))
spring_dataset = spring_dataset.assign(vht=(('depth','distance'), vht(spring_dataset)[0]))
fall_dataset = fall_dataset.assign(vht=(('depth','distance'), vht(fall_dataset)[0]))


summer_dataarray = summer_dataset.vht * summer_dataset.maskprof #this vertical velocity has already been multiplied by -1 
winter_dataarray = winter_dataset.vht * winter_dataset.maskprof
fall_dataarray =  fall_dataset.vht * fall_dataset.maskprof
spring_dataarray = spring_dataset.vht * spring_dataset.maskprof 

summer_ct = t_prime(summer_dataset)
# print(summer_ct.mean(dim = 'distance'))
winter_ct = t_prime(winter_dataset)
fall_ct = t_prime(fall_dataset)
spring_ct = t_prime(spring_dataset)

#create lists of the variables that are relevant and the colors used to represent them
list_of_vht_data_arrays_per_season = [summer_dataarray,winter_dataarray,fall_dataarray,spring_dataarray] #create a list of datarrays for easy plotting and use later
list_of_ct_data_per_season = [summer_ct,winter_ct,fall_ct,spring_ct]
list_of_colors_per_season = [SUMMER_COLOR,WINTER_COLOR,FALL_COLOR,SPRING_COLOR]


dir_out = f'/Users/kat/Desktop/Siegelman_Lab/figures/Figure_4/fig4.pdf'

def plot_figure4(summer_dataarray,winter_dataarray,fall_dataarray,spring_dataarray,dir_out = dir_out): 
    '''  This is the main funciton of this script. This function plots the statistics necessary
          for the vertical velocity data. By calling the other helper functions here. 
    
    '''
    plt.figure(figsize = (20,20))
    plt.rcParams["font.size"] = 24
    
    ### PLOT THE DISTRIBUTION OF VHT
    ax = plt.subplot(2,2,1)
    ax.text(0.0, 1.06, "a", transform=ax.transAxes,fontsize=26, fontweight="bold", va="top")
    ax.hist(summer_dataarray.values.ravel(),bins=100, density=True, histtype='step', linewidth=3,label='Summer [DJF]', color = SUMMER_COLOR)
    ax.hist(winter_dataarray.values.ravel(),bins=100, density=True, histtype='step', linewidth=3,label='Winter [JJA]', color = WINTER_COLOR)
    ax.hist(spring_dataarray.values.ravel(),bins=100, density=True, histtype='step', linewidth=3,label='Spring [SON]',color = SPRING_COLOR)
    ax.hist(fall_dataarray.values.ravel(),bins=100, density=True, histtype='step', linewidth=3,label='Fall [MAM]' ,color = FALL_COLOR)
    ax.set_ylabel('PDF',fontname='Times New Roman',fontsize = 28)
    ax.set_xlabel(r"VHT $[W.m^{-2}]$", fontname='Times New Roman',fontsize = 28)
    plt.legend(loc = 'upper left',fontsize = 20)
    ax.set_yscale('log')
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
    ax.set_ylim([10**(-7.5), 10**(-1.5)])
    plt.grid()
    
    ### PLOT THE RMS AND THE CONFIDENCE INTERVAL 
    ax2 = plt.subplot(2,2,2)
    ax2.text(0.0, 1.06, "b", transform=ax2.transAxes,fontsize=26, fontweight="bold", va="top")
    vht_rms = [call_rms_function_per_season(season_array)[0] for season_array in list_of_vht_data_arrays_per_season] #order of summer, winter,fall,spring
    vht_confidence_interval = [call_rms_function_per_season(season_array)[1] for season_array in list_of_vht_data_arrays_per_season] #summer, winter,fall,spring
    [plot_variable_vs_depth_and_ci(ax2,vht_rms[idx],vht_confidence_interval[idx],list_of_colors_per_season[idx]) for idx in range(0,4)]
    ax2.invert_yaxis()
    # ax2.set_xlim([0,25])
    ax2.set_ylim([500,15])
    # ax2.set_yticks([100, 200, 300, 400, 500])
    # ax2.set_yticklabels([100, 200, 300, 400, 500])
    # ax2.set_xticks([5, 10, 15, 20, 25])
    # ax2.set_xticklabels([5, 10, 15, 20, 25])
    ax2.set_ylabel('Depth [m]',fontname='Times New Roman',fontsize = 28)
    ax2.set_xlabel(r"rms VHT $[W.m^{-2}]$", fontname='Times New Roman',fontsize = 28)
    plt.grid()
    
    # PLOT THE VHT MEAN
    ax3 = plt.subplot(2,2,3)
    ax3.text(0.0, 1.06, "c", transform=ax3.transAxes,fontsize=26, fontweight="bold", va="top")
    vht_mean = [call_mean_function_per_season(season_array)[0] for season_array in list_of_vht_data_arrays_per_season] #order of summer, winter,fall,spring
    # print(vht_mean)
    vht_mean_confidence_interval = [call_mean_function_per_season(season_array)[1] for season_array in list_of_vht_data_arrays_per_season] #summer, winter,fall,spring
    [plot_variable_vs_depth_and_ci(ax3,vht_mean[idx],vht_mean_confidence_interval[idx],list_of_colors_per_season[idx]) for idx in range(0,4)]
    ax3.invert_yaxis() 
    ax3.set_ylim([500,15])
    ax3.set_yticks([100, 200, 300, 400, 500])
    ax3.set_yticklabels([100, 200, 300, 400, 500])
    ax3.set_xlabel(r"<VHT> $[w.m^{-2}]$", fontname='Times New Roman',fontsize = 28)
    ax3.set_ylabel('Depth [m]', fontname = 'Times New Roman',fontsize = 28)
    plt.grid()
    
    
    #PLOT THE CT VARIABLE rms
    ax4 = plt.subplot(2,2,4)
    ax4.text(0.0, 1.06, "d", transform=ax4.transAxes,fontsize=26, fontweight="bold", va="top")
    ct_mean = [call_rms_function_per_season(season_array)[0] for season_array in list_of_ct_data_per_season] #order of summer, winter,fall,spring
    # print(ct_mean)
    ct_mean_confidence_interval = [call_rms_function_per_season(season_array)[1] for season_array in list_of_ct_data_per_season] #summer, winter,fall,spring

    [plot_variable_vs_depth_and_ci(ax4,ct_mean[idx],ct_mean_confidence_interval[idx],list_of_colors_per_season[idx]) for idx in range(0,4)]
    ax4.invert_yaxis() 
    ax4.set_ylim([500,15])
    ax4.set_yticks([100, 200, 300, 400, 500])
    ax4.set_yticklabels([100, 200, 300, 400, 500])
    ax4.set_xlabel(r"rms CT'[°C]", fontname='Times New Roman',fontsize = 28)
    ax4.set_ylabel('Depth [m]', fontname = 'Times New Roman',fontsize = 28)
    plt.grid()

    
    #PLOT THE CONDITIONAL VHT
    
    # cond_pos = VHT > 0
    # cond_neg = VHT < 0
    # ax4 = plt.subplot(2,2,4)
    # ax4.text(0.0, 1.06, "d", transform=ax3.transAxes,fontsize=26, fontweight="bold", va="top")
    # vht_mean = [call_conditional_mean_function_per_season(season_array)[0] for season_array in list_of_vht_data_arrays_per_season] #order of summer, winter,fall,spring
    # print(vht_mean)
    # vht_mean_confidence_interval = [call_conditional_mean_function_per_season(season_array)[1] for season_array in list_of_vht_data_arrays_per_season] #summer, winter,fall,spring
    # [plot_variable_vs_depth_and_ci(ax3,vht_mean[idx],vht_mean_confidence_interval[idx],list_of_colors_per_season[idx]) for idx in range(0,4)]
    # ax3.invert_yaxis() 
    # ax3.set_ylim([500,15])
    # ax3.set_yticks([100, 200, 300, 400, 500])
    # ax3.set_yticklabels([100, 200, 300, 400, 500])
    # ax3.set_xlabel(r"<VHT> $[w.m^{-2}]$", fontname='Times New Roman',fontsize = 28)
    # ax3.set_ylabel('Depth [m]', fontname = 'Times New Roman',fontsize = 28)
    # plt.grid()
    
    
    plt.savefig(dir_out)
    
    return None


plot_figure4(summer_dataarray,winter_dataarray,fall_dataarray,spring_dataarray)



###### PLOT PANEL KURTOSIS FOR SI ####### 
def calculate_vhf(ds):
    rcp = 4e6  
    w_neg = -1 * ds.vertical_velocity #since the dataset already has the negative 
    VHF_step1 = ds.temperature - (ds.temperature).mean(dim = 'distance') 
    VHF_step2 = (w_neg - (w_neg).mean(dim = 'distance'))* rcp
    VHF = VHF_step1.values * VHF_step2.values ## the issue is in broadcasting step1 and step2 together
    vhf_mean = (np.mean((VHF * ds.maskprof), axis=1))
    
    return VHF, vhf_mean

vht = lambda ds:calculate_vhf(ds)
t_prime = lambda ds: (ds.temperature * ds.maskprof) - (ds.temperature * ds.maskprof).mean(dim = 'distance')

call_rms_function_per_season = lambda dataarray :rms_ci95_depth_t(dataarray)
call_mean_function_per_season = lambda dataarray :mean_ci95_depth_distance(dataarray)
call_conditional_mean_function_per_season = lambda dataarray :conditional_seasonal_mean_ci(dataarray)




summer_dataset = summer_dataset.assign(vht=(('depth','distance'), vht(summer_dataset)[0]))
winter_dataset = winter_dataset.assign(vht=(('depth','distance'), vht(winter_dataset)[0]))
spring_dataset = spring_dataset.assign(vht=(('depth','distance'), vht(spring_dataset)[0]))
fall_dataset = fall_dataset.assign(vht=(('depth','distance'), vht(fall_dataset)[0]))
