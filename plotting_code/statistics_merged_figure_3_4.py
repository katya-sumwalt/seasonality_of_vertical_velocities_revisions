''' This script plots a 2 x 3 figure that shows the pdf of w the rms of w the strain mean and rms plot
    The pdf VHT, the rms VHT, and the mean VHT  

'''

import xarray as xr 
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import matplotlib.ticker as ticker

import lmoments3 as lm
from calculate_statistics import rms_ci95_depth_t 
from calculate_statistics import rms_over_all_dims 
from calculate_statistics import mean_over_all_dims
from calculate_statistics import conditional_seasonal_mean_ci
from calculate_statistics import mean_ci95_depth_distance
from matplotlib.gridspec import GridSpec 
from time_series_of_mean_upward_vertcal_heat_transport import plot_time_series_of_mean_upward_vertical_heat_transport_function




###### DATA STUFF  ################## ################## ##################   ################## ################## ##################   ################## ################## ##################
# LAMBDA FUNCTIONS
call_function_per_season = lambda dataarray :rms_ci95_depth_t(dataarray)
vht = lambda ds:calculate_vhf(ds)
t_prime = lambda ds: (ds.temperature * ds.maskprof) - (ds.temperature * ds.maskprof).mean(dim = 'distance')

call_rms_function_per_season = lambda dataarray :rms_ci95_depth_t(dataarray)
call_mean_function_per_season = lambda dataarray :mean_ci95_depth_distance(dataarray)
call_conditional_mean_function_per_season = lambda dataarray :conditional_seasonal_mean_ci(dataarray)
    
plt.rcParams.update({
    "text.usetex": False,          # enable full LaTeX rendering
    "font.family": "serif",       # use a serif font
    "font.size": 22
})

dir_out = f'/Users/kat/Desktop/Siegelman_Lab/figures/Figure_3/fig3_6panel_SUBSETS.png'
    
# summer_dataset = xr.open_dataset('/Users/kat/Desktop/LS/siegelman-lab-ksumwalt/seal-vertical-velocities/Data/east/seal_data_with_vertical_velocity_east_summer_200m_not_removed_3.nc')
# winter_dataset = xr.open_dataset('/Users/kat/Desktop/LS/siegelman-lab-ksumwalt/seal-vertical-velocities/Data/east/seal_data_with_vertical_velocity_east_winter_200m_not_removed_3.nc')
# fall_dataset = xr.open_dataset('/Users/kat/Desktop/LS/siegelman-lab-ksumwalt/seal-vertical-velocities/Data/east/seal_data_with_vertical_velocity_east_fall_200m_not_removed_4.nc') 
# spring_dataset = xr.open_dataset('/Users/kat/Desktop/LS/siegelman-lab-ksumwalt/seal-vertical-velocities/Data/east/seal_data_with_vertical_velocity_east_spring_200m_not_removed.nc') #5 is most clean 

#### FOR SUBSETS OF THE DATA 
spring_dataset = xr.open_dataset('/Users/kat/Desktop/All_of_Work_With_Lia/LS/siegelman-lab-ksumwalt/seal-vertical-velocities/Data/east/spring_subset.nc') 
winter_dataset = xr.open_dataset('/Users/kat/Desktop/All_of_Work_With_Lia/LS/siegelman-lab-ksumwalt/seal-vertical-velocities/Data/east/winter_subset.nc') 
fall_dataset = xr.open_dataset('/Users/kat/Desktop/All_of_Work_With_Lia/LS/siegelman-lab-ksumwalt/seal-vertical-velocities/Data/east/fall_subset.nc') 
summer_dataset = xr.open_dataset('/Users/kat/Desktop/All_of_Work_With_Lia/LS/siegelman-lab-ksumwalt/seal-vertical-velocities/Data/east/summer_subset.nc') 

DEPTH = -1 * summer_dataset.depth
SUMMER_COLOR = 'tab:blue' #'#FE420F'
WINTER_COLOR = 'tab:green' #'#15B01A' #'#004B23'   # deeper, cooler green
FALL_COLOR   = 'tab:orange'#'#004B23' #'#9FBF3B'   # more yellow-green (brighter)
SPRING_COLOR  = 'tab:red'#FFB000'   # more golden/yellow 

#create the relevent dataarrays of masked vertical velocity in correct units for each season
summer_dataarray = 86400* summer_dataset.vertical_velocity * summer_dataset.maskprof #this vertical velocity has already been multiplied by -1 
winter_dataarray = 86400 * winter_dataset.vertical_velocity * winter_dataset.maskprof
fall_dataarray = 86400 * fall_dataset.vertical_velocity * fall_dataset.maskprof
spring_dataarray = 86400 * spring_dataset.vertical_velocity * spring_dataset.maskprof 

#create lists of the variables that are relevant and the colors used to represent them
list_of_data_arrays_per_season = [summer_dataarray,winter_dataarray,fall_dataarray,spring_dataarray] #create a list of datarrays for easy plotting and use later
list_of_colors_per_season = [SUMMER_COLOR,WINTER_COLOR,FALL_COLOR,SPRING_COLOR]
list_of_datasets = [summer_dataset,winter_dataset,fall_dataset,spring_dataset]

def calculate_kurtosis(row):
    row = row[np.isfinite(row)]
    return lm.lmom_ratios(row, nmom=4)[3]

def calculate_vhf(ds):
    rcp = 4e6  
    w_neg = -1 * ds.vertical_velocity #since the dataset already has the negative 
    VHF_step1 = ds.temperature - (ds.temperature).mean(dim = 'distance') 
    VHF_step2 = (w_neg - (w_neg).mean(dim = 'distance'))* rcp
    VHF = VHF_step1.values * VHF_step2.values ## the issue is in broadcasting step1 and step2 together
    vhf_mean = (np.mean((VHF * ds.maskprof), axis=1))
    
    return VHF, vhf_mean

summer_dataset = summer_dataset.assign(vht=(('depth','distance'), vht(summer_dataset)[0]))
winter_dataset = winter_dataset.assign(vht=(('depth','distance'), vht(winter_dataset)[0]))
spring_dataset = spring_dataset.assign(vht=(('depth','distance'), vht(spring_dataset)[0]))
fall_dataset = fall_dataset.assign(vht=(('depth','distance'), vht(fall_dataset)[0]))


summer_dataarray_vht = summer_dataset.vht * summer_dataset.maskprof #this vertical velocity has already been multiplied by -1 
winter_dataarray_vht = winter_dataset.vht * winter_dataset.maskprof
fall_dataarray_vht =  fall_dataset.vht * fall_dataset.maskprof
spring_dataarray_vht = spring_dataset.vht * spring_dataset.maskprof 

summer_ct = t_prime(summer_dataset)
winter_ct = t_prime(winter_dataset)
fall_ct = t_prime(fall_dataset)
spring_ct = t_prime(spring_dataset)

#create lists of the variables that are relevant and the colors used to represent them
list_of_vht_data_arrays_per_season = [summer_dataarray_vht,winter_dataarray_vht,fall_dataarray_vht,spring_dataarray_vht] #create a list of datarrays for easy plotting and use later
list_of_ct_data_per_season = [summer_ct,winter_ct,fall_ct,spring_ct]
list_of_colors_per_season = [SUMMER_COLOR,WINTER_COLOR,FALL_COLOR,SPRING_COLOR]


# FUNCTIONS FOR STATISTICS AND PLOTTING
def plot_variable_vs_depth_and_ci(ax,rms,confidence_interval,color,depth= DEPTH,confidence_interval_bool = True): 
    if confidence_interval_bool == True:
        upper_bound = rms + confidence_interval
        lower_bound = rms - confidence_interval
        ax.fill_betweenx(DEPTH, upper_bound, lower_bound, alpha=0.3, color=color)
        ax.plot(rms,DEPTH, color = color,linewidth = 3)
    else: 
        ax.plot(rms,DEPTH, color = color,linewidth = 3) 
    return None







def plot_figure3_6_panel(summer_dataarray,winter_dataarray,fall_dataarray,spring_dataarray,dir_out = dir_out): 
    plt.clf()
    #### SET UP ORIENTATION OF FIGURE
    fig = plt.figure(figsize = (25,25))
    plt.rcParams.update({
    "text.usetex": False,          # enable full LaTeX rendering
    "font.family": "serif",       # use a serif font
    "font.size": 22
    })
    gs = GridSpec(3, 3,wspace= 0.25,hspace = 0.25)
    ax = plt.subplot(gs[0, 0])
    ax2 = plt.subplot(gs[0,1])
    ax3 = plt.subplot(gs[0, 2])
    ax4 = plt.subplot(gs[1, 0])
    ax5 = plt.subplot(gs[1,1])
    ax6 = plt.subplot(gs[1, 2])
    ax7 = plt.subplot(gs[2,:])

    ################## PLOT THE  W DISTRIBUTION  ################## ################## ##################  ################## ##################  ##################  ################## ################## ##################
  
    ax.text(0.0, 1.08, "a", transform=ax.transAxes,fontweight="bold", va="top", fontsize = 27)
    ax.hist(summer_dataarray.values.ravel(),bins=100, density=True, histtype='step', linewidth=3,label='Summer [DJF]', color = SUMMER_COLOR)
    ax.hist(winter_dataarray.values.ravel(),bins=100, density=True, histtype='step', linewidth=3,label='Winter [JJA]', color = WINTER_COLOR)
    ax.hist(spring_dataarray.values.ravel(),bins=100, density=True, histtype='step', linewidth=3,label='Spring [SON]',color = SPRING_COLOR)
    ax.hist(fall_dataarray.values.ravel(),bins=100, density=True, histtype='step', linewidth=3,label='Fall [MAM]',color = FALL_COLOR)
    # plt.rcParams['mathtext.fontset'] = 'stix'
    ax.set_ylabel('PDF',fontname='Times New Roman',fontsize = 25)
    ax.set_xlabel(r"w [m.d$^{-1}$]",fontsize = 25)
    plt.legend(loc = 'upper left')#, fontsize = 20)
    ax.set_yscale('log')
    ax.set_xlim([-150,150])
    ax.set_ylim([10**(-5), 10**0])
    ax.set_xticks([-100,-50, 0,50,100])
    ax.set_xticklabels([-100,-50,0,50,100])
    ax.legend(loc = 'upper left', fontsize = 16)#,fontsize = 20)
    ax.grid()
    
    ### PLOT THE RMS AND THE CONFIDENCE INTERVAL ################## ################## ##################  ################## ################## ##################  ################## ################## ##################
    # ax2 = plt.subplot(2,3,2)
    ax2.text(0.0, 1.08, "b", transform=ax2.transAxes,fontweight="bold", va="top", fontsize = 27)
    w_rms = [call_function_per_season(season_array)[0] for season_array in list_of_data_arrays_per_season] #order of summer, winter,fall,spring
    w_confidence_interval = [call_function_per_season(season_array)[1] for season_array in list_of_data_arrays_per_season] #summer, winter,fall,spring
    [plot_variable_vs_depth_and_ci(ax2,w_rms[idx],w_confidence_interval[idx],list_of_colors_per_season[idx]) for idx in range(0,4)]
    ax2.invert_yaxis()
    ax2.set_xlim([0,25])
    ax2.set_ylim([500,15])
    ax2.set_yticks([100, 200, 300, 400, 500])
    ax2.set_yticklabels([100, 200, 300, 400, 500])
    ax2.set_xticks([5, 10, 15, 20, 25])
    ax2.set_xticklabels([5, 10, 15, 20, 25])
    ax2.set_ylabel('Depth [m]',fontname='Times New Roman',fontsize = 25)
    ax2.set_xlabel(r"rms w [m.d$^{-1}$]",fontsize = 25)
    ax2.grid()
    
    #PLOT THE STRAIN AND BX   ################## ################## ##################   ################## ################## ##################   ################## ################## ##################
    # ax3 = plt.subplot(2,3,3)
    ax3.text(0.0, 1.08, "c", transform=ax3.transAxes,fontsize=27, fontweight="bold", va="top")
    
    summer_bx = summer_dataset.bx * summer_dataset.maskprof
    winter_bx = winter_dataset.bx * winter_dataset.maskprof
    fall_bx = fall_dataset.bx * fall_dataset.maskprof
    spring_bx = spring_dataset.bx * spring_dataset.maskprof
    
    summer_strain = summer_dataset.strain_neuro 
    winter_strain = winter_dataset.strain_neuro 
    fall_strain = fall_dataset.strain_neuro 
    spring_strain = spring_dataset.strain_neuro 
    
    season_array_bx = [spring_bx,summer_bx,fall_bx,winter_bx,]
    list_season_array_strain = [winter_strain,spring_strain,summer_strain,fall_strain,]
    season = ['winter','spring','summer','fall']
    season_colors_for_scatter = [WINTER_COLOR,SPRING_COLOR,SUMMER_COLOR,FALL_COLOR]
    
    means_strain = [mean_over_all_dims(strain_array)[0] for strain_array in list_season_array_strain]
    ci_mean_strain = [mean_over_all_dims(strain_array)[1] for strain_array in list_season_array_strain]
    rms_strain = [rms_over_all_dims(strain_array)[0] for strain_array in list_season_array_strain]
    ci_rms_strain = [rms_over_all_dims(strain_array)[1] for strain_array in list_season_array_strain]
    
    means_bx = [mean_over_all_dims(strain_array)[0] for strain_array in season_array_bx]
    ci_mean_bx = [mean_over_all_dims(strain_array)[1] for strain_array in season_array_bx]
    rms_bx = [rms_over_all_dims(strain_array)[0] for strain_array in season_array_bx]
    ci_rms_bx = [rms_over_all_dims(strain_array)[1] for strain_array in season_array_bx]
    
    x_seasons = range(len(season_array_bx))
    #plot strain rms
    [ax3.errorbar(x_seasons[idx], 1e5 * rms_strain[idx], yerr= 1e5 * ci_rms_strain[idx],color = season_colors_for_scatter[idx],ms = 7,elinewidth = 3, fmt='D-', capsize=5, label=r"rms $\sigma$")
    for idx in range(len(list_of_colors_per_season))]
    ax3.plot(x_seasons, 1e5 * np.array(rms_strain), color = 'grey',alpha = 0.5) 
    # ax4.set_ylabel(r"rms $\sigma$ [s$^{-1}$]")
    # ax4.set_ylim([0.8,1.45])
    #plot strain mean
    # ax4_2 = ax4.twinx()
    [ax3.errorbar(x_seasons[idx], 1e5 * means_strain[idx], yerr= 1e5 *ci_mean_strain[idx],color = season_colors_for_scatter[idx],ms = 7, elinewidth = 3,fmt='o-', capsize=5, label=r"mean $\sigma$")
    for idx in range(len(list_of_colors_per_season))]
    ax3.plot(x_seasons, 1e5 * np.array(means_strain),color = 'grey',alpha = 0.5) 
    
    ax3.set_ylabel(r"mean,rms $\sigma$ [s$^{-1}$]")
    ax3.set_xticks(x_seasons,season)
    # plt.rcParams['mathtext.fontset'] = 'stix'
    ax3.text(-0.13, 0.992, r"$\times 10^{-5}$", transform=ax3.transAxes,fontsize = 14, va="top")# fontsize= 18, va="top")
    legend_elements = [Line2D([0], [0],marker ='D',markersize= 5, lw=1,color = 'k', label=r"rms $\sigma$ "),
                       Line2D([0], [0], marker='o',lw=1,markersize= 5, color = 'k' ,label=r"mean $\sigma$")]
    # ax4_2.set_ylim([0.6,1.1])
    ax3.legend(handles=legend_elements, loc='lower left')
    # print('completed the change ')
    plt.grid(axis = 'both')
    ax3.xaxis.grid(True)
    ax3.yaxis.grid(True)
    # plt.grid()

    
    # ax4 = plt.subplot(2,3,4)
    ax4.text(0.0, 1.08, "d", transform=ax4.transAxes,fontweight="bold", va="top", fontsize = 27)
    ax4.hist(summer_dataarray_vht.values.ravel(),bins=100, density=True, histtype='step', linewidth=3,label='Summer [DJF]', color = SUMMER_COLOR)
    ax4.hist(winter_dataarray_vht.values.ravel(),bins=100, density=True, histtype='step', linewidth=3,label='Winter [JJA]', color = WINTER_COLOR)
    ax4.hist(spring_dataarray_vht.values.ravel(),bins=100, density=True, histtype='step', linewidth=3,label='Spring [SON]',color = SPRING_COLOR)
    ax4.hist(fall_dataarray_vht.values.ravel(),bins=100, density=True, histtype='step', linewidth=3,label='Fall [MAM]' ,color = FALL_COLOR)
    ax4.set_ylabel('PDF',fontname='Times New Roman',fontsize = 25)
    ax4.set_xlabel(r"VHT [W.m$^{-2}$]",fontsize = 25)
    # plt.legend(loc = 'upper left',fontsize = 20)
    ax4.set_yscale('log')
    ax4.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax4.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
    ax4.xaxis.get_offset_text().set_visible(False)
    ax4.text(0.920, -0.05, r"$\times 10^{4}$", transform=ax4.transAxes,fontsize = 14,va="bottom")#, fontsize= 18, va="bottom")
    ax4.set_ylim([10**(-7.5), 10**(-1.5)])
    ax4.set_xlim([-1.4e4, 1.4e4])
    ax4.grid()
    
    ###VHT -----> RMS AND THE CONFIDENCE INTERVAL  ################## ################## ##################   ################## ################## ##################   ################## ################## ##################
    # ax5 = plt.subplot(2,3,5)
    ax5.text(0.0, 1.08, "e", transform=ax5.transAxes,fontsize=27, fontweight="bold", va="top")
    vht_rms = [call_rms_function_per_season(season_array)[0] for season_array in list_of_vht_data_arrays_per_season] #order of summer, winter,fall,spring
    vht_confidence_interval = [call_rms_function_per_season(season_array)[1] for season_array in list_of_vht_data_arrays_per_season] #summer, winter,fall,spring
    [plot_variable_vs_depth_and_ci(ax5,vht_rms[idx],vht_confidence_interval[idx],list_of_colors_per_season[idx]) for idx in range(0,4)]
    ax5.invert_yaxis()
    # ax2.set_xlim([0,25])
    ax5.set_ylim([500,15])
    # ax2.set_yticks([100, 200, 300, 400, 500])
    # ax2.set_yticklabels([100, 200, 300, 400, 500])
    # ax2.set_xticks([5, 10, 15, 20, 25])
    # ax2.set_xticklabels([5, 10, 15, 20, 25])
    ax5.set_ylabel('Depth [m]',fontname='Times New Roman',fontsize = 25)
    ax5.set_xlabel(r"rms VHT [W.m$^{-2}$]",fontsize = 25)
    ax5.grid()
    
    
    ###VHT -----> MEAN  ################## ################## ##################   ################## ################## ##################   ################## ################## ##################
    # ax6 = plt.subplot(2,3,6)
    ax6.text(0.0, 1.08, "f", transform=ax6.transAxes,fontsize=27, fontweight="bold", va="top")
    vht_mean = [call_mean_function_per_season(season_array)[0] for season_array in list_of_vht_data_arrays_per_season] #order of summer, winter,fall,spring
    vht_mean_confidence_interval = [call_mean_function_per_season(season_array)[1] for season_array in list_of_vht_data_arrays_per_season] #summer, winter,fall,spring
    [plot_variable_vs_depth_and_ci(ax6,vht_mean[idx],vht_mean_confidence_interval[idx],list_of_colors_per_season[idx]) for idx in range(0,4)]
    ax6.invert_yaxis() 
    ax6.set_ylim([500,15])
    ax6.set_yticks([100, 200, 300, 400, 500])
    ax6.set_yticklabels([100, 200, 300, 400, 500])
    ax6.set_xlabel(r"<VHT> [W.m$^{-2}$]",fontsize = 25)
    ax6.set_ylabel('Depth [m]', fontname = 'Times New Roman',fontsize = 25)
    ax6.grid()
    
    
    ### ADDED TIME SERIES PLOT 
    plot_time_series_of_mean_upward_vertical_heat_transport_function(ax7)
    ax7.text(0.0, 1.09, "g", transform=ax7.transAxes,fontsize=27, fontweight="bold", va="top")

    
    # plt.show()
    
    # plt.savefig(dir_out) 
    plt.savefig('/Users/kat/Desktop/All_of_Work_With_Lia/Siegelman_Lab_Initial_Submit/figures/Subsetted_Statistics/fig3_6panel_1.pdf')
    plt.savefig('/Users/kat/Desktop/All_of_Work_With_Lia/Siegelman_Lab_Initial_Submit/figures/Subsetted_Statistics/fig3_6panel_1.png')
    
    # plt.tight_layout()
    return None
        






plot_figure3_6_panel(summer_dataarray,winter_dataarray,fall_dataarray,spring_dataarray) 