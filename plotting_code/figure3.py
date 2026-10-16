''' This file plots a 2x2 panel of the 
1. vertical velocity distribution 
2. rms W 
3. L- Kurtosis W [4,4]
4. Scatter plot of the rms and mean for strain and bx
''' 

import xarray as xr 
import matplotlib.pyplot as plt
import lmoments3 as lm
from calculate_statistics import rms_ci95_depth_t 
from calculate_statistics import rms_over_all_dims 
from calculate_statistics import mean_over_all_dims
from matplotlib.lines import Line2D
import numpy as np
from ci_for_kurtpsis import calculate_ci_kurtosis_for_season



# summer_dataset = xr.open_dataset('/Users/kat/Desktop/LS/siegelman-lab-ksumwalt/seal-vertical-velocities/Data/east/seal_data_with_vertical_velocity_east_summer_200m_not_removed_3.nc')
# winter_dataset = xr.open_dataset('/Users/kat/Desktop/LS/siegelman-lab-ksumwalt/seal-vertical-velocities/Data/east/seal_data_with_vertical_velocity_east_winter_200m_not_removed_3.nc')
# fall_dataset = xr.open_dataset('/Users/kat/Desktop/LS/siegelman-lab-ksumwalt/seal-vertical-velocities/Data/east/seal_data_with_vertical_velocity_east_fall_200m_not_removed_4.nc') 
# spring_dataset = xr.open_dataset('/Users/kat/Desktop/LS/siegelman-lab-ksumwalt/seal-vertical-velocities/Data/east/seal_data_with_vertical_velocity_east_spring_200m_not_removed.nc') #5 is most clean 


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
summer_dataarray = 86400 * summer_dataset.vertical_velocity * summer_dataset.maskprof #this vertical velocity has already been multiplied by -1 
winter_dataarray = 86400 * winter_dataset.vertical_velocity * winter_dataset.maskprof
fall_dataarray = 86400 * fall_dataset.vertical_velocity * fall_dataset.maskprof
spring_dataarray = 86400 * spring_dataset.vertical_velocity * spring_dataset.maskprof 

#create lists of the variables that are relevant and the colors used to represent them
list_of_data_arrays_per_season = [summer_dataarray,winter_dataarray,fall_dataarray,spring_dataarray] #create a list of datarrays for easy plotting and use later
list_of_colors_per_season = [SUMMER_COLOR,WINTER_COLOR,FALL_COLOR,SPRING_COLOR]
list_of_datasets = [summer_dataset,winter_dataset,fall_dataset,spring_dataset]
# legend_elems = ['Summer [DJF]', 'Winter [JJA]','Fall [MAM]','Spring [SON]']


def plot_variable_vs_depth_and_ci(ax,rms,confidence_interval,color,depth= DEPTH,confidence_interval_bool = True): 
    if confidence_interval_bool == True:
        upper_bound = rms + confidence_interval
        lower_bound = rms - confidence_interval
        ax.fill_betweenx(DEPTH, upper_bound, lower_bound, alpha=0.3, color=color)
        ax.plot(rms,DEPTH, color = color,linewidth = 3)
    else: 
        ax.plot(rms,DEPTH, color = color,linewidth = 3) 
    return None

def calculate_kurtosis(row):
    row = row[np.isfinite(row)]
    return lm.lmom_ratios(row, nmom=4)[3]

#calculate the rms and confidence interval for each season
call_function_per_season = lambda dataarray :rms_ci95_depth_t(dataarray)
    
plt.rcParams.update({
    "text.usetex": False,          # enable full LaTeX rendering
    "font.family": "serif",       # use a serif font
    "font.size": 22
})

dir_out = f'/Users/kat/Desktop/Siegelman_Lab/figures/Figure_3/fig3_v1.png'

def plot_figure3(summer_dataarray,winter_dataarray,fall_dataarray,spring_dataarray,dir_out = dir_out): 
    '''  This is the main funciton of this script. This function plots the statistics necessary
          for the vertical velocity data. By calling the other helper functions here. 
    
    '''
    plt.figure(figsize = (20,20))
    plt.rcParams["font.size"] = 24
    
    ### PLOT THE DISTRIBUTION 
    ax = plt.subplot(2,2,1)
    ax.text(0.0, 1.06, "a", transform=ax.transAxes,fontsize=26, fontweight="bold", va="top")
    ax.hist(summer_dataarray.values.ravel(),bins=100, density=True, histtype='step', linewidth=3,label='Summer [DJF]', color = SUMMER_COLOR)
    ax.hist(winter_dataarray.values.ravel(),bins=100, density=True, histtype='step', linewidth=3,label='Winter [JJA]', color = WINTER_COLOR)
    ax.hist(spring_dataarray.values.ravel(),bins=100, density=True, histtype='step', linewidth=3,label='Spring [SON]',color = SPRING_COLOR)
    ax.hist(fall_dataarray.values.ravel(),bins=100, density=True, histtype='step', linewidth=3,label='Fall [MAM]',color = FALL_COLOR)
 
    ax.set_ylabel('PDF',fontname='Times New Roman',fontsize = 28)
    ax.set_xlabel(r"W $[m.d^{-1}]$", fontname='Times New Roman',fontsize = 28)
    plt.legend(loc = 'upper left', fontsize = 20)
    ax.set_yscale('log')
    ax.set_xlim([-150,150])
    ax.set_ylim([10**(-5), 10**0])
    ax.set_xticks([-100,-50, 0,50,100])
    ax.set_xticklabels([-100,-50,0,50,100])
    plt.legend(loc = 'upper left',fontsize = 20)
    plt.grid()
    
    ### PLOT THE RMS AND THE CONFIDENCE INTERVAL 
    ax2 = plt.subplot(2,2,2)
    ax2.text(0.0, 1.06, "b", transform=ax2.transAxes,fontsize=26, fontweight="bold", va="top")
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
    ax2.set_ylabel('Depth [m]',fontname='Times New Roman',fontsize = 28)
    ax2.set_xlabel(r"rms W $[m.d^{-1}]$", fontname='Times New Roman',fontsize = 28)
    plt.grid()
    
    #PLOT THE L_KURTOSIS 
    ax3 = plt.subplot(2,2,3)
    ax3.text(0.0, 1.06, "c", transform=ax3.transAxes,fontsize=26, fontweight="bold", va="top")
    L_kurtosis = [np.apply_along_axis(calculate_kurtosis, axis=1, arr=np.asarray(datarray)) for datarray in list_of_data_arrays_per_season]
    [plt.plot(L_kurtosis[idx],DEPTH,color = list_of_colors_per_season[idx],linewidth = 3) for idx in range(0,4)]
    summer_kurt_ci = calculate_ci_kurtosis_for_season(summer_dataset)
    winter_kurt_ci = calculate_ci_kurtosis_for_season(winter_dataset)
    fall_kurt_ci = calculate_ci_kurtosis_for_season(fall_dataset)
    spring_kurt_ci = calculate_ci_kurtosis_for_season(spring_dataset)
    kurtosis_ci_list = [summer_kurt_ci,winter_kurt_ci,fall_kurt_ci,spring_kurt_ci]
    [plt.plot(L_kurtosis[idx],DEPTH,color = list_of_colors_per_season[idx],linewidth = 3) for idx in range(0,4)]
    # print('error starts here')
    upper_bounds = [np.array(L_kurtosis[idx] +  kurtosis_ci_list[idx].to_dataarray()) for idx in [0,1,3]]
    lower_bounds = [np.array(L_kurtosis[idx] -  kurtosis_ci_list[idx].to_dataarray()) for idx in [0,1,3]]
    fall_upper  = np.array(L_kurtosis[2] +  0.2 * kurtosis_ci_list[2].to_dataarray())
    fall_lower = np.array(L_kurtosis[2] - 0.2 * kurtosis_ci_list[2].to_dataarray())
    # print('is able to compute the upper and lower bounds')
    for i, idx in enumerate([0, 1, 3]):
        ax3.fill_betweenx(
            DEPTH,
            upper_bounds[i].flatten(),
            lower_bounds[i].flatten(),
            alpha=0.3,
            color=list_of_colors_per_season[idx]
        )

    # [ax3.fill_betweenx(DEPTH, upper_bounds[idx].flatten(), lower_bounds[idx].flatten(), alpha=0.3, color=list_of_colors_per_season[idx]) for idx in [0,1,3]]
    ax3.fill_betweenx(DEPTH, fall_upper.flatten(), fall_lower.flatten(), alpha=0.3, color=list_of_colors_per_season[2])

    ax3.invert_yaxis() 
    ax3.set_ylim([500,15])
    ax3.set_xlim([0,0.55])
    ax3.set_yticks([100, 200, 300, 400, 500])
    ax3.set_yticklabels([100, 200, 300, 400, 500])
    ax3.set_xlabel('L-Kurtosis W', fontname = 'Times New Roman',fontsize = 28)
    ax3.set_ylabel('Depth [m]', fontname = 'Times New Roman',fontsize = 28)
    plt.grid()
    
    
    #PLOT THE STRAIN AND BX
    ax4 = plt.subplot(2,2,4)
    ax4.text(0.0, 1.06, "d", transform=ax4.transAxes,fontsize=26, fontweight="bold", va="top")
    
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
    [ax4.errorbar(x_seasons[idx], 1e5 * rms_strain[idx], yerr= 1e5 * ci_rms_strain[idx],color = season_colors_for_scatter[idx],ms = 10,elinewidth = 3,  fmt='D', capsize=5, label=r"rms $\sigma$")
    for idx in range(len(list_of_colors_per_season))]
    # ax4.set_ylabel(r"rms $\sigma$ [s$^{-1}$]")
    # ax4.set_ylim([0.8,1.45])
    #plot strain mean
    # ax4_2 = ax4.twinx()
    [ax4.errorbar(x_seasons[idx], 1e5 * means_strain[idx], yerr= 1e5 *ci_mean_strain[idx],color = season_colors_for_scatter[idx],ms = 10, elinewidth = 3,fmt='o', capsize=5, label=r"mean $\sigma$")
    for idx in range(len(list_of_colors_per_season))]
    
    ax4.set_ylabel(r"mean,rms $\sigma$ [s$^{-1}$] $\times$ 1e-5" )
    ax4.set_xticks(x_seasons,season)
    legend_elements = [Line2D([0], [0],marker ='D',markersize= 10, lw=1,color = 'k', label=r"rms $\sigma$ "),
                       Line2D([0], [0], marker='o',lw=1,markersize= 10, color = 'k' ,label=r"mean $\sigma$")]
    # ax4_2.set_ylim([0.6,1.1])
    ax4.legend(handles=legend_elements, loc='lower left')
    plt.grid(axis = 'both')
    ax4.xaxis.grid(True)
    
    # ax4_2 = ax4.twinx()
    # #plot bx rms 
    # [ax4_2.errorbar(x_seasons[idx], 0.3 * rms_bx[idx], yerr= ci_rms_bx[idx],color = list_of_colors_per_season[idx], fmt='D', capsize=5, label=r"rms $\sigma$")
    #  for idx in range(len(list_of_colors_per_season))]
    # #plot strain mean
    # [ax4_2.errorbar(x_seasons[idx], means_bx[idx], yerr= ci_mean_bx[idx],color = list_of_colors_per_season[idx], fmt='o', capsize=5, label=r"mean $\sigma$")
    # for idx in range(len(list_of_colors_per_season))]
    # ax4_2.set_ylabel('bx')
    
    
    
    
    

    plt.savefig(dir_out)
    
    return None
    
######## PLOT FIGURE WITH THE 3-PANEL AND THE FOURTH PANEL AS SOMETHING THAT GOES IN SI ############################################################################################################
######## PLOT FIGURE WITH THE 3-PANEL AND THE FOURTH PANEL AS SOMETHING THAT GOES IN SI ############################################################################################################


######## PLOT FIGURE WITH THE 3-PANEL AND THE FOURTH PANEL AS SOMETHING THAT GOES IN SI ############################################################################################################
######## PLOT FIGURE WITH THE 3-PANEL AND THE FOURTH PANEL AS SOMETHING THAT GOES IN SI ############################################################################################################

######## PLOT FIGURE WITH THE 3-PANEL AND THE FOURTH PANEL AS SOMETHING THAT GOES IN SI ############################################################################################################
######## PLOT FIGURE WITH THE 3-PANEL AND THE FOURTH PANEL AS SOMETHING THAT GOES IN SI ############################################################################################################


dir_out = f'/Users/kat/Desktop/Siegelman_Lab/figures/Figure_3/fig3_1panel_kurtosis_vertical_velocity.png'

def plot_figure3_3_panel(summer_dataarray,winter_dataarray,fall_dataarray,spring_dataarray,dir_out = dir_out): 
    '''  This is the main funciton of this script. This function plots the statistics necessary
          for the vertical velocity data. By calling the other helper functions here. 
    
    '''
    # plt.figure(figsize = (30,10))
    # plt.rcParams["font.size"] = 24
    
    # ### PLOT THE DISTRIBUTION 
    # ax = plt.subplot(1,3,1)
    # ax.text(0.0, 1.06, "a", transform=ax.transAxes,fontsize=26, fontweight="bold", va="top")
    # ax.hist(summer_dataarray.values.ravel(),bins=100, density=True, histtype='step', linewidth=3,label='Summer [DJF]', color = SUMMER_COLOR)
    # ax.hist(winter_dataarray.values.ravel(),bins=100, density=True, histtype='step', linewidth=3,label='Winter [JJA]', color = WINTER_COLOR)
    # ax.hist(spring_dataarray.values.ravel(),bins=100, density=True, histtype='step', linewidth=3,label='Spring [SON]',color = SPRING_COLOR)
    # ax.hist(fall_dataarray.values.ravel(),bins=100, density=True, histtype='step', linewidth=3,label='Fall [MAM]',color = FALL_COLOR)
 
    # ax.set_ylabel('PDF',fontname='Times New Roman',fontsize = 28)
    # ax.set_xlabel(r"W $[m.d^{-1}]$", fontname='Times New Roman',fontsize = 28)
    # plt.legend(loc = 'upper left', fontsize = 20)
    # ax.set_yscale('log')
    # ax.set_xlim([-150,150])
    # ax.set_ylim([10**(-5), 10**0])
    # ax.set_xticks([-100,-50, 0,50,100])
    # ax.set_xticklabels([-100,-50,0,50,100])
    # plt.legend(loc = 'upper left',fontsize = 20)
    # plt.grid()
    
    # ### PLOT THE RMS AND THE CONFIDENCE INTERVAL 
    # ax2 = plt.subplot(1,3,2)
    # ax2.text(0.0, 1.06, "b", transform=ax2.transAxes,fontsize=26, fontweight="bold", va="top")
    # w_rms = [call_function_per_season(season_array)[0] for season_array in list_of_data_arrays_per_season] #order of summer, winter,fall,spring
    # w_confidence_interval = [call_function_per_season(season_array)[1] for season_array in list_of_data_arrays_per_season] #summer, winter,fall,spring
    # [plot_variable_vs_depth_and_ci(ax2,w_rms[idx],w_confidence_interval[idx],list_of_colors_per_season[idx]) for idx in range(0,4)]
    # ax2.invert_yaxis()
    # ax2.set_xlim([0,25])
    # ax2.set_ylim([500,15])
    # ax2.set_yticks([100, 200, 300, 400, 500])
    # ax2.set_yticklabels([100, 200, 300, 400, 500])
    # ax2.set_xticks([5, 10, 15, 20, 25])
    # ax2.set_xticklabels([5, 10, 15, 20, 25])
    # ax2.set_ylabel('Depth [m]',fontname='Times New Roman',fontsize = 28)
    # ax2.set_xlabel(r"rms W $[m.d^{-1}]$", fontname='Times New Roman',fontsize = 28)
    # plt.grid()
    
    
    
    # #PLOT THE STRAIN AND BX
    # ax4 = plt.subplot(1,3,3)
    # ax4.text(0.0, 1.06, "c", transform=ax4.transAxes,fontsize=26, fontweight="bold", va="top")
    
    # summer_bx = summer_dataset.bx * summer_dataset.maskprof
    # winter_bx = winter_dataset.bx * winter_dataset.maskprof
    # fall_bx = fall_dataset.bx * fall_dataset.maskprof
    # spring_bx = spring_dataset.bx * spring_dataset.maskprof
    
    # summer_strain = summer_dataset.strain_neuro 
    # winter_strain = winter_dataset.strain_neuro 
    # fall_strain = fall_dataset.strain_neuro 
    # spring_strain = spring_dataset.strain_neuro 
    
    # season_array_bx = [spring_bx,summer_bx,fall_bx,winter_bx,]
    # list_season_array_strain = [winter_strain,spring_strain,summer_strain,fall_strain,]
    # season = ['winter','spring','summer','fall']
    # season_colors_for_scatter = [WINTER_COLOR,SPRING_COLOR,SUMMER_COLOR,FALL_COLOR]
    
    # means_strain = [mean_over_all_dims(strain_array)[0] for strain_array in list_season_array_strain]
    # ci_mean_strain = [mean_over_all_dims(strain_array)[1] for strain_array in list_season_array_strain]
    # rms_strain = [rms_over_all_dims(strain_array)[0] for strain_array in list_season_array_strain]
    # ci_rms_strain = [rms_over_all_dims(strain_array)[1] for strain_array in list_season_array_strain]
    
    # means_bx = [mean_over_all_dims(strain_array)[0] for strain_array in season_array_bx]
    # ci_mean_bx = [mean_over_all_dims(strain_array)[1] for strain_array in season_array_bx]
    # rms_bx = [rms_over_all_dims(strain_array)[0] for strain_array in season_array_bx]
    # ci_rms_bx = [rms_over_all_dims(strain_array)[1] for strain_array in season_array_bx]
    
    # x_seasons = range(len(season_array_bx))
    # #plot strain rms
    # [ax4.errorbar(x_seasons[idx], 1e5 * rms_strain[idx], yerr= 1e5 * ci_rms_strain[idx],color = season_colors_for_scatter[idx],ms = 10,elinewidth = 3, fmt='D-', capsize=5, label=r"rms $\sigma$")
    # for idx in range(len(list_of_colors_per_season))]
    # ax4.plot(x_seasons, 1e5 * np.array(rms_strain), color = 'grey',alpha = 0.5) 
    # # ax4.set_ylabel(r"rms $\sigma$ [s$^{-1}$]")
    # # ax4.set_ylim([0.8,1.45])
    # #plot strain mean
    # # ax4_2 = ax4.twinx()
    # [ax4.errorbar(x_seasons[idx], 1e5 * means_strain[idx], yerr= 1e5 *ci_mean_strain[idx],color = season_colors_for_scatter[idx],ms = 10, elinewidth = 3,fmt='o-', capsize=5, label=r"mean $\sigma$")
    # for idx in range(len(list_of_colors_per_season))]
    # ax4.plot(x_seasons, 1e5 * np.array(means_strain),color = 'grey',alpha = 0.5) 
    
    # ax4.set_ylabel(r"mean,rms $\sigma$ [s$^{-1}$]" )
    # ax4.set_xticks(x_seasons,season)
    # plt.rcParams['mathtext.fontset'] = 'stix'
    # ax4.text(-0.09, 0.995, r"$\times 10^{5}$", transform=ax4.transAxes, fontsize= 20, va="top")
    # legend_elements = [Line2D([0], [0],marker ='D',markersize= 10, lw=1,color = 'k', label=r"rms $\sigma$ "),
    #                    Line2D([0], [0], marker='o',lw=1,markersize= 10, color = 'k' ,label=r"mean $\sigma$")]
    # # ax4_2.set_ylim([0.6,1.1])
    # ax4.legend(handles=legend_elements, loc='lower left')
    # plt.grid(axis = 'both')
    # ax4.xaxis.grid(True)
    # plt.savefig(dir_out)
    
    
    
    
    #PLOT THE L_KURTOSIS 
    # ax3 = plt.subplot(2,2,3)
    plt.figure(figsize = (10,10))
    ax3 = plt.subplot()
    ax3.text(0.0, 1.06, "a", transform=ax3.transAxes,fontsize=26, fontweight="bold", va="top")
    L_kurtosis = [np.apply_along_axis(calculate_kurtosis, axis=1, arr=np.asarray(datarray)) for datarray in list_of_data_arrays_per_season]
    [plt.plot(L_kurtosis[idx],DEPTH,color = list_of_colors_per_season[idx],linewidth = 3) for idx in range(0,4)]
    summer_kurt_ci = calculate_ci_kurtosis_for_season(summer_dataset)
    winter_kurt_ci = calculate_ci_kurtosis_for_season(winter_dataset)
    fall_kurt_ci = calculate_ci_kurtosis_for_season(fall_dataset)
    spring_kurt_ci = calculate_ci_kurtosis_for_season(spring_dataset)
    kurtosis_ci_list = [summer_kurt_ci,winter_kurt_ci,fall_kurt_ci,spring_kurt_ci]
    [plt.plot(L_kurtosis[idx],DEPTH,color = list_of_colors_per_season[idx],linewidth = 3) for idx in range(0,4)]
    # print('error starts here')
    upper_bounds = [np.array(L_kurtosis[idx] +  kurtosis_ci_list[idx].to_dataarray()) for idx in [0,1,3]]
    lower_bounds = [np.array(L_kurtosis[idx] -  kurtosis_ci_list[idx].to_dataarray()) for idx in [0,1,3]]
    fall_upper  = np.array(L_kurtosis[2] + 0.2 * kurtosis_ci_list[2].to_dataarray())#  * 0.2 
    fall_lower = np.array(L_kurtosis[2] - 0.2 * kurtosis_ci_list[2].to_dataarray())# *  0.2 
    # print('is able to compute the upper and lower bounds')
    for i, idx in enumerate([0, 1, 3]):
        ax3.fill_betweenx(DEPTH,upper_bounds[i].flatten(),lower_bounds[i].flatten(),alpha=0.3,color=list_of_colors_per_season[idx])

    # [ax3.fill_betweenx(DEPTH, upper_bounds[idx].flatten(), lower_bounds[idx].flatten(), alpha=0.3, color=list_of_colors_per_season[idx]) for idx in [0,1,3]]
    ax3.fill_betweenx(DEPTH, fall_upper.flatten(), fall_lower.flatten(), alpha=0.3, color=list_of_colors_per_season[2])

    ax3.invert_yaxis() 
    ax3.set_ylim([500,15])
    # ax3.set_xlim([0,0.55])
    ax3.set_yticks([100, 200, 300, 400, 500])
    ax3.set_yticklabels([100, 200, 300, 400, 500])
    ax3.set_xlabel('L-Kurtosis w', fontname = 'Times New Roman',fontsize = 28)
    ax3.set_ylabel('Depth [m]', fontname = 'Times New Roman',fontsize = 28)
    plt.grid()
    # dir_out = f'/Users/kat/Desktop/Siegelman_Lab/figures/Figure_3/fig3_1_panel_kurtosis.png'
    plt.savefig(dir_out)
    

    
    return None

    

    
    



plot_figure3(summer_dataarray,winter_dataarray,fall_dataarray,spring_dataarray)
# plot_figure3_3_panel(summer_dataarray,winter_dataarray,fall_dataarray,spring_dataarray)


    

