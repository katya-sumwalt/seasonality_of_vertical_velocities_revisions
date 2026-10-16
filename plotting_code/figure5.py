''' This script plots the conditional of the VHT''' 

import xarray as xr 
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import matplotlib.ticker as ticker 
from calculate_statistics import mean_ci95_depth_distance


summer_dataset = xr.open_dataset('/Users/kat/Desktop/LS/siegelman-lab-ksumwalt/seal-vertical-velocities/Data/east/seal_data_with_vertical_velocity_east_summer_200m_not_removed_3.nc')
winter_dataset = xr.open_dataset('/Users/kat/Desktop/LS/siegelman-lab-ksumwalt/seal-vertical-velocities/Data/east/seal_data_with_vertical_velocity_east_winter_200m_not_removed_3.nc')
fall_dataset = xr.open_dataset('/Users/kat/Desktop/LS/siegelman-lab-ksumwalt/seal-vertical-velocities/Data/east/seal_data_with_vertical_velocity_east_fall_200m_not_removed_4.nc') 
spring_dataset = xr.open_dataset('/Users/kat/Desktop/LS/siegelman-lab-ksumwalt/seal-vertical-velocities/Data/east/seal_data_with_vertical_velocity_east_spring_200m_not_removed.nc') #5 is most clean  

summer_dataset['w_neg'] = summer_dataset.vertical_velocity * -1
winter_dataset['w_neg'] = winter_dataset.vertical_velocity * -1
fall_dataset['w_neg'] = fall_dataset.vertical_velocity * -1
spring_dataset['w_neg'] = spring_dataset.vertical_velocity * -1

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
prime = lambda ds,var: ds[var] - ds[var].mean(dim = 'distance')

def split_pos_neg(dataset,var): 
    '''This function takes an array and splits it into a positive and negative mean''' 
    data = dataset[var] * dataset.maskprof 
    positive_data = data.where(data > 0)
    negative_data = data.where(data < 0) 
    return positive_data,negative_data  

def split_pos_neg_count(dataset,var): 
    '''This function takes an array and splits it into a positive and negative mean''' 
    data = dataset[var] * dataset.maskprof 
    positive_data = data.where(data > 0).count()
    negative_data = data.where(data < 0).count() 
    return positive_data,negative_data  

def split_pos_array(datarray): 
    '''This function takes an array and splits it into a positive and negative mean''' 
    positive_data = datarray.where(datarray > 0)
    negative_data = datarray.where(datarray < 0) 
    return positive_data,negative_data  

def split_pos_array_count(datarray): 
    '''This function takes an array and splits it into a positive and negative mean''' 
    positive_data = datarray.where(datarray > 0).count()
    negative_data = datarray.where(datarray < 0).count() 
    return positive_data,negative_data  

def calculate_conditional_mean_vht(dataset): 
    dataset_vht_pos,dataset_vht_neg = split_pos_neg(dataset,'vht')
    dataset_vht_pos_mean, dataset_vht_pos_ci95 = mean_ci95_depth_distance(dataset_vht_pos)
    dataset_vht_neg_mean, dataset_neg_pos_ci95 = mean_ci95_depth_distance(dataset_vht_neg)
    return dataset_vht_pos_mean,dataset_vht_pos_ci95, dataset_vht_neg_mean,dataset_neg_pos_ci95

def calculate_conditional_mean_datarray(array): 
    dataset_vht_pos,dataset_vht_neg = split_pos_array(array)
    dataset_vht_pos_mean, dataset_vht_pos_ci95 = mean_ci95_depth_distance(dataset_vht_pos)
    dataset_vht_neg_mean, dataset_neg_pos_ci95 = mean_ci95_depth_distance(dataset_vht_neg)
    return dataset_vht_pos_mean,dataset_vht_pos_ci95, dataset_vht_neg_mean,dataset_neg_pos_ci95

def plot_conditional_mean(ax,dataset,season_color,season_label): 
    dataset_conditional_mean_and_ci = calculate_conditional_mean_vht(dataset)
    pos_mean = dataset_conditional_mean_and_ci[0]
    neg_mean = dataset_conditional_mean_and_ci[2]
    pos_ci = dataset_conditional_mean_and_ci[1]
    neg_ci = dataset_conditional_mean_and_ci[3]
    upper_bound_pos = pos_mean + pos_ci
    lower_bound_pos = pos_mean - pos_ci
    upper_bound_neg = neg_mean + neg_ci
    lower_bound_neg = neg_mean - neg_ci
    ax.plot(dataset_conditional_mean_and_ci[0],DEPTH, color = season_color,linewidth = 3,label = season_label)
    ax.plot(dataset_conditional_mean_and_ci[2],DEPTH,linestyle = '--' ,color = season_color,linewidth = 3)
    ax.vlines(0,DEPTH.min(),DEPTH.max(), color = 'grey', linewidth = 3)
    ax.fill_betweenx(DEPTH, upper_bound_pos, lower_bound_pos, alpha=0.3, color=season_color)
    ax.fill_betweenx(DEPTH, upper_bound_neg, lower_bound_neg, alpha=0.3, color=season_color)
    return None

def plot_conditional_mean_array(ax,array,season_color,season_label): 
    dataset_conditional_mean_and_ci = calculate_conditional_mean_datarray(array)
    pos_mean = dataset_conditional_mean_and_ci[0]
    neg_mean = dataset_conditional_mean_and_ci[2]
    pos_ci = dataset_conditional_mean_and_ci[1]
    neg_ci = dataset_conditional_mean_and_ci[3]
    upper_bound_pos = pos_mean + pos_ci
    lower_bound_pos = pos_mean - pos_ci
    upper_bound_neg = neg_mean + neg_ci
    lower_bound_neg = neg_mean - neg_ci
    ax.plot(dataset_conditional_mean_and_ci[0],DEPTH, color = season_color,linewidth = 3,label = season_label)
    ax.plot(dataset_conditional_mean_and_ci[2],DEPTH,linestyle = '--' ,color = season_color,linewidth = 3)
    ax.vlines(0,DEPTH.min(),DEPTH.max(), color = 'grey', linewidth = 3)
    ax.fill_betweenx(DEPTH, upper_bound_pos, lower_bound_pos, alpha=0.3, color=season_color)
    ax.fill_betweenx(DEPTH, upper_bound_neg, lower_bound_neg, alpha=0.3, color=season_color)
    return None

    
# def plot_variable_vs_depth_and_ci(ax,rms,confidence_interval,color,depth= DEPTH,confidence_interval_bool = True): 
#     if confidence_interval_bool == True:
#         upper_bound = rms + confidence_interval
#         lower_bound = rms - confidence_interval
#         ax.fill_betweenx(DEPTH, upper_bound, lower_bound, alpha=0.3, color=color)
#         ax.plot(rms,DEPTH, color = color,linewidth = 3)
#     else: 
#         ax.plot(rms,DEPTH, color = color,linewidth = 3) 
#     return None
    

summer_dataset = summer_dataset.assign(vht=(('depth','distance'), vht(summer_dataset)[0]))
winter_dataset = winter_dataset.assign(vht=(('depth','distance'), vht(winter_dataset)[0]))
spring_dataset = spring_dataset.assign(vht=(('depth','distance'), vht(spring_dataset)[0]))
fall_dataset = fall_dataset.assign(vht=(('depth','distance'), vht(fall_dataset)[0]))

wb_prime_summer = prime(summer_dataset,'w_neg') * prime(summer_dataset,'buoyancy') * summer_dataset.maskprof
wb_prime_winter = prime(winter_dataset,'w_neg') * prime(winter_dataset,'buoyancy') * winter_dataset.maskprof
wb_prime_fall = prime(fall_dataset,'w_neg') * prime(fall_dataset,'buoyancy') * fall_dataset.maskprof
wb_prime_spring = prime(spring_dataset,'w_neg') * prime(spring_dataset,'buoyancy') * spring_dataset.maskprof

ws_prime_summer = prime(summer_dataset,'w_neg') * prime(summer_dataset,'salinity') * summer_dataset.maskprof
ws_prime_winter = prime(winter_dataset,'w_neg') * prime(winter_dataset,'salinity') * winter_dataset.maskprof
ws_prime_fall = prime(fall_dataset,'w_neg') * prime(fall_dataset,'salinity') * fall_dataset.maskprof
ws_prime_spring = prime(spring_dataset,'w_neg') * prime(spring_dataset,'salinity') * spring_dataset.maskprof

summer_dataset['wb_prime'] = wb_prime_summer
winter_dataset['wb_prime'] = wb_prime_winter
spring_dataset['wb_prime'] = wb_prime_spring
fall_dataset['wb_prime'] = wb_prime_fall


pos_summer_count_vht,neg_summer_count_vht  = split_pos_neg_count(summer_dataset,'vht')
pos_winter_count_vht,neg_winter_count_vht  = split_pos_neg_count(winter_dataset,'vht')
pos_fall_count_vht,neg_fall_count_vht  = split_pos_neg_count(fall_dataset,'vht')
pos_spring_count_vht,neg_spring_count_vht  = split_pos_neg_count(spring_dataset,'vht')

pos_summer_count_wb,neg_summer_count_wb  = split_pos_neg_count(summer_dataset,'wb_prime')
pos_winter_count_wb,neg_winter_count_wb  = split_pos_neg_count(winter_dataset,'wb_prime')
pos_fall_count_wb,neg_fall_count_wb  = split_pos_neg_count(fall_dataset,'wb_prime')
pos_spring_count_wb,neg_spring_count_wb  = split_pos_neg_count(spring_dataset,'wb_prime')



plt.figure(figsize = (30,10))
plt.rcParams['mathtext.fontset'] = 'stix'
ax = plt.subplot(131)

### PLOT VHT #####################################################################################################################################################################
ax.text(0.0, 1.06, "a", transform=ax.transAxes,fontsize=26, fontweight="bold", va="top")
plot_conditional_mean(ax,summer_dataset,SUMMER_COLOR,'DFJ')
plot_conditional_mean(ax,winter_dataset,WINTER_COLOR,'JJA')
plot_conditional_mean(ax,spring_dataset,SPRING_COLOR,'SON')
plot_conditional_mean(ax,fall_dataset,FALL_COLOR,'MAM')

ax.invert_yaxis()
ax.set_ylim([500,15])
ax.set_xlim([-740,740])
ax.set_yticks([100, 200, 300, 400, 500])
ax.set_yticklabels([100, 200, 300, 400, 500])

ax.set_xlabel(r"Conditional <VHT> [W$^{}$.m$^{-2}$]", fontname='Times New Roman',fontsize = 28)
ax.set_ylabel('Depth [m]', fontname = 'Times New Roman',fontsize = 28)

box_style = dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="black", alpha=0.8)
ax.text(0.03, 0.25, 'n+/n-', transform=ax.transAxes, va='bottom', fontsize=15,bbox=box_style, color = 'k')
ax.text(0.03, 0.20, f'{pos_summer_count_vht.values:,}/{neg_summer_count_vht.values:,}',transform=ax.transAxes, va='bottom', fontsize=15, bbox=box_style, color = SUMMER_COLOR)
ax.text(0.03, 0.15, f'{pos_winter_count_vht.values:,}/{neg_winter_count_vht.values:,}',transform=ax.transAxes, va='bottom', fontsize=15, bbox=box_style,color = WINTER_COLOR)
ax.text(0.03, 0.05, f'{pos_spring_count_vht.values:,}/{neg_spring_count_vht.values:,}',transform=ax.transAxes, va='bottom', fontsize=15, bbox=box_style,color = SPRING_COLOR)
ax.text(0.03, 0.10, f'{pos_fall_count_vht.values:,}/{neg_fall_count_vht.values:,}',transform=ax.transAxes, va='bottom', fontsize=15, bbox=box_style, color = FALL_COLOR)

plt.legend(loc = 'upper left', fontsize = 20)
plt.grid()

## add a textbox 




### PLOT CONDITIONAL W'B' #####################################################################################################################################################################

ax2 = plt.subplot(132)
ax2.text(0.0, 1.06, "b", transform=ax2.transAxes,fontsize=26, fontweight="bold", va="top")
plot_conditional_mean_array(ax2,wb_prime_summer,SUMMER_COLOR,'Summer [DFJ]')
plot_conditional_mean_array(ax2,wb_prime_winter,WINTER_COLOR,'Winter [JJA]')
plot_conditional_mean_array(ax2,wb_prime_spring,SPRING_COLOR,'Spring [SON]')
plot_conditional_mean_array(ax2,wb_prime_fall,FALL_COLOR,'Fall [MAM]')
ax2.invert_yaxis()
ax2.set_ylim([500,15])
ax2.set_xlim([-2.5e-7,2.5e-7])
ax2.set_yticks([100, 200, 300, 400, 500])
ax2.set_yticklabels([100, 200, 300, 400, 500])
ax2.set_xlabel(r"Conditional <w'b'>  [m$^{2}$.s$^{-3}$]", fontname='Times New Roman',fontsize = 28)
ax2.set_ylabel('Depth [m]', fontname = 'Times New Roman',fontsize = 28)

box_style = dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="black", alpha=0.8)
ax2.text(0.03, 0.25, 'n+/n-', transform=ax2.transAxes, va='bottom', fontsize=15,bbox=box_style, color = 'k')
ax2.text(0.03, 0.20, f'{pos_summer_count_wb.values:,}/{neg_summer_count_wb.values:,}',transform=ax2.transAxes, va='bottom', fontsize=15, bbox=box_style, color = SUMMER_COLOR)
ax2.text(0.03, 0.15, f'{pos_winter_count_wb.values:,}/{neg_winter_count_wb.values:,}',transform=ax2.transAxes, va='bottom', fontsize=15, bbox=box_style,color = WINTER_COLOR)
ax2.text(0.03, 0.05, f'{pos_spring_count_wb.values:,}/{neg_spring_count_wb.values:,}',transform=ax2.transAxes, va='bottom', fontsize=15, bbox=box_style,color = SPRING_COLOR)
ax2.text(0.03, 0.10, f'{pos_fall_count_wb.values:,}/{neg_fall_count_wb.values:,}',transform=ax2.transAxes, va='bottom', fontsize=15, bbox=box_style, color = FALL_COLOR)
ax2.set_ylabel('Depth [m]', fontname = 'Times New Roman',fontsize = 28)
ax2.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
ax2.xaxis.get_offset_text().set_visible(False)
# plt.rcParams['mathtext.fontset'] = 'stix'
ax2.text(0.9,-0.065, r"$\times 10^{-7}$", transform=ax2.transAxes, fontsize= 20, va="top")
plt.grid()




### PLOT  W'B' MEAN #####################################################################################################################################################################

ax3 = plt.subplot(133)
ax3.text(0.0, 1.06, "c", transform=ax3.transAxes,fontsize=26, fontweight="bold", va="top")
mean_summer,ci_summer = mean_ci95_depth_distance(wb_prime_summer)
plot_variable_vs_depth_and_ci(ax3,mean_summer,ci_summer,SUMMER_COLOR)

mean_winter,ci_winter = mean_ci95_depth_distance(wb_prime_winter)
plot_variable_vs_depth_and_ci(ax3,mean_winter,ci_winter,WINTER_COLOR)

mean_fall,ci_fall = mean_ci95_depth_distance(wb_prime_fall)
plot_variable_vs_depth_and_ci(ax3,mean_fall,ci_fall,FALL_COLOR)

mean_spring,ci_spring = mean_ci95_depth_distance(wb_prime_spring)
plot_variable_vs_depth_and_ci(ax3,mean_spring,ci_spring,SPRING_COLOR)

ax3.invert_yaxis()
ax3.set_ylim([500,15])
# ax3.set_xlim([-2.5e-7,2.5e-7])
ax3.set_yticks([100, 200, 300, 400, 500])
ax3.set_yticklabels([100, 200, 300, 400, 500])
ax3.set_xlabel(r"<w'b'> [m$^{2}$.s$^{-3}$]", fontname='Times New Roman',fontsize = 28)
ax3.set_ylabel('Depth [m]', fontname = 'Times New Roman',fontsize = 28)
ax3.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
ax3.xaxis.get_offset_text().set_visible(False)
# plt.rcParams['mathtext.fontset'] = 'stix'
ax3.text(0.9,-0.065, r"$\times 10^{-7}$", transform=ax3.transAxes, fontsize= 20, va="top")


    
# plt.legend()
plt.grid()
plt.savefig('/Users/kat/Desktop/Siegelman_Lab/figures/Figure_5/fig5.png')
plt.savefig('/Users/kat/Desktop/Siegelman_Lab/figures/Figure_5/fig5.pdf')


    