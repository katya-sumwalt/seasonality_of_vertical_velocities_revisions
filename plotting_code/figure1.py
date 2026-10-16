# Core scientific stack
import numpy as np
import pandas as pd
import xarray as xr
import scipy as sp
import gsw
import scipy.stats as st
import glob

# Plotting
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as plt_colors
import matplotlib.ticker as ticker
from matplotlib.patches import Patch
from matplotlib.ticker import ScalarFormatter
from cartopy.mpl.geoaxes import GeoAxes 
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.basemap import Basemap
import seaborn as sns
import cmocean

# Cartopy / mapping
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from geopy.distance import geodesic
import matplotlib.patches as mpatches
import os
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import cartopy.crs as ccrs
from cartopy.mpl.geoaxes import GeoAxes
from mpl_toolkits.basemap import Basemap
import matplotlib.gridspec as gspec


############### DEFINE CONSTANTS ########################################################################################################################
LON_KUERGUELEN_PENINSULA = 70.2167
LAT_KUERGUELEN_PENINSULA = -49.3500
############### LOAD IN THE DATASETS ################################################################################################################################################
ds_satellite_test = xr.open_dataset('/Users/kat/Desktop/LS_organized/my_code/untitled folder/all_seals_processed.nc')
ds_bathy = xr.open_dataset('/Users/kat/Desktop/GEBCO_14_Apr_2026_62e651901193/gebco_2025_n-18.785_s-72.897_w21.167_e154.905.nc')  
summer = xr.open_dataset('/Users/kat/Desktop/LS/siegelman-lab-ksumwalt/seal-vertical-velocities/Data/east/seal_data_with_vertical_velocity_east_summer_200m_not_removed_3.nc')
winter = xr.open_dataset('/Users/kat/Desktop/LS/siegelman-lab-ksumwalt/seal-vertical-velocities/Data/east/seal_data_with_vertical_velocity_east_winter_200m_not_removed_3.nc')
fall = xr.open_dataset('/Users/kat/Desktop/LS/siegelman-lab-ksumwalt/seal-vertical-velocities/Data/east/seal_data_with_vertical_velocity_east_fall_200m_not_removed_4.nc') 
spring = xr.open_dataset('/Users/kat/Desktop/LS/siegelman-lab-ksumwalt/seal-vertical-velocities/Data/east/seal_data_with_vertical_velocity_east_spring_200m_not_removed.nc') #5 is most clean
raw_seal_data = xr.open_dataset('/Users/kat/Desktop/LS/siegelman-lab-ksumwalt/Seals_NEUROST/Data/seal_data.nc')

# print(ds_satellite_test.strain_neuro)
# def mask_out_bad_seals(dataset):
#     '''bad seal = CT131-035BAT2-15, CT139-624-17 which are seals where the vertical velocity was not computed'''
#     print(np.unique(dataset.name.values))
#     mask = (dataset.name == 'ct131-035BAT2-15') | (dataset.name == 'ct139-624-17')
#     masked_dataset = dataset.where(~mask, drop=True)
#     print(np.unique(masked_dataset.name.values))
#     return masked_dataset

def mask_out_bad_seals(dataset):
    '''bad seal = ct131-035bat2-15, ct139-624-17 — seals where vertical velocity was not computed'''
    bad_seals = ['ct131-035BAT2-15', 'ct139-624-17']
    keep = ~dataset.name.isin(bad_seals)
    return dataset.isel({dataset.name.dims[0]: keep.values})

summer = mask_out_bad_seals(summer)
winter = mask_out_bad_seals(winter)
fall = mask_out_bad_seals(fall)
spring = mask_out_bad_seals(spring)

print(summer.distance.max())
print(winter.distance.max())
print(fall.distance.max())
print(spring.distance.max())

# print(summer)
# print(winter)
# print(fall)
# print(spring)
# ds_satellite_test = mask_out_bad_seals(ds_satellite_test)

    
############### CROP THE BATHYMETRY DATASET ################################################################################################################################################
extent = [67, 117, -56.7, -45.7]
bathy = ds_bathy.sel(lon=slice(extent[0], extent[1]), lat=slice(extent[2], extent[3]))
#define important variables
f = 2 * (2*np.pi/86400) * (np.sin(np.pi/180 * -50)) # Ω = 2*np.pi/86400, # rad = np.pi/180 * -50
strain = ds_satellite_test['strain'].mean(dim = 'time')

############### GENERAL PLOT SETTINGS ########################################################################################################################
############### SET THE TEXT SETTINGS FOR ALL OF THE PLOTS ########################################################################################################################
plt.rcParams['font.family'] = 'Times New Roman' #set fontsize to times new roman
plt.rcParams['mathtext.fontset'] = 'stix'
formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_powerlimits((0, 0))  # always use scientific notation
############### DEFINE COLORS FOR LEGENGS FOR ALL PLOTS ########################################################################################################################
summer_color = 'tab:blue' 
fall_color   = 'tab:orange'
winter_color = 'tab:green' 
spring_color = 'tab:red'

def plot_main_figure1():
    print('plotting main figure........')
    ############### SET UP THE GRIDSPEC FOR THE PLOT ########################################################################################################################
    widths = [4,2.2]
    heights = [6,6,6]
    full_fig = plt.figure(figsize = (10,12),constrained_layout = True)
    gridspec = gspec.GridSpec(ncols = 2,nrows = 3,width_ratios=widths,height_ratios=heights,wspace = 0.20,hspace = 0.20) #wspace is
    ############### SUBPLOT A ########################################################################################################################
    ############### SUBPLOT A: PLOT MAP OF STRAIN ########################################################################################################################
    ax0 = full_fig.add_subplot(gridspec[0,:],projection =ccrs.PlateCarree() )
    ax0.set_title('a', loc = 'left', fontsize = 15, fontweight = 'bold') 
    strain_pcolormesh = ax0.pcolormesh(strain.longitude,strain.latitude,np.abs(strain),transform=ccrs.PlateCarree(),cmap=cmocean.cm.dense,shading="auto", vmin = 0, vmax = 2e-5,alpha = 0.9) #strain
    cb = plt.colorbar(strain_pcolormesh,shrink = 1)
    cb.ax.yaxis.get_offset_text().set_visible(False)
    cb.ax.text(0.5, 1.01, r"$\times 10^{-5}$", transform=cb.ax.transAxes,
            fontsize=8, va="bottom", ha="center", fontname="Times New Roman")
    cb.set_label(label= r" $\sigma$ [s$^{-1}$]",fontsize = 15)
    cb.ax.yaxis.set_major_formatter(formatter)
    cb.update_ticks()
    ############### SUBPLOT A: PLOT MAP OF BATHYMETRRY ########################################################################################################################
    bathymetry_contours = ax0.contour(bathy.lon,bathy.lat,bathy.elevation, levels=[-1500,-1000,-500],linestyles = '-' ,colors='black', linewidths=0.5, transform=ccrs.PlateCarree()) #countour
    ############### SUBPLOT A: SCATTER PLOT TRAJECTORY OF SEALS ########################################################################################################################
    ax0.scatter(summer.lon, summer.lat, s=0.2, color= summer_color,label='Summer', transform=ccrs.PlateCarree())
    ax0.scatter(spring.lon, spring.lat, s=0.2, color=spring_color,label='Spring', transform=ccrs.PlateCarree())
    ax0.scatter(winter.lon, winter.lat, s=0.2, color=winter_color,label='Winter', transform=ccrs.PlateCarree())
    ax0.scatter(fall.lon, fall.lat, s=0.2, color= fall_color,label='Fall', transform=ccrs.PlateCarree())
    ax0.set_extent(extent, crs=ccrs.PlateCarree())
    ax0.add_feature(cfeature.COASTLINE) 
    ax0.add_feature(cfeature.LAND) 
    ax0.set_aspect(2)
    gl = ax0.gridlines(draw_labels=True, linewidth=0.3, color='gray', alpha=0.5) 
    gl.top_labels = False
    gl.right_labels = False
    sum_lbl = 'Summer [DJF]'
    wnt_lbl = 'Winter [JJA]'
    fll_lbl = 'Fall [MAM]'
    spr_lbl = 'Spring [SON]'
    legend_elements = [Patch(facecolor=summer_color, edgecolor='black', label=sum_lbl), Patch(facecolor=fall_color, edgecolor='black', label=fll_lbl),Patch(facecolor=winter_color, edgecolor='black', label=wnt_lbl),Patch(facecolor=spring_color, edgecolor='black', label=spr_lbl)]
    ############### SUBPLOT B ################################################################################################################################################################################################
    ############### SUBPLOT B: PLOT MAP OF DISTRIBUTION OF DATA + BATHYMETRY ########################################################################################################################
    ax1 = full_fig.add_subplot(gridspec[1,:],projection =ccrs.PlateCarree())
    ax1.set_title('b', loc = 'left', fontsize = 15, fontweight = 'bold') 
    lon_bins = np.linspace(extent[0], extent[1], 18)  
    lat_bins = np.linspace(extent[2], extent[3], 10) 
    bathymetry_contours = ax1.contour(bathy.lon,bathy.lat,bathy.elevation, levels=[-1500,-1000,-500],linestyles = '-' ,colors='black', linewidths=0.5, transform=ccrs.PlateCarree()) #countour
    counts, lon_edges, lat_edges = np.histogram2d(raw_seal_data.lon,raw_seal_data.lat,bins=[lon_bins, lat_bins])
    pcm = ax1.pcolormesh(lon_edges,lat_edges,counts.T,transform=ccrs.PlateCarree(),cmap=cmocean.cm.tempo,norm=plt_colors.LogNorm(vmin = 10,vmax = 10000))
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    cb2 = plt.colorbar(pcm, orientation='vertical',shrink = 1)
    cb2.set_label("# of Profiles", fontsize = 15)
    ax1.add_feature(cfeature.COASTLINE)
    ax1.add_feature(cfeature.LAND)
    ax1.set_aspect(2)
    gl1 = ax1.gridlines(draw_labels=True,linewidth=0.3, color='gray', alpha=0.9) #gl = ax.gridlines(draw_labels=True, linewidth=0.3, color='gray', alpha=0.5)
    gl1.top_labels = False
    gl1.right_labels = False
    ############### SUBPLOT D ################################################################################################################################################################################################
    ############### SUBPLOT D: PLOT VERTICAL DISTRIBUTION OF OBSERVATIONS  ########################################################################################################################
    ax2 = full_fig.add_subplot(gridspec[2,1]) 
    strain_pcolormesh_fake = ax2.pcolormesh(strain.longitude,strain.latitude,np.abs(strain),cmap=cmocean.cm.dense,shading="auto")#,vmin = 0, vmax = 0.20,alpha = 0.9) #strain
    cb2 = plt.colorbar(strain_pcolormesh_fake,shrink = 1)
    cb2.ax.set_visible(False)
    ax2.set_title('d', loc = 'left', fontsize = 15, fontweight = 'bold') 
    depth = -1 * summer.depth
    ax2.plot((summer.vertical_velocity * summer.maskprof).count(dim = 'distance'),depth, color = summer_color) #counts the number of profiles
    ax2.plot((spring.vertical_velocity * spring.maskprof).count(dim = 'distance'),depth, color = spring_color)
    ax2.plot((winter.vertical_velocity * winter.maskprof).count(dim = 'distance'),depth, color = winter_color)
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
    ############### SUBPLOT C ################################################################################################################################################################################################
    ############### SUBPLOT C: BAR CHART OF COUNTS PER MONTH  ########################################################################################################################
    ax3 = full_fig.add_subplot(gridspec[2,0])
    ax3.set_title('c', loc = 'left', fontsize = 15, fontweight = 'bold') 
    month_names = ['Dec','Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                'Jul', 'Aug', 'Sep', 'Oct', 'Nov']
    months = [12,1,2,3,4,5,6,7,8,9,10,11]
    positions = np.arange(1, 13)
    colors = []
    for month in months:
        if month in [12, 1, 2]:
            colors.append(summer_color)   # Summer
        elif month in [3, 4, 5]:
            colors.append(fall_color)    # Fall
        elif month in [6, 7, 8]:
            colors.append(winter_color)  # Winter
        else:
            colors.append(spring_color)    # Spring #SON

    winter['month'] = winter.time.dt.month
    summer['month'] = summer.time.dt.month
    fall['month'] = fall.time.dt.month
    spring['month'] = spring.time.dt.month

    #SUMMER
    dec_data_count = summer.distance.where(summer['month'] == 12).count()
    jan_data_count = summer.distance.where(summer['month'] == 1).count()
    feb_data_count = summer.distance.where(summer['month'] == 2).count()
    #FALL
    mar_data_count = fall.distance.where(fall['month'] == 3).count()
    april_data_count = fall.distance.where(fall['month'] == 4).count()
    may_data_count = fall.distance.where(fall['month'] == 5).count()

    #WINTER
    jun_data_count = winter.distance.where(winter['month'] == 6).count()
    jul_data_count = winter.distance.where(winter['month'] == 7).count()
    august_data_count = winter.distance.where(winter['month'] == 8).count()

    #SPRING
    september_data_count = spring.distance.where(spring['month'] == 9).count()
    october_data_count = spring.distance.where(spring['month'] == 10).count()
    november_data_count = spring.distance.where(spring['month'] == 11).count()

    counts_per_month = [dec_data_count,jan_data_count,feb_data_count,mar_data_count,april_data_count,may_data_count,jun_data_count,jul_data_count,august_data_count,september_data_count,october_data_count,november_data_count]

    bars = ax3.bar(positions, counts_per_month, color=colors, edgecolor='black', width=0.8)
    ax3.set_xticks(ticks=positions, labels=month_names)
    ax3.set_xlim([0, 13])
    ax3.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    ax3.yaxis.get_offset_text().set_visible(False) #turn axis off once youve seen it and add textbox in presentation 1d4
    ax3.set_xlim([0,13])
    ax3.grid()
    ax3.set_ylabel('# of Profiles',fontsize = 15)
    plt.rcParams['mathtext.fontset'] = 'stix'
    ax3.text(-0.06, 1.02, r"$\times 10^{4}$", transform=ax3.transAxes, fontsize= 8, va="top")
    plt.savefig('/Users/kat/Desktop/Siegelman_Lab/figures/Figure_1/fig1_main.png')
    print('saved main plots of figure 1 as png at /Users/kat/Desktop/Siegelman_Lab/figures/Figure_1/')
    plt.savefig('/Users/kat/Desktop/Siegelman_Lab/figures/Figure_1/fig1_main.pdf')
    print('saved main plots of figure 1 as pdf at /Users/kat/Desktop/Siegelman_Lab/figures/Figure_1/')
    return None 





############### INSET 1: PLOT OF MAP ###########################################################################################################################################################################################################################################################################
def plot_inset_map():
    print('plotting inset of map........')
    fig = plt.figure(figsize = (4,4))
    ax = plt.subplot(projection=ccrs.PlateCarree())
    ax.set_extent([-180,180,-90,0],crs= ccrs.PlateCarree())
    ax.coastlines()
    ax.gridlines(draw_labels=True, linewidth=0.3, color='gray', alpha=0.5)
    # plt.show()
    # sub_ax = inset_axes(ax, width= 3.5, height= 1.5, loc='center',bbox_to_anchor=(0.31, - 0.02, 1, 1), bbox_transform=ax.transAxes, axes_class=GeoAxes, axes_kwargs=dict(projection=ccrs.PlateCarree()))
    # sub_ax.set_extent([0,150,-90,0],crs=ccrs.PlateCarree())
    # sub_ax.coastlines()       
    # gl_inset = sub_ax.gridlines(draw_labels=True, linewidth=0.3, color='gray', alpha=0.5)
    # gl_inset.xlabel_style = {'size': 10} #adjust inset labels
    # gl_inset.ylabel_style = {'size': 10} #adjust inset labels
    # gl_inset.top_labels = False
    # gl_inset.right_labels = True
    # gl_inset.bottom_labels = False
    # gl_inset.left_labels = False
    # sub_ax.add_patch(mpatches.Rectangle(xy=[LON_KUERGUELEN_PENINSULA-10, LAT_KUERGUELEN_PENINSULA-12], width=50,height=22,facecolor='blue',alpha=0.2,transform=ccrs.PlateCarree()))
    # plt.savefig("/Users/kat/Desktop/Siegelman_Lab/figures/Figure_1/fig1_inset1.png", dpi=300, transparent=True, bbox_inches="tight")
    # print('saved inset 1 plot of figure 1 as png at /Users/kat/Desktop/Siegelman_Lab/figures/Figure_1/')
    plt.savefig("/Users/kat/Desktop/Siegelman_Lab/figures/Figure_1/fig1_inset1.pdf")
    print('saved inset 1 plot of figure 1 as pdf at /Users/kat/Desktop/Siegelman_Lab/figures/Figure_1/')
    return None 




############### INSET 2: TEXTBOX ###########################################################################################################################################################################################################################################################################
def plot_inset_textbox():
    print('plotting inset of textbox........')
    fig, ax = plt.subplots(figsize=(2, 2))
    ax.axis("off")
    total_number_of_profiles = summer.distance.count(dim = 'distance') + winter.distance.count(dim = 'distance') + fall.distance.count(dim = 'distance')+ spring.distance.count(dim = 'distance')
    # total_number_of_profiles = 50068
    pop_spring = (spring.distance.count(dim = 'distance')/total_number_of_profiles )* 100
    pop_fall = (fall.distance.count(dim = 'distance')/total_number_of_profiles) * 100
    pop_winter = (winter.distance.count(dim = 'distance')/total_number_of_profiles) * 100
    pop_summer = (summer.distance.count(dim = 'distance')/total_number_of_profiles) * 100
    lines = [
        (f"Summer Profiles: {summer.distance.count(dim='distance').values:,} / {total_number_of_profiles.values:,} - {round(float(pop_summer.values), 1)} % of total profiles", 'tab:blue'),
        (f"Winter Profiles: {winter.distance.count(dim='distance').values:,} / {total_number_of_profiles.values:,} - {round(float(pop_winter.values),1 )} % of total profiles", 'tab:green'),
        (f"Spring Profiles: {spring.distance.count(dim='distance').values:,} / {total_number_of_profiles.values:,} - {round(float(pop_spring.values), 1)} % of total profiles", 'tab:red'),
        (f"Fall Profiles: {fall.distance.count(dim='distance').values:,} / {total_number_of_profiles.values:,} - {round(float(pop_fall.values), 1)} % of total profiles", 'tab:orange'),]

    y_start = 0.75   # top position
    y_step  = 0.15   # spacing between lines

    for i, (text, color) in enumerate(lines):
        ax.text(
            0.5, y_start - i * y_step,
            text,
            ha="center", va="center",
            fontsize=12,
            color=color
            )
    plt.savefig("/Users/kat/Desktop/Siegelman_Lab/figures/Figure_1/fig1_inset2.png", dpi=300, transparent=False, bbox_inches="tight")
    print('saved inset 2 plot of figure 1 as png at /Users/kat/Desktop/Siegelman_Lab/figures/Figure_1/')
    plt.savefig("/Users/kat/Desktop/Siegelman_Lab/figures/Figure_1/fig1_inset2.pdf", dpi=300, transparent=False, bbox_inches="tight")
    print('saved inset 2 plot of figure 1 as pdf at /Users/kat/Desktop/Siegelman_Lab/figures/Figure_1/')

    return None


def plot_figure1(): 
    plot_main_figure1() 
    # plot_inset_map() 
    plot_inset_textbox()
    print('done generating all plots for figure 1')
    return None 
    

plot_main_figure1()