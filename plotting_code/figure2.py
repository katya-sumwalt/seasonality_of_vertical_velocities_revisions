'''  This script plots the 5 panel figure containing  CT',Bx, N2,W, VHT in a 5 panel plot
'''


# Core scientific stack
import numpy as np
import pandas as pd
import xarray as xr
import scipy as sp
import gsw

# Plotting
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.basemap import Basemap
import seaborn as sns
import cmocean

# Cartopy / mapping
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from geopy.distance import geodesic

# NetCDF / time
from netCDF4 import Dataset as NetCDFFile
from astropy.time import Time, TimezoneInfo
from datetime import datetime, timedelta

# Signal / statistics
from scipy.ndimage import uniform_filter1d
from scipy.signal import welch
from scipy.stats import (
    linregress, sem, t, chi2, gaussian_kde
)

# Skill metrics
import xskillscore as xs

# Dask / parallel
import dask
from dask import delayed, compute
from dask.distributed import Client, as_completed
from dask.diagnostics import ProgressBar

# Utilities
import glob
import os
import gc
from tqdm import tqdm

def calculate_vhf(ds):
    rcp = 4e6  
    w_neg = -1 * ds.vertical_velocity
    VHF_step1 = ds.temperature - (ds.temperature).mean(dim = 'distance') 
    VHF_step2 = (w_neg - (w_neg).mean(dim = 'distance'))* rcp
    VHF = VHF_step1.values * VHF_step2.values ## the issue is in broadcasting step1 and step2 together
    vhf_mean = (np.mean((VHF * ds.maskprof), axis=1))
    return VHF, vhf_mean

vht = lambda ds:calculate_vhf(ds) 

plt.rcParams.update({
    "text.usetex": False,          # enable full LaTeX rendering
    "font.family": "serif",       # use a serif font
    "font.size": 22
})

def plot_figure_2(ds_tag:xr.Dataset,time_slice: tuple,name: str,dir_out:str): 
    ''' Plots the variables strain, CT', bx, N2,W, VHT in a five panel plot. 
        Input Variable: dataset_of_seal_tag, time_slice, name, out_directory
        Output: None
    '''
    sub = ds_tag.sel(time=time_slice) #gets a subset of the data according to the time
    depth = sub.depth #gets the depth variable
    time_ax = np.arange(sub.time.size) #
    t0 = str(sub.time.values[0])[:10]
    t1 = str(sub.time.values[-1])[:10]
    
    #apply the masks to the relevant variables 
    strain = sub.strain_neuro
    CT = (sub.temperature - sub.temperature.mean("time")) * sub.maskprof
    bx = sub.bx * sub.maskprof
    N2 = sub.N2 * sub.maskprof
    W = sub.vertical_velocity * sub.maskprof
    VHT = sub.vht * sub.maskprof
    print(np.max(-1 * W *86400))
    
    
    fig,axes = plt.subplots(6,1,figsize = (30,40), sharex=True,gridspec_kw={'height_ratios':[4,4,4,4,4,4]})
    plt.rcParams["font.size"] = 24
    pad = 0.2
    
    ### PLOT STAIN
    ax = axes[0]
    ax.text(0.0, 1.1, "a", transform=ax.transAxes,
                    fontsize=26, fontweight="bold", va="top")
    # Temporary image + colorbar to fix horizontal extent
    tmp_img = ax.imshow(np.zeros((10, len(time_ax))), origin='lower', aspect='auto')
    ax.set_xlim(tmp_img.get_extent()[0], tmp_img.get_extent()[1])
    divider = make_axes_locatable(ax)
    cax_tmp = divider.append_axes('right', size='1%', pad=pad)
    tmp_cbar = plt.colorbar(tmp_img, cax=cax_tmp)
    tmp_cbar.remove()  # delete temporary colorbar
    tmp_img.remove()   # delete temporary image
    ax.plot(time_ax, sub.strain_neuro * 1e5, color='black',lw=2, label = r'$\sigma$ $\times 1e5$ [s$^{-1}$]')#$\times 5.10^3$ [s$^{-1}$]')
    ax.set_title(f"{name} | {t0} → {t1}")
    ax.tick_params(axis='x', labelbottom=False)
    ax.grid(True, lw=0.5)
    # print(np.max(sub.strain_neuro))
    ax.set_ylim(-0.001, 2.83)  # change to whatever range you want
    # ax.legend(loc = 'upper right')
    ax.set_ylabel(r'$\sigma$ [s$^{-1}$]')
    ax.text(-0.049, 1.0, r" $\times$10$^{-5}$", transform=ax.transAxes,fontsize=20, va="top")

    
    ### PLOT CT
    ax = axes[1]
    ax.text(0.0, 1.1, "b", transform=ax.transAxes,fontsize=26, fontweight="bold", va="top")
    img = ax.imshow(CT, origin='lower', aspect='auto',cmap=cmocean.cm.curl, vmin=-1.5, vmax=1.5)
    ax.plot(sub.MLD - 15, lw=2, color='k')
    ax.set_ylabel('Depth [m]')
    ax.invert_yaxis()
    ax.set_yticks([ 85, 185, 285, 385, 485])
    ax.set_yticklabels([ 100, 200, 300, 400, 500])
    ax.tick_params(axis='x', labelbottom=False)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='1%', pad=pad)
    fig.colorbar(img, cax=cax, label="CT′ [°C]")
    
    ### PLOT BX 
    ax = axes[2]
    ax.text(0.0, 1.1, "c", transform=ax.transAxes,
                    fontsize=26, fontweight="bold", va="top")
    img = ax.imshow(bx, origin='lower', aspect='auto',
                    cmap='RdBu_r', vmin=-2.22e-7, vmax=2.22e-7)
    ax.plot(sub.MLD - 15, lw=2, color='k')
    ax.invert_yaxis()
    ax.set_ylabel("Depth [m]")
    ax.set_yticks([ 85, 185, 285, 385, 485])
    ax.set_yticklabels([ 100, 200, 300, 400, 500])
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='1%', pad=pad)
    cbar = plt.colorbar(img, cax=cax)
    cbar.set_label("bₓ [s⁻²]")
    limits = 0.4 * np.nanmax(np.abs(bx))
    # cbar.mappable.set_clim(-1 * limits, limits)
    cbar.formatter = ScalarFormatter(useMathText=True)
    cbar.formatter.set_powerlimits((-2,2))
    cbar.update_ticks()
    
    ### PLOT N2 
    ax = axes[3]
    ax.text(0.0, 1.1, "d", transform=ax.transAxes,
                    fontsize=26, fontweight="bold", va="top")
    img = ax.imshow(N2, origin='lower', aspect='auto',
                    cmap=cmocean.cm.matter, vmin = 0* 10**(-5), vmax = 5.3* 10**(-5))
    ax.plot(sub.MLD - 15, color="k", lw=2)
    ax.set_ylabel("Depth [m]")
    ax.set_ylim((15,486))
    ax.invert_yaxis()
    ax.set_yticks([ 85, 185, 285, 385, 485])
    ax.set_yticklabels([ 100, 200, 300, 400, 500])
    ax.tick_params(axis="x", labelbottom=False)
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='1%', pad=pad)
    cbar = plt.colorbar(img, cax=cax)
    cbar.set_label(r"N$^2$ [s$^{-2}$]")
    cbar.formatter = ScalarFormatter(useMathText=True)
    cbar.formatter.set_powerlimits((-2,2))
    cbar.update_ticks()
    
    
    ### PLOT W 
    ax = axes[4]
    ax.text(0.0, 1.1, "e", transform=ax.transAxes,
                    fontsize=26, fontweight="bold", va="top")
    img = ax.imshow(-1* W * 86400, origin='lower', aspect='auto',
                    cmap='RdBu_r',vmin=-100, vmax=100)
    ax.plot(sub.MLD - 15, color="k", lw=2)
    ax.set_ylabel("Depth [m]")
    ax.set_ylim((15,486))
    # ax.set_ylim((15,486))
    ax.invert_yaxis()
    ax.set_yticks([ 85, 185, 285, 385, 485])
    ax.set_yticklabels([ 100, 200, 300, 400, 500])
    ax.tick_params(axis="x", labelbottom=False)
    
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes('right', size='1%', pad=pad)
    # cbar = plt.colorbar(img, cax=cax)
    # cbar.set_label(r"W [m.d$^{-1}$]")
    # cbar.formatter = ScalarFormatter(useMathText=True)
    # cbar.formatter.set_powerlimits((-1,1))
    # cbar.update_ticks()
    
    # #### CLAUDE AI FOR FORMATING THE SCALIN FACTOR #### 
    
    class FixedOrderFormatter(ScalarFormatter):
        def __init__(self, order=1, **kwargs):
            self._order = order
            super().__init__(**kwargs)
        def _set_order_of_magnitude(self):
            self.orderOfMagnitude = self._order
        _set_orderOfMagnitude = _set_order_of_magnitude

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='1%', pad=pad)
    cbar = plt.colorbar(img, cax=cax)
    limits = 0.25 * np.nanmax(np.abs(-1* W * 86400))
    
    cbar.mappable.set_clim(-1 * 32, 32)
    cbar.set_label(r"w [m.d$^{-1}$]")
    # fmt = FixedOrderFormatter(order=1, useMathText=True)
    # fmt.set_scientific(True)        
    # cbar.formatter = fmt
    # cbar.update_ticks()
    
    
    ### PLOT VHT 
    ax = axes[5]
    ax.text(0.0, 1.1, "f", transform=ax.transAxes,
                    fontsize=26, fontweight="bold", va="top")
    img = ax.imshow(VHT, origin='lower', aspect='auto',
                    cmap='RdBu_r')
    ax.plot(sub.MLD - 15, color="k", lw=2)
    ax.set_ylabel("Depth [m]")
    ax.set_ylim((15,486))
    ax.invert_yaxis()
    ax.set_yticks([ 85, 185, 285, 385, 485])
    ax.set_yticklabels([ 100, 200, 300, 400, 500])
    ax.tick_params(axis="x", labelbottom=False)
    
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes('right', size='1%', pad=pad)
    # cbar = plt.colorbar(img, cax=cax)
    # cbar.set_label(r"VHT [W.m$^{2}$]")
    # cbar.formatter = ScalarFormatter(useMathText=True)
    # cbar.formatter.set_powerlimits((-2,2))
    # cbar.update_ticks()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='1%', pad=pad)
    cbar = plt.colorbar(img, cax=cax)
    cbar.set_label(r"VHT [W.m$^{2}$]")
    limits = 0.25 * np.nanmax(np.abs(VHT))
    cbar.mappable.set_clim(-1 * 443 * 10, 443 * 10)
    # fmt = FixedOrderFormatter(order=1, useMathText=True)
    # fmt.set_scientific(True)        
    # cbar.formatter = fmt
    # cbar.update_ticks()
    
    ax.tick_params(axis="x", labelbottom=True)
    ax.set_xlabel("Along-track distance [km]")
    plt.savefig(dir_out)

    return None 


#load in dataset
ds_east = xr.open_dataset('/Users/kat/Desktop/LS/siegelman-lab-ksumwalt/seal-vertical-velocities/Data/east/seal_data_with_vertical_velocity_east_without_200removed.nc') 
# ds_east = xr.open_dataset('/Users/kat/Desktop/seal_data_with_vertical_velocity_error_removed_final.nc')
# add VHT as a variable 

ds_east_vht = ds_east.assign(vht=(('depth','distance'), vht(ds_east)[0]))
#define the tag that you are interested in 
name = 'ft22-879-18'#'ft22-878-18'
# #select the subset of the data corresponding to the tag
ds_tag = ds_east_vht.where(ds_east_vht.name == name, drop = True ) 
#swap the primary dimension from distance to time to enable for use of LS function to plot
ds_tag_time = ds_tag.swap_dims({'distance': 'time'}) 
start_time = '2018-11-06'
end_time =  '2018-11-15'
# #select the time slice you are interested in 
time_slice = slice(start_time,end_time )
# #set the directory where you want to save your plot to 
dir_out = f'/Users/kat/Desktop/Siegelman_Lab/figures/Figure_2/panel_plot_{name}_{start_time}_{end_time}.pdf'
#clear figure  
plt.clf()
#call function
# plot_figure_2(ds_tag_time,time_slice,name,dir_out)



names = ['ft22-879-18','ct112-035-14','ct112-048-14','ct131-048BAT2-15','ft22-876-18','ft22-878-18','ft22-879-18'] #
start_times = ['2018-11-06','2014-10-21', '2014-12-08','2016-11-15','2018-10-25','2018-11-11','2018-11-06']
end_times =  ['2018-11-15', '2014-11-04','2014-12-29','2016-11-25','2018-10-31','2018-11-20','2018-11-15']
# #select the subset of the data corresponding to the tag

for idx in range(len(names)):
    print(names[idx])
    ds_tag = ds_east_vht.where(ds_east_vht.name == names[idx], drop = True ) 
    #swap the primary dimension from distance to time to enable for use of LS function to plot
    ds_tag_time = ds_tag.swap_dims({'distance': 'time'}) 

    # #select the time slice you are interested in 
    time_slice = slice(start_times[idx],end_times[idx])
    # #set the directory where you want to save your plot to 
    dir_out = f'/Users/kat/Desktop/Siegelman_Lab/figures/Figure_2/panel_plot_{names[idx]}_{start_times[idx]}_{end_times[idx]}.png'
    dir_out2 = f'/Users/kat/Desktop/Siegelman_Lab/figures/Figure_2/panel_plot_{names[idx]}_{start_times[idx]}_{end_times[idx]}.pdf'
    #clear figure  
    # plt.clf()
    #call function
    plot_figure_2(ds_tag_time,time_slice,names[idx],dir_out)
    plot_figure_2(ds_tag_time,time_slice,names[idx],dir_out2)