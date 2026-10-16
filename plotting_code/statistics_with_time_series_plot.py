''' This script plots a 2 x 3 figure + 1 panel plot that shows the pdf of w the rms of w the strain mean and rms plot
    The pdf VHT, the rms VHT, and the mean VHT  and the time series

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




ig = plt.figure()
gs = GridSpec(3, 3)
ax1 = plt.subplot(gs[0, 0])
ax2 = plt.subplot(gs[0,1])
ax3 = plt.subplot(gs[0, 2])
ax4 = plt.subplot(gs[1, 0])
ax5 = plt.subplot(gs[1,1])
ax6 = plt.subplot(gs[1, 2])
ax7 = plt.subplot(gs[2,:])

