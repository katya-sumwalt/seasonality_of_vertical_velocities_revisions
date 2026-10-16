## plot the timeseries 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import xarray as xr
import os
from calculate_statistics import rms_ci95_depth_t 
from calculate_statistics import rms_over_all_dims 
from calculate_statistics import mean_over_all_dims
from calculate_statistics import conditional_seasonal_mean_ci
from calculate_statistics import mean_ci95_depth_distance
from matplotlib.gridspec import GridSpec 

# summer_dataset = xr.open_dataset('/Users/kat/Desktop/LS/siegelman-lab-ksumwalt/seal-vertical-velocities/Data/east/seal_data_with_vertical_velocity_east_summer_200m_not_removed_3.nc')
# winter_dataset = xr.open_dataset('/Users/kat/Desktop/LS/siegelman-lab-ksumwalt/seal-vertical-velocities/Data/east/seal_data_with_vertical_velocity_east_winter_200m_not_removed_3.nc')
# fall_dataset = xr.open_dataset('/Users/kat/Desktop/LS/siegelman-lab-ksumwalt/seal-vertical-velocities/Data/east/seal_data_with_vertical_velocity_east_fall_200m_not_removed_4.nc') 
# spring_dataset = xr.open_dataset('/Users/kat/Desktop/LS/siegelman-lab-ksumwalt/seal-vertical-velocities/Data/east/seal_data_with_vertical_velocity_east_spring_200m_not_removed.nc') #5 is most clean 

spring_dataset = xr.open_dataset('/Users/kat/Desktop/All_of_Work_With_Lia/LS/siegelman-lab-ksumwalt/seal-vertical-velocities/Data/east/spring_subset.nc') 
winter_dataset = xr.open_dataset('/Users/kat/Desktop/All_of_Work_With_Lia/LS/siegelman-lab-ksumwalt/seal-vertical-velocities/Data/east/winter_subset.nc') 
fall_dataset = xr.open_dataset('/Users/kat/Desktop/All_of_Work_With_Lia/LS/siegelman-lab-ksumwalt/seal-vertical-velocities/Data/east/fall_subset.nc') 
summer_dataset = xr.open_dataset('/Users/kat/Desktop/All_of_Work_With_Lia/LS/siegelman-lab-ksumwalt/seal-vertical-velocities/Data/east/summer_subset.nc') 

def calculate_vhf(ds):
    rcp = 4e6  
    w_neg = -1 * ds.vertical_velocity #since the dataset already has the negative 
    VHF_step1 = ds.temperature - (ds.temperature).mean(dim = 'distance') 
    VHF_step2 = (w_neg - (w_neg).mean(dim = 'distance'))* rcp
    VHF = VHF_step1.values * VHF_step2.values ## the issue is in broadcasting step1 and step2 together
    vhf_mean = (np.mean((VHF * ds.maskprof), axis=1))
    
    return VHF, vhf_mean

# def mean_ci95_depth_distance(da, sample_dim="distance"):
#     """
#     Compute mean(depth) and 95% CI(depth) for the mean using SEM + t critical value.
#     da dims: (depth, distance)
#     """
#     mu = da.mean(dim=sample_dim, skipna=True)
#     sd = da.std(dim=sample_dim, skipna=True)
#     n  = da.count(dim=sample_dim)

#     sem = sd / np.sqrt(n)

#     df = xr.where(n > 1, n - 1, np.nan)
#     tcrit = xr.apply_ufunc(
#         lambda x: t.ppf(0.975, x),
#         df,
#         vectorize=True,
#         dask="allowed",
#         output_dtypes=[float],
#     )

#     ci95 = tcrit * sem
#     return mu, ci95

vht = lambda ds:calculate_vhf(ds)


summer_dataset = summer_dataset.assign(vht=(('depth','distance'), vht(summer_dataset)[0]))
winter_dataset = winter_dataset.assign(vht=(('depth','distance'), vht(winter_dataset)[0]))
spring_dataset = spring_dataset.assign(vht=(('depth','distance'), vht(spring_dataset)[0]))
fall_dataset = fall_dataset.assign(vht=(('depth','distance'), vht(fall_dataset)[0]))

summer_dataset = summer_dataset.swap_dims({'distance': 'time'})
winter_dataset = winter_dataset.swap_dims({'distance': 'time'})
spring_dataset = spring_dataset.swap_dims({'distance': 'time'})
fall_dataset = fall_dataset.swap_dims({'distance': 'time'})



ds_E = xr.concat([summer_dataset,winter_dataset,fall_dataset,spring_dataset], dim = 'time')
print(ds_E)
ds_E = ds_E.swap_dims({'time': 'distance'})

plt.rcParams.update({
    "text.usetex": False,          # enable full LaTeX rendering
    "font.family": "serif",       # use a serif font
    "font.size": 15
    })

# ============================================================
# 1. Prepare VHT
# ============================================================

vht = (
    ds_E['vht']
    .where(ds_E['maskprof'].notnull())
    .swap_dims({'distance': 'time'})
    .sortby('time')
)


# ============================================================
# 2. Depth-average VHT
# ============================================================

vht_15_200 = (
    vht
    .sel(depth=slice(-15, -100))
    .mean(dim='depth', skipna=True)
)

vht_200_500 = (
    vht
    .sel(depth=slice(-200, -500))
    .mean(dim='depth', skipna=True)
)

###  CONFIDENCE INTERVALS FOR DEPTH AVERAGED VHT
ci_VHT_15_200 = mean_ci95_depth_distance(vht.sel(depth=slice(-15, -100)),'depth')[1]
ci_VHT_200_500 = mean_ci95_depth_distance(vht.sel(depth=slice(-200, -500)),'depth')[1]
# print(f'confidence_interval_vht_15_200 = {ci_VHT_15_200}')
# print(f'confidence_interval_vht_15_200 shape = {np.shape(ci_VHT_15_200)}')


# ============================================================
# 3. Convert to pandas
# ============================================================

df = pd.DataFrame({
    'VHT_15_200': vht_15_200.to_pandas(),
    'VHT_200_500': vht_200_500.to_pandas(),
    'ci_VHT_15_200': ci_VHT_15_200.to_pandas(),
    'ci_VHT_200_500': ci_VHT_200_500.to_pandas()
    
})

df.index = pd.to_datetime(df.index)
df = df.sort_index()


# ============================================================
# 4. Average observations occurring on the same date
# ============================================================

daily = df.groupby(df.index.normalize()).mean()


# ============================================================
# 5. Create daily climatology by month and day
# ============================================================

daily['month_day'] = daily.index.strftime('%m-%d')

clim = (
    daily
    .groupby('month_day')[['VHT_15_200', 'VHT_200_500','ci_VHT_15_200','ci_VHT_200_500']]
    .mean()
)

# Remove February 29 so the climatology has 365 days
clim = clim.drop(index='02-29', errors='ignore')

# Assign a dummy non-leap-year index
clim.index = pd.to_datetime(
    '2001-' + clim.index,
    format='%Y-%m-%d'
)

# Reindex to retain all calendar days, including days with no data
full_calendar = pd.date_range(
    '2001-01-01',
    '2001-12-31',
    freq='D'
)

clim = clim.reindex(full_calendar)


# ============================================================
# 6. Reorder climatology so winter is first: June to May
# ============================================================

clim_jun_dec = clim.loc['2001-06-01':'2001-12-31'].copy()
clim_jan_may = clim.loc['2001-01-01':'2001-05-31'].copy()

# Assign continuous dummy dates from June 2000 to May 2001
clim_jun_dec.index = pd.date_range(
    '2000-06-01',
    '2000-12-31',
    freq='D'
)

clim_jan_may.index = pd.date_range(
    '2001-01-01',
    '2001-05-31',
    freq='D'
)

clim_plot = pd.concat([clim_jun_dec, clim_jan_may])


# ============================================================
# 7. Calculate centered 30-day running mean
# ============================================================

# Use zeros during smoothing so isolated data segments taper
# toward zero, but restore NaNs where there were no observations.
original_nan = clim_plot.isna()

clim_plot_7d = (
    clim_plot
    .fillna(0)
    .rolling(
        window=30,
        center=True,
        min_periods=1
    )
    .mean()
)

# Plot NaN wherever the original climatology had no value
clim_plot_7d = clim_plot_7d.mask(original_nan)



# ============================================================
# 8. Plot
# ============================================================



def plot_time_series_of_mean_upward_vertical_heat_transport_function(ax):

    season_colors = {
        'DJF': 'tab:blue',
        'MAM': 'tab:orange',
        'JJA': 'tab:green',
        'SON': 'tab:red'
    }

    season_labels = {
        'DJF': 'Summer',
        'MAM': 'Fall',
        'JJA': 'Winter',
        'SON': 'Spring'
    }

    season_ranges = {
        'JJA': (
            pd.Timestamp('2000-06-01'),
            pd.Timestamp('2000-09-01')
        ),
        'SON': (
            pd.Timestamp('2000-09-01'),
            pd.Timestamp('2000-12-01')
        ),
        'DJF': (
            pd.Timestamp('2000-12-01'),
            pd.Timestamp('2001-03-01')
        ),
        'MAM': (
            pd.Timestamp('2001-03-01'),
            pd.Timestamp('2001-06-01')
        )
    }

    # Seasonal shading and annotations
    for season, (start, end) in season_ranges.items():

        ax.axvspan(
            start,
            end,
            color=season_colors[season],
            alpha=0.12,
            zorder=0
        )

        midpoint = start + (end - start) / 2

        ax.text(
            midpoint,
            0.97,
            season_labels[season],
            transform=ax.get_xaxis_transform(),
            ha='center',
            va='top',
            fontsize=22,
            fontweight='bold',
            color=season_colors[season],
            alpha=0.9,
            zorder=5
        )

       

    ### VHT climatologies
    ax.plot(
        clim_plot_7d.index,
        clim_plot_7d['VHT_15_200'],
        c='grey',
        lw=3,
        label='15–100 m'
    )
    print(f'what the ci looks like {clim_plot_7d['ci_VHT_15_200']}')
    print(f'what the vht time series looks like {clim_plot_7d['VHT_15_200']}') #(1,365)
    
    ax.fill_between(clim_plot_7d.index, clim_plot_7d['VHT_15_200'] + clim_plot_7d['ci_VHT_15_200'], clim_plot_7d['VHT_15_200'] - clim_plot_7d['ci_VHT_15_200'] , alpha=0.26, color='grey')
    
    ax.plot(
        clim_plot_7d.index,
        clim_plot_7d['VHT_200_500'],
        c='k',
        lw=3,
        label='200–500 m'
    )

    ax.fill_between(clim_plot_7d.index, clim_plot_7d['VHT_200_500'] + clim_plot_7d['ci_VHT_200_500'], clim_plot_7d['VHT_200_500'] - clim_plot_7d['ci_VHT_200_500'] , alpha=0.26, color='black')


    ax.axhline(
        0,
        color='k',
        lw=0.8
    )

    ax.set_xlim(
        pd.Timestamp('2000-06-01'),
        pd.Timestamp('2001-05-31')
    )

    ax.set_ylabel(
        r'$\langle$VHT$\rangle$ [W.m$^{-2}$]'
    )

    ax.xaxis.set_major_locator(
        mdates.MonthLocator()
    )

    ax.xaxis.set_major_formatter(
        mdates.DateFormatter('%b')
    )

    ax.legend(
        frameon=False
    )

    # plt.tight_layout()
    # plt.show() 
    # plt.savefig('/Users/kat/Desktop/Siegelman_Lab/figures/Time_Series_Of_Vht_Fig/time_series_of_vht_fig.pdf')
    # plt.savefig('/Users/kat/Desktop/Siegelman_Lab/figures/Time_Series_Of_Vht_Fig/time_series_of_vht_fig.png')
    # print('plotted time series plot')
    # plt.show()
    
    return None 
# outname = os.path.join(dir_out, f"ts_VHT.png")
# # outname = os.path.join(dir_out, f"ts_VHT.pdf")
# fig.savefig(outname, dpi=300, bbox_inches="tight")

# fig, ax = plt.subplots(figsize=(11, 4)) 
# plot_time_series_of_mean_upward_vertical_heat_transport_function(ax)