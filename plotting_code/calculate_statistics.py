# Core scientific stack
import numpy as np
import pandas as pd
import xarray as xr
import scipy as sp
import gsw

# Plotting
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
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
import lmoments3 as lm
# Skill metrics
# !pip install xskillscore
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

import numpy as np
import xarray as xr
from scipy.stats import t 


def rms_ci95_depth_t(da, time_dim="distance"):
    """
    Compute RMS(depth) and 95% CI(depth) using SEM + t-distribution.
    Input:
        da(depth, time)
    Output:
        rms(depth), ci95(depth)
    """

    # Step 1: squared signal
    sq = da ** 2

    # Step 2: statistics across time
    mean_sq = sq.mean(dim=time_dim)
    std_sq  = sq.std(dim=time_dim)
    n = sq.count(dim=time_dim)

    # Step 3: RMS
    rms = np.sqrt(mean_sq)

    # Step 4: SEM of mean-square
    sem_sq = std_sq / np.sqrt(n)

    # t critical value
    tcrit = xr.apply_ufunc(lambda x: t.ppf(0.975, x-1), n)

    # 95% CI of mean-square
    ci95_sq = tcrit * sem_sq

    # Step 5: propagate CI to RMS
    ci95_rms = 0.5 * ci95_sq / rms

    return rms, ci95_rms



def mean_ci95_depth_t(da, time_dim="time"):
    """
    Compute mean(depth) and 95% CI(depth) using SEM + t-distribution.
    Input:
        da(depth, time)
    Output:
        mean(depth), ci95(depth)
    """

    # Step 1: mean and std across time
    mean_ = da.mean(dim=time_dim)
    std_  = da.std(dim=time_dim)
    n     = da.count(dim=time_dim)

    # Step 2: SEM (standard error of the mean)
    sem = std_ / np.sqrt(n)

    # Step 3: t critical value for 95% CI
    tcrit = xr.apply_ufunc(lambda x: t.ppf(0.975, x - 1), n)

    # Step 4: compute CI
    ci95 = tcrit * sem

    return mean_, ci95



def rms_over_all_dims(da):
    """
    Compute RMS(depth) and 95% CI(depth) using SEM + t-distribution.
    Input:
        da(depth, time)
    Output:
        rms(depth), ci95(depth)
    """

    # Step 1: squared signal
    sq = da ** 2

    # Step 2: statistics across time
    mean_sq = sq.mean()
    std_sq  = sq.std()
    n = sq.count()

    # Step 3: RMS
    rms = np.sqrt(mean_sq)

    # Step 4: SEM of mean-square
    sem_sq = std_sq / np.sqrt(n)

    # t critical value
    tcrit = xr.apply_ufunc(lambda x: t.ppf(0.975, x-1), n)

    # 95% CI of mean-square
    ci95_sq = tcrit * sem_sq

    # Step 5: propagate CI to RMS
    ci95_rms = 0.5 * ci95_sq / rms

    return rms, ci95_rms



def mean_over_all_dims(da):
    """
    Compute mean(depth) and 95% CI(depth) using SEM + t-distribution.
    Input:
        da(depth, time)
    Output:
        mean(depth), ci95(depth)
    """

    # Step 1: mean and std across time
    mean_ = da.mean()
    print(mean_)
    std_  = da.std()
    print(std_)
    n     = da.count()

    # Step 2: SEM (standard error of the mean)
    sem = std_ / np.sqrt(n)

    # Step 3: t critical value for 95% CI
    tcrit = xr.apply_ufunc(lambda x: t.ppf(0.975, x - 1), n)

    # Step 4: compute CI
    ci95 = tcrit * sem

    return mean_, ci95


def mean_ci95_depth_distance(da, sample_dim="distance"):
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


def conditional_seasonal_mean_ci(ds, var,
                                 cond,                       # boolean mask DataArray like (depth,distance)
                                 season_coord="season",
                                 sample_dim="distance",
                                 time_coord="time"):
    """
    Conditional mean(depth) and 95% CI(depth) per season, computed over sample_dim,
    using only points where cond==True.
    """
    if time_coord not in ds.coords:
        raise KeyError(f"{time_coord} coordinate not found in ds.coords")
    if season_coord not in ds.coords:
        ds = ds.assign_coords({season_coord: ds[time_coord].dt.season})

    da = ds[var].where(cond)  # mask to conditional subset

    g = da.groupby(season_coord)

    mu = g.mean(dim=sample_dim, skipna=True)
    sd = g.std(dim=sample_dim, skipna=True)
    n  = g.count(dim=sample_dim)

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

    out = xr.Dataset({"mean": mu, "ci95": ci95, "n": n})
    return out


