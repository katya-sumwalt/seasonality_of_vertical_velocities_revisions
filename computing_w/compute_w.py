import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import gsw
from xinvert import invert_omega
from xinvert import FiniteDiff
import os

import scipy.stats as st
from matplotlib.ticker import FormatStrFormatter
from xinvert.apps import __template, __mask_FS, inv_standard2D


###ADAPTED XINVERT FUNCTION DEVELOPED BY LIAS COLLEAGUE (WRITE NAME )

### New set of 2D omega equation based on Poisson solver in xinvert (to git push in xinvert project in the future ?)
default_iParams = {
    # boundary conditions for the 2D slice
    # if 3D, should be ['fixed', 'fixed', 'fixed']
    'BCs'      : ['reflect', 'fixed'], #['reflect', 'fixed'] #############
    # undefined value in the array
    'undef'    : np.nan,
    # max loop count, exceed which the iteration is stopped
    'mxLoop'   : 5000,
    # tolerance, smaller than which the iteration is stopped
    'tolerance': 1e-8,
    # optimal argument for SOR, 1 stand for G-S iteration.
    # This argument will be automatically updated according to grids
    'optArg'   : None,
    # Whether or not print the information of the iteration
    'printInfo': True,
    # Whether or not print out debug info.
    'debug'    : False,
}

default_mParams = {
    'f0'     : 1e-5 , # Coriolis parameter at south BC on beta plane
    'beta'   : 2e-11, # meridional derivative of f
    'Phi'    : 1e4  , # background geopotential in Gill-Matsuno model
    'epsilon': 7e-6 , # linear damping coefficient in Gill-Matsuno model
    'N2'     : 2e-4 , # stratification or buoyancy frequency
    'A'      : 1e5  , # Laplacian viscosity of momentum in Munk model
    'R'      : 5e-5 , # linear drag coefficient in Stommel-Munk model
    'depth'  : 100  , # depth of the ocean or mixed layer in Stommel-Munk model
    'rho0'   : 1027 , # constant density of seawater in Stommel-Munk model
    'ang0'   : 2e5  , # background angular momentum
    'lambda' : 1e-8 , # used in Bretherton-Haidvogel model
    'c0'     : 8e-9 , # for Fofonoff model
    'c1'     : 8e-5 , # for Fofonoff model
    'Rearth' : 6371200.0, # Radius of Earth
    'Omega'  : 7.292e-5 , # angular speed of Earth's rotation
    'g'      : 9.80665  , # gravitational acceleration
}


_undeftmp = -9.99e8
def invert_omega_2D(F, dims, coords='cartesian', icbc=None,
                   mParams=default_mParams, iParams=default_iParams):

    return __template(__coeffs_omega_2D, inv_standard2D, 2, F, dims, coords,
                      icbc, ['f', 'N2'], mParams, iParams)


def __coeffs_omega_2D(force, dims, coords, mParams, iParams, icbc):
    """Calculating coefficients for QG omega equation."""
    f    = mParams['f']
    N2    = mParams['N2']
    
    maskF, initS, zero = __mask_FS(force, dims, iParams, icbc)
    
    if coords.lower() == 'cartesian': # dims[0] is z, dims[1] is x
        
        A = zero + f**2
        B = zero 
        C = zero + N2
        F = maskF.where(maskF!=_undeftmp, _undeftmp)
        
    elif coords.lower() == 'z-lat': # dims[0] is z, dims[1] is lat
        cosG = np.cos(np.deg2rad(maskF[dims[1]]))
        
        A = zero + f**2
        B = zero
        C = zero + N2
        F = (maskF*cosG).where(maskF!=_undeftmp, _undeftmp)

        
    elif coords.lower() == 'z-lon': # dims[0] is z, dims[1] is lon
        # assuming at the equator and in this case cosLat = 1.0
        # which is exactly the same as cartesian case
        A = zero + f**2
        B = zero
        C = zero + N2
        F = maskF.where(maskF!=_undeftmp, _undeftmp)
        
    else:
        raise Exception('unsupported coords ' + coords +
                        ', should be in [lat-lon, cartesian]')
    
    return F, initS, (A, B, C)


def compute_w(subsetted_seal_dataset_distance):
    iParams = {
        'BCs'      : ['reflect', 'fixed'],#['reflect', 'fixed']
        'undef'    : np.nan,
        'mxLoop'   : 2000,
        'tolerance': 1e-10,
    }

    subsetted_seal_dataset_distance['depth'] =   -1 * subsetted_seal_dataset_distance['depth'].values #flip so the sign convention is correct

    # # calculate QG forcings
    fd = FiniteDiff(
        {'Z': 'depth', 'X': 'distance'},
        BCs={'Z': 'reflect', 'X': 'fixed'}, #had as reflect, fixed
        coords='cartesian'
    )

    strain = subsetted_seal_dataset_distance['strain_neuro']
    bx = subsetted_seal_dataset_distance['bx']
    # bx_flipped = np.flip(bx,0)
    Q =  bx * strain #since the bx is the buoyancy along the trajectory and the strain is along track
    print(Q.shape) #has got all depths, and all of the time steps to create a cartesian grid (486 depth, 5270 time steps ~ 1000kms)

    divQ = -2 * fd.divg(Q, dims=['X']).load()
    FQvec = xr.where(np.isfinite(divQ), divQ, np.nan).load()


    f = subsetted_seal_dataset_distance['f'] 
    N2 = subsetted_seal_dataset_distance['N2']
    N2 = N2.where(N2<0, np.nanmin(np.abs(N2)))
    # N2_flipped = np.flip(N2,0)
    iParams = {
        'BCs'      : ['reflect', 'fixed'],#had as reflect, fixed
        'undef'    : np.nan,
        'mxLoop'   : 2000,
        'tolerance': 1e-10, #tolerance sets acceptable error and the output should have a lower tolerance output than what we put in
    }
    mParams = {'N2': N2, 'f':f}

    W_start = invert_omega_2D(FQvec, dims=['depth','distance'], coords='cartesian', iParams=iParams, mParams=mParams)
    WBC1 = xr.where(FQvec.depth == -15, 0, W_start).load() #returns values from x otherwise returns them from y
    ### try to set the initial boundary condition where the values are 
    ##the values should be 
    # WBC2 = xr.where(FQvec.depth == -500, 0, W_start).load() #returns values from x otherwise returns them from y
    W = invert_omega_2D(FQvec, dims=['depth','distance'], coords='cartesian',icbc=WBC1, iParams=iParams, mParams=mParams)
    return -1 * W