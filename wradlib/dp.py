#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Copyright (c) 2011-2020, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

"""
Dual-Pol and Differential Phase
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Overview
--------

This module provides algorithms to process polarimetric radar moments,
namely the differential phase, :math:`Phi_{{DP}}`, and, based on successful
:math:`Phi_{{DP}}` retrieval, also the specific differential phase,
:math:`K_{{DP}}`.
Please note that the actual application of polarimetric moments is implemented
in the corresponding wradlib modules, e.g.:

    - fuzzy echo classification from polarimetric moments
      (:func:`wradlib.clutter.classify_echo_fuzzy`)
    - attenuation correction (:func:`wradlib.atten.pia_from_kdp`)
    - direct precipitation retrieval from Kdp (:func:`wradlib.trafo.kdp_to_r`)

Establishing a valid :math:`Phi_{{DP}}` profile for :math:`K_{{DP}}` retrieval
involves despeckling (linear_despeckle), phase unfolding, and iterative
retrieval of :math:`Phi_{{DP}}` form :math:`K_{{DP}}`.
The main workflow and its single steps is based on a publication by
:cite:`Vulpiani2012`. For convenience, the entire workflow has been
put together in the function :func:`wradlib.dp.process_raw_phidp_vulpiani`.

Once a valid :math:`Phi_{{DP}}` profile has been established, the
`kdp_from_phidp` functions can be used to retrieve :math:`K_{{DP}}`.

Please note that so far, the functions in this module were designed to increase
performance. This was mainly achieved by allowing the simultaneous application
of functions over multiple array dimensions. The only requirement to apply
these function is that the **range dimension must be the last dimension** of
all input arrays.


.. autosummary::
   :nosignatures:
   :toctree: generated/

   {}
"""
__all__ = ['process_raw_phidp_vulpiani', 'kdp_from_phidp',
           'unfold_phi_vulpiani', 'unfold_phi', 'linear_despeckle', 'texture',
           'depolarization']
__doc__ = __doc__.format('\n   '.join(__all__))

import numpy as np
from scipy import interpolate, ndimage, stats
import xarray as xr

from wradlib import io, trafo, util


def process_raw_phidp_vulpiani(phidp, dr=None, ndespeckle=5, winlen=7,
                               niter=2, **kwargs):
    """Establish consistent :math:`Phi_{DP}` profiles from raw data.

    This approach is based on :cite:`Vulpiani2012` and involves a
    two step procedure of :math:`Phi_{DP}` reconstruction.

    Processing of raw :math:`Phi_{DP}` data contains the following steps:

        - Despeckle
        - Initial :math:`K_{DP}` estimation
        - Removal of artifacts
        - Phase unfolding
        - :math:`Phi_{DP}` reconstruction using iterative estimation
          of :math:`K_{DP}`

    Parameters
    ----------
    phidp : :class:`xarray.DataArray`
        Differential Phase
    ndespeckle : int
        ``ndespeckle`` parameter of :func:`~wradlib.dp.linear_despeckle`
    winlen : integer
        ``winlen`` parameter of :func:`~wradlib.dp.kdp_from_phidp`
    niter : int
        Number of iterations in which :math:`Phi_{DP}` is retrieved from
        :math:`K_{DP}` and vice versa

    Returns
    -------
    phidp : :class:`xarray.DataArray`
        reconstructed :math:`Phi_{DP}`
    kdp : :class:`xarray.DataArray`
        ``kdp`` estimate corresponding to ``phidp`` output

    Examples
    --------

    See :ref:`/notebooks/verification/wradlib_verify_example.ipynb`.

    """
    # numpy -> xarray
    is_numpy = False
    if not isinstance(phidp, xr.DataArray):
        is_numpy = True
        shape = phidp.shape
        phidp = xr.DataArray(phidp, name='PHIDP')
        dims = phidp.dims
        phidp = phidp.to_dataset().rename_dims({dims[-1]: 'range'}).PHIDP
        phidp = phidp.assign_coords(
            {'range': (['range'], np.arange(shape[-1]) * dr * 1000.)})

    # despeckle
    phidp_attrs = phidp.attrs
    phidp = linear_despeckle(phidp, ndespeckle=3)

    # retrieve dr gatelength in km
    dr = phidp.range.diff(dim='range')[0] / 1000.

    # kdp retrieval first guess
    kdp = kdp_from_phidp(phidp, winlen=winlen, **kwargs)

    # remove extreme values by masking-filling
    kdp = kdp.where(kdp <= 20)
    kdp = kdp.where((kdp >= -2) | (kdp <= -20))
    kdp = kdp.fillna(0)

    # unfold phidp
    phidp = unfold_phi_vulpiani(phidp, kdp)

    # clean up unfolded PhiDP
    # phidp[phidp > 360] = np.nan

    # kdp retrieval second guess
    kdp = kdp_from_phidp(phidp, winlen=winlen, **kwargs)
    # interpolate nans along 'range'
    kdp = kdp.interpolate_na(dim='range')

    # remove remaining extreme values (by masking-filling)
    kdp = kdp.where((kdp <= 20) & (kdp >= -2)).fillna(0)

    # start the actual phidp/kdp iteration
    for i in range(niter):
        # phidp from kdp through integration
        phidp = 2 * kdp.cumsum(dim='range') * dr
        # kdp from phidp by convolution
        kdp = kdp_from_phidp(phidp, winlen=winlen, **kwargs)

    # we can output a dataset here with both dataarrays combined
    phidp.attrs = phidp_attrs
    phidp.name = 'PHIDP'

    kdp_attrs = {}
    kdp_attrs.update(io.xarray.moments_mapping['KDP'])
    kdp.name = kdp_attrs.pop('short_name')
    kdp_attrs.pop('gamic')
    kdp.attrs = kdp_attrs

    if is_numpy:
        phidp = phidp.values
        kdp = kdp.values

    return phidp, kdp


def unfold_phi_vulpiani(phidp, kdp):
    """Alternative phase unfolding which completely relies on :math:`K_{DP}`.

    This unfolding should be used in oder to iteratively reconstruct
    :math:`Phi_{DP}` and :math:`K_{DP}` (see :cite:`Vulpiani2012`).

    Parameters
    ----------
    phidp : :class:`xarray.DataArray`
        Differential Phase
    kdp : :class:`xarray.DataArray`
        Specific Differential Phase
    """
    # numpy -> xarray
    is_numpy = False
    if not isinstance(phidp, xr.DataArray):
        is_numpy = True
        shape = phidp.shape
        phidp = xr.DataArray(phidp, name='PHIDP')
        dims = phidp.dims
        phidp = phidp.to_dataset().rename_dims({dims[-1]: 'range'}).PHIDP
        phidp = phidp.assign_coords(
            {'range': (['range'], np.arange(shape[-1]))})

    if not isinstance(kdp, xr.DataArray):
        is_numpy = True
        shape = kdp.shape
        kdp = xr.DataArray(kdp, name='KDP')
        dims = kdp.dims
        kdp = kdp.to_dataset().rename_dims({dims[-1]: 'range'}).KDP
        kdp = kdp.assign_coords(
            {'range': (['range'], np.arange(shape[-1]))})

    # find first location of
    amax = xr.where(kdp < -20, 1, 0).argmax(dim='range')
    dims = phidp.dims

    phidp = phidp.assign_coords(
        {'range_idx': (['range'], np.arange(phidp.range.size))})
    phidp = xr.where(phidp.range_idx >= amax, phidp + 360, phidp)
    #print(phidp)
    phidp = phidp.transpose(*dims)

    if is_numpy:
        phidp = phidp.values

    return phidp


def _fill_sweep(dat, kind="nan_to_num", fill_value=0.):
    """Fills missing data in a 1D profile.

    Parameters
    ----------
    dat : :class:`numpy:numpy.ndarray`
        array of shape (n azimuth angles, n range gates)
    kind : string
        Defines how the filling is done.
    fill_value : float
        Fill value in areas of extrapolation.

    """
    if kind == "nan_to_num":
        return np.nan_to_num(dat)

    if not np.any(np.isnan(dat)):
        return dat

    shape = dat.shape
    dat = dat.reshape((-1, shape[-1]))

    for beam in range(len(dat)):
        invalid = np.isnan(dat[beam])
        validx = np.where(~invalid)[0]
        if len(validx) < 2:
            dat[beam, invalid] = 0.
            continue
        f = interpolate.interp1d(validx, dat[beam, validx], kind=kind,
                                 bounds_error=False, fill_value=fill_value)
        invalidx = np.where(invalid)[0]
        dat[beam, invalidx] = f(invalidx)
    return dat.reshape(shape)


def kdp_from_phidp(phidp, winlen=7, dr=None, padding='constant',
                   method='lanczos',
                   pad_kwargs={},
                   rolling_kwargs={'min_periods': 1, 'center': True}):
    """Retrieves :math:`K_{DP}` from :math:`Phi_{DP}`.

    This functions uses xarray's rolling window implementation. Handling of NaN
    can be set via `rolling_kwargs`.

    In normal operation (`method='lanczos'`) the method uses
    convolution to estimate :math:`K_{DP}` (the derivative of :math:`Phi_{DP}`
    with Low-noise Lanczos differentiators. The results are very similar to the
    fallback moving window linear regression (`method='polyfit'`), but the
    former is *reasonable* faster.

    For further reading please see `Differentiation by integration using \
    orthogonal polynomials, a survey <https://arxiv.org/pdf/1102.5219>`_ \
    and `Low-noise Lanczos differentiators \
    <http://www.holoborodko.com/pavel/numerical-methods/numerical-derivative/\
lanczos-low-noise-differentiators/>`_.

    Please note that the moving window size ``winlen`` is specified as the
    number of range gates. Thus, this argument might need adjustment in case
    the range resolution changes.
    In the original publication (:cite:`Vulpiani2012`), the value ``winlen=7``
    was chosen for a range resolution of 1km.

    Parameters
    ----------
    phidp : :class:`xarray.DataArray`
        Differential Phase
    winlen : int
        Width of the window (as number of range gates)
    padding : str
        padding mode, see xarray.Dataarray.pad().
    method : str
        method of the derivation calculation, `lanczos` or `polyfit`.
    """
    assert (winlen % 2) == 1, \
        "Window size N for function kdp_from_phidp must be an odd number."

    # numpy -> xarray
    is_numpy = False
    if not isinstance(phidp, xr.DataArray):
        is_numpy = True
        shape = phidp.shape
        phidp = xr.DataArray(phidp, name='PHIDP')
        dims = phidp.dims
        phidp = phidp.to_dataset().rename_dims({dims[-1]: 'range'}).PHIDP
        phidp = phidp.assign_coords({'range': (['range'], np.arange(shape[-1]) * dr * 1000.)})

    # retrieve dr gatelength in km
    dr = phidp.range.diff(dim='range').values[0] / 1000.

    # Make really sure winlen is an integer
    winlen = int(winlen)

    # mode='mean' is nice, it pads with the mean of the pad-length at the edges
    if method == 'lanczos':
        # pad half window
        pad = winlen // 2

        pad_kwargs = {}
        if padding in ["maximum", "mean", "median", "minimum"]:
            pad_kwargs.update('stat_length',
                              pad_kwargs.get('stat_length', pad))
        phidp = phidp.pad(range=pad, mode=padding, **pad_kwargs)

    kdp = phidp.rolling({'range': winlen}, **rolling_kwargs)
    kdp = kdp.construct('window')

    # calculate weights
    if method == 'lanczos':
        weight = lanczos_differentiator(winlen)
        # actual convolution using dot-product
        kdp = kdp.dot(weight)
    elif method in ['polyfit', 'slow']:
        kdp = kdp.polyfit(dim='window', deg=1)
        kdp = kdp.sel(degree=1)
        kdp = kdp.polyfit_coefficients
    else:
        raise ValueError(f"wradlib: Unknown method {method}")

    if method == 'lanczos':
        # remove padding
        kdp = kdp.isel(range=slice(pad, -pad))

    if is_numpy:
        kdp = kdp.values

    return kdp / 2. / dr


def lanczos_differentiator(winlen):
    m = (winlen - 1) / 2
    denom = m * (m + 1.) * (2 * m + 1.)
    k = np.arange(1, m + 1)
    f = 3 * k / denom
    out = xr.DataArray(np.r_[f[::-1], [0], -f] * -1, dims=['window'])
    return out


def unfold_phi(phidp, rho, width=5, copy=False):
    """Unfolds differential phase by adjusting values that exceeded maximum \
    ambiguous range.

    Accepts arbitrarily dimensioned arrays, but THE LAST DIMENSION MUST BE
    THE RANGE.

    This is the fast Fortran-based implementation (RECOMMENDED).

    The algorithm is based on the paper of :cite:`Wang2009`.

    Parameters
    ----------
    phidp : :class:`numpy:numpy.ndarray`
        array of shape (...,nr) with nr being the number of range bins
    rho : :class:`numpy:numpy.ndarray`
        array of same shape as ``phidp``
    width : int
       Width of the analysis window
    copy : bool
       Leaves original ``phidp`` array unchanged if set to True
       (default: False)
    """
    # Check whether fast Fortran implementation is available
    speedup = util.import_optional("wradlib.speedup")

    shape = phidp.shape
    assert rho.shape == shape, "rho and phidp must have the same shape."

    phidp = phidp.reshape((-1, shape[-1]))
    if copy:
        phidp = phidp.copy()
    rho = rho.reshape((-1, shape[-1]))
    gradphi = util.gradient_from_smoothed(phidp)

    beams, rs = phidp.shape

    # Compute the standard deviation within windows of 9 range bins
    stdarr = np.zeros(phidp.shape, dtype=np.float32)
    for r in range(rs - 9):
        stdarr[..., r] = np.std(phidp[..., r:r + 9], -1)

    phidp = speedup.f_unfold_phi(phidp=phidp.astype("f4"),
                                 rho=rho.astype("f4"),
                                 gradphi=gradphi.astype("f4"),
                                 stdarr=stdarr.astype("f4"),
                                 beams=beams, rs=rs, w=width)

    return phidp.reshape(shape)


def unfold_phi_naive(phidp, rho, width=5, copy=False):
    """Unfolds differential phase by adjusting values that exceeded maximum \
    ambiguous range.

    Accepts arbitrarily dimensioned arrays, but THE LAST DIMENSION MUST BE
    THE RANGE.

    This is the slow Python-based implementation (NOT RECOMMENDED).

    The algorithm is based on the paper of :cite:`Wang2009`.

    Parameters
    ----------
    phidp : :class:`numpy:numpy.ndarray`
        array of shape (...,nr) with nr being the number of range bins
    rho : :class:`numpy:numpy.ndarray`
        array of same shape as ``phidp``
    width : int
       Width of the analysis window
    copy : bool
        Leaves original ``phidp`` array unchanged if set to True
        (default: False)

    """
    shape = phidp.shape
    assert rho.shape == shape, "rho and phidp must have the same shape."

    phidp = phidp.reshape((-1, shape[-1]))
    if copy:
        phidp = phidp.copy()
    rho = rho.reshape((-1, shape[-1]))
    gradphi = util.gradient_from_smoothed(phidp)

    beams, rs = phidp.shape

    # Compute the standard deviation within windows of 9 range bins
    stdarr = np.zeros(phidp.shape, dtype=np.float32)
    for r in range(rs - 9):
        stdarr[..., r] = np.std(phidp[..., r:r + 9], -1)

    # phi_corr = np.zeros(phidp.shape)
    for beam in range(beams):

        if np.all(phidp[beam] == 0):
            continue

        # step 1: determine location where meaningful PhiDP profile begins
        for j in range(0, rs - width):
            if (np.sum(stdarr[beam, j:j + width] < 5) == width) and \
                    (np.sum(rho[beam, j:j + 5] > 0.9) == width):
                break

        ref = np.mean(phidp[beam, j:j + width])
        for k in range(j + width, rs):
            if np.sum(stdarr[beam, k - width:k] < 5) and \
                    np.logical_and(gradphi[beam, k] > -5,
                                   gradphi[beam, k] < 20):
                ref += gradphi[beam, k] * 0.5
                if phidp[beam, k] - ref < -80:
                    if phidp[beam, k] < 0:
                        phidp[beam, k] += 360
            elif phidp[beam, k] - ref < -80:
                if phidp[beam, k] < 0:
                    phidp[beam, k] += 360
    return phidp


def linear_despeckle(data, ndespeckle=3):
    """Remove floating pixels in between NaNs in a multi-dimensional array.

    Parameters
    ----------
    data : :class:`xarray.DataArray`
        Note that the range dimension must be the last dimension of the
        input array.
    ndespeckle : int
        (must be either 3 or 5, 3 by default),
        Width of the window in which we check for speckle
    """
    assert ndespeckle in (3, 5), \
        "Window size ndespeckle for function xr_linear_despeckle must be 3 or 5."

    # numpy -> xarray
    is_numpy = False
    if not isinstance(data, xr.DataArray):
        is_numpy = True
        shape = data.shape
        data = xr.DataArray(data, name='DATA')
        dims = data.dims
        data = data.to_dataset().rename_dims({dims[-1]: 'range'}).DATA

    pad = ndespeckle // 2
    # mode='constant' pads with nan which is needed here
    arr = data.pad(range=pad, mode='constant')

    arr = arr.rolling({'range': ndespeckle}, min_periods=1, center=True)
    arr = arr.count()

    # remove padding
    arr = arr.isel(range=slice(pad, -pad))

    # return masked data
    out = data.where(arr > 1)

    if is_numpy:
        out = out.values

    return out


def texture(data):
    """Compute the texture of data.

    Compute the texture of the data by comparing values with a 3x3 neighborhood
    (based on :cite:`Gourley2007`). NaN values in the original array have
    NaN textures.

    Parameters
    ----------
    data : :class:`numpy:numpy.ndarray`
        multi-dimensional array with shape (..., number of beams, number
        of range bins)

    Returns
    ------
    texture : :class:`numpy:numpy.ndarray`
        array of textures with the same shape as data

    """
    # numpy -> xarray
    is_numpy = False
    if not isinstance(data, xr.DataArray):
        data = xr.DataArray(data)
        is_numpy = True

    # remove indices/coords
    plain = data.reset_index(list(data.indexes), drop=True)
    plain = plain.reset_coords(drop=True)

    # one-element wrap-around padding for last two axes
    padding = [(0,)] * (data.ndim - 2) + [(1,), (1,)]
    pads = {dim: pad[0] for dim, pad in zip(data.dims[:-1], padding[:-1])}
    x = plain.pad(pads, mode='wrap')
    x = x.pad({data.dims[-1]: padding[-1][0]}, mode='constant')

    # get neighbours using views into padded array
    x1 = x[..., :-2, 1:-1]  # center:2
    x2 = x[..., 1:-1, :-2]  # 4
    x3 = x[..., 2:, 1:-1]  # 8
    x4 = x[..., 1:-1, 2:]  # 6
    x5 = x[..., :-2, :-2]  # 1
    x6 = x[..., :-2, 2:]  # 3
    x7 = x[..., 2:, 2:]  # 9
    x8 = x[..., 2:, :-2]  # 7

    # concat along new dimension
    xb = xr.concat([x1, x2, x3, x4, x5, x6, x7, x8], dim='texture')

    # root mean squared calculation
    cnt = xb.count(dim='texture')
    sd = (xb - plain) ** 2
    mean = sd.sum(dim='texture') / cnt
    rmsd = np.sqrt(mean)
    rmsd = rmsd.where(~np.isnan(data))

    if is_numpy:
        rmsd = rmsd.values
    else:
        # return with indexes and coords
        rmsd = data.copy(data=rmsd.data)

    return rmsd


def depolarization(zdr, rho):
    """Compute the depolarization ration.

    Compute the depolarization ration using differential
    reflectivity :math:`Z_{DR}` and crosscorrelation coefficient
    :math:`Rho_{HV}` of a radar sweep (:cite:`Kilambi2018`,
    :cite:`Melnikov2013`, :cite:`Ryzhkov2017`).

    Parameters
    ----------
    zdr : float or :class:`numpy:numpy.ndarray`
        differential reflectivity
    rho : float or :class:`numpy:numpy.ndarray`
        crosscorrelation coefficient

    Returns
    ------
    depolarization : :class:`numpy:numpy.ndarray`
        array of depolarization ratios with the same shape as input data,
        numpy broadcasting rules apply
    """
    if not isinstance(zdr, xr.DataArray):
        zdr = np.asanyarray(zdr)
    if not isinstance(rho, xr.DataArray):
        rho = np.asanyarray(rho)
    zdr = trafo.idecibel(zdr)
    m = 2 * rho * zdr ** 0.5

    return trafo.decibel((1 + zdr - m) / (1 + zdr + m))


if __name__ == '__main__':
    print('wradlib: Calling module <dp> as main...')
