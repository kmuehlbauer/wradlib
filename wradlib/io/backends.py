#!/usr/bin/env python
# Copyright (c) 2011-2021, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

"""
wradlib backends for xarray
^^^^^^^^^^^^^^^^^^^^^^^^^^
Reading radar data into xarray Datasets using xr.open_dataset

.. autosummary::
   :nosignatures:
   :toctree: generated/

   {}
"""
__all__ = ["RadolanBackendEntrypoint", "OdimH5BackendEntrypoint"]

__doc__ = __doc__.format("\n   ".join(__all__))

import os

import datetime as dt
import numpy as np

from xarray import decode_cf, Dataset, DataArray
from xarray.core import indexing
from xarray.core.utils import Frozen, FrozenDict, close_on_error, read_magic_number
from xarray.core.variable import Variable
from xarray.backends import H5NetCDFStore
from xarray.backends.common import (
    AbstractDataStore,
    BackendArray,
    BackendEntrypoint,
)
from xarray.backends.locks import SerializableLock, ensure_lock
from xarray.backends.store import StoreBackendEntrypoint

from wradlib.georef import rect
from wradlib.io import radolan

from .xarray import GAMIC_NAMES, moment_attrs, moments_mapping, _reindex_angle

# FIXME: Add a dedicated lock
RADOLAN_LOCK = SerializableLock()


class RadolanArrayWrapper(BackendArray):
    def __init__(self, datastore, array):
        self.datastore = datastore
        self.shape = array.shape
        self.dtype = array.dtype
        self.array = array

    def __getitem__(self, key):
        return indexing.explicit_indexing_adapter(
            key, self.shape, indexing.IndexingSupport.BASIC, self._getitem
        )

    def _getitem(self, key):
        with self.datastore.lock:
            return self.array[key]


class WradlibVariable(object):
    def __init__(self, dims, data, attrs):
        self._dimensions = dims
        self._data = data
        self._attrs = attrs

    @property
    def dimensions(self):
        return self._dimensions

    @property
    def data(self):
        return self._data

    @property
    def attributes(self):
        return self._attrs


class WradlibDataset(object):
    def __init__(self, dims, vars, attrs, encoding):
        self._dimensions = dims
        self._variables = vars
        self._attrs = attrs
        self._encoding = encoding

    @property
    def dimensions(self):
        return self._dimensions

    @property
    def variables(self):
        return self._variables

    @property
    def attributes(self):
        return self._attrs

    @property
    def encoding(self):
        return self._encoding


def radolan_to_xarray(data, attrs):
    """Converts RADOLAN data to xarray Dataset

    Parameters
    ----------
    data : :func:`numpy:numpy.array`
        array of shape (number of rows, number of columns)
    attrs : dict
        dictionary of metadata information from the file header

    Returns
    -------
    dset : xarray.Dataset
        RADOLAN data and coordinates
    """
    product = attrs["producttype"]
    pattrs = radolan._get_radolan_product_attributes(attrs)
    radolan_grid_xy = rect.get_radolan_grid(attrs["nrow"], attrs["ncol"])
    radolan_grid_ll = rect.get_radolan_grid(attrs["nrow"], attrs["ncol"],
                                            wgs84=True)
    xlocs = radolan_grid_xy[0, :, 0]
    ylocs = radolan_grid_xy[:, 0, 1]
    if pattrs:
        if "nodatamask" in attrs:
            data.flat[attrs["nodatamask"]] = pattrs["_FillValue"]
        if "cluttermask" in attrs:
            data.flat[attrs["cluttermask"]] = pattrs["_FillValue"]

    time_attrs = {
        "standard_name": "time",
        "units": "seconds since 1970-01-01T00:00:00Z",
    }
    variables = []
    time = np.array(
        [attrs["datetime"].replace(tzinfo=dt.timezone.utc).timestamp()])
    time_var = WradlibVariable(("time"), data=time, attrs=time_attrs)
    variables.append(("time", time_var))

    xattrs = {'units': 'km', 'long_name': 'x coordinate of projection',
              'standard_name': 'projection_x_coordinate'}
    x_var = WradlibVariable(('x',), xlocs, xattrs)
    variables.append(("x", x_var))
    yattrs = {'units': 'm', 'long_name': 'y coordinate of projection',
              'standard_name': 'projection_y_coordinate'}
    y_var = WradlibVariable(('y',), ylocs, yattrs)
    variables.append(("y", y_var))

    lon_var = WradlibVariable(('y', 'x'), data=radolan_grid_ll[..., 0],
                       attrs={'long_name': 'longitude', 'units': 'degrees_east'})
    lat_var = WradlibVariable(('y', 'x'), data=radolan_grid_ll[..., 1],
                       attrs={'long_name': 'latitude', 'units': 'degrees_north'})
    variables.append(("lon", lon_var))
    variables.append(("lat", lat_var))

    pattrs.update({"long_name": product, "coordinates": "lat lon y x time"})
    data_var = WradlibVariable(('time', 'y', 'x'), data=data[None, ...], attrs=pattrs)
    variables.append((product, data_var))
    # data_arr = DataArray(data, coords={
    #     "y": ylocs,
    #     "x": xlocs},
    #     dims=["y", "x"])
    # dset = Dataset({product: (["y", "x"], data_arr, pattrs)},
    #                coords={
    #                    "x": (["x"], xlocs, xattrs),
    #                    "y": (["y"], ylocs, yattrs),
    #                    "lon": (["y", "x"], radolan_grid_ll[..., 0], {'long_name': 'longitude', 'units': 'degrees_east'}),
    #                    "lat": (["y", "x"], radolan_grid_ll[..., 0],
    #                            {'long_name': 'latitude', 'units': 'degrees_north'}),
    #                    "time": (["time"], time, time_attrs)
    #                })

    return WradlibDataset({}, dict(variables), {}, {})


class RadolanDataStore(AbstractDataStore):
    """
    Implements the ``xr.AbstractDataStore`` read-only API for a Radolan file.
    """

    def __init__(self, filename, lock=None, **backend_kwargs):

        if lock is None:
            lock = RADOLAN_LOCK
        self.lock = ensure_lock(lock)
        self.ds = radolan_to_xarray(*radolan.read_radolan_composite(filename,
                                                                   loaddata="backend",
                                                                   **backend_kwargs))
        for k, v in self.ds.variables.items():
            print(k, v)
            print(v.dimensions)

    def open_store_variable(self, name, var):
        if isinstance(var.data, np.ndarray):
            print("numpy")
            data = var.data
        else:
            wrapped_array = RadolanArrayWrapper(self, var.data)
            data = indexing.LazilyOuterIndexedArray(wrapped_array)
            print("other")

        encoding = self.ds.encoding.copy()
        encoding["original_shape"] = var.data.shape

        return Variable(var.dimensions, data, var.attributes, encoding)

    def get_variables(self):
        return FrozenDict(
            (k, self.open_store_variable(k, v)) for k, v in self.ds.variables.items()
        )

    def get_attrs(self):
        return Frozen(self.ds.attributes)

    def get_dimensions(self):
        return Frozen(self.ds.dimensions)

    def get_encoding(self):
        dims = self.get_dimensions()
        encoding = {"unlimited_dims": {k for k, v in dims.items() if v is None}}
        return encoding

    # def get_variables(self):
    #     variables = []
    #     product = self.attrs["producttype"]
    #     pattrs = radolan._get_radolan_product_attributes(self.attrs)
    #     radolan_grid_xy = rect.get_radolan_grid(self.attrs["nrow"], self.attrs["ncol"])
    #     radolan_grid_ll = rect.get_radolan_grid(self.attrs["nrow"], self.attrs["ncol"],
    #                                             wgs84=True)
    #     xlocs = radolan_grid_xy[0, :, 0]
    #     ylocs = radolan_grid_xy[:, 0, 1]
    #     if self.data is not None:
    #         if pattrs:
    #             if "nodatamask" in self.attrs:
    #                 self.data.flat[self.attrs["nodatamask"]] = pattrs["_FillValue"]
    #             if "cluttermask" in self.attrs:
    #                 self.data.flat[self.attrs["cluttermask"]] = pattrs["_FillValue"]
    #
    #     time_attrs = {
    #         "standard_name": "time",
    #         "units": "seconds since 1970-01-01T00:00:00Z",
    #     }
    #     time = np.array([self.attrs["datetime"].replace(tzinfo=dt.timezone.utc).timestamp()])
    #     time_var = Variable(("time"), data=time, attrs=time_attrs)
    #     variables.append(("time", time_var))
    #
    #     xattrs = {'units': 'km', 'long_name': 'x coordinate of projection',
    #              'standard_name': 'projection_x_coordinate'}
    #     x_var = Variable(('x',), xlocs, xattrs)
    #     variables.append(("x", x_var))
    #     yattrs = {'units': 'm', 'long_name': 'y coordinate of projection',
    #              'standard_name': 'projection_y_coordinate'}
    #     y_var = Variable(('y',), ylocs, yattrs)
    #     variables.append(("y", y_var))
    #
    #     lon_var = Variable(('y', 'x'), data=radolan_grid_ll[..., 0],
    #                        attrs={'long_name': 'longitude', 'units': 'degrees_east'})
    #     lat_var = Variable(('y', 'x'), data=radolan_grid_ll[..., 1],
    #                        attrs={'long_name': 'latitude', 'units': 'degrees_north'})
    #     variables.append(("lon", lon_var))
    #     variables.append(("lat", lat_var))
    #
    #     pattrs.update({"long_name": product, "coordinates": "lat lon y x time"})
    #     data_var = Variable(('time', 'y', 'x'), data=self.data[None, ...], attrs=pattrs)
    #     variables.append((product, data_var))
    #
    #     return FrozenDict(variables)
    #
    # def get_attrs(self):
    #     return Frozen(self.attrs)


class RadolanBackendEntrypoint(BackendEntrypoint):
    def open_dataset(
        self,
        filename_or_obj,
        *,
        mask_and_scale=True,
        decode_times=None,
        concat_characters=None,
        decode_coords=None,
        drop_variables=None,
        use_cftime=None,
        decode_timedelta=None,
        lock=None,
    ):

        store = RadolanDataStore(
            filename_or_obj,
            lock=lock,
        )
        store_entrypoint = StoreBackendEntrypoint()
        with close_on_error(store):
            ds = store_entrypoint.open_dataset(
                store,
                mask_and_scale=mask_and_scale,
                decode_times=decode_times,
                concat_characters=concat_characters,
                decode_coords=decode_coords,
                drop_variables=drop_variables,
                use_cftime=use_cftime,
                decode_timedelta=decode_timedelta,
            )
        return ds


class OdimH5BackendEntrypoint(BackendEntrypoint):
    def guess_can_open(self, store_spec):
        try:
            return read_magic_number(store_spec).startswith(b"\211HDF\r\n\032\n")
        except TypeError:
            pass

        try:
            _, ext = os.path.splitext(store_spec)
        except TypeError:
            return False

        return ext in {".h5", ".hdf5", ".mvol"}

    def open_dataset(
        self,
        filename_or_obj,
        *,
        mask_and_scale=True,
        decode_times=None,
        concat_characters=None,
        decode_coords=None,
        drop_variables=None,
        use_cftime=None,
        decode_timedelta=None,
        format=None,
        group=None,
        lock=None,
        invalid_netcdf=None,
        phony_dims=None,
    ):

        store = H5NetCDFStore.open(
            filename_or_obj,
            format=format,
            group=group,
            lock=lock,
            invalid_netcdf=invalid_netcdf,
            phony_dims=phony_dims,
        )

        store_entrypoint = StoreBackendEntrypoint()

        ds = store_entrypoint.open_dataset(
            store,
            #mask_and_scale=mask_and_scale,
            #decode_times=decode_times,
            concat_characters=concat_characters,
            #decode_coords=decode_coords,
            drop_variables=drop_variables,
            use_cftime=use_cftime,
            decode_timedelta=decode_timedelta,
        )

        header = ds["ray_header"].copy()
        ds = ds.drop_vars("ray_header", errors="ignore")
        for mom in ds.variables:
            #mom_name = mom.ncpath.split("/")[-1]
            dmom = ds[mom]
            name = dmom.moment.lower()
            try:
                name = GAMIC_NAMES[name]
            except KeyError:
                ds = ds.drop_vars(mom)
                continue

            # extract and translate attributes to cf
            attrs = {}
            dmax = np.iinfo(dmom.dtype).max
            dmin = np.iinfo(dmom.dtype).min
            minval = dmom.dyn_range_min
            maxval = dmom.dyn_range_max
            if maxval != minval:
                gain = (maxval - minval) / dmax
            else:
                gain = (dmax - dmin) / dmax
                minval = dmin
            undetect = float(dmin)
            attrs["scale_factor"] = gain
            attrs["add_offset"] = minval
            attrs["_FillValue"] = float(dmax)
            attrs["_Undetect"] = undetect

            if decode_coords:
                attrs["coordinates"] = "elevation azimuth range"

            mapping = moments_mapping[name]
            attrs.update({key: mapping[key] for key in moment_attrs})
            # assign attributes to moment
            dmom.attrs = {}
            dmom.attrs.update(attrs)
            ds = ds.rename({mom: name.upper()})

        # fix dimensions
        dims = sorted(list(ds.dims.keys()), key=lambda x: int(x[len("phony_dim_"):]))
        dim0 = ("azimuth", "elevation")
        dim1 = "range"
        ds = ds.rename({dims[0]: dim0[0], dims[1]: dim1})

        # todo: this sorts and reindexes the unsorted GAMIC dataset by azimuth
        # only if `decode_coords` is False
        # adding coord ->  sort -> reindex -> remove coord
        # if not decode_coords:  # and (self._dim0[0] == "azimuth"):
        #     ds = (
        #         ds.assign_coords({self._dim0[0]: getattr(self, self._dim0[0])})
        #             .sortby(dim0[0])
        #             #.pipe(_reindex_angle, self)
        #             .drop_vars(dim0[0])
        #     )
        if mask_and_scale | decode_coords | decode_times:
            ds = decode_cf(ds)

        return ds
