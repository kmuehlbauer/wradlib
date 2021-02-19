#!/usr/bin/env python
# Copyright (c) 2011-2021, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

"""
wradlib backends for xarray
^^^^^^^^^^^^^^^^^^^^^^^^^^^
Reading radar data into xarray Datasets using ``xarray.open_dataset``
and ``xarray.open_mfdataset``

Coonvenience functions `wradlib.open_dataset`` and ``wradlib.open_mfdataset``
are available.

.. autosummary::
   :nosignatures:
   :toctree: generated/

   {}
"""
__all__ = [
    "open_dataset",
    "open_mfdataset",
    "RadolanBackendEntrypoint",
    "OdimBackendEntrypoint",
    "GamicBackendEntrypoint",
    "CfRadial1BackendEntrypoint",
    "CfRadial2BackendEntrypoint",
]

__doc__ = __doc__.format("\n   ".join(__all__))

import datetime as dt
import glob
import io
import os
from pathlib import Path

import numpy as np
import xarray
from xarray import Dataset, decode_cf, merge
from xarray.backends import NetCDF4DataStore, H5NetCDFStore
from xarray.backends.common import AbstractDataStore, BackendArray, BackendEntrypoint
from xarray.backends.locks import SerializableLock, ensure_lock
from xarray.backends.store import StoreBackendEntrypoint
from xarray.core import indexing
from xarray.core.utils import (
    Frozen,
    FrozenDict,
    close_on_error,
    read_magic_number,
)
from xarray.core.variable import Variable

from wradlib.georef import rect
from wradlib.io import radolan

from .xarray import (
    _reindex_angle,
    tqdm,
    GamicStore,
    OdimStore,
    RadarVolume,
    _assign_data_radial,
    _assign_data_radial2,
)

try:
    import h5netcdf

    has_h5netcdf = True
except ModuleNotFoundError:
    has_h5netcdf = False

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
    radolan_grid_ll = rect.get_radolan_grid(attrs["nrow"], attrs["ncol"], wgs84=True)
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
    time = np.array([attrs["datetime"].replace(tzinfo=dt.timezone.utc).timestamp()])
    time_var = WradlibVariable(("time"), data=time, attrs=time_attrs)
    variables.append(("time", time_var))

    xattrs = {
        "units": "km",
        "long_name": "x coordinate of projection",
        "standard_name": "projection_x_coordinate",
    }
    x_var = WradlibVariable(("x",), xlocs, xattrs)
    variables.append(("x", x_var))
    yattrs = {
        "units": "m",
        "long_name": "y coordinate of projection",
        "standard_name": "projection_y_coordinate",
    }
    y_var = WradlibVariable(("y",), ylocs, yattrs)
    variables.append(("y", y_var))

    lon_var = WradlibVariable(
        ("y", "x"),
        data=radolan_grid_ll[..., 0],
        attrs={"long_name": "longitude", "units": "degrees_east"},
    )
    lat_var = WradlibVariable(
        ("y", "x"),
        data=radolan_grid_ll[..., 1],
        attrs={"long_name": "latitude", "units": "degrees_north"},
    )
    variables.append(("lon", lon_var))
    variables.append(("lat", lat_var))

    pattrs.update({"long_name": product, "coordinates": "lat lon y x time"})
    data_var = WradlibVariable(("time", "y", "x"), data=data[None, ...], attrs=pattrs)
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
        self.ds = radolan_to_xarray(
            *radolan.read_radolan_composite(
                filename, loaddata="backend", **backend_kwargs
            )
        )
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


class OdimBackendEntrypoint(BackendEntrypoint):
    def guess_can_open(self, store_spec):
        try:
            return read_magic_number(store_spec).startswith(b"\211HDF\r\n\032\n")
        except TypeError:
            pass

        try:
            _, ext = os.path.splitext(store_spec)
        except TypeError:
            return False

        return ext in {".hdf5", ".h5"}

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
        phony_dims="access",
        decode_vlen_strings=True,
        keep_elevation=None,
        keep_azimuth=None,
    ):
        if isinstance(filename_or_obj, io.IOBase):
            filename_or_obj.seek(0)
        with OdimStore.open(
            filename_or_obj,
            format=format,
            group=group,
            lock=lock,
            invalid_netcdf=invalid_netcdf,
            phony_dims=phony_dims,
            decode_vlen_strings=decode_vlen_strings,
        ) as store:

            with store._manager.acquire_context(True) as root:
                groups = root[store._group].groups
                variables = [k for k in groups if "data" in k]
                vars_idx = np.argsort([int(v[len("data") :]) for v in variables])
                variables = np.array(variables)[vars_idx].tolist()
                ds_list = ["/".join([store._group, var]) for var in variables]
            store.close()

        stores = []
        for grp in ds_list:
            if isinstance(filename_or_obj, io.IOBase):
                filename_or_obj.seek(0)
            store = OdimStore.open(
                filename_or_obj,
                format=format,
                group=grp,
                lock=lock,
                invalid_netcdf=invalid_netcdf,
                phony_dims=phony_dims,
                decode_vlen_strings=decode_vlen_strings,
            )
            stores.append(store)

        store_entrypoint = StoreBackendEntrypoint()

        if keep_azimuth == True:

            def reindex_angle(ds, store):
                return ds

        else:

            def reindex_angle(ds, store):
                return _reindex_angle(ds, store)

        ds = merge(
            [
                store_entrypoint.open_dataset(
                    store,
                    mask_and_scale=mask_and_scale,
                    decode_times=decode_times,
                    concat_characters=concat_characters,
                    decode_coords=decode_coords,
                    drop_variables=drop_variables,
                    use_cftime=use_cftime,
                    decode_timedelta=decode_timedelta,
                ).pipe(reindex_angle, store=store)
                for store in stores
            ],
            combine_attrs="override",
        )

        return ds


class GamicBackendEntrypoint(BackendEntrypoint):
    def guess_can_open(self, store_spec):
        try:
            return read_magic_number(store_spec).startswith(b"\211HDF\r\n\032\n")
        except TypeError:
            pass

        try:
            _, ext = os.path.splitext(store_spec)
        except TypeError:
            return False

        return ext in {".mvol", ".h5"}

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
        group="scan0",
        lock=None,
        invalid_netcdf=None,
        phony_dims="access",
        decode_vlen_strings=True,
        keep_elevation=None,
        keep_azimuth=None,
    ):

        if isinstance(filename_or_obj, io.IOBase):
            filename_or_obj.seek(0)

        store = GamicStore.open(
            filename_or_obj,
            format=format,
            group=group,
            lock=lock,
            invalid_netcdf=invalid_netcdf,
            phony_dims=phony_dims,
            decode_vlen_strings=decode_vlen_strings,
        )

        store_entrypoint = StoreBackendEntrypoint()

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

        ds = ds.sortby(list(ds.dims.keys())[0])

        if keep_azimuth is not True:
            ds = ds.pipe(_reindex_angle)

        return ds


def get_group_names(filename, groupname):
    with h5netcdf.File(filename, "r") as fh:
        groups = [grp for grp in fh.groups if groupname in grp.lower()]
    if isinstance(filename, io.BytesIO):
        filename.seek(0)
    return groups


class CfRadial1BackendEntrypoint(BackendEntrypoint):

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
        phony_dims="access",
        decode_vlen_strings=True,
        #keep_elevation=None,
        #keep_azimuth=None,
    ):

        if isinstance(filename_or_obj, io.IOBase):
            filename_or_obj.seek(0)

        store = H5NetCDFStore.open(
            filename_or_obj,
            format=format,
            group=None,
            lock=lock,
            invalid_netcdf=invalid_netcdf,
            phony_dims=phony_dims,
            decode_vlen_strings=decode_vlen_strings,
        )

        store_entrypoint = StoreBackendEntrypoint()

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

        if group is not None:
            ds = _assign_data_radial(ds, sweep=group)[0]
            ds = ds.sortby(list(ds.dims.keys())[0])

        return ds


class CfRadial2BackendEntrypoint(BackendEntrypoint):

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
        phony_dims="access",
        decode_vlen_strings=True,
        #keep_elevation=None,
        #keep_azimuth=None,
    ):

        if isinstance(filename_or_obj, io.IOBase):
            filename_or_obj.seek(0)

        store = H5NetCDFStore.open(
            filename_or_obj,
            format=format,
            group=group,
            lock=lock,
            invalid_netcdf=invalid_netcdf,
            phony_dims=phony_dims,
            decode_vlen_strings=decode_vlen_strings,
        )

        if group is not None:
            variables = store.get_variables()
            site = Dataset(variables)
            site = {key: loc for key, loc in site.items() if key in ["longitude", "latitude", "altitude"]}

            store.close()

            if isinstance(filename_or_obj, io.IOBase):
                filename_or_obj.seek(0)

            store = H5NetCDFStore.open(
                filename_or_obj,
                format=format,
                group=group,
                lock=lock,
                invalid_netcdf=invalid_netcdf,
                phony_dims=phony_dims,
                decode_vlen_strings=decode_vlen_strings,
            )

        store_entrypoint = StoreBackendEntrypoint()

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

        if group is not None:
            ds = ds.assign_coords(site)
            ds = _assign_data_radial2(ds)
            ds = ds.sortby(list(ds.dims.keys())[0])

        return ds


def open_dataset(filename_or_obj, **kwargs):
    """Open and decode a radar sweep or volume from a file or file-like object.

    This function uses ``xarray.open_dataset`` under the hood. Please refer for
    details to the documentation of ``xarray.open_dataset``.

    Parameters
    ----------
    filename_or_obj : str, Path, file-like or DataStore
        Strings and Path objects are interpreted as a path to a local or remote
        radar file and opened with an appropriate engine.
    engine : {"odim", "gamic", "cfradial"}
        Engine to use when reading files.
    group : str, optional
        Path to a sweep group in the given file to open.
    **kwargs : optional
        Additional arguments passed on to :py:func:`xarray.open_dataset`.

    Returns
    -------
    dataset : xarray.Dataset | wradlib.Volume
        The newly created radar dataset or radar volume.

    See Also
    --------
    wradlib.open_mfdataset
    """

    engine = kwargs.pop("engine")
    group = kwargs.pop("group", "all")

    if "gamic" in engine:
        groupname = "scan"
    elif "odim" in engine:
        groupname = "dataset"
    elif "cfradial1" in engine:
        groupname = "None"
        if group == "all":
            group = None
            groupname = None
    elif "cfradial2" in engine:
        groupname = "sweep"
    else:
        raise ValueError(f"wradlib: unknown engine {engine}.")

    if group == "all" and groupname is not None:
        groups = get_group_names(filename_or_obj, groupname)
    elif isinstance(group, str) or group is None:
        groups = [group]
    else:
        pass

    ds = [
        xarray.open_dataset(filename_or_obj, engine=engine, backend_kwargs=dict(group=grp), **kwargs)
        for grp in groups
    ]

    # wrap out sweeps after reading
    if engine == "cfradial1" and groupname is None:
        ds = _assign_data_radial(ds[0], groupname)

    if len(ds) > 1:
        vol = RadarVolume()
        vol.extend(ds)
        vol.sort(key=lambda x: x.time.min().values)
        ds = vol
    else:
        ds = ds[0]

    return ds


def open_mfdataset(paths, **kwargs):
    """Open multiple radar files as a single radar sweep dataset or radar volume.

    This function uses ``xarray.open_mfdataset`` under the hood. Please refer for
    details to the documentation of ``xarray.open_mfdataset``.

    Parameters
    ----------
    paths : str or sequence
        Either a string glob in the form ``"path/to/my/files/*"`` or an explicit list of
        files to open. Paths can be given as strings or as pathlib Paths. If
        concatenation along more than one dimension is desired, then ``paths`` must be a
        nested list-of-lists (see ``xarray.combine_nested`` for details). (A string glob will
        be expanded to a 1-dimensional list.)
    chunks : int or dict, optional
        Dictionary with keys given by dimension names and values given by chunk sizes.
        In general, these should divide the dimensions of each dataset. If int, chunk
        each dimension by ``chunks``. By default, chunks will be chosen to load entire
        input files into memory at once. This has a major impact on performance: please
        see the full documentation for more details [2]_.
    concat_dim : str, or list of str, DataArray, Index or None, optional
        Dimensions to concatenate files along.  You only need to provide this argument
        if ``combine='by_coords'``, and if any of the dimensions along which you want to
        concatenate is not a dimension in the original datasets, e.g., if you want to
        stack a collection of 2D arrays along a third dimension. Set
        ``concat_dim=[..., None, ...]`` explicitly to disable concatenation along a
        particular dimension. Default is None, which for a 1D list of filepaths is
        equivalent to opening the files separately and then merging them with
        ``xarray.merge``.
    combine : {"by_coords", "nested"}, optional
        Whether ``xarray.combine_by_coords`` or ``xarray.combine_nested`` is used to
        combine all the data. Default is to use ``xarray.combine_by_coords``.
    engine : {"odim", "gamic", "cfradial"}
        Engine to use when reading files.
    **kwargs : optional
        Additional arguments passed on to :py:func:`xarray.open_mfdataset`.

    Returns
    -------
    dataset : xarray.Dataset | wradlib.Volume

    See Also
    --------
    wradlib.open_dataset
    """
    if isinstance(paths, str):
        paths = sorted(glob.glob(paths))
    else:
        paths = [
            str(p)
            if isinstance(p, Path)
            else p
            if os.path.isfile(p)
            else sorted(glob.glob(p))
            for p in paths
        ]

    patharr = np.array(paths)

    if patharr.ndim == 2 and len(patharr) == 1:
        patharr = patharr[0]

    concat_dim = kwargs.pop("concat_dim", "time")
    combine = kwargs.pop("combine", "nested")
    if concat_dim and patharr.ndim > 1:
        concat_dim = ["time"] + (patharr.ndim - 1) * [None]
    if concat_dim is None:
        combine = "by_coords"

    engine = kwargs.pop("engine")

    if "gamic" in engine:
        groupname = "scan"
    elif "odim" in engine:
        groupname = "dataset"
    else:
        raise ValueError(f"wradlib: unknown engine {engine}.")

    group = kwargs.pop("group", None)
    if group is None:
        group = get_group_names(patharr.flat[0], groupname)
    elif isinstance(group, str):
        group = [group]
    else:
        pass

    ds = [
        xarray.open_mfdataset(
            patharr.tolist(),
            engine=engine,
            group=grp,
            concat_dim=concat_dim,
            combine=combine,
            **kwargs,
        )
        for grp in tqdm(group)
    ]

    if len(ds) > 1:
        vol = RadarVolume()
        vol.extend(ds)
        vol.sort(key=lambda x: x.time.min().values)
        ds = vol
    else:
        ds = ds[0]

    return ds
