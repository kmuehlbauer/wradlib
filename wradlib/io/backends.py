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
__all__ = ["RadolanBackendEntrypoint", "OdimH5BackendEntrypoint",
           "GamicBackendEntrypoint"]

__doc__ = __doc__.format("\n   ".join(__all__))

import functools
import io
import os
from distutils.version import LooseVersion

import datetime as dt
import numpy as np

from xarray import decode_cf, Dataset, DataArray
from xarray.core import indexing
from xarray.core.utils import Frozen, FrozenDict, close_on_error, is_remote_uri, read_magic_number
from xarray.core.variable import Variable
from xarray.backends import H5NetCDFStore
from xarray.backends.common import (
    AbstractDataStore,
    BackendArray,
    BackendEntrypoint,
    WritableCFDataStore,
    find_root_and_group,
)
from xarray.backends.file_manager import CachingFileManager, DummyFileManager
from xarray.backends.locks import HDF5_LOCK, combine_locks, ensure_lock, get_write_lock
from xarray.backends.netCDF4_ import (
    BaseNetCDF4Array,
    _encode_nc4_variable,
    _extract_nc4_variable_encoding,
    _get_datatype,
    _nc4_require_group,
)
from xarray.backends.locks import SerializableLock, ensure_lock
from xarray.backends.store import StoreBackendEntrypoint

from wradlib.georef import rect
from wradlib.io import radolan

from .xarray import GAMIC_NAMES, moment_attrs, moments_mapping, _reindex_angle

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


class GamicArrayWrapper(BaseNetCDF4Array):
    def get_array(self, needs_lock=True):
        ds = self.datastore._acquire(needs_lock)
        variable = ds.variables[self.variable_name]
        return variable

    def __getitem__(self, key):
        return indexing.explicit_indexing_adapter(
            key, self.shape, indexing.IndexingSupport.OUTER_1VECTOR, self._getitem
        )

    def _getitem(self, key):
        # h5py requires using lists for fancy indexing:
        # https://github.com/h5py/h5py/issues/992
        key = tuple(list(k) if isinstance(k, np.ndarray) else k for k in key)
        with self.datastore.lock:
            array = self.get_array(needs_lock=False)
            return array[key]


def maybe_decode_bytes(txt):
    if isinstance(txt, bytes):
        return txt.decode("utf-8")
    else:
        return txt


def _read_attributes(h5netcdf_var):
    # GH451
    # to ensure conventions decoding works properly on Python 3, decode all
    # bytes attributes to strings
    attrs = {}
    for k, v in h5netcdf_var.attrs.items():
        if k not in ["_FillValue", "missing_value"]:
            v = maybe_decode_bytes(v)
        attrs[k] = v
    return attrs


_extract_h5nc_encoding = functools.partial(
    _extract_nc4_variable_encoding, lsd_okay=False, h5py_okay=True, backend="h5netcdf"
)


def _h5netcdf_create_group(dataset, name):
    return dataset.create_group(name)


class GamicStore(WritableCFDataStore):
    """Store for reading and writing data via h5netcdf"""

    __slots__ = (
        "autoclose",
        "format",
        "is_remote",
        "lock",
        "_filename",
        "_group",
        "_manager",
        "_mode",
    )

    def __init__(self, manager, group=None, mode=None, lock=HDF5_LOCK, autoclose=False):

        if isinstance(manager, (h5netcdf.File, h5netcdf.Group)):
            if group is None:
                root, group = find_root_and_group(manager)
            else:
                if not type(manager) is h5netcdf.File:
                    raise ValueError(
                        "must supply a h5netcdf.File if the group "
                        "argument is provided"
                    )
                root = manager
            manager = DummyFileManager(root)

        self._manager = manager
        self._group = group
        self._mode = mode
        self.format = None
        # todo: utilizing find_root_and_group seems a bit clunky
        #  making filename available on h5netcdf.Group seems better
        self._filename = find_root_and_group(self.ds)[0].filename
        self.is_remote = is_remote_uri(self._filename)
        self.lock = ensure_lock(lock)
        self.autoclose = autoclose

    @classmethod
    def open(
        cls,
        filename,
        mode="r",
        format=None,
        group=None,
        lock=None,
        autoclose=False,
        invalid_netcdf=None,
        phony_dims=None,
        decode_vlen_strings=None,
    ):

        if isinstance(filename, bytes):
            raise ValueError(
                "can't open netCDF4/HDF5 as bytes "
                "try passing a path or file-like object"
            )
        elif isinstance(filename, io.IOBase):
            magic_number = read_magic_number(filename)
            if not magic_number.startswith(b"\211HDF\r\n\032\n"):
                raise ValueError(
                    f"{magic_number} is not the signature of a valid netCDF file"
                )

        if format not in [None, "NETCDF4"]:
            raise ValueError("invalid format for h5netcdf backend")

        kwargs = {"invalid_netcdf": invalid_netcdf}
        if phony_dims is not None:
            if LooseVersion(h5netcdf.__version__) >= LooseVersion("0.8.0"):
                kwargs["phony_dims"] = phony_dims
            else:
                raise ValueError(
                    "h5netcdf backend keyword argument 'phony_dims' needs "
                    "h5netcdf >= 0.8.0."
                )
        if LooseVersion(h5netcdf.__version__) >= LooseVersion(
            "0.10.0"
        ) and LooseVersion(h5netcdf.core.h5py.__version__) >= LooseVersion("3.0.0"):
            kwargs["decode_vlen_strings"] = decode_vlen_strings

        if lock is None:
            if mode == "r":
                lock = HDF5_LOCK
            else:
                lock = combine_locks([HDF5_LOCK, get_write_lock(filename)])

        manager = CachingFileManager(h5netcdf.File, filename, mode=mode, kwargs=kwargs)
        return cls(manager, group=group, mode=mode, lock=lock, autoclose=autoclose)

    def _acquire(self, needs_lock=True):
        with self._manager.acquire_context(needs_lock) as root:
            ds = _nc4_require_group(
                root, self._group, self._mode, create_group=_h5netcdf_create_group
            )
        return ds


    @property
    def ds(self):
        return self._acquire()

    def open_store_variable(self, name, var):
        import h5py

        print("Manager:", self._manager)


        dims = var.dimensions
        dimensions = []
        for n, _ in enumerate(dims):
            if n == 0:
                dimensions.append("azimuth")
            elif n == 1:
                dimensions.append("range")
            else:
                pass
        dimensions = tuple(dimensions)
        data = indexing.LazilyOuterIndexedArray(GamicArrayWrapper(name, self))
        attrs = _read_attributes(var)

        # netCDF4 specific encoding
        encoding = {
            "chunksizes": var.chunks,
            "fletcher32": var.fletcher32,
            "shuffle": var.shuffle,
        }
        # Convert h5py-style compression options to NetCDF4-Python
        # style, if possible
        if var.compression == "gzip":
            encoding["zlib"] = True
            encoding["complevel"] = var.compression_opts
        elif var.compression is not None:
            encoding["compression"] = var.compression
            encoding["compression_opts"] = var.compression_opts

        # save source so __repr__ can detect if it's local or not
        encoding["source"] = self._filename
        encoding["original_shape"] = var.shape

        vlen_dtype = h5py.check_dtype(vlen=var.dtype)
        if vlen_dtype is str:
            encoding["dtype"] = str
        elif vlen_dtype is not None:  # pragma: no cover
            # xarray doesn't support writing arbitrary vlen dtypes yet.
            pass
        else:
            encoding["dtype"] = var.dtype

        # cheat attributes
        #dmom = ds[mom]
        if "moment" in name:
            name = attrs.pop("moment").lower()
            try:
                name = GAMIC_NAMES[name]
            except KeyError:
                #ds = ds.drop_vars(mom)
                pass

            # extract and translate attributes to cf
            dmax = np.iinfo(var.dtype).max
            dmin = np.iinfo(var.dtype).min
            minval = attrs.pop("dyn_range_min")
            maxval = attrs.pop("dyn_range_max")
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

        elif "ray_header" in name:
            variables = {}
            recarray = Variable(dimensions, data, attrs, encoding)
            print(dir(recarray))
            #print(recarray.values["azimuth_start"])
            ray_header = {recname: recarray.values[recname] for recname in recarray.values.dtype.names}

            azstart = ray_header["azimuth_start"]
            azstop = ray_header["azimuth_stop"]
            #if self._dim0[0] == "azimuth":
            zero_index = np.where(azstop < azstart)
            azstop[zero_index[0]] += 360
            azimuth = (azstart + azstop) / 2.0

            elstart = ray_header["elevation_start"]
            elstop = ray_header["elevation_stop"]
            elevation = (elstart + elstop) / 2.0

            rtime = ray_header["timestamp"] / 1e6
            rtime_attrs = {"units": "seconds since 1970-01-01T00:00:00Z",
                     "standard_name": "time"}

            with self._manager.acquire_context(False) as root:
                #print(root)
                #print(root["scan0"]["how"])
                #print(root["scan0"])
                range_samples = root["scan0"]["how"].attrs["range_samples"]
                range_step = root["scan0"]["how"].attrs["range_step"]
                bin_range = range_step * range_samples
                range_data = np.arange(
                    bin_range / 2.0, bin_range * root["scan0"]["how"].attrs["bin_count"], bin_range, dtype="float32"
                )

            from .xarray import az_attrs, el_attrs, range_attrs

            print("dims", dimensions)
            range_attrs["meters_to_center_of_first_gate"] = bin_range / 2.0

            variables["azimuth"] = Variable(("azimuth",), azimuth, az_attrs)
            variables["elevatiom"] = Variable(("azimuth",), elevation, el_attrs)
            variables["rtime"] = Variable(("azimuth",), rtime, rtime_attrs)
            variables["range"] = Variable(("range",), range_data, range_attrs)
            return variables

        return dict({name: Variable(dimensions, data, attrs, encoding)})


    def get_variables(self):
        return FrozenDict(
            (k1, v1) for k, v in self.ds.variables.items() for k1, v1 in self.open_store_variable(k, v).items()
        )

    def get_attrs(self):
        #print(dir(self))
        return FrozenDict(_read_attributes(self.ds))

    def get_dimensions(self):
        #print(self.ds)
        return self.ds.dimensions

    def get_encoding(self):
        encoding = {}
        encoding["unlimited_dims"] = {
            k for k, v in self.ds.dimensions.items() if v is None
        }
        return encoding

    def set_dimension(self, name, length, is_unlimited=False):
        if is_unlimited:
            self.ds.dimensions[name] = None
            self.ds.resize_dimension(name, length)
        else:
            self.ds.dimensions[name] = length

    def set_attribute(self, key, value):
        self.ds.attrs[key] = value

    def encode_variable(self, variable):
        return _encode_nc4_variable(variable)

    def prepare_variable(
        self, name, variable, check_encoding=False, unlimited_dims=None
    ):
        import h5py

        attrs = variable.attrs.copy()
        dtype = _get_datatype(variable, raise_on_invalid_encoding=check_encoding)

        fillvalue = attrs.pop("_FillValue", None)
        if dtype is str and fillvalue is not None:
            raise NotImplementedError(
                "h5netcdf does not yet support setting a fill value for "
                "variable-length strings "
                "(https://github.com/shoyer/h5netcdf/issues/37). "
                "Either remove '_FillValue' from encoding on variable %r "
                "or set {'dtype': 'S1'} in encoding to use the fixed width "
                "NC_CHAR type." % name
            )

        if dtype is str:
            dtype = h5py.special_dtype(vlen=str)

        encoding = _extract_h5nc_encoding(variable, raise_on_invalid=check_encoding)
        kwargs = {}

        # Convert from NetCDF4-Python style compression settings to h5py style
        # If both styles are used together, h5py takes precedence
        # If set_encoding=True, raise ValueError in case of mismatch
        if encoding.pop("zlib", False):
            if check_encoding and encoding.get("compression") not in (None, "gzip"):
                raise ValueError("'zlib' and 'compression' encodings mismatch")
            encoding.setdefault("compression", "gzip")

        if (
            check_encoding
            and "complevel" in encoding
            and "compression_opts" in encoding
            and encoding["complevel"] != encoding["compression_opts"]
        ):
            raise ValueError("'complevel' and 'compression_opts' encodings mismatch")
        complevel = encoding.pop("complevel", 0)
        if complevel != 0:
            encoding.setdefault("compression_opts", complevel)

        encoding["chunks"] = encoding.pop("chunksizes", None)

        # Do not apply compression, filters or chunking to scalars.
        if variable.shape:
            for key in [
                "compression",
                "compression_opts",
                "shuffle",
                "chunks",
                "fletcher32",
            ]:
                if key in encoding:
                    kwargs[key] = encoding[key]
        if name not in self.ds:
            nc4_var = self.ds.create_variable(
                name,
                dtype=dtype,
                dimensions=variable.dims,
                fillvalue=fillvalue,
                **kwargs,
            )
        else:
            nc4_var = self.ds[name]

        for k, v in attrs.items():
            nc4_var.attrs[k] = v

        target = GamicArrayWrapper(name, self)

        return target, variable.data

    def sync(self):
        self.ds.sync()

    def close(self, **kwargs):
        self._manager.close(**kwargs)


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
        group=None,
        lock=None,
        invalid_netcdf=None,
        phony_dims=None,
        decode_vlen_strings=True,
    ):

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
        return ds
