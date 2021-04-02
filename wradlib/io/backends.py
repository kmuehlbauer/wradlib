#!/usr/bin/env python
# Copyright (c) 2021, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

"""
wradlib backends for xarray
^^^^^^^^^^^^^^^^^^^^^^^^^^^
Reading radar data into xarray Datasets using ``xarray.open_dataset``
and ``xarray.open_mfdataset``

.. autosummary::
   :nosignatures:
   :toctree: generated/

   {}
"""
__all__ = [
    "RadolanBackendEntrypoint",
    "OdimBackendEntrypoint",
]

__doc__ = __doc__.format("\n   ".join(__all__))

import io
import os
from distutils.version import LooseVersion

import h5netcdf
import numpy as np
from xarray.backends import H5NetCDFStore
from xarray.backends.common import (
    AbstractDataStore,
    BackendArray,
    BackendEntrypoint,
    find_root_and_group,
)
from xarray.backends.file_manager import CachingFileManager, DummyFileManager
from xarray.backends.locks import (
    HDF5_LOCK,
    SerializableLock,
    combine_locks,
    ensure_lock,
    get_write_lock,
)
from xarray.backends.store import StoreBackendEntrypoint
from xarray.core import indexing
from xarray.core.utils import (
    Frozen,
    FrozenDict,
    close_on_error,
    is_remote_uri,
    read_magic_number,
)
from xarray.core.variable import Variable

from wradlib.io import radolan
from wradlib.io.xarray import (
    GamicH5NetCDFMetadata,
    OdimH5NetCDFMetadata,
    GAMIC_NAMES,
    _fix_angle,
    _reindex_angle,
    moment_attrs,
    moments_mapping,
)

RADOLAN_LOCK = SerializableLock()


class RadolanArrayWrapper(BackendArray):
    """Wraps array of RADOLAN data."""

    def __init__(self, datastore, array):
        self.datastore = datastore
        self.shape = array.shape
        self.dtype = array.dtype

    def _getitem(self, key):
        return self.datastore.ds.data[key]

    def __getitem__(self, key):
        return indexing.explicit_indexing_adapter(
            key,
            self.shape,
            indexing.IndexingSupport.BASIC,
            self._getitem,
        )


class RadolanDataStore(AbstractDataStore):
    """Implements ``xarray.AbstractDataStore`` read-only API for a RADOLAN files."""

    def __init__(self, filename_or_obj, lock=None, fillmissing=False, copy=False):
        if lock is None:
            lock = RADOLAN_LOCK
        self.lock = ensure_lock(lock)

        if isinstance(filename_or_obj, str):
            manager = CachingFileManager(
                radolan._radolan_file,
                filename_or_obj,
                lock=lock,
                kwargs=dict(fillmissing=fillmissing, copy=copy),
            )
        else:
            if isinstance(filename_or_obj, bytes):
                filename_or_obj = io.BytesIO(filename_or_obj)
            dataset = radolan._radolan_file(
                filename_or_obj, fillmissing=fillmissing, copy=copy
            )
            manager = DummyFileManager(dataset)

        self._manager = manager
        self._filename = self.ds.filename

    def _acquire(self, needs_lock=True):
        with self._manager.acquire_context(needs_lock) as ds:
            return ds

    @property
    def ds(self):
        return self._acquire()

    def open_store_variable(self, name, var):
        encoding = dict(source=self._filename)
        if isinstance(var.data, np.ndarray):
            data = var.data
        else:
            data = indexing.LazilyOuterIndexedArray(RadolanArrayWrapper(self, var.data))
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

    def close(self, **kwargs):
        self._manager.close(**kwargs)


class RadolanBackendEntrypoint(BackendEntrypoint):
    """Xarray BackendEntrypoint for RADOLAN data."""

    def open_dataset(
        self,
        filename_or_obj,
        *,
        mask_and_scale=True,
        decode_times=True,
        concat_characters=True,
        decode_coords=True,
        drop_variables=None,
        use_cftime=None,
        decode_timedelta=None,
        lock=None,
        fillmissing=False,
        copy=False,
    ):

        store = RadolanDataStore(
            filename_or_obj,
            lock=lock,
            fillmissing=fillmissing,
            copy=copy,
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


HDF5_LOCK = SerializableLock()


class H5NetCDFArrayWrapper(BackendArray):
    """H5NetCDFArrayWrapper

    adapted from https://github.com/pydata/xarray/
    """

    __slots__ = ("datastore", "dtype", "shape", "variable_name")

    def __init__(self, variable_name, datastore):
        self.datastore = datastore
        self.variable_name = variable_name

        array = self.get_array()
        self.shape = array.shape

        dtype = array.dtype
        if dtype is str:
            # use object dtype because that's the only way in numpy to
            # represent variable length strings; it also prevents automatic
            # string concatenation via conventions.decode_cf_variable
            dtype = np.dtype("O")
        self.dtype = dtype

    def __setitem__(self, key, value):
        with self.datastore.lock:
            data = self.get_array(needs_lock=False)
            data[key] = value
            if self.datastore.autoclose:
                self.datastore.close(needs_lock=False)

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


def _get_encoding(self, var):
    import h5py
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
    return encoding


class OdimSubStore(AbstractDataStore):
    """Store for reading ODIM data-moments via h5netcdf."""

    def __init__(
        self,
        store,
        group=None,
        lock=HDF5_LOCK,
    ):

        if not isinstance(store, OdimStore):
            raise TypeError(
                f"Wrong type {type(store)} for parameter store, "
                f"expected 'OdimStore'."
            )

        self._manager = store._manager
        self._group = group
        self._filename = store.filename
        self.is_remote = is_remote_uri(self._filename)
        self.lock = ensure_lock(lock)

    @property
    def root(self):
        with self._manager.acquire_context(False) as root:
            return OdimH5NetCDFMetadata(root, self._group.lstrip("/"))

    def _acquire(self, needs_lock=True):
        with self._manager.acquire_context(needs_lock) as root:
            ds = root[self._group.lstrip("/")]
        return ds

    @property
    def ds(self):
        return self._acquire()

    def open_store_variable(self, name, var):


        dimensions = self.root.get_variable_dimensions(var.dimensions)
        data = indexing.LazilyOuterIndexedArray(H5NetCDFArrayWrapper(name, self))
        encoding = _get_encoding(self, var)

        # cheat attributes
        if "data" in name:
            encoding["group"] = self._group
            attrs = self.root.what
            name = attrs.pop("quantity")

            # handle non-standard moment names
            try:
                mapping = moments_mapping[name]
            except KeyError:
                pass
            else:
                attrs.update({key: mapping[key] for key in moment_attrs})
            attrs[
                "coordinates"
            ] = "elevation azimuth range latitude longitude altitude time rtime sweep_mode"

        return name, Variable(dimensions, data, attrs, encoding)

    def open_store_coordinates(self):
        return self.root.coordinates

    def get_variables(self):
        return FrozenDict(
            (k1, v1)
            for k1, v1 in {
                **dict(
                    [
                        self.open_store_variable(k, v)
                        for k, v in self.ds.variables.items()
                    ]
                ),
            }.items()
        )


class OdimStore(AbstractDataStore):
    """Store for reading ODIM dataset groups via h5netcdf."""

    def __init__(self, manager, group=None, lock=HDF5_LOCK):

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
        self._filename = self.filename
        self.is_remote = is_remote_uri(self._filename)
        self.lock = ensure_lock(lock)
        self._substore = None

    @classmethod
    def open(
        cls,
        filename,
        mode="r",
        format=None,
        group=None,
        lock=None,
        invalid_netcdf=None,
        phony_dims=None,
        decode_vlen_strings=True,
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
        return cls(manager, group=group, lock=lock)

    @property
    def filename(self):
        with self._manager.acquire_context(False) as root:
            return root.filename

    @property
    def substore(self):
        if self._substore is None:
            with self._manager.acquire_context(False) as root:
                subgroups = [
                    "/".join([self._group, k])
                    for k in root[self._group].groups
                    if "data" in k
                ]
                substore = []
                substore.extend(
                    [
                        OdimSubStore(
                            self,
                            group=group,
                            lock=self.lock,
                        )
                        for group in subgroups
                    ]
                )
                self._substore = substore

        return self._substore

    def open_store_coordinates(self):
        coords = self.substore[0].open_store_coordinates()
        return coords

    def get_variables(self):
        return FrozenDict(
            (k1, v1)
            for k1, v1 in {
                **dict(
                    [
                        (k, v)
                        for substore in self.substore
                        for k, v in substore.get_variables().items()
                    ]
                ),
                **self.open_store_coordinates(),
            }.items()
        )

    def get_attrs(self):
        dim, angle = self.substore[0].root.fixed_dim_and_angle
        attributes = {}
        attributes["fixed_angle"] = angle.item()
        return FrozenDict(attributes)


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
        group="dataset1",
        lock=None,
        invalid_netcdf=None,
        phony_dims="access",
        decode_vlen_strings=True,
        keep_elevation=False,
        keep_azimuth=False,
    ):

        if isinstance(filename_or_obj, io.IOBase):
            filename_or_obj.seek(0)

        store = OdimStore.open(
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

        if decode_coords:
            ds = ds.pipe(_reindex_angle, store=store)

        if not keep_azimuth:
            if ds.azimuth.dims[0] == "elevation":
                ds = ds.assign_coords({"azimuth": ds.azimuth.pipe(_fix_angle)})
        if not keep_elevation:
            if ds.elevation.dims[0] == "azimuth":
                ds = ds.assign_coords({"elevation": ds.elevation.pipe(_fix_angle)})

        return ds

# todo: clean-up, move all possible processing into .xarray
class GamicStore(AbstractDataStore):
    """Store for reading ODIM dataset groups via h5netcdf."""

    def __init__(self, manager, group=None, lock=HDF5_LOCK):

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
        self._filename = self.filename
        self.is_remote = is_remote_uri(self._filename)
        self.lock = ensure_lock(lock)
        self._need_time_recalc = False
        #self._substore = None

    @classmethod
    def open(
        cls,
        filename,
        mode="r",
        format=None,
        group=None,
        lock=None,
        invalid_netcdf=None,
        phony_dims=None,
        decode_vlen_strings=True,
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
        return cls(manager, group=group, lock=lock)

    @property
    def filename(self):
        with self._manager.acquire_context(False) as root:
            return root.filename

    @property
    def root(self):
        with self._manager.acquire_context(False) as root:
            return GamicH5NetCDFMetadata(root, self._group.lstrip("/"))

    def _acquire(self, needs_lock=True):
        with self._manager.acquire_context(needs_lock) as root:
            ds = root[self._group.lstrip("/")]
        return ds

    @property
    def ds(self):
        return self._acquire()

    def open_store_variable(self, name, var):

        dimensions = self.root.get_variable_dimensions(var.dimensions)
        data = indexing.LazilyOuterIndexedArray(H5NetCDFArrayWrapper(name, self))
        encoding = _get_encoding(self, var)

        # cheat attributes
        if "moment" in name:
            encoding["group"] = self._group
            attrs = {**var.attrs}

            name = attrs.pop("moment").lower()
            try:
                name = GAMIC_NAMES[name]
            except KeyError:
                # ds = ds.drop_vars(mom)
                pass

            dmax = np.iinfo(var.dtype).max
            dmin = np.iinfo(var.dtype).min
            minval = attrs.pop("dyn_range_min")
            maxval = attrs.pop("dyn_range_max")
            dtype = minval.dtype
            dyn_range = maxval - minval
            if maxval != minval:
                gain = dyn_range / (dmax - 1)
                minval -= gain
            else:
                gain = (dmax - dmin) / dmax
                minval = dmin
            gain = gain.astype(dtype)
            minval = minval.astype(dtype)
            undetect = np.array([dmin])[0].astype(dtype)
            attrs["scale_factor"] = gain
            attrs["add_offset"] = minval
            attrs["_FillValue"] = undetect
            attrs["_Undetect"] = undetect

            attrs[
                "coordinates"
            ] = "elevation azimuth range latitude longitude altitude time rtime sweep_mode"
        elif "ray_header" in name:
            variables = self.open_store_coordinates(name, dimensions, data, encoding)
            return variables
        else:
            return {}
        return {name: Variable(dimensions, data, attrs, encoding)}

    def open_store_coordinates(self, name, dimensions, data, encoding):
        from .xarray import az_attrs, el_attrs, range_attrs, time_attrs

        def _get_ray_header_data(dimensions, data, encoding):

            ray_header = Variable(dimensions, data, {}, encoding)

            azstart = ray_header.values["azimuth_start"]
            azstop = ray_header.values["azimuth_stop"]
            zero_index = np.where(azstop < azstart)
            azstop[zero_index[0]] += 360
            azimuth = (azstart + azstop) / 2.0

            elstart = ray_header.values["elevation_start"]
            elstop = ray_header.values["elevation_stop"]
            elevation = (elstart + elstop) / 2.0

            rtime = ray_header.values["timestamp"] / 1e6

            return azimuth, elevation, rtime

        azimuth, elevation, rtime = _get_ray_header_data(dimensions, data, encoding)
        dim, angle = self.root.fixed_dim_and_angle
        angle_res = np.round(np.nanmedian(np.diff(locals()[dim])), decimals=1)
        dims = ("azimuth", "elevation")
        if dim == dims[1]:
            dims = (dims[1], dims[0])

        sort_idx = np.argsort(locals()[dim])
        a1gate = np.argsort(rtime[sort_idx])[0]

        az_attrs["a1gate"] = a1gate

        if dim == "azimuth":
            az_attrs["angle_res"] = angle_res
        else:
            el_attrs["angle_res"] = angle_res

        sweep_mode = "azimuth_surveillance" if dim == "azimuth" else "rhi"

        rtime_attrs = {
            "units": "seconds since 1970-01-01T00:00:00Z",
            "standard_name": "time",
        }

        range_data, cent_first, bin_range = self.root.range
        range_attrs["meters_to_center_of_first_gate"] = cent_first
        range_attrs["meters_between_gates"] = bin_range

        lon_attrs = dict(
            long_name="longitude", units="degrees_east", standard_name="longitude"
        )
        lat_attrs = dict(
            long_name="latitude",
            units="degrees_north",
            positive="up",
            standard_name="latitude",
        )
        alt_attrs = dict(long_name="altitude", units="meters", standard_name="altitude")

        lon, lat, alt = self.root.site_coords

        coordinates = dict(
            azimuth=Variable((dims[0],), azimuth, az_attrs),
            elevation=Variable((dims[0],), elevation, el_attrs),
            rtime=Variable((dims[0],), rtime, rtime_attrs),
            range=Variable(("range",), range_data, range_attrs),
            time=Variable((), self.root.time, time_attrs),
            sweep_mode=Variable((), sweep_mode),
            longitude=Variable((), lon, lon_attrs),
            latitude=Variable((), lat, lat_attrs),
            altitude=Variable((), alt, alt_attrs),
        )

        return coordinates

    def get_variables(self):
        # return FrozenDict(
        #     (k1, v1)
        #     for k1, v1 in {
        #         **dict(
        #             [
        #                 (k, v)
        #                 for k, v in self.ds.variables.items()
        #             ]
        #         ),
        #         #**self.open_store_coordinates(),
        #     }.items()
        # )
        return FrozenDict(
            (k1, v1)
            for k, v in self.ds.variables.items()
            for k1, v1 in {
                **self.open_store_variable(k, v),
            }.items()
        )

    def get_attrs(self):
        dim, angle = self.root.fixed_dim_and_angle
        attributes = {}
        attributes["fixed_angle"] = angle.item()
        return FrozenDict(attributes)


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
        keep_elevation=False,
        keep_azimuth=False,
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

        if decode_coords:
            ds = ds.pipe(_reindex_angle, store=store)

        if not keep_azimuth:
            if ds.azimuth.dims[0] == "elevation":
                ds = ds.assign_coords({"azimuth": ds.azimuth.pipe(_fix_angle)})
        if not keep_elevation:
            if ds.elevation.dims[0] == "azimuth":
                ds = ds.assign_coords({"elevation": ds.elevation.pipe(_fix_angle)})

        return ds

