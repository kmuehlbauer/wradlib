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
__all__ = ["RadolanBackendEntrypoint", "OdimBackendEntrypoint",
           "GamicBackendEntrypoint"]

__doc__ = __doc__.format("\n   ".join(__all__))

import os

import datetime as dt
import numpy as np

from xarray import decode_cf, merge
from xarray.core import indexing
from xarray.core.utils import Frozen, FrozenDict, close_on_error, read_magic_number
from xarray.core.variable import Variable
from xarray.backends import H5NetCDFStore
from xarray.backends.common import (
    AbstractDataStore,
    BackendArray,
    BackendEntrypoint,
)
from xarray.backends.netCDF4_ import (
    BaseNetCDF4Array,
    _nc4_require_group
)

from xarray.backends.h5netcdf_ import (
    _read_attributes,
    maybe_decode_bytes,
    _h5netcdf_create_group,
    H5NetCDFArrayWrapper,
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
        keep_azimuth=None,
        keep_elevation=None,
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


# class OdimArrayWrapper(BaseNetCDF4Array):
#     def get_array(self, needs_lock=True):
#         ds = self.datastore._acquire(needs_lock)
#         variable = ds.variables[self.variable_name]
#         return variable
#
#     def __getitem__(self, key):
#         return indexing.explicit_indexing_adapter(
#             key, self.shape, indexing.IndexingSupport.OUTER_1VECTOR, self._getitem
#         )
#
#     def _getitem(self, key):
#         # h5py requires using lists for fancy indexing:
#         # https://github.com/h5py/h5py/issues/992
#         key = tuple(list(k) if isinstance(k, np.ndarray) else k for k in key)
#         with self.datastore.lock:
#             array = self.get_array(needs_lock=False)
#             return array[key]

class OdimStore(H5NetCDFStore):
    """Store for reading and writing GAMIC data via h5netcdf"""

    def _get_fixed_dim_and_angle(self):
        with self._manager.acquire_context(False) as root:
            grp = self._group.split("/")[0]
            dim = "elevation"

            # try RHI first
            angle_keys = ["az_angle", "azangle"]
            for ak in angle_keys:
               angle = root[grp]["where"].attrs.get(ak, None)
               if angle is not None:
                   break
            if angle is not None:
               angle = np.round(angle, decimals=1)
            else:
               dim = "azimuth"
               angle = np.round(root[grp]["where"].attrs["elangle"], decimals=1)

        return dim, angle

    def open_store_variable(self, group, name, var):
        import h5py

        print("Manager:", self._manager)

        with self._manager.acquire_context(False) as root:
            print(root[group])
            print(root[group]["what"])


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
        data = indexing.LazilyOuterIndexedArray(H5NetCDFArrayWrapper(name, self))
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

        variables = {}
        # cheat attributes
        if "data" in name:
            encoding["group"] = group

            with self._manager.acquire_context(False) as root:
                attrs = {}
                attrs["scale_factor"] = root[group]["what"].attrs["gain"]
                attrs["add_offset"] = root[group]["what"].attrs["offset"]
                attrs["_FillValue"] = root[group]["what"].attrs["nodata"]
                attrs["_Undetect"] = root[group]["what"].attrs["undetect"]
                name = maybe_decode_bytes(root[group]["what"].attrs["quantity"])

            # handle non-standard moment names
            try:
                mapping = moments_mapping[name]
            except KeyError:
                pass
            else:
                from .xarray import moment_attrs
                attrs.update({key: mapping[key] for key in moment_attrs})

            variables[name] = Variable(dimensions, data, attrs, encoding)

        # add coordinate
        def _get_azimuth_how(self):
            with self._manager.acquire_context(False) as root:
                grp = self._group.split("/")[0]
                startaz = root[grp]["how"].attrs["startazA"]
                stopaz = root[grp]["how"].attrs["stopazA"]
                zero_index = np.where(stopaz < startaz)
                stopaz[zero_index[0]] += 360
                azimuth_data = (startaz + stopaz) / 2.0
            return azimuth_data

        def _get_azimuth_where(self):
            with self._manager.acquire_context(False) as root:
                grp = self._group.split("/")[0]
                nrays = root[grp]["where"].attrs["nrays"]
                res = 360.0 / nrays
                azimuth_data = np.arange(res / 2.0, 360.0, res, dtype="float32")
            return azimuth_data

        try:
            azimuth = _get_azimuth_how(self)
        except (AttributeError, KeyError, TypeError):
            azimuth = _get_azimuth_where(self)

        # add coordinate
        def _get_elevation_how(self):
            with self._manager.acquire_context(False) as root:
                grp = self._group.split("/")[0]
                startaz = root[grp]["how"].attrs["startelA"]
                stopaz = root[grp]["how"].attrs["stopelA"]
                elevation_data = (startaz + stopaz) / 2.0
            return elevation_data

        def _get_elevation_where(self):
            with self._manager.acquire_context(False) as root:
                grp = self._group.split("/")[0]
                nrays = root[grp]["where"].attrs["nrays"]
                elangle = root[grp]["where"].attrs["elangle"]
                elevation_data = np.ones(nrays, dtype="float32") * elangle
            return elevation_data

        def _get_time_how(self):
            with self._manager.acquire_context(False) as root:
                grp = self._group.split("/")[0]
                startT = root[grp]["how"].attrs["startazT"]
                stopT = root[grp]["how"].attrs["stopazT"]
                time_data = (startT + stopT) / 2.0
            return time_data

        def _get_time_what(self, nrays=None):
            with self._manager.acquire_context(False) as root:
                grp = self._group.split("/")[0]
            what = root[grp]["how"].attrs
            startdate = what["startdate"]
            starttime = what["starttime"]
            enddate = what["enddate"]
            endtime = what["endtime"]
            start = dt.datetime.strptime(startdate + starttime, "%Y%m%d%H%M%S")
            end = dt.datetime.strptime(enddate + endtime, "%Y%m%d%H%M%S")
            start = start.replace(tzinfo=dt.timezone.utc).timestamp()
            end = end.replace(tzinfo=dt.timezone.utc).timestamp()
            if nrays is None:
                nrays = root[grp]["where"]["nrays"]
            if start == end:
                import warnings
                warnings.warn(
                    "WRADLIB: Equal ODIM `starttime` and `endtime` "
                    "values. Can't determine correct sweep start-, "
                    "end- and raytimes.",
                    UserWarning,
                )

                time_data = np.ones(nrays) * start
            else:
                delta = (end - start) / nrays
                time_data = np.arange(start + delta / 2.0, end, delta)
                time_data = np.roll(time_data, shift=+self.a1gate)
            return time_data

        def _get_ray_times(self, nrays=None):

            try:
                time_data = _get_time_how(self)
                self._need_time_recalc = False
            except (AttributeError, KeyError, TypeError):
                time_data = _get_time_what(self, nrays=nrays)
                self._need_time_recalc = True
            return time_data

        try:
            elevation = _get_elevation_how(self)
        except (AttributeError, KeyError, TypeError):
            elevation = _get_elevation_where(self)

        rtime = _get_ray_times(self)

        from .xarray import az_attrs, el_attrs, range_attrs, time_attrs

        dim, angle = self._get_fixed_dim_and_angle()
        print(dim, angle)
        dims = ("azimuth", "elevation")
        if dim == dims[1]:
            dims = (dims[1], dims[0])
        print(dims)

        variables["azimuth"] = Variable((dims[0],), azimuth, az_attrs)
        variables["elevation"] = Variable((dims[0],), elevation, el_attrs)

        rtime_attrs = {"units": "seconds since 1970-01-01T00:00:00Z",
                       "standard_name": "time"}
        variables["rtime"] = Variable((dims[0],), rtime, rtime_attrs)
        #
        #     with self._manager.acquire_context(False) as root:
        #         #print(root)
        #         #print(root["how"])
        #         #print(root["what"])
        #         #print(root["where"])
        #         #print(root["scan0"]["how"])
        #         #print(root["scan0"])
        #         range_samples = root[group]["how"].attrs["range_samples"]
        #         range_step = root[group]["how"].attrs["range_step"]
        #         bin_range = range_step * range_samples
        #         range_data = np.arange(
        #             bin_range / 2.0, bin_range * root[group]["how"].attrs["bin_count"], bin_range, dtype="float32"
        #         )
        #         longitude = root["where"].attrs["lon"]
        #         latitude = root["where"].attrs["lat"]
        #         altitude = root["where"].attrs["height"]
        #         start = root[group]["how"].attrs["timestamp"]
        #         import dateutil
        #         start = dateutil.parser.parse(start)
        #         start = start.replace(tzinfo=dt.timezone.utc).timestamp()
        #         angle_res = root[group]["how"].attrs["angle_step"]
        #
        #     dim, angle = self._get_fixed_dim_and_angle()
        #     dims = ("azimuth", "elevation")
        #     if dim == dims[0]:
        #         dims = (dims[1], dims[0])
        #
        #     sort_idx = np.argsort(azimuth)
        #     a1gate = np.argsort(rtime[sort_idx])[0]
        #
        #
        #     sweep_mode = (
        #         "azimuth_surveillance" if dims[0] == "azimuth" else "rhi"
        #     )
        #
        #     from .xarray import az_attrs, el_attrs, range_attrs, time_attrs
        #
        #     #print("dims", dimensions)
        #     range_attrs["meters_to_center_of_first_gate"] = bin_range / 2.0
        #
        #     lon_attrs = dict(long_name="longitude", units="degrees_east", standard_name="longitude")
        #     lat_attrs = dict(long_name="latitude", units="degrees_north", positive="up",
        #                      standard_name="latitude")
        #     alt_attrs = dict(long_name="altitude", units="meters",
        #                      standard_name="altitude")
        #     variables["longitude"] = Variable((), longitude, lon_attrs)
        #     variables["latitude"] = Variable((), latitude, lat_attrs)
        #     variables["altitude"] = Variable((), altitude, alt_attrs)
        #     variables["time"] = Variable((), start, time_attrs)
        #     variables["sweep_mode"] = Variable((), sweep_mode)
        #
        #     az_attrs["a1gate"] = a1gate
        #     if dims[0] == "azimuth":
        #         az_attrs["angle_res"] = angle_res
        #     else:
        #         el_attrs["angle_res"] = angle_res
        #
        #     variables["azimuth"] = Variable((dims[0],), azimuth, az_attrs)
        #     variables["elevation"] = Variable((dims[0],), elevation, el_attrs)
        #     variables["rtime"] = Variable((dims[0],), rtime, rtime_attrs)
        #     variables["range"] = Variable(("range",), range_data, range_attrs)
        #     return variables

        return variables

    # def _acquire(self, needs_lock=True):
    #     # with self._manager.acquire_context(needs_lock) as root:
    #     #     ds = _nc4_require_group(
    #     #         root, self._group, self._mode, create_group=_h5netcdf_create_group
    #     #     )
    #     with self._manager.acquire_context(needs_lock) as root:
    #         groups = root[self._group].groups
    #         variables = [k for k in groups if "data" in k]
    #         vars_idx = np.argsort([int(v[len("data"):]) for v in variables])
    #         variables = np.array(variables)[vars_idx].tolist()
    #         print(variables)
    #
    #         ds_list = {"/".join([self._group, var]): _nc4_require_group(
    #             root, "/".join([self._group, var]), self._mode,
    #             create_group=_h5netcdf_create_group
    #         ) for var in variables}
    #         print(ds_list)
    #     return ds_list



    def get_variables(self):
        return FrozenDict(
            (k1, v1) for k, v in self.ds.variables.items() for k1, v1 in self.open_store_variable(self._group, k, v).items()
        )


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
        phony_dims="access",
        decode_vlen_strings=True,
        keep_elevation=None,
        keep_azimuth=None,
    ):

        store = OdimStore.open(
            filename_or_obj,
            format=format,
            group=group,
            lock=lock,
            invalid_netcdf=invalid_netcdf,
            phony_dims=phony_dims,
            decode_vlen_strings=decode_vlen_strings,
        )

        with store._manager.acquire_context(True) as root:
            print("store.group:", store._group)
            print("root-groups:", root[store._group].groups)
            groups = root[store._group].groups
            variables = [k for k in groups if "data" in k]
            vars_idx = np.argsort([int(v[len("data"):]) for v in variables])
            variables = np.array(variables)[vars_idx].tolist()
            print(variables)

            ds_list = ["/".join([store._group, var]) for var in variables]
            print(ds_list)
        store.close()

        stores = [OdimStore.open(filename_or_obj,
                                 format=format,
                                 group=grp,
                                 lock=lock,
                                 invalid_netcdf=invalid_netcdf,
                                 phony_dims=phony_dims,
                                 decode_vlen_strings=decode_vlen_strings,) for grp in ds_list]

        store_entrypoint = StoreBackendEntrypoint()

        ds = merge([store_entrypoint.open_dataset(
            store,
            mask_and_scale=mask_and_scale,
            decode_times=decode_times,
            concat_characters=concat_characters,
            decode_coords=decode_coords,
            drop_variables=drop_variables,
            use_cftime=use_cftime,
            decode_timedelta=decode_timedelta,
        ) for store in stores])

        #ds = ds.sortby(list(ds.dims.keys())[0])
        #ds = ds.pipe(_reindex_angle)

        return ds


class GamicStore(H5NetCDFStore):
    """Store for reading and writing GAMIC data via h5netcdf"""

    def _get_fixed_dim_and_angle(self):
       with self._manager.acquire_context(False) as root:
           dim = "azimuth"
           try:
               angle = np.round(root[self._group]["how"].attrs[dim], decimals=1)
           except KeyError:
               dim = "elevation"
               angle = np.round(root[self._group]["how"].attrs[dim], decimals=1)
       return dim, angle

    def open_store_variable(self, group, name, var):
        import h5py

        #print("Manager:", self._manager)


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
        data = indexing.LazilyOuterIndexedArray(H5NetCDFArrayWrapper(name, self))
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
            encoding["group"] = group
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
            attrs["coordinates"] = "elevation azimuth range latitude longitude altitude time rtime sweep_mode"

        elif "ray_header" in name:
            variables = {}
            recarray = Variable(dimensions, data, attrs, encoding)
            #print(dir(recarray))
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
                #print(root["how"])
                #print(root["what"])
                #print(root["where"])
                #print(root["scan0"]["how"])
                #print(root["scan0"])
                range_samples = root[group]["how"].attrs["range_samples"]
                range_step = root[group]["how"].attrs["range_step"]
                bin_range = range_step * range_samples
                range_data = np.arange(
                    bin_range / 2.0, bin_range * root[group]["how"].attrs["bin_count"], bin_range, dtype="float32"
                )
                longitude = root["where"].attrs["lon"]
                latitude = root["where"].attrs["lat"]
                altitude = root["where"].attrs["height"]
                start = root[group]["how"].attrs["timestamp"]
                import dateutil
                start = dateutil.parser.parse(start)
                start = start.replace(tzinfo=dt.timezone.utc).timestamp()
                angle_res = root[group]["how"].attrs["angle_step"]

            dim, angle = self._get_fixed_dim_and_angle()
            dims = ("azimuth", "elevation")
            if dim == dims[0]:
                dims = (dims[1], dims[0])

            sort_idx = np.argsort(azimuth)
            a1gate = np.argsort(rtime[sort_idx])[0]


            sweep_mode = (
                "azimuth_surveillance" if dims[0] == "azimuth" else "rhi"
            )

            from .xarray import az_attrs, el_attrs, range_attrs, time_attrs

            #print("dims", dimensions)
            range_attrs["meters_to_center_of_first_gate"] = bin_range / 2.0

            lon_attrs = dict(long_name="longitude", units="degrees_east", standard_name="longitude")
            lat_attrs = dict(long_name="latitude", units="degrees_north", positive="up",
                             standard_name="latitude")
            alt_attrs = dict(long_name="altitude", units="meters",
                             standard_name="altitude")
            variables["longitude"] = Variable((), longitude, lon_attrs)
            variables["latitude"] = Variable((), latitude, lat_attrs)
            variables["altitude"] = Variable((), altitude, alt_attrs)
            variables["time"] = Variable((), start, time_attrs)
            variables["sweep_mode"] = Variable((), sweep_mode)

            az_attrs["a1gate"] = a1gate
            if dims[0] == "azimuth":
                az_attrs["angle_res"] = angle_res
            else:
                el_attrs["angle_res"] = angle_res

            variables["azimuth"] = Variable((dims[0],), azimuth, az_attrs)
            variables["elevation"] = Variable((dims[0],), elevation, el_attrs)
            variables["rtime"] = Variable((dims[0],), rtime, rtime_attrs)
            variables["range"] = Variable(("range",), range_data, range_attrs)
            return variables

        return dict({name: Variable(dimensions, data, attrs, encoding)})

    def get_variables(self):
        return FrozenDict(
            (k1, v1) for k, v in self.ds.variables.items() for k1, v1 in self.open_store_variable(self._group, k, v).items()
        )


def _reindex_angle(ds, force=False):
    # Todo: The current code assumes to have PPI's of 360deg and RHI's of 90deg,
    #       make this work also for sectorized measurements
    full_range = dict(azimuth=360, elevation=90)
    dimname = list(ds.dims)[0]
    secname = "elevation"
    dim = ds[dimname]
    diff = dim.diff(dimname)
    # this captures different angle spacing
    # catches also missing rays and double rays
    # and other erroneous ray alignments which result in different diff values
    diffset = set(diff.values)
    non_uniform_angle_spacing = len(diffset) > 1
    # this captures missing and additional rays in case the angle differences
    # are equal
    non_full_circle = False
    if not non_uniform_angle_spacing:
        res = list(diffset)[0]
        non_full_circle = ((res * ds.dims[dimname]) % full_range[dimname]) != 0

    # fix issues with ray alignment
    if force | non_uniform_angle_spacing | non_full_circle:
        # create new array and reindex
        res = ds.azimuth.angle_res
        new_rays = int(np.round(full_range[dimname] / res, decimals=0))

        # find exact duplicates and remove
        _, idx = np.unique(ds[dimname], return_index=True)
        if len(idx) < len(ds[dimname]):
            ds = ds.isel({dimname: idx})
            # if ray_time was errouneously created from wrong dimensions
            # we need to recalculate it
            # if sweep._need_time_recalc:
            #     ray_times = sweep._get_ray_times(nrays=len(idx))
            #     ray_times = sweep._decode_cf(ray_times)
            #     ds = ds.assign({"rtime": ray_times})

        # todo: check if assumption that beam center points to
        #       multiples of res/2. is correct in any case
        azr = np.arange(res / 2.0, new_rays * res, res, dtype=diff.dtype)
        ds = ds.reindex(
            {dimname: azr},
            method="nearest",
            tolerance=res / 4.0,
            # fill_value=xr.core.dtypes.NA,
        )
        # check other coordinates
        # check secondary angle coordinate (no nan)
        # set nan values to reasonable median
        if np.count_nonzero(np.isnan(ds[secname])):
            ds[secname] = ds[secname].fillna(ds[secname].median(skipna=True))
        # todo: rtime is also affected, might need to be treated accordingly

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
        group=None,
        lock=None,
        invalid_netcdf=None,
        phony_dims="access",
        decode_vlen_strings=True,
        keep_elevation=None,
        keep_azimuth=None,
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

        ds = ds.sortby(list(ds.dims.keys())[0])
        ds = ds.pipe(_reindex_angle)

        return ds
