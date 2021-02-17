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
__all__ = [
    "RadolanBackendEntrypoint",
    "OdimBackendEntrypoint",
    "GamicBackendEntrypoint",
]

__doc__ = __doc__.format("\n   ".join(__all__))

import datetime as dt
import glob
import os

import numpy as np
import xarray
from xarray import DataArray, Dataset, decode_cf, merge
from xarray.backends import H5NetCDFStore
from xarray.backends.common import AbstractDataStore, BackendArray, BackendEntrypoint
from xarray.backends.h5netcdf_ import (
    H5NetCDFArrayWrapper,
    _h5netcdf_create_group,
    _read_attributes,
    maybe_decode_bytes,
)
from xarray.backends.locks import SerializableLock, ensure_lock
from xarray.backends.netCDF4_ import BaseNetCDF4Array, _nc4_require_group
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

from wradlib.georef import rect
from wradlib.io import radolan

from .xarray import GAMIC_NAMES, XRadBase, _reindex_angle, moment_attrs, moments_mapping, tqdm

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


def _reindex_angle(ds, store=None, force=False):
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
        res = ds[dimname].angle_res
        new_rays = int(np.round(full_range[dimname] / res, decimals=0))

        # find exact duplicates and remove
        _, idx = np.unique(ds[dimname], return_index=True)
        if len(idx) < len(ds[dimname]):
            ds = ds.isel({dimname: idx})
            # if ray_time was errouneously created from wrong dimensions
            # we need to recalculate it
            if store and store._need_time_recalc:
                ray_times = store._get_ray_times(nrays=len(idx))
                # need to decode only if ds is decoded
                if "units" in ds.rtime.encoding:
                    ray_times = decode_cf(Dataset({"rtime": ray_times})).rtime
                ds = ds.assign({"rtime": ray_times})

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


class HDFStoreAttribute(object):
    def __init__(self):
        super(HDFStoreAttribute, self).__init__()

    @property
    def site_coords(self):
        return self._get_site_coords()

    @property
    def time(self):
        return self._get_time()

    @property
    def fixed_dim_and_angle(self):
        return self._get_fixed_dim_and_angle()

    @property
    def range(self):
        return self._get_range()

    @property
    def what(self):
        return self._get_dset_what()


class OdimStoreAttributeMixin(HDFStoreAttribute):
    def __init__(self):
        super(OdimStoreAttributeMixin, self).__init__()

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
            if angle is None:
                dim = "azimuth"
                angle = root[grp]["where"].attrs["elangle"]

            angle = np.round(angle, decimals=1)
        return dim, angle

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
            time_data = self._get_time_how()
            self._need_time_recalc = False
        except (AttributeError, KeyError, TypeError):
            time_data = self._get_time_what(nrays=nrays)
            self._need_time_recalc = True
        return time_data

    def _get_range(self):
        with self._manager.acquire_context(False) as root:
            grp = self._group.split("/")[0]
            where = root[grp]["where"].attrs
            ngates = where["nbins"]
            range_start = where["rstart"] * 1000.0
            bin_range = where["rscale"]
            cent_first = range_start + bin_range / 2.0
            range_data = np.arange(
                cent_first, range_start + bin_range * ngates, bin_range, dtype="float32"
            )
        return range_data, cent_first, bin_range

    def _get_time(self, point="start"):
        with self._manager.acquire_context(False) as root:
            grp = self._group.split("/")[0]
            what = root[grp]["what"].attrs
            startdate = what[f"{point}date"].item().decode()
            starttime = what[f"{point}time"].item().decode()
            start = dt.datetime.strptime(startdate + starttime, "%Y%m%d%H%M%S")
            start = start.replace(tzinfo=dt.timezone.utc).timestamp()
        return start

    def _get_a1gate(self):
        with self._manager.acquire_context(False) as root:
            grp = self._group.split("/")[0]
            a1gate = root[grp]["where"].attrs["a1gate"]
        return a1gate

    def _get_site_coords(self):
        with self._manager.acquire_context(False) as root:
            lon = root["where"].attrs["lon"]
            lat = root["where"].attrs["lat"]
            alt = root["where"].attrs["height"]
        return lon, lat, alt

    def _get_dset_what(self):
        with self._manager.acquire_context(False) as root:
            attrs = {}
            what = root[self._group]["what"].attrs
            attrs["scale_factor"] = what["gain"]
            attrs["add_offset"] = what["offset"]
            attrs["_FillValue"] = what["nodata"]
            attrs["_Undetect"] = what["undetect"]
            attrs["quantity"] = maybe_decode_bytes(what["quantity"])
        return attrs

    @property
    def a1gate(self):
        return self._get_a1gate()

    @property
    def azimuth(self):
        try:
            azimuth = self._get_azimuth_how()
        except (AttributeError, KeyError, TypeError):
            azimuth = self._get_azimuth_where()
        return azimuth

    @property
    def elevation(self):
        try:
            elevation = self._get_elevation_how()
        except (AttributeError, KeyError, TypeError):
            elevation = self._get_elevation_where()
        return elevation

    @property
    def ray_times(self):
        return self._get_ray_times()


class OdimStore(H5NetCDFStore, OdimStoreAttributeMixin):
    """Store for reading and writing ODIM data via h5netcdf"""

    def open_store_variable(self, name, var):

        dimensions, data, attrs, encoding = _prepare_open_variable(self, name, var)

        # cheat attributes
        if "data" in name:
            encoding["group"] = self._group
            attrs = self.what
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

        return {name: Variable(dimensions, data, attrs, encoding)}

    def open_store_coordinates(self):
        from .xarray import az_attrs, el_attrs, range_attrs, time_attrs

        azimuth = self.azimuth
        elevation = self.elevation
        a1gate = self.a1gate
        rtime = self.ray_times
        dim, angle = self.fixed_dim_and_angle
        angle_res = np.round(np.nanmedian(np.diff(locals()[dim])), decimals=1)

        dims = ("azimuth", "elevation")
        if dim == dims[1]:
            dims = (dims[1], dims[0])

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

        range_data, cent_first, bin_range = self.range
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

        lon, lat, alt = self.site_coords

        coordinates = dict(
            azimuth=Variable((dims[0],), azimuth, az_attrs),
            elevation=Variable((dims[0],), elevation, el_attrs),
            rtime=Variable((dims[0],), rtime, rtime_attrs),
            range=Variable(("range",), range_data, range_attrs),
            time=Variable((), self.time, time_attrs),
            sweep_mode=Variable((), sweep_mode),
            longitude=Variable((), lon, lon_attrs),
            latitude=Variable((), lat, lat_attrs),
            altitude=Variable((), alt, alt_attrs),
        )

        return coordinates

    def get_variables(self):
        return FrozenDict(
            (k1, v1)
            for k, v in self.ds.variables.items()
            for k1, v1 in {
                **self.open_store_variable(k, v),
                **self.open_store_coordinates(),
            }.items()
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
            vars_idx = np.argsort([int(v[len("data") :]) for v in variables])
            variables = np.array(variables)[vars_idx].tolist()
            print(variables)

            ds_list = ["/".join([store._group, var]) for var in variables]
            print(ds_list)
        store.close()

        stores = [
            OdimStore.open(
                filename_or_obj,
                format=format,
                group=grp,
                lock=lock,
                invalid_netcdf=invalid_netcdf,
                phony_dims=phony_dims,
                decode_vlen_strings=decode_vlen_strings,
            )
            for grp in ds_list
        ]

        store_entrypoint = StoreBackendEntrypoint()

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
                ).pipe(_reindex_angle, store=store)
                for store in stores
            ]
        )

        return ds


class GamicStoreAttributeMixin(HDFStoreAttribute):
    def __init__(self):
        super(GamicStoreAttributeMixin, self).__init__()

    def _get_fixed_dim_and_angle(self):
        with self._manager.acquire_context(False) as root:
            how = root[self._group]["how"].attrs
            dims = {0: "elevation", 1: "azimuth"}
            try:
                dim = 1
                angle = np.round(how[dims[0]], decimals=1)
            except KeyError:
                dim = 0
                angle = np.round(how[dims[1]], decimals=1)

        return dims[dim], angle

    def _get_range(self):
        with self._manager.acquire_context(False) as root:
            how = root[self._group]["how"].attrs
            range_samples = how["range_samples"]
            range_step = how["range_step"]
            ngates = how["bin_count"]
            bin_range = range_step * range_samples
            cent_first = bin_range / 2.0
            range_data = np.arange(
                cent_first,
                bin_range * ngates,
                bin_range,
                dtype="float32",
            )
        return range_data, cent_first, bin_range

    def _get_time(self):
        with self._manager.acquire_context(False) as root:
            start = root[self._group]["how"].attrs["timestamp"]
            import dateutil

            start = dateutil.parser.parse(start)
            start = start.replace(tzinfo=dt.timezone.utc).timestamp()
        return start

    def _get_site_coords(self):
        with self._manager.acquire_context(False) as root:
            lon = root["where"].attrs["lon"]
            lat = root["where"].attrs["lat"]
            alt = root["where"].attrs["height"]
        return lon, lat, alt

    def _get_dset_what(self):
        with self._manager.acquire_context(False) as root:
            attrs = {}
            what = root[self._group]["what"].attrs
            attrs["scale_factor"] = what["gain"]
            attrs["add_offset"] = what["offset"]
            attrs["_FillValue"] = what["nodata"]
            attrs["_Undetect"] = what["undetect"]
            attrs["quantity"] = maybe_decode_bytes(what["quantity"])
        return attrs


def _prepare_open_variable(self, name, var):
    import h5py

    dim, _ = self._get_fixed_dim_and_angle()
    dims = var.dimensions
    dimensions = []
    for n, _ in enumerate(dims):
        if n == 0:
            dimensions.append(dim)
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

    return dimensions, data, attrs, encoding


def _get_ray_header_data(self, name, var):

    dimensions, data, attrs, encoding = _prepare_open_variable(self, name, var)

    recarray = Variable(dimensions, data, attrs, encoding)
    ray_header = {
        recname: recarray.values[recname] for recname in recarray.values.dtype.names
    }

    azstart = ray_header["azimuth_start"]
    azstop = ray_header["azimuth_stop"]
    # todo: RHI
    # if self._dim0[0] == "azimuth":
    zero_index = np.where(azstop < azstart)
    azstop[zero_index[0]] += 360
    azimuth = (azstart + azstop) / 2.0

    elstart = ray_header["elevation_start"]
    elstop = ray_header["elevation_stop"]
    elevation = (elstart + elstop) / 2.0

    rtime = ray_header["timestamp"] / 1e6

    return azimuth, elevation, rtime


class GamicStore(H5NetCDFStore, GamicStoreAttributeMixin):
    """Store for reading GAMIC data via h5netcdf"""

    def open_store_variable(self, name, var):

        # fix moment attributes
        if "moment" in name:
            dimensions, data, attrs, encoding = _prepare_open_variable(self, name, var)

            encoding["group"] = self._group
            name = attrs.pop("moment").lower()
            try:
                name = GAMIC_NAMES[name]
            except KeyError:
                # ds = ds.drop_vars(mom)
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
            attrs[
                "coordinates"
            ] = "elevation azimuth range latitude longitude altitude time rtime sweep_mode"
        elif "ray_header" in name:
            variables = self.open_store_coordinates(name, var)
            return variables
        else:
            return {}

        return {name: Variable(dimensions, data, attrs, encoding)}

    def open_store_coordinates(self, name, var):
        from .xarray import az_attrs, el_attrs, range_attrs, time_attrs

        azimuth, elevation, rtime = _get_ray_header_data(self, name, var)
        dim, angle = self.fixed_dim_and_angle
        angle_res = np.round(np.nanmedian(np.diff(locals()[dim])), decimals=1)
        # print(dim, angle)
        dims = ("azimuth", "elevation")
        if dim == dims[1]:
            dims = (dims[1], dims[0])

        sort_idx = np.argsort(locals()[dim])
        # print(sort_idx)
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

        range_data, cent_first, bin_range = self.range
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

        lon, lat, alt = self.site_coords

        coordinates = dict(
            azimuth=Variable((dims[0],), azimuth, az_attrs),
            elevation=Variable((dims[0],), elevation, el_attrs),
            rtime=Variable((dims[0],), rtime, rtime_attrs),
            range=Variable(("range",), range_data, range_attrs),
            time=Variable((), self.time, time_attrs),
            sweep_mode=Variable((), sweep_mode),
            longitude=Variable((), lon, lon_attrs),
            latitude=Variable((), lat, lat_attrs),
            altitude=Variable((), alt, alt_attrs),
        )

        return coordinates

    def get_variables(self):
        return FrozenDict(
            (k1, v1)
            for k, v in self.ds.variables.items()
            for k1, v1 in {
                **self.open_store_variable(k, v),
            }.items()
        )

    def get_attrs(self):
        dim, angle = self.fixed_dim_and_angle
        attributes = {}
        attributes["fixed_angle"] = angle
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


def get_group_names(filename, groupname):
    with h5netcdf.File(filename, "r") as fh:
        groups = [grp for grp in fh.groups if groupname in grp]
    return groups


class XRadVolume(XRadBase):
    """Class for holding a volume of radar sweeps"""

    def __init__(self, **kwargs):
        super(XRadVolume, self).__init__()
        self._data = None
        self._root = None
        self._dims = dict(azimuth="elevation", elevation="azimuth")

    def __repr__(self):
        summary = ["<wradlib.{}>".format(type(self).__name__)]
        dims = "Dimension(s):"
        dims_summary = f"sweep: {len(self)}"
        summary.append("{} ({})".format(dims, dims_summary))
        dim = list(self[0].dims.keys())[0]
        angle = f"{self._dims[dim].capitalize()}(s):"
        angle_summary = [f"{v.attrs['fixed_angle']:.1f}" for v in self]
        angle_summary = ", ".join(angle_summary)
        summary.append("{} ({})".format(angle, angle_summary))

        return "\n".join(summary)

    @property
    def root(self):
        """Return root object."""
        if self._root is None:
            self.assign_root()
        return self._root

    def assign_root(self):
        """(Re-)Create root object according CfRadial2 standard"""
        # assign root variables
        sweep_group_names = [f"sweep_{i}" for i in range(len(self))]

        sweep_fixed_angles = [ts.attrs["fixed_angle"] for ts in self]

        # extract time coverage
        times = np.array(
            [[ts.rtime.values.min(), ts.rtime.values.max()] for ts in self]
        ).flatten()
        time_coverage_start = min(times)
        time_coverage_end = max(times)

        time_coverage_start_str = str(time_coverage_start)[:19] + "Z"
        time_coverage_end_str = str(time_coverage_end)[:19] + "Z"

        # create root group from scratch
        root = Dataset()  # data_vars=wrl.io.xarray.global_variables,
        # attrs=wrl.io.xarray.global_attrs)

        # take first dataset/file for retrieval of location
        # site = self.site

        # assign root variables
        root = root.assign(
            {
                "volume_number": 0,
                "platform_type": str("fixed"),
                "instrument_type": "radar",
                "primary_axis": "axis_z",
                "time_coverage_start": time_coverage_start_str,
                "time_coverage_end": time_coverage_end_str,
                "latitude": self[0]["latitude"],
                "longitude": self[0]["longitude"],
                "altitude": self[0]["altitude"],
                "sweep_group_name": (["sweep"], sweep_group_names),
                "sweep_fixed_angle": (["sweep"], sweep_fixed_angles),
            }
        )

        # assign root attributes
        attrs = {}
        attrs.update(
            {
                "version": "None",
                "title": "None",
                "institution": "None",
                "references": "None",
                "source": "None",
                "history": "None",
                "comment": "im/exported using wradlib",
                "instrument_name": "None",
            }
        )
        # attrs["version"] = self[0].attrs["version"]
        root = root.assign_attrs(attrs)
        # todo: pull in only CF attributes
        root = root.assign_attrs(self[0].attrs)
        self._root = root

    @property
    def site(self):
        """Return coordinates of radar site."""
        return self[0][["latitude", "longitude", "altitude"]]

    @property
    def Conventions(self):
        """Return Conventions string."""
        try:
            conv = self[0].attrs["Conventions"]
        except KeyError:
            conv = None
        return conv

    def to_odim(self, filename, timestep=0):
        """Save volume to ODIM_H5/V2_2 compliant file.

        Parameters
        ----------
        filename : str
            Name of the output file
        timestep : int
            timestep of wanted volume
        """
        if self.root:
            to_odim(self, filename, timestep=timestep)
        else:
            warnings.warn(
                "WRADLIB: No OdimH5-compliant data structure " "available. Not saving.",
                UserWarning,
            )

    def to_cfradial2(self, filename, timestep=0):
        """Save volume to CfRadial2 compliant file.

        Parameters
        ----------
        filename : str
            Name of the output file
        timestep : int
            timestep wanted volume
        """
        if self.root:
            to_cfradial2(self, filename, timestep=timestep)
        else:
            warnings.warn(
                "WRADLIB: No CfRadial2-compliant data structure "
                "available. Not saving.",
                UserWarning,
            )

    def to_netcdf(self, filename, timestep=None, keys=None):
        """Save volume to netcdf compliant file.

        Parameters
        ----------
        filename : str
            Name of the output file
        timestep : int, slice
            timestep/slice of wanted volume
        keys : list
            list of sweep_group_names which should be written to the file
        """
        if self.root:
            to_netcdf(self, filename, keys=keys, timestep=timestep)
        else:
            warnings.warn(
                "WRADLIB: No netcdf-compliant data structure " "available. Not saving.",
                UserWarning,
            )


def open_dataset(filename_or_obj, **kwargs):

    engine = kwargs.pop("engine")

    if "gamic" in engine:
        groupname = "scan"
    elif "odim" in engine:
        groupname = "dataset"
    else:
        raise ValueError(
            f"wradlib: groupname {groupname} not allowed for engine {engine}."
        )

    group = kwargs.pop("group", None)
    if group is None:
        group = get_group_names(filename_or_obj, groupname)
    elif isinstance(group, str):
        group = [group]
    else:
        pass

    ds = [
        xarray.open_dataset(filename_or_obj, engine=engine, group=grp, **kwargs)
        for grp in group
    ]

    if len(ds) > 1:
        vol = XRadVolume()
        vol.extend(ds)
        vol.sort(key=lambda x: x.time.min().values)
        ds = vol
    else:
        ds = ds[0]

    return ds


def open_mfdataset(paths, **kwargs):

    if isinstance(paths, str):
        if is_remote_uri(paths):
            raise ValueError(
                "cannot do wild-card matching for paths that are remote URLs: "
                "{!r}. Instead, supply paths as an explicit list of strings.".format(
                    paths
                )
            )
        paths = sorted(glob.glob(paths))
    else:
        paths = [str(p) if isinstance(p, Path) else p for p in paths]

    engine = kwargs.pop("engine")

    if "gamic" in engine:
        groupname = "scan"
    elif "odim" in engine:
        groupname = "dataset"
    else:
        raise ValueError(
            f"wradlib: groupname {groupname} not allowed for engine {engine}."
        )

    group = kwargs.pop("group", None)
    if group is None:
        group = get_group_names(paths[0], groupname)
    elif isinstance(group, str):
        group = [group]
    else:
        pass

    concat_dim = kwargs.pop("concat_dim", "time")
    combine = kwargs.pop("combine", "nested")

    ds = [
        xarray.open_mfdataset(
            paths,
            engine=engine,
            group=grp,
            concat_dim=concat_dim,
            combine=combine,
            **kwargs,
        )
        for grp in tqdm(group)
    ]

    if len(ds) > 1:
        vol = XRadVolume()
        vol.extend(ds)
        vol.sort(key=lambda x: x.time.min().values)
        ds = vol
    else:
        ds = ds[0]

    return ds


# def _open_odim_sweep1(filename, **kwargs):
#     """Returns list of XRadSweep objects
#
#     Every sweep will be put into it's own class instance.
#     """
#     # open file
#     ds = open_dataset(filename, **kwargs)
#
#     # iterate over single sweeps
#     # todo: if sorting does not matter, we can skip this
#     sweeps = [k for k in groups if dsdesc in k]
#     sweeps_idx = np.argsort([int(s[len(dsdesc) :]) for s in sweeps])
#     sweeps = np.array(sweeps)[sweeps_idx].tolist()
#     return [sweep_cls(handle, k, **kwargs) for k in sweeps]
#
#
# def open_odim(paths, **kwargs):
#     """Open multiple ODIM files as a XRadVolume structure.
#
#     Parameters
#     ----------
#     paths : str or sequence
#         Either a filename or string glob in the form `'path/to/my/files/*.h5'`
#         or an explicit list of files to open.
#
#     loader : {'netcdf4', 'h5py', 'h5netcdf'}
#         Loader used for accessing file metadata, defaults to 'netcdf4'.
#
#     kwargs : optional
#         Additional arguments passed on to :py:class:`wradlib.io.XRadSweep`.
#     """
#     import glob
#     if isinstance(paths, str):
#         paths = glob.glob(paths)
#     else:
#         paths = np.array(paths).flatten().tolist()
#
#     sweeps = []
#     [
#         sweeps.extend(_open_odim_sweep1(f, **kwargs))
#         for f in tqdm(paths, desc="Open", unit=" Files", leave=None)
#     ]
#     angles = collect_by_angle(sweeps)
#     for i in tqdm(range(len(angles)), desc="Collecting", unit=" Angles", leave=None):
#         angles[i] = collect_by_time(angles[i])
#     angles.sort(key=lambda x: x[0].time)
#     for f in angles:
#         f._parent = angles
#     angles._ncfile = angles[0].ncfile
#     angles._ncpath = "/"
#     return angles
