#!/usr/bin/env python
# Copyright (c) 2022, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

"""
Furuno binary Data I/O
^^^^^^^^^^^^^^^^^^^^^^

Reads data from Furuno's SCNX data formats

To read from Furuno SCNX files :class:`numpy:numpy.memmap` is used to get access to
the data. The Furuno header is read in any case into dedicated OrderedDict's.
Reading sweep data can be skipped by setting `loaddata=False`.
By default the data is decoded on the fly.
Using `rawdata=True` the data will be kept undecoded.

.. autosummary::
   :nosignatures:
   :toctree: generated/

   {}
"""
__all__ = [
    "FurunoHeaderBase",
    "FurunoMainHeader",
    "FurunoFile",
]
__doc__ = __doc__.format("\n   ".join(__all__))

import contextlib
import copy
import datetime as dt
import io
import struct
import warnings
from collections import OrderedDict

import numpy as np

from wradlib.io.xarray import (
    open_radar_dataset,
    open_radar_mfdataset,
    raise_on_missing_xarray_backend,
)
from wradlib.io.iris import (_unpack_dictionary, _get_fmt_string,
                             UINT1, UINT2, UINT4, SINT2, SINT4)

def decode_time(data):
    """Decode `YMDS_TIME` into datetime object."""
    time = _unpack_dictionary(data, YMDS_TIME)
    print(time)
    try:
        t = dt.datetime(time["year"], time["month"], time["day"],
                        time["hour"], time["minute"], time["second"])
        return t
    except ValueError:
        return None

YMDS_TIME = OrderedDict(
    [
        ("year", UINT2),
        ("month", UINT1),
        ("day", UINT1),
        ("hour", UINT1),
        ("minute", UINT1),
        ("second", UINT1),
        ("spare", {"fmt": "1s"}),
    ]
)

LEN_YMDS_TIME = struct.calcsize(_get_fmt_string(YMDS_TIME))
_YMDS_TIME = {"size": f"{LEN_YMDS_TIME}s", "func": decode_time, "fkw": {}}


MAIN_HEADER = OrderedDict(
    [
        ("size_of_header", UINT2),
        ("format_version", UINT2),
        ("scan_start_time", _YMDS_TIME),
        ("scan_stop_time", _YMDS_TIME),
        ("time_zone", SINT2),
        ("product_number", UINT2),
        ("model_type", UINT2),
        ("latitude", SINT4),
        ("longitude", SINT4),
        ("altitude", SINT4),
        ("azimuth_offset", UINT2),
        ("tx_frequency", UINT4),
        ("polarization_mode", UINT2),
        ("antenna_gain_h", UINT2),
        ("antenna_gain_v", UINT2),
        ("half_power_beam_width_h", UINT2),
        ("half_power_beam_width_v", UINT2),
        ("tx_power_h", UINT2),
        ("tx_power_v", UINT2),
        ("radar_constant_h", SINT2),
        ("radar_constant_v", SINT2),
        ("noise_power_short_pulse_h", SINT2),
        ("noise_power_long_pulse_h", SINT2),
        ("threshold_power_short_pulse", SINT2),
        ("threshold_power_long_pulse", SINT2),
        ("tx_pulse_specification", UINT2),
        ("prf_mode", UINT2),
        ("prf_1", UINT2),
        ("prf_2", UINT2),
        ("prf_3", UINT2),
        ("nyquist_velocity", UINT2),
        ("sample_number", UINT2),
        ("tx_pulse_blind_length", UINT2),
        ("short_pulse_width", UINT2),
        ("short_pulse_modulation_bandwidth", UINT2),
        ("long_pulse_width", UINT2),
        ("long_pulse_modulation_bandwidth", UINT2),
        ("pulse_switch_point", UINT2),
        ("observation_mode", UINT2),
        ("antenna_rotation_speed", UINT2),
        ("number_sweep_direction_data", UINT2),
        ("number_range_direction_data", UINT2),
        ("resolution_range_direction", UINT2),
        ("current_scan_number", UINT2),
        ("total_number_scans_volume", UINT2),
        ("rainfall_intensity_estimation_method", UINT2),
        ("z_r_coefficient_b", UINT2),
        ("z_r_coefficient_beta", UINT2),
        ("kdp_r_coefficient_a", UINT2),
        ("kdp_r_coefficient_b", UINT2),
        ("kdp_r_coefficient_c", UINT2),
        ("zh_attenuation_correction_method", UINT2),
        ("zh_attenuation_coefficient_b1", UINT2),
        ("zh_attenuation_coefficient_b2", UINT2),
        ("zh_attenuation_coefficient_d1", UINT2),
        ("zh_attenuation_coefficient_d2", UINT2),
        ("air_attenuation_one_way", UINT2),
        ("output_threshold_rain", UINT2),
        ("record_item", UINT2),
        ("signal_processing_flag", UINT2),
        ("clutter_reference_file", _YMDS_TIME),
        ("reserved", {"fmt": "8s"}),
    ]
)

LEN_MAIN_HEADER = struct.calcsize(_get_fmt_string(MAIN_HEADER))

# ds.filepos
# dlen = 936 * 2
# moff = 0
# start = 156
# rr = ds._fh[start:].view(dtype="uint16").reshape(722, -1)#[start+6+moff:start+722*(dlen*8+6):8*dlen+6]#.view(dtype=np.uint16).reshape(722,936)
# rr = rr[:, 4:].reshape(722, 9, 936)
# rr.shape
# wrl.vis.plot_ppi(rr[:, 1, :]/100 - 32768/100, vmin=0, vmax=50, cmap="turbo")

class FurunoHeaderBase:
    """Base Class for Furuno Headers."""

    def __init__(self, **kwargs):
        super().__init__()

    def init_header(self):
        pass


class FurunoMainHeader(FurunoHeaderBase):
    """Furuno Main Header class."""

    len = LEN_MAIN_HEADER
    structure = MAIN_HEADER
    name = "_main_header"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._main_header = None

    @property
    def header(self):
        """Returns ingest_header dictionary."""
        return self._main_header

    @property
    def version(self):
        return self.header["format_version"]

    @property
    def site_coords(self):
        return (
            self.header["longitude"]/1e5,
            self.header["latitude"] / 1e5,
            self.header["altitude"] / 1e2,
        )


class FurunoFileBase:
    """Base class for Iris Files."""

    def __init__(self, **kwargs):
        super().__init__()


class FurunoFile(FurunoFileBase, FurunoMainHeader):
    """IrisFile class"""

    identifier = ["MAIN_HEADER"]

    def __init__(self, filename, **kwargs):
        self._debug = kwargs.get("debug", False)
        self._rawdata = kwargs.get("rawdata", False)
        self._loaddata = kwargs.get("loaddata", True)
        self._fp = None
        self._filename = filename
        if isinstance(filename, str):
            self._fp = open(filename, "rb")
            self._fh = np.memmap(self._fp, mode="r")
        else:
            if isinstance(filename, io.BytesIO):
                filename.seek(0)
                filename = filename.read()
            self._fh = np.frombuffer(filename, dtype=np.uint8)
        self._filepos = 0
        self._data = None
        super().__init__(**kwargs)
        # read first structure header
        self.get_header(FurunoMainHeader)
        self._filepos = 0
        if self._loaddata:
            self.get_data()

    def get_data(self):
        if self._data is None:
            moments = ["RR", "DBZH", "VRADH", "ZDR", "KDP", "PHIDP", "RHOHV", "WRADH",
                       "QUAL", "RES1", "RES2", "RES3", "RES4", "RES5", "RES6", "FIX"]
            items = dict()
            for i in range(9):
                if (self.header["record_item"] & 2 ** i) == 2 ** i:
                    items[i] = moments[i]
            rays = self.header["number_sweep_direction_data"]
            rng = self.header["number_range_direction_data"]
            start = 156
            cnt = len(items)
            data = self._fh[start:].view(dtype="uint16").reshape(rays, -1)[:, 4:].reshape(rays, cnt, rng)
            self._data = dict()
            for i in range(cnt):
                self._data[items[i]] = data[:, i , :]
        return self._data

    def close(self):
        if self._fp is not None:
            self._fp.close()

    __del__ = close

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    @property
    def data(self):
        return self._data

    @property
    def loaddata(self):
        """Returns `loaddata` switch."""
        return self._loaddata

    @property
    def rawdata(self):
        """Returns `rawdata` switch."""
        return self._rawdata

    @property
    def debug(self):
        return self._debug

    @property
    def filename(self):
        return self._filename

    @property
    def first_dimension(self):
        obs_mode = self.header["observation_mode"]
        if obs_mode in [1, 3, 4]:
            return "azimuth"
        elif obs_mode == 2:
            return "elevation"
        else:
            raise TypeError(f"Unknown Furuno Observation Mode: {obs_mode}")

    @property
    def fixed_angle(self):
        return 0

    @property
    def fh(self):
        return self._fh

    @property
    def filepos(self):
        return self._filepos

    @filepos.setter
    def filepos(self, pos):
        self._filepos = pos

    def read_from_file(self, size):
        """Read from file.

        Parameters
        ----------
        size : int
            Number of data words to read.

        Returns
        -------
        data : array-like
            numpy array of data
        """
        start = self._filepos
        self._filepos += size
        return self._fh[start : self._filepos]

    def get_header(self, header):
        head = _unpack_dictionary(
            self.read_from_file(header.len), header.structure, self._rawdata
        )
        setattr(self, header.name, head)
        header.init_header(self)


from xarray.backends.common import (
    AbstractDataStore,
    BackendArray,
    BackendEntrypoint,
    find_root_and_group,
)
from xarray.backends.file_manager import CachingFileManager, DummyFileManager
from wradlib.io.furuno import FurunoFile
from xarray.backends.store import StoreBackendEntrypoint
from xarray.core.utils import Frozen, FrozenDict, close_on_error, is_remote_uri
from xarray.core import indexing
from xarray.core.variable import Variable
from wradlib.io.xarray import (
    _assign_data_radial,
    _assign_data_radial2,
    _fix_angle,
    _GamicH5NetCDFMetadata,
    _get_gamic_variable_name_and_attrs,
    _get_odim_variable_name_and_attrs,
    _OdimH5NetCDFMetadata,
    _reindex_angle,
    az_attrs,
    el_attrs,
    iris_mapping,
    moment_attrs,
    moments_mapping,
    rainbow_mapping,
    range_attrs,
    time_attrs,
)


class FurunoArrayWrapper(BackendArray):
    def __init__(
            self,
            data,
    ):
        self.data = data
        self.shape = data.shape
        self.dtype = np.uint16

    def __getitem__(self, key: tuple):
        return xr.core.indexing.explicit_indexing_adapter(
            key,
            xr.core.indexing.IndexingSupport.BASIC,
            self._raw_indexing_method,
        )

    def _raw_indexing_method(self, key: tuple):
        return self.data[key]


class FurunoStore(AbstractDataStore):
    """Store for reading RAINBOW5 sweeps via wradlib."""

    def __init__(self, manager, group=None):

        self._manager = manager
        self._group = group
        self._filename = self.filename
        self._need_time_recalc = False

    @classmethod
    def open(cls, filename, mode="r", group=None, **kwargs):
        manager = CachingFileManager(FurunoFile, filename, mode=mode, kwargs=kwargs)
        return cls(manager, group=group)

    @property
    def filename(self):
        with self._manager.acquire_context(False) as root:
            return root.filename

    @property
    def root(self):
        with self._manager.acquire_context(False) as root:
            return root

    def _acquire(self, needs_lock=True):
        with self._manager.acquire_context(needs_lock) as root:
            return root
            # ds = root#.header["scan"]["slice"][self._group]
            # except KeyError:
            #    ds = root.header["scan"]["slice"]
        # return ds

    @property
    def ds(self):
        return self._acquire()

    def open_store_variable(self, name, var):
        print(name)
        dim = self.root.first_dimension

        data = indexing.LazilyOuterIndexedArray(FurunoArrayWrapper(var))
        encoding = {"group": self._group}
        if name == "PHIDP":
            add_offset = 360 * -32768 / 65535
            scale_factor = 360 / 65535
        elif name == "RHOHV":
            add_offset = 2 * -1 / 65534
            scale_factor = 2 / 65534
        elif name == "WRADH":
            add_offset = -1e-2
            scale_factor = 1e-2
        elif name == "QUAL":
            add_offset = 0
            scale_factor = 1
        else:
            add_offset = -327.68
            scale_factor = 1e-2

        mapping = moments_mapping.get(name, {})
        attrs = {key: mapping[key] for key in moment_attrs if key in mapping}

        attrs["add_offset"] = add_offset
        attrs["scale_factor"] = scale_factor
        attrs["_FillValue"] = 0
        # attrs[
        #    "coordinates"
        # ] = "elevation azimuth range latitude longitude altitude time rtime sweep_mode"
        print(attrs)
        print(Variable((dim, "range"), data, attrs, encoding))
        return Variable((dim, "range"), data, attrs, encoding)

    #     def open_store_coordinates(self, var):

    #         dim = self.root.first_dimension
    #         ray = var["slicedata"]["rayinfo"]

    #         if not isinstance(ray, list):
    #             var["slicedata"]["rayinfo"] = [ray]
    #             ray = var["slicedata"]["rayinfo"]

    #         start = next(filter(lambda x: x["@refid"] == "startangle", ray), False)
    #         start_idx = ray.index(start)
    #         stop = next(filter(lambda x: x["@refid"] == "stopangle", ray), False)

    #         anglestep = self.root._get_rbdict_value(var, "anglestep", dtype=float)
    #         antdirection = self.root._get_rbdict_value(
    #             var, "antdirection", default=0, dtype=bool
    #         )

    #         encoding = {"group": self._group}
    #         startangle = indexing.LazilyOuterIndexedArray(
    #             RainbowArrayWrapper(self, start_idx, start)
    #         )

    #         step = anglestep
    #         # antdirection == True ->> negative angles
    #         # antdirection == False ->> positive angles
    #         if antdirection:
    #             step = -anglestep

    #         if dim == "azimuth":
    #             startaz = Variable((dim,), startangle, az_attrs, encoding)

    #             if stop:
    #                 stop_idx = ray.index(stop)
    #                 stopangle = indexing.LazilyOuterIndexedArray(
    #                     RainbowArrayWrapper(self, stop_idx, stop)
    #                 )
    #                 stopaz = Variable((dim,), stopangle, az_attrs, encoding)
    #                 zero_index = np.where(startaz - stopaz > 5)
    #                 stopazv = stopaz.values
    #                 stopazv[zero_index[0]] += 360
    #                 azimuth = (startaz + stopazv) / 2.0
    #                 azimuth[azimuth >= 360] -= 360
    #             else:
    #                 azimuth = startaz + step / 2.0

    #             elevation = np.ones_like(azimuth) * float(var["posangle"])
    #         else:
    #             startel = Variable((dim,), startangle, el_attrs, encoding)

    #             if stop:
    #                 stop_idx = ray.index(stop)
    #                 stopangle = indexing.LazilyOuterIndexedArray(
    #                     RainbowArrayWrapper(self, stop_idx, stop)
    #                 )
    #                 stopel = Variable((dim,), stopangle, el_attrs, encoding)
    #                 elevation = (startel + stopel) / 2.0
    #             else:
    #                 elevation = startel + step / 2.0

    #             azimuth = np.ones_like(elevation) * float(var["posangle"])

    #         dstr = var["slicedata"]["@date"]
    #         tstr = var["slicedata"]["@time"]

    #         timestr = f"{dstr}T{tstr}Z"
    #         time = dt.datetime.strptime(timestr, "%Y-%m-%dT%H:%M:%SZ")
    #         total_seconds = (time - dt.datetime(1970, 1, 1)).total_seconds()

    #         # range is in km
    #         start_range = self.root._get_rbdict_value(
    #             var, "startrange", default=0, dtype=float
    #         )
    #         start_range *= 1000.0

    #         stop_range = self.root._get_rbdict_value(var, "stoprange", dtype=float)
    #         stop_range *= 1000.0

    #         range_step = self.root._get_rbdict_value(var, "rangestep", dtype=float)
    #         range_step *= 1000.0
    #         rng = np.arange(
    #             start_range + range_step / 2,
    #             stop_range + range_step / 2,
    #             range_step,
    #             dtype="float32",
    #         )[: int(var["slicedata"]["rawdata"]["@bins"])]

    #         range_attrs["meters_to_center_of_first_gate"] = start_range + range_step / 2
    #         range_attrs["meters_between_gates"] = range_step

    #         # making-up ray times
    #         antspeed = self.root._get_rbdict_value(var, "antspeed", dtype=float)
    #         raytime = anglestep / antspeed
    #         raytimes = np.array(
    #             [
    #                 dt.timedelta(seconds=x * raytime).total_seconds()
    #                 for x in range(azimuth.shape[0] + 1)
    #             ]
    #         )

    #         diff = np.diff(raytimes) / 2.0
    #         rtime = raytimes[:-1] + diff
    #         rtime_attrs = {
    #             "units": f"seconds since {time.isoformat()}Z",
    #             "standard_name": "time",
    #         }

    #         rng = Variable(("range",), rng, range_attrs)
    #         azimuth = Variable((dim,), azimuth, az_attrs, encoding)
    #         elevation = Variable((dim,), elevation, el_attrs, encoding)
    #         rtime = Variable((dim,), rtime, rtime_attrs, encoding)
    #         time = Variable((), total_seconds, time_attrs, encoding)

    #         # get coordinates from RainbowFile
    #         sweep_mode = "azimuth_surveillance" if dim == "azimuth" else "rhi"
    #         lon_attrs = {
    #             "long_name": "longitude",
    #             "units": "degrees_east",
    #             "standard_name": "longitude",
    #         }
    #         lat_attrs = {
    #             "long_name": "latitude",
    #             "units": "degrees_north",
    #             "positive": "up",
    #             "standard_name": "latitude",
    #         }
    #         alt_attrs = {
    #             "long_name": "altitude",
    #             "units": "meters",
    #             "standard_name": "altitude",
    #         }
    #         lon, lat, alt = self.root.site_coords

    #         coords = {
    #             "azimuth": azimuth,
    #             "elevation": elevation,
    #             "range": rng,
    #             "time": time,
    #             "rtime": rtime,
    #             "longitude": Variable((), lon, lon_attrs),
    #             "latitude": Variable((), lat, lat_attrs),
    #             "altitude": Variable((), alt, alt_attrs),
    #             "sweep_mode": Variable((), sweep_mode),
    #         }

    #         # a1gate, this might be off by 1 if reindexing is applied
    #         if dim == "azimuth":
    #             a1gate = np.argmin(azimuth[::-1].values)
    #         else:
    #             a1gate = np.argmin(elevation[::-1].values)
    #         coords[dim].attrs["a1gate"] = a1gate
    #         # angle_res
    #         coords[dim].attrs["angle_res"] = anglestep
    #         return coords

    def get_variables(self):
        return FrozenDict(
            # (k1, v1)
            # for k1, v1 in dict(
            (k, self.open_store_variable(k, v))
            for k, v in self.ds.data.items() if k != "QUAL"
            # **self.open_store_coordinates(self.ds),
            #            }.items()
        )

    def get_attrs(self):
        attributes = {"fixed_angle": float(self.ds.fixed_angle)}
        return FrozenDict(attributes)


class FurunoBackendEntrypoint(BackendEntrypoint):
    """Xarray BackendEntrypoint for Rainbow5 data."""

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
            group=None,
            reindex_angle=None,
    ):
        store = FurunoStore.open(
            filename_or_obj,
            group=group,
            loaddata=True,
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

        # if decode_coords and reindex_angle is not False:
        #    ds = ds.pipe(_reindex_angle, store=store, tol=reindex_angle)

        return ds