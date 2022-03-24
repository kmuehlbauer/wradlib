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
        ("record_item", UINT2),
        ("signal_processing_flag", UINT2),
        ("missing", UINT2),
        ("clutter_reference_file", _YMDS_TIME),
        ("reserved", {"fmt": "8s"}),
    ]
)

ANGLE_HEADER = OrderedDict(
    [
        ("angle_header_size", UINT2),
        ("azimuth_angle", UINT2),
        ("elevation_angle", UINT2),
    ]
)

LEN_MAIN_HEADER = struct.calcsize(_get_fmt_string(MAIN_HEADER))
LEN_ANGLE_HEADER = struct.calcsize(_get_fmt_string(ANGLE_HEADER))

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

    def close(self):
        if self._fp is not None:
            self._fp.close()

    __del__ = close

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def check_identifier(self):
        if self.structure_identifier in self.identifier:
            return self.structure_identifier
        else:
            raise IOError(
                f"Cannot read {self.structure_identifier} with "
                f"{self.__class__.__name__} class"
            )

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
