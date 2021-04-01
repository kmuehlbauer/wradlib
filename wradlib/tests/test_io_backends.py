# Copyright (c) 2011-2021, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

import contextlib
import gc
import os

import h5py
import numpy as np
import pytest
import xarray as xr

from wradlib import io, util

from . import (
    file_or_filelike,
    get_wradlib_data_file,
    has_data,
    requires_data,
    requires_xarray_backend_api,
)
from .test_io_odim import write_group


@pytest.fixture(params=["v1", "v2"])
def xarray_backend_api(request, monkeypatch):
    monkeypatch.setenv("XARRAY_BACKEND_API", request.param)
    return request.param


@requires_data
@requires_xarray_backend_api
def test_radolan_backend(file_or_filelike, xarray_backend_api):
    filename = "radolan/misc/raa01-rw_10000-1408030950-dwd---bin.gz"
    test_attrs = {
        "radarlocations": [
            "boo",
            "ros",
            "emd",
            "hnr",
            "pro",
            "ess",
            "asd",
            "neu",
            "nhb",
            "oft",
            "tur",
            "isn",
            "fbg",
            "mem",
        ],
        "radolanversion": "2.13.1",
        "radarid": "10000",
    }
    with get_wradlib_data_file(filename, file_or_filelike) as rwfile:
        if xarray_backend_api == "v1":
            with pytest.raises(ValueError):
                with io.radolan.open_radolan_dataset(rwfile, engine="radolan") as ds:
                    pass
        else:
            data, meta = io.read_radolan_composite(rwfile)
            data[data == -9999.0] = np.nan
            with xr.open_dataset(rwfile, engine="radolan") as ds:
                assert ds["RW"].encoding["dtype"] == np.uint16
                if file_or_filelike == "file":
                    assert ds["RW"].encoding["source"] == os.path.abspath(rwfile)
                else:
                    assert ds["RW"].encoding["source"] == "None"
                assert ds.attrs == test_attrs
                assert ds["RW"].shape == (900, 900)
                np.testing.assert_almost_equal(
                    ds["RW"].values,
                    (data * 3600.0 / meta["intervalseconds"]),
                    decimal=5,
                )


@contextlib.contextmanager
def get_measured_volume(file, format, fileobj, **kwargs):
    # h5file = util.get_wradlib_data_file(file)
    with get_wradlib_data_file(file, fileobj) as h5file:
        engine = format.lower()
        if engine == "odim":
            open_ = io.xarray.open_odim_dataset
        yield open_(h5file, **kwargs)


@contextlib.contextmanager
def get_synthetic_volume(name, file_or_filelike, **kwargs):
    import tempfile

    tmp_local = tempfile.NamedTemporaryFile(suffix=".h5", prefix=name).name
    if "gamic" in name:
        format = "GAMIC"
    else:
        format = "ODIM"
    with h5py.File(str(tmp_local), "w") as f:
        data = globals()[name]()
        write_group(f, data)
    with get_wradlib_data_file(tmp_local, file_or_filelike) as h5file:
        engine = format.lower()
        if engine == "odim":
            open_ = io.xarray.open_odim_dataset
        yield open_(h5file, **kwargs)


def create_volume_repr(swp, ele, cls):
    repr = "".join(
        [
            f"<wradlib.{cls}>\n",
            f"Dimension(s): (sweep: {swp})\n",
            f"Elevation(s): {tuple(ele)}",
        ]
    )
    return repr


class DataVolume:
    def test_volumes(self, file_or_filelike):
        with self.get_volume_data(file_or_filelike) as vol:
            assert isinstance(vol, io.xarray.RadarVolume)
            repr = create_volume_repr(self.sweeps, self.elevations, type(vol).__name__)
            assert vol.__repr__() == repr
            print(vol[0])
        del vol
        gc.collect()

    def test_sweeps(self, file_or_filelike):
        with self.get_volume_data(file_or_filelike, backend_kwargs=dict(keep_azimuth=True)) as vol:
            for i, ds in enumerate(vol):
                assert isinstance(ds, xr.Dataset)
                assert self.azimuths[i] == ds.dims["azimuth"]
                assert self.ranges[i] == ds.dims["range"]

    def test_odim_output(self, file_or_filelike):
        with self.get_volume_data(file_or_filelike) as vol:
            import tempfile
            tmp_local = tempfile.NamedTemporaryFile(suffix=".h5", prefix="odim").name
            vol.to_odim(tmp_local)
        del vol
        gc.collect()

    def test_cfradial2_output(self, file_or_filelike):
        with self.get_volume_data(file_or_filelike) as vol:
            import tempfile
            tmp_local = tempfile.NamedTemporaryFile(
                suffix=".nc", prefix="cfradial"
            ).name
            vol.to_cfradial2(tmp_local)
        del vol
        gc.collect()

    def test_netcdf_output(self, file_or_filelike):
        with self.get_volume_data(file_or_filelike) as vol:
            import tempfile
            tmp_local = tempfile.NamedTemporaryFile(
                suffix=".nc", prefix="cfradial"
            ).name
            vol.to_netcdf(tmp_local, timestep=slice(0, None))
        del vol
        gc.collect()


class MeasuredDataVolume(DataVolume):
    @contextlib.contextmanager
    def get_volume_data(self, fileobj, **kwargs):
        with get_measured_volume(self.name, self.format, fileobj, **kwargs) as vol:
            yield vol


class SyntheticDataVolume(DataVolume):
    @contextlib.contextmanager
    def get_volume_data(self, fileobj, **kwargs):
        with get_synthetic_volume(self.name, fileobj, **kwargs) as vol:
            yield vol


@requires_data
@requires_xarray_backend_api
class TestKNMIVolume(MeasuredDataVolume):
    if has_data:
        name = "hdf5/knmi_polar_volume.h5"
        format = "ODIM"
        volumes = 1
        sweeps = 14
        moments = ["DBZH"]
        elevations = [
            0.3,
            0.4,
            0.8,
            1.1,
            2.0,
            3.0,
            4.5,
            6.0,
            8.0,
            10.0,
            12.0,
            15.0,
            20.0,
            25.0,
        ]
        azimuths = [360] * sweeps
        ranges = [320, 240, 240, 240, 240, 340, 340, 300, 300, 240, 240, 240, 240, 240]

        data = io.read_generic_hdf5(util.get_wradlib_data_file(name))

        dsdesc = "dataset{}"
        mdesc = "data{}"


@requires_data
@requires_xarray_backend_api
class TestGamicVolume(MeasuredDataVolume):
    if has_data:
        name = "hdf5/DWD-Vol-2_99999_20180601054047_00.h5"
        format = "GAMIC"
        volumes = 1
        sweeps = 10
        moments = [
            "DBZH",
            "DBZV",
            "DBTH",
            "DBTV",
            "ZDR",
            "VRADH",
            "VRADV",
            "WRADH",
            "WRADV",
            "PHIDP",
            "KDP",
            "RHOHV",
        ]
        elevations = [28.0, 18.0, 14.0, 11.0, 8.2, 6.0, 4.5, 3.1, 1.7, 0.6]
        azimuths = [361, 361, 361, 360, 361, 360, 360, 361, 360, 360]
        ranges = [360, 500, 620, 800, 1050, 1400, 1000, 1000, 1000, 1000]

        data = io.read_generic_hdf5(util.get_wradlib_data_file(name))

        dsdesc = "scan{}"
        mdesc = "moment_{}"


