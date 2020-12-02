# Copyright (c) 2011-2020, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

import contextlib
import gc
import io as sio
import tempfile
import datetime as dt

import h5py
import numpy as np
import pytest
import xarray as xr

from wradlib import io, util

from . import has_data, requires_data


def decode_dict(attrs):
    out = {}
    for k, v in attrs.items():
        try:
            v = v.item()
        except (ValueError, AttributeError):
            pass
        try:
            v = v.decode()
        except (UnicodeDecodeError, AttributeError):
            pass
        out[k] = v
    return out


@contextlib.contextmanager
def get_wradlib_data_file(file, fileobj):
    datafile = util.get_wradlib_data_file(file)
    if fileobj == "filelike":
        _open = open
        if datafile[-3:] == ".gz":
            gzip = util.import_optional("gzip")
            _open = gzip.open
        with _open(datafile, mode="r+b") as f:
            yield sio.BytesIO(f.read())
    else:
        yield datafile


@contextlib.contextmanager
def get_measured_volume(file, loader, format, fileobj):
    with get_wradlib_data_file(file, fileobj) as h5file:
        if format == "CfRadial1":
            yield io.xarray.open_cfradial(h5file, loader=loader)
        else:
            yield io.xarray.open_odim(h5file, loader=loader, flavour=format)


@contextlib.contextmanager
def get_synthetic_volume(data, tmp_local, loader, fileobj, **kwargs):
    with get_wradlib_data_file(tmp_local, fileobj) as h5file:
        yield io.xarray.open_odim(h5file, loader=loader, flavour=data.format, **kwargs)


def create_volume_repr(vol, swp, ele):
    repr = "".join(
        [
            f"<wradlib.{vol}>\n",
            f"Dimension(s): (sweep: {swp})\n",
            f"Elevation(s): {tuple(ele)}",
        ]
    )
    return repr


def create_timeseries_repr(ts, time, azi, range, ele):
    repr = "".join(
        [
            f"<wradlib.{ts}>\n",
            f"Dimension(s): (time: {time}, azimuth: {azi}, ",
            f"range: {range})\n",
            f"Elevation(s): ({ele})",
        ]
    )
    return repr


def create_sweep_repr(format, azi, range, ele, mom):
    format = format.lower().capitalize()
    repr = "".join(
        [
            f"<wradlib.XRadSweep{format}>\n",
            f"Dimension(s): (azimuth: {azi}, range: {range})\n",
            f"Elevation(s): ({ele})\n",
            f'Moment(s): ({", ".join(mom)})',
        ]
    )
    return repr


def create_moment_repr(mclass, azi, range, ele, mom):
    repr = "".join(
        [
            f"<wradlib.{mclass}>\n",
            f"Dimension(s): (azimuth: {azi}, range: {range})\n",
            f"Elevation(s): ({ele})\n",
            f"Moment: ({mom})",
        ]
    )
    return repr


def check_moment(test, swp, i, mom, k):
    repr = create_moment_repr(
        test.momclass,
        test.wazimuths[i],
        test.ranges[i],
        test.elevations[i],
        test.moments[k],
    )
    assert isinstance(mom, io.xarray.XRadMoment)
    assert mom.__repr__() == repr
    assert mom.quantity == test.moments[k]
    assert mom.parent == swp


def check_volume(test, vol):
    assert isinstance(vol, io.xarray.XRadVolume)
    repr = create_volume_repr(test.volclass, test.sweeps, test.elevations)
    assert vol.__repr__() == repr
    if "_io.BytesIO" not in vol.filename:
        assert test.name.split("/")[-1] in vol.filename
    assert vol.ncid == vol.ncfile
    assert vol.ncpath == "/"
    assert vol.parent is None


def check_timeseries(test, vol, i):
    repr = create_timeseries_repr(test.tsclass,
                                  test.volumes, test.wazimuths[i], test.ranges[i],
                                  test.elevations[i],
                                  )
    assert isinstance(vol[i], io.xarray.XRadTimeSeries)
    assert vol[i].__repr__() == repr
    assert vol[i].parent == vol


def check_sweep(test, vol, i, swp):
    repr = create_sweep_repr(
        test.format,
        test.wazimuths[i],
        test.ranges[i],
        test.elevations[i],
        test.moments,
    )
    assert isinstance(swp, io.xarray.XRadSweep)
    assert swp.__repr__() == repr
    assert swp.parent == vol[i]


def check_cfradial_volume(test, vol):
    check_volume(test, vol)
    for i, ts in enumerate(vol):
        check_timeseries(test, vol, i)
        for j, swp in enumerate(ts):
            check_sweep(test, vol, i, swp)
            for k, mom in enumerate(swp):
                check_moment(test, swp, i, mom, k)
    del mom
    del ts
    del swp


def check_volume_data(test, vol):
    for i, ts in enumerate(vol):
        ds = test.trg[i]
        for j, swp in enumerate(ts):
            ds = ds.isel(time=j)
            xr.testing.assert_identical(swp.data, ds)
            for k, mom in enumerate(swp):
                xr.testing.assert_identical(mom.data, ds[mom.quantity])



@pytest.fixture(params=[dict(decode_coords=False, mask_and_scale=False, decode_times=False),
                        dict(decode_coords=True, mask_and_scale=False, decode_times=False),
                        dict(decode_coords=True, mask_and_scale=False, decode_times=True),
                        dict(decode_coords=True, mask_and_scale=True, decode_times=True),
                        ])
def decode_cf(request):
    return request.param


class MeasuredDataVolume:
    @contextlib.contextmanager
    def get_volume_data(self, loader, fileobj, **kwargs):
        with get_measured_volume(self.name, loader, self.format, fileobj) as vol:
            yield vol

    def check_metadata(self, vol, loader):
        engine = "h5netcdf" if "h5" in loader else "netcdf4"
        num = 1 if self.format == "ODIM" else 0
        check_volume(self, vol)
        for i, ts in enumerate(vol):
            check_timeseries(self, vol, i)
            assert ts.ncid == ts.ncfile[ts.ncpath]
            assert ts.ncpath == self.dsdesc.format(i + num)
            if "_io.BytesIO" not in ts.filename:
                assert self.name.split("/")[-1] in ts.filename
            for j, swp in enumerate(ts):
                check_sweep(self, vol, i, swp)
                if "_io.BytesIO" not in swp.filename:
                    assert self.name.split("/")[-1] in swp.filename
                for k, mom in enumerate(swp):
                    check_moment(self, swp, i, mom, k)
                    assert mom.engine == engine
                    if "_io.BytesIO" not in mom.filename:
                        assert self.name.split("/")[-1] in mom.filename

                    ncpath = "/".join([self.dsdesc, self.mdesc]).format(
                        i + num, k + num
                    )
                    assert mom.ncpath == ncpath
                    assert mom.ncid == mom.ncfile[mom.ncpath]

    @pytest.mark.parametrize("loader, fileobj",
                             [("h5py", "file"), ("h5netcdf", "file"),
                              ("h5netcdf", "filelike"),
                              ("netcdf4", "file"), ("netcdf4", "filelike")])
    def test_volume(self, loader, fileobj):
        with self.get_volume_data(loader, fileobj) as vol:
            self.check_metadata(vol, loader)

        del vol
        gc.collect()


class MeasuredGamicDataVolume(MeasuredDataVolume):

    @pytest.mark.parametrize("loader, fileobj",
                             [("h5py", "file"), ("h5netcdf", "file"),
                              ("h5netcdf", "filelike")])
    def test_volume(self, loader, fileobj):
        with self.get_volume_data(loader, fileobj) as vol:
            self.check_metadata(vol, loader)

        del vol
        gc.collect()


class SyntheticDataVolume:
    def setup_class(self):
        self.src = self.get_synthetic_data(self)
        # todo: add erroneous data for file export here
        import tempfile
        tmp_local = tempfile.NamedTemporaryFile(suffix=".h5", prefix=self.name).name
        with h5py.File(str(tmp_local), "w") as f:
            io.odim.write_group(f, self.src)
        self.filename = str(tmp_local)


    @contextlib.contextmanager
    def get_volume_data(self, loader, fileobj, **kwargs):
        with get_synthetic_volume(self, self.filename, loader, fileobj,
                                  **kwargs) as vol:
            yield vol

    def check_metadata(self, vol, get_loader):
        engine = "h5netcdf" if "h5" in get_loader else "netcdf4"
        num = 1 if self.format == "ODIM" else 0
        check_volume(self, vol)
        for i, ts in enumerate(vol):
            check_timeseries(self, vol, i)
            assert ts.ncid == ts.ncfile[ts.ncpath]
            assert ts.ncpath == self.dsdesc.format(i + num)
            if "_io.BytesIO" not in ts.filename:
                assert self.name.split("/")[-1] in ts.filename
            for j, swp in enumerate(ts):
                check_sweep(self, vol, i, swp)
                if "_io.BytesIO" not in swp.filename:
                    assert self.name.split("/")[-1] in swp.filename
                # mixins
                dataset = self.src[self.dsdesc.format(i + num)]
                if swp.how is not None:
                    np.testing.assert_equal(swp.how, decode_dict(dataset["how"]))
                if self.format == "ODIM":
                    assert swp.what == decode_dict(dataset["what"])
                    assert swp.where == decode_dict(dataset["where"])
                for k, mom in enumerate(swp):
                    check_moment(self, swp, i, mom, k)
                    assert mom.engine == engine
                    if "_io.BytesIO" not in mom.filename:
                        assert self.name.split("/")[-1] in mom.filename

                    ncpath = "/".join([self.dsdesc, self.mdesc]).format(
                        i + num, k + num
                    )
                    assert mom.ncpath == ncpath
                    assert mom.ncid == mom.ncfile[mom.ncpath]

    def check_volume_data(self, vol, trg):
        for i, ts in enumerate(vol):
            ds = trg[i]
            for j, swp in enumerate(ts):
                ds = ds.isel(time=j)
                xr.testing.assert_identical(swp.data, ds)
                for k, mom in enumerate(swp):
                    xr.testing.assert_identical(mom.data, ds[mom.quantity])


class SyntheticOdimDataVolume(SyntheticDataVolume):

    # todo: add function which creates xarray datasets per root and sweeps
    def get_synthetic_data(self):
            return io.create_synthetic_odim_volume(self, v=self.test_version)

    def get_synthetic_ds(self, **kwargs):
        return io.create_synthetic_odim_xarray_volume(self.src, **kwargs)

    @pytest.mark.parametrize("loader, fileobj",
                             [("h5py", "file"), ("h5netcdf", "file"), ("h5netcdf", "filelike"),
                              ("netcdf4", "file"), ("netcdf4", "filelike")])
    def test_volume(self, loader, fileobj):
        with self.get_volume_data(loader, fileobj) as vol:
            print(vol[0])
            print(vol[0].data)
            self.check_metadata(vol, loader)

        del vol
        gc.collect()

    @pytest.mark.parametrize("loader, fileobj",
                             [("h5py", "file"), ("h5netcdf", "file"),
                              ("h5netcdf", "filelike"),
                              ("netcdf4", "file"), ("netcdf4", "filelike")])
    def test_volume_data(self, loader, fileobj, decode_cf):
        with self.get_volume_data(loader,
                                  fileobj,
                                  chunks=None,
                                  parallel=False,
                                  **decode_cf
                                  ) as vol:
            trg = self.get_synthetic_ds(**decode_cf)
            self.check_volume_data(vol, trg)

        del vol
        gc.collect()


class SyntheticGamicDataVolume(SyntheticDataVolume):

    # todo: add function which creates xarray datasets per root and sweeps
    def get_synthetic_data(self):
            return io.create_synthetic_gamic_volume(self)

    def get_synthetic_ds(self, **kwargs):
        return io.create_synthetic_gamic_xarray_volume(self.src, **kwargs)

    @pytest.mark.parametrize("loader, fileobj",
                             [("h5py", "file"), ("h5netcdf", "file"),
                              ("h5netcdf", "filelike")])
    def test_volume(self, loader, fileobj):
        with self.get_volume_data(loader, fileobj) as vol:
            self.check_metadata(vol, loader)

        del vol
        gc.collect()

    @pytest.mark.parametrize("loader, fileobj",
                             [("h5py", "file"), ("h5netcdf", "file"),
                              ("h5netcdf", "filelike")])
    def test_volume_data(self, loader, fileobj, decode_cf):
        with self.get_volume_data(loader,
                                  fileobj,
                                  chunks=None,
                                  parallel=False,
                                  **decode_cf
                                  ) as vol:
            trg = self.get_synthetic_ds(**decode_cf)
            self.check_volume_data(vol, trg)

        del vol
        gc.collect()

@requires_data
class TestKNMIVolume(MeasuredDataVolume):
    if has_data:
        name = "hdf5/knmi_polar_volume.h5"
        format = "ODIM"
        volclass = "XRadVolumeOdim"
        tsclass = "XRadTimeSeriesOdim"
        momclass = "XRadMomentOdim"
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
        wazimuths = [360] * sweeps
        ranges = [320, 240, 240, 240, 240, 340, 340, 300, 300, 240, 240, 240, 240, 240]
        dsdesc = "dataset{}"
        mdesc = "data{}"


@requires_data
class TestGamicVolume(MeasuredGamicDataVolume):
    if has_data:
        name = "hdf5/DWD-Vol-2_99999_20180601054047_00.h5"
        format = "GAMIC"
        volclass = "XRadVolumeOdim"
        tsclass = "XRadTimeSeriesOdim"
        momclass = "XRadMomentGamic"
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
        wazimuths = [361, 361, 361, 360, 361, 360, 360, 361, 360, 360]
        ranges = [360, 500, 620, 800, 1050, 1400, 1000, 1000, 1000, 1000]

        dsdesc = "scan{}"
        mdesc = "moment_{}"


class SyntheticOdimVolume(SyntheticOdimDataVolume):
    name = "base_odim_data_00"
    format = "ODIM"
    volclass = "XRadVolumeOdim"
    tsclass = "XRadTimeSeriesOdim"
    momclass = "XRadMomentOdim"
    volumes = 1
    sweeps = 2
    moments = ["DBZH", "DBZV"]
    elevations = [1.0, 2.5]
    azimuths = [360, 361]
    wazimuths = [360, 360]
    ranges = [100, 100]
    height = 99.5
    lon = 7.071624
    lat = 50.730599
    version = "9"
    conv = np.array(b"ODIM_H5/V2_0", dtype="|S13")
    start_time = dt.datetime(2011, 6, 10, 10, 10, 10)
    stop_time = start_time + dt.timedelta(seconds=50)
    time_diff = dt.timedelta(minutes=1)
    dsdesc = "dataset{}"
    mdesc = "data{}"

class TestSyntheticOdimVolume01(SyntheticOdimVolume):
    test_version = 0

class TestSyntheticOdimVolume02(SyntheticOdimVolume):
    test_version = 1

#
#
# class TestSyntheticOdimVolume03(SyntheticDataVolume):
#     name = "base_odim_data_02"
#     format = "ODIM"
#     volclass = "XRadVolumeOdim"
#     tsclass = "XRadTimeSeriesOdim"
#     momclass = "XRadMomentOdim"
#     volumes = 1
#     sweeps = 2
#     moments = ["DBZH"]
#     elevations = [0.5, 1.5]
#     azimuths = [361, 361]
#     ranges = [100, 100]
#
#     data = globals()[name]()
#
#     dsdesc = "dataset{}"
#     mdesc = "data{}"
#
#
# class TestSyntheticOdimVolume04(SyntheticDataVolume):
#     name = "base_odim_data_03"
#     format = "ODIM"
#     volclass = "XRadVolumeOdim"
#     tsclass = "XRadTimeSeriesOdim"
#     momclass = "XRadMomentOdim"
#     volumes = 1
#     sweeps = 2
#     moments = ["DBZH"]
#     elevations = [0.5, 1.5]
#     azimuths = [360, 360]
#     ranges = [100, 100]
#
#     data = globals()[name]()
#
#     dsdesc = "dataset{}"
#     mdesc = "data{}"
#
#
class TestSyntheticGamicVolume01(SyntheticGamicDataVolume):
    name = "base_gamic_data"
    format = "GAMIC"
    volclass = "XRadVolumeOdim"
    tsclass = "XRadTimeSeriesOdim"
    momclass = "XRadMomentGamic"
    volumes = 1
    sweeps = 2
    moments = ["DBZH", "DBZV"]
    elevations = [0.5, 1.5]
    azimuths = [360, 361]
    wazimuths = [360, 360]
    ranges = [100, 100]
    height = 99.5
    lon = 7.071624
    lat = 50.730599
    version = "9"
    start_time = dt.datetime(2011, 6, 10, 10, 10, 10)
    stop_time = start_time + dt.timedelta(seconds=50)
    time_diff = dt.timedelta(minutes=1)
    dsdesc = "scan{}"
    mdesc = "moment_{}"
#
#
# @requires_data
# class TestCfRadial1Volume(MeasuredCfRadial1DataVolume):
#     if has_data:
#         name = "netcdf/cfrad.20080604_002217_000_SPOL_v36_SUR.nc"
#         format = "CfRadial1"
#         volclass = "XRadVolumeCfradial"
#         tsclass = "XRadTimeSeriesOdim"
#         momclass = "XRadMomentCfRadial"
#         volumes = 1
#         sweeps = 9
#         moments = ["DBZ", "VR"]
#         elevations = [
#             0.5, 1.1, 1.8, 2.6, 3.6, 4.7, 6.5, 9.1, 12.8,
#         ]
#         azimuths = [482, 482, 481, 482, 480, 481, 481, 483, 482]
#         ranges = [996] * sweeps
#
#         data = io.read_generic_netcdf(util.get_wradlib_data_file(name))
#
#         #dsdesc = "dataset{}"
#         #mdesc = "data{}"
#


def test_create_synthetic_odim_file():
    tmp_local = tempfile.NamedTemporaryFile(suffix=".h5", prefix="test").name

    height = 99.5
    lon = 7.071624
    lat = 50.730599
    start_time = dt.datetime(2011, 6, 10, 10, 10, 10)
    stop_time = start_time + dt.timedelta(seconds=50)
    time_diff = dt.timedelta(minutes=1)
    moments = ["DBZH", "DBZV"]
    elevations = [0.5, 1.5]
    sweeps = 2

    odim = dict(where=dict(height=height, lon=lon, lat=lat),
                what=dict(version="9"),
                Conventions=np.array(b"ODIM_H5/V2_0", dtype="|S13"),
                )

    for swp in range(sweeps):
        dset = io.create_odim_sweep_dset(moments,
                                      start_time + swp * time_diff,
                                      stop_time + swp * time_diff,
                                      elangle=elevations[swp], v=1)
        odim.update({f"dataset{swp + 1}": dset})

    io.create_synthetic_odim_file(tmp_local, odim)
    actual = io.read_opera_hdf5(tmp_local)

    assert odim["Conventions"] == actual["attrs"]["Conventions"]
    assert odim["what"] == actual["what"]
    assert odim["where"] == actual["where"]
    assert odim["dataset1"]["what"] == actual["dataset1/what"]
    assert odim["dataset1"]["where"] == actual["dataset1/where"]
    np.testing.assert_array_equal(
        odim["dataset1"]["how"]["startazT"], actual["dataset1/how"]["startazT"]
    )
    assert odim["dataset2"]["what"] == actual["dataset2/what"]
    assert odim["dataset2"]["where"] == actual["dataset2/where"]
    np.testing.assert_array_equal(
        odim["dataset2"]["how"]["startazT"], actual["dataset2/how"]["startazT"]
    )
    assert odim["dataset1"]["data1"]["what"] == actual["dataset1/data1/what"]
    assert odim["dataset2"]["data1"]["what"] == actual["dataset2/data1/what"]
    np.testing.assert_array_equal(
        odim["dataset1"]["data1"]["data"], actual["dataset1/data1/data"]
    )
    np.testing.assert_array_equal(
        odim["dataset2"]["data1"]["data"], actual["dataset2/data1/data"]
    )
