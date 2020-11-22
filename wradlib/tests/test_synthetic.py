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
from wradlib.synthetic import (
    create_data,
    create_root_where,
    create_root_what,
    create_dset_how,
    create_dset_where,
    create_dset_what,
    create_ray_time,
    create_dbz_what,
    create_synthetic_odim_file,
    write_odim_dataset,
    write_gamic_dataset,
    write_gamic_ray_header,
    write_group,
    create_synthetic_odim_dataset,
)

from . import has_data, requires_data


def string_to_char(arr):
    """Like nc4.stringtochar, but faster and more flexible.
    """
    # ensure the array is contiguous
    arr = np.array(arr, copy=False, order="C")
    kind = arr.dtype.kind
    if kind not in ["U", "S"]:
        raise ValueError("argument must be a string")
    return arr.reshape(arr.shape + (1,)).view(kind + "1")


def test_create_synthetic_odim_file():
    tmp_local = tempfile.NamedTemporaryFile(suffix=".h5", prefix="test").name

    height = 99.5
    lon = 7.071624
    lat = 50.730599
    start_time = dt.datetime(2011, 6, 10, 10, 10, 10)
    stop_time = start_time + dt.timedelta(seconds=50)
    print(create_ray_time(start_time, stop_time)[0].shape)
    odim = dict(root=dict(
        where=dict(height=height, lon=lon, lat=lat),
        what=dict(version="9"),
        attrs=dict(Conventions=np.array(b"ODIM_H5/V2_0", dtype="|S13")),
    ),
        dataset1=dict(
            where=create_dset_where(elangle=1.0),
            what=create_dset_what(start_time, stop_time),
            how=create_dset_how(start_time, stop_time, elangle=1.0),
            data=dict(data1="DBZH", data2="DBZV"),
        ),
        dataset2=dict(
            where=create_dset_where(elangle=2.5),
            what=create_dset_what(
                start_time + dt.timedelta(minutes=1), stop_time + dt.timedelta(minutes=1)
            ),
            how=create_dset_how(start_time + dt.timedelta(minutes=1),
                                stop_time + dt.timedelta(minutes=1), elangle=2.5),
            data=dict(data1="DBZH", data2="DBZV"),
        ),
    )

    data = create_synthetic_odim_dataset(odim)
    create_synthetic_odim_file(tmp_local, data)
    actual = io.read_opera_hdf5(tmp_local)

    assert data["attrs"] == actual["attrs"]
    assert data["what"]["attrs"] == actual["what"]
    assert data["where"]["attrs"] == actual["where"]
    assert data["dataset1"]["what"]["attrs"] == actual["dataset1/what"]
    assert data["dataset1"]["where"]["attrs"] == actual["dataset1/where"]
    np.testing.assert_array_equal(
        data["dataset1"]["how"]["attrs"]["startazT"], actual["dataset1/how"]["startazT"]
    )
    assert data["dataset2"]["what"]["attrs"] == actual["dataset2/what"]
    assert data["dataset2"]["where"]["attrs"] == actual["dataset2/where"]
    np.testing.assert_array_equal(
        data["dataset2"]["how"]["attrs"]["startazT"], actual["dataset2/how"]["startazT"]
    )
    assert data["dataset1"]["data1"]["what"]["attrs"] == actual["dataset1/data1/what"]
    assert data["dataset2"]["data1"]["what"]["attrs"] == actual["dataset2/data1/what"]
    np.testing.assert_array_equal(
        data["dataset1"]["data1"]["data"], actual["dataset1/data1/data"]
    )
    np.testing.assert_array_equal(
        data["dataset2"]["data1"]["data"], actual["dataset2/data1/data"]
    )
