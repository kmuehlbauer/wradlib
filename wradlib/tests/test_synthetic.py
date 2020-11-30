# Copyright (c) 2011-2020, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

import tempfile
import datetime as dt

import numpy as np

from wradlib import io
from wradlib.io.odim import (
    create_odim_sweep_dset,
    create_synthetic_odim_dataset,
    create_synthetic_odim_file,
)


def test_create_synthetic_odim_file():
    tmp_local = tempfile.NamedTemporaryFile(suffix=".h5", prefix="test").name

    height = 99.5
    lon = 7.071624
    lat = 50.730599
    start_time = dt.datetime(2011, 6, 10, 10, 10, 10)
    stop_time = start_time + dt.timedelta(seconds=50)

    odim = dict(root=dict(
        where=dict(height=height, lon=lon, lat=lat),
        what=dict(version="9"),
        attrs=dict(Conventions=np.array(b"ODIM_H5/V2_0", dtype="|S13")),
    ),
        dataset1=create_odim_sweep_dset(["DBZH", "DBZV"],
                                        start_time,
                                        stop_time,
                                        elangle=1.0),
        dataset2=create_odim_sweep_dset(["DBZH", "DBZV"],
                                        start_time + dt.timedelta(minutes=1),
                                        stop_time + dt.timedelta(minutes=1),
                                        elangle=2.5),
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
