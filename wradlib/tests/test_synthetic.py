# Copyright (c) 2011-2020, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

import contextlib
import gc
import io as sio

import h5py
import numpy as np
import pytest
import xarray as xr

from wradlib import io, util
from wradlib.synthetic import (create_data, create_root_where, create_root_what, create_dset_where, create_dset_what, create_dbz_what, create_synthetic_odim_file, write_group)

from . import has_data, requires_data


def base_odim_data_00(nrays=360):
    data = {}
    root_attrs = [("Conventions", np.array([b"ODIM_H5/V2_0"], dtype="|S13"))]

    foo_data = create_data(nrays=nrays)

    dataset = ["dataset1", "dataset2"]
    datas = ["data1"]

    data["where"] = {}
    data["where"]["attrs"] = create_root_where()
    data["what"] = {}
    data["what"]["attrs"] = create_root_what()
    for i, grp in enumerate(dataset):
        sub = {}
        sub["how"] = {}
        sub["where"] = {}
        sub["where"]["attrs"] = create_dset_where(i, nrays=nrays)
        sub["what"] = {}
        sub["what"]["attrs"] = create_dset_what()
        for j, mom in enumerate(datas):
            sub2 = {}
            sub2["data"] = foo_data
            sub2["what"] = {}
            sub2["what"]["attrs"] = create_dbz_what()
            sub[mom] = sub2
        data[grp] = sub
    data["attrs"] = root_attrs
    return data


def test_create_odim_file():
    import tempfile
    tmp_local = tempfile.NamedTemporaryFile(suffix=".h5", prefix="test").name
    data = base_odim_data_00()
    print(data)
    with h5py.File(str(tmp_local), "w") as f:
        write_group(f, data)
    data = io.read_generic_hdf5(tmp_local)
    print(data)
