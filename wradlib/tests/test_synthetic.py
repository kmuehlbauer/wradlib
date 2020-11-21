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
from wradlib.synthetic import (create_data, create_root_where, create_root_what, create_dset_where, create_dset_what, create_dbz_what, create_synthetic_odim_file,
                               write_odim_dataset, write_gamic_dataset, write_gamic_ray_header, write_group,
                               create_synthetic_odim_dataset)

from . import has_data, requires_data


def string_to_char(arr):
    """Like nc4.stringtochar, but faster and more flexible.
    """
    # ensure the array is contiguous
    arr = np.array(arr, copy=False, order='C')
    kind = arr.dtype.kind
    if kind not in ['U', 'S']:
        raise ValueError('argument must be a string')
    return arr.reshape(arr.shape + (1,)).view(kind + '1')


def test_create_synthetic_odim_file():
    import tempfile
    tmp_local = tempfile.NamedTemporaryFile(suffix=".h5", prefix="test").name
    data = create_synthetic_odim_dataset(nrays=360)
    create_synthetic_odim_file(tmp_local, data)
    actual = io.read_opera_hdf5(tmp_local)

    assert data["attrs"] == actual["attrs"]
    assert data["what"]["attrs"] == actual["what"]
    assert data["where"]["attrs"] == actual["where"]
    assert data["dataset1"]["what"]["attrs"] == actual["dataset1/what"]
    assert data["dataset1"]["where"]["attrs"] == actual["dataset1/where"]
    assert data["dataset2"]["what"]["attrs"] == actual["dataset2/what"]
    assert data["dataset2"]["where"]["attrs"] == actual["dataset2/where"]
    assert data["dataset1"]["data1"]["what"]["attrs"] == actual["dataset1/data1/what"]
    assert data["dataset2"]["data1"]["what"]["attrs"] == actual["dataset2/data1/what"]
    np.testing.assert_array_equal(data["dataset1"]["data1"]["data"],
                                  actual["dataset1/data1/data"])
    np.testing.assert_array_equal(data["dataset2"]["data1"]["data"],
                                  actual["dataset2/data1/data"])
