#!/usr/bin/env python
# Copyright (c) 2011-2020, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

"""
"""

__all__ = [
    "write_group",
    "write_odim_dataset",
    "write_gamic_ray_header",
    "write_gamic_dataset",
    "create_data",
    "create_a1gate",
    "create_dbz_what",
    "create_dset_how",
    "create_dset_what",
    "create_dset_where",
    "create_root_what",
    "create_root_where",
    "create_startazA",
    "create_startazT",
    "create_startelA",
    "create_stopazA",
    "create_stopazT",
    "create_stopelA",
    "create_synthetic_odim_file",
]
__doc__ = __doc__.format("\n   ".join(__all__))

import h5py
import numpy as np


def write_odim_dataset(grp, data):
    grp.create_dataset("data", data=data)


def write_gamic_dataset(grp, name, data):
    da = grp.create_dataset(name, data=data["data"])
    da.attrs.update(data["attrs"])


def write_gamic_ray_header(grp, data):
    dt_type = np.dtype(
        {
            "names": [
                "azimuth_start",
                "azimuth_stop",
                "elevation_start",
                "elevation_stop",
                "timestamp",
            ],
            "formats": ["<f8", "<f8", "<f8", "<f8", "<i8"],
            "offsets": [0, 8, 16, 24, 32],
            "itemsize": 40,
        }
    )
    rh = grp.create_dataset("ray_header", (360,), dtype=dt_type)
    rh[...] = data


def write_group(grp, data):
    print("###########################################################################")
    for k, v in data.items():
        print(k, v)
        if k == "attrs":
            grp.attrs.update(v)
        elif k == "data":
            write_odim_dataset(grp, v)
        elif "moment" in k:
            write_gamic_dataset(grp, k, v)
        elif "ray_header" in k:
            write_gamic_ray_header(grp, v)
        else:
            if v:
                subgrp = grp.create_group(k)
                write_group(subgrp, v)


def create_a1gate(i):
    return i + 20


def create_time():
    return xr.DataArray(1307700610.0, attrs=io.xarray.time_attrs)


def create_startazT(i, nrays=361):
    start = 1307700610.0
    arr = np.linspace(start, start + 360, 360, endpoint=False, dtype=np.float64)
    arr = np.roll(arr, shift=create_a1gate(i))
    if nrays == 361:
        arr = np.insert(arr, 10, arr[-1], axis=0)
    return arr


def create_stopazT(i, nrays=360):
    start = 1307700611.0
    arr = np.linspace(start, start + 360, 360, endpoint=False, dtype=np.float64)
    arr = np.roll(arr, shift=create_a1gate(i))
    if nrays == 361:
        arr = np.insert(arr, 10, arr[-1], axis=0)
    return arr


def create_startazA(nrays=360):
    arr = np.linspace(0, 360, 360, endpoint=False, dtype=np.float32)
    if nrays == 361:
        arr = np.insert(arr, 10, (arr[10] + arr[9]) / 2, axis=0)
    return arr


def create_stopazA(nrays=360):
    arr = np.linspace(1, 361, 360, endpoint=False, dtype=np.float32)
    # arr = np.arange(1, 361, 1, dtype=np.float32)
    arr[arr >= 360] -= 360
    if nrays == 361:
        arr = np.insert(arr, 10, (arr[10] + arr[9]) / 2, axis=0)
    return arr


def create_startelA(i, nrays=360):
    arr = np.ones(360, dtype=np.float32) * (i + 0.5)
    if nrays == 361:
        arr = np.insert(arr, 10, arr[-1], axis=0)
    return arr


def create_stopelA(i, nrays=360):
    arr = np.ones(360, dtype=np.float32) * (i + 0.5)
    if nrays == 361:
        arr = np.insert(arr, 10, arr[-1], axis=0)
    return arr



def create_dset_how(i, nrays=360):
    return {
        "startazA": create_startazA(nrays=nrays),
        "stopazA": create_stopazA(nrays=nrays),
        "startelA": create_startelA(i, nrays=nrays),
        "stopelA": create_stopelA(i, nrays=nrays),
        "startazT": create_startazT(i, nrays=nrays),
        "stopazT": create_stopazT(i, nrays=nrays),
    }


def create_dset_where(i, nrays=360):
    return {
        "a1gate": np.array([create_a1gate(i)], dtype=np.int),
        "elangle": np.array([i + 0.5], dtype=np.float32),
        "nrays": np.array([nrays], dtype=np.int),
        "nbins": np.array([100], dtype=np.int),
        "rstart": np.array([0], dtype=np.float32),
        "rscale": np.array([1000], dtype=np.float32),
    }


def create_dset_what():
    return {
        "startdate": np.array([b"20110610"], dtype="|S9"),
        "starttime": np.array([b"101010"], dtype="|S7"),
        "enddate": np.array([b"20110610"], dtype="|S9"),
        "endtime": np.array([b"101610"], dtype="|S7"),
    }


def create_dbz_what():
    return {
        "gain": np.array([0.5], dtype=np.float32),
        "nodata": np.array([255.0], dtype=np.float32),
        "offset": np.array([-31.5], dtype=np.float32),
        "quantity": np.array([b"DBZH"], dtype="|S5"),
        "undetect": np.array([0.0], dtype=np.float32),
    }

def create_data(nrays=360):
    np.random.seed(42)
    data = np.random.randint(0, 255, (360, 100), dtype=np.uint8)
    if nrays == 361:
        data = np.insert(data, 10, data[-1], axis=0)
    return data


def create_root_where():
    return {"height": 99.5, "lon": 7.071624, "lat": 50.730599}


def create_root_what():
    return {"version": "9"}


def create_synthetic_odim_file(tmp_local, data):
    with h5py.File(str(tmp_local), "w") as f:
        write_group(f, data)



