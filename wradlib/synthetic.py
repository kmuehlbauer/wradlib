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
    for k, v in data.items():
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


def create_ray_time(start, stop, a1gate=0, nrays=360):
    arr = np.linspace(start.timestamp(), stop.timestamp(), nrays + 1, endpoint=True, dtype=np.float64)
    arr = np.roll(arr, shift=a1gate)
    return arr[:-1], arr[1:]


def create_ray_azimuth(nrays=360):
    arr = np.linspace(0, 360, nrays + 1, endpoint=True, dtype=np.float32)
    return arr[:-1], arr[1:]


def create_ray_elevation(elangle=0, nrays=360):
    print("rays:", nrays)
    print(type(nrays))
    arr = np.ones(nrays + 1, dtype=np.float32) * elangle
    return arr[:-1], arr[1:]


def create_startazT(i, nrays=361):
    start = 1307700610.0
    arr = np.linspace(0, 360, 361, endpoint=False, dtype=np.float64)
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


def create_dset_how(start, stop, nrays=360, a1gate=0, elangle=0):
    startazT, stopazT = create_ray_time(start, stop, nrays=nrays, a1gate=a1gate)
    startazA, stopazA = create_ray_azimuth(nrays=nrays)
    startelA, stopelA = create_ray_elevation(nrays=nrays, elangle=elangle)

    return {
        "startazA": startazA,
        "stopazA": stopazA,
        "startelA": startelA,
        "stopelA": stopelA,
        "startazT": startazT,
        "stopazT": stopazT,
    }


def create_dset_where(a1gate=0, elangle=0, nrays=360,
                      nbins=100, rstart=0, rscale=1000):
    return {
        "a1gate": np.array([a1gate], dtype=np.int),
        "elangle": np.array([elangle], dtype=np.float32),
        "nrays": np.array([nrays], dtype=np.int),
        "nbins": np.array([nbins], dtype=np.int),
        "rstart": np.array([rstart], dtype=np.float32),
        "rscale": np.array([rscale], dtype=np.float32),
    }


def create_dset_what(start, stop):
    return {
        "startdate": np.array([f"{start:%y%m%d}"], dtype="|S9"),
        "starttime": np.array([f"{start:%H%M%S}"], dtype="|S7"),
        "enddate": np.array([f"{stop:%y%m%d}"], dtype="|S9"),
        "endtime": np.array([f"{stop:%H%M%S}"], dtype="|S7"),
    }


moment_map = dict(DBZH=dict(gain=0.5, nodata=255., offset=-31.5, undetect=0.),
                  DBZV=dict(gain=0.5, nodata=255., offset=-31.5, undetect=0.))

def create_mom_what(mom):
    mm = moment_map[mom]
    return {
        "gain": np.array([mm["gain"]], dtype=np.float32),
        "nodata": np.array([mm["nodata"]], dtype=np.float32),
        "offset": np.array([mm["offset"]], dtype=np.float32),
        "quantity": np.array([mom], dtype=f"|S{len(mom)+1}"),
        "undetect": np.array([mm["undetect"]], dtype=np.float32),
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


def create_synthetic_odim_dataset(obj):
    odim = {}
    for k, v in obj.items():
        if k == "root":
            for k1, v1 in v.items():
                if "attrs" not in k1:
                    odim[k1] = {}
                    if v1:
                        odim[k1]["attrs"] = v1
                else:
                    odim[k1] = v1
        else:
            sub = {}
            for k2, v2 in v.items():
                if "data" not in k2:
                    sub[k2] = {}
                    if v2:
                        sub[k2]["attrs"] = v2
                else:
                    for data, mom in v2.items():
                        sub2 = {}
                        sub2["what"] = {}
                        sub2["what"]["attrs"] = create_mom_what(mom)
                        sub2["data"] = create_data(nrays=v["where"]["nrays"])
                        sub[data] = sub2
            odim[k] = sub
    return odim


def create_synthetic_odim_file(tmp_local, data):
    with h5py.File(str(tmp_local), "w") as f:
        write_group(f, data)



