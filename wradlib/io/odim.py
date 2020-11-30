#!/usr/bin/env python
# Copyright (c) 2011-2020, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

"""
"""

__all__ = [
    "create_odim_sweep_dset",
    "create_synthetic_odim_dataset",
    "create_synthetic_odim_file",
]
__doc__ = __doc__.format("\n   ".join(__all__))

import h5py
import numpy as np


GAMIC_RAY_HEADER = np.dtype(
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

moment_map = dict(DBZH=dict(gain=0.5, nodata=255., offset=-31.5, undetect=0.),
                  DBZV=dict(gain=0.5, nodata=255., offset=-31.5, undetect=0.))


def write_group(grp, data):
    for k, v in data.items():
        if k == "attrs":
            grp.attrs.update(v)
        elif k == "data":
            grp.create_dataset("data", data=v)
        elif "moment" in k:
            da = grp.create_dataset(k, data=v["data"])
            da.attrs.update(v["attrs"])
        elif "ray_header" in k:
            rh = grp.create_dataset("ray_header", (360,), dtype=GAMIC_RAY_HEADER)
            rh[...] = v
        else:
            if v:
                subgrp = grp.create_group(k)
                write_group(subgrp, v)


def create_ray_time(start, stop, a1gate=0, nrays=360):
    arr = np.linspace(start.timestamp(), stop.timestamp(), nrays + 1, endpoint=True, dtype=np.float64)
    arr = np.roll(arr, shift=a1gate)
    return arr[:-1], arr[1:]


def create_ray_azimuth(nrays=360):
    arr = np.linspace(0, 360, nrays + 1, endpoint=True, dtype=np.float32)
    return arr[:-1], arr[1:]


def create_ray_elevation(elangle=0, nrays=360):
    arr = np.ones(nrays + 1, dtype=np.float32) * elangle
    return arr[:-1], arr[1:]


def create_odim_dset_how(start, stop, nrays=360, a1gate=0, elangle=0):
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


def create_odim_dset_where(a1gate=0, elangle=0, nrays=360,
                           nbins=100, rstart=0, rscale=1000):
    return {
        "a1gate": np.array([a1gate], dtype=np.int),
        "elangle": np.array([elangle], dtype=np.float32),
        "nrays": np.array([nrays], dtype=np.int),
        "nbins": np.array([nbins], dtype=np.int),
        "rstart": np.array([rstart], dtype=np.float32),
        "rscale": np.array([rscale], dtype=np.float32),
    }


def create_odim_dset_what(start, stop):
    return {
        "startdate": np.array([f"{start:%y%m%d}"], dtype="|S9"),
        "starttime": np.array([f"{start:%H%M%S}"], dtype="|S7"),
        "enddate": np.array([f"{stop:%y%m%d}"], dtype="|S9"),
        "endtime": np.array([f"{stop:%H%M%S}"], dtype="|S7"),
    }



def create_odim_moment_what(mom):
    mm = moment_map[mom]
    return {
        "gain": np.array([mm["gain"]], dtype=np.float32),
        "nodata": np.array([mm["nodata"]], dtype=np.float32),
        "offset": np.array([mm["offset"]], dtype=np.float32),
        "quantity": np.array([mom], dtype=f"|S{len(mom)+1}"),
        "undetect": np.array([mm["undetect"]], dtype=np.float32),
    }


def create_data(nrays=360, seed=42):
    np.random.seed(seed)
    data = np.random.randint(0, 255, (360, 100), dtype=np.uint8)
    if nrays == 361:
        data = np.insert(data, 10, data[-1], axis=0)
    return data


def create_odim_sweep_dset(moments, start, stop, a1gate=0, elangle=0, nrays=360,
                           nbins=100, rstart=0, rscale=1000, seed=42):
    dset = dict(where=create_odim_dset_where(a1gate=a1gate, elangle=elangle,
                                             nrays=nrays, nbins=nbins,
                                             rstart=rstart, rscale=rscale),
                what=create_odim_dset_what(start, stop),
                how=create_odim_dset_how(start, stop, nrays=nrays, a1gate=a1gate,
                                         elangle=elangle),
                )
    for i, mom in enumerate(moments):
        dset.update({f"data{i+1}": create_odim_moment_dset(mom, nrays, seed=seed)})

    return dset


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
                    sub[k2] = v2
            odim[k] = sub
    return odim


def create_synthetic_odim_file(tmp_local, data):
    with h5py.File(str(tmp_local), "w") as f:
        write_group(f, data)


def create_odim_moment_dset(moment, nrays, seed=42):
    return dict(what=dict(attrs=create_odim_moment_what(moment)),
                data=create_data(nrays=nrays, seed=seed))
