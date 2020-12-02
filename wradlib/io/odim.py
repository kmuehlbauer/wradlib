#!/usr/bin/env python
# Copyright (c) 2011-2020, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

"""
"""

__all__ = [
    "create_odim_sweep_dset",
    #"create_synthetic_odim_dataset",
    "create_synthetic_odim_file",
    "create_synthetic_gamic_volume",
    "create_synthetic_odim_volume",
    "create_synthetic_gamic_xarray_volume",
    "create_synthetic_odim_xarray_volume"
]
__doc__ = __doc__.format("\n   ".join(__all__))

import h5py
import numpy as np

import xarray as xr
import dateutil
import datetime as dt

import wradlib.io as io

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
gamic_map = dict(DBZH=dict(dyn_range_min=-32.0, dyn_range_max=95.5, format=b"UV8",
                           moment=b"Zh", unit=b"dBZ"),
                 DBZV=dict(dyn_range_min=-32.0, dyn_range_max=95.5, format=b"UV8",
                           moment=b"Zv", unit=b"dBZ"))


def write_group(grp, data):
    for k, v in data.items():
        if k == "attrs":
            grp.attrs.update(v)
        elif k == "data":
            grp.create_dataset("data", data=v)
        elif "moment" in k:
            data = v.copy()
            da = grp.create_dataset(k, data=data.pop("data"))
            da.attrs.update(data)
        elif "ray_header" in k:
            rh = grp.create_dataset("ray_header", (360,), dtype=GAMIC_RAY_HEADER)
            rh[...] = v
        else:
            if isinstance(v, dict):
                subgrp = grp.create_group(k)
                write_group(subgrp, v)
            else:
                grp.attrs[k] = v


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


def create_gamic_dset_how(start, elangle=0, nrays=360, nbins=100, rscale=1000., rsamples=1):
    timestamp = "{}:{:.3f}Z".format(start.strftime('%Y-%m-%dT%H:%M'),
                                    float("{:.3f}".format(start.second + start.microsecond / 1e6)))

    return {
        "range_samples": rsamples,
        "range_step": rscale,
        "ray_count": nrays,
        "bin_count": nbins,
        "timestamp": timestamp.encode(),
        "elevation": elangle,
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
        "startdate": np.array([f"{start:%Y%m%d}"], dtype="|S9"),
        "starttime": np.array([f"{start:%H%M%S}"], dtype="|S7"),
        "enddate": np.array([f"{stop:%Y%m%d}"], dtype="|S9"),
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


def create_odim_moment_dset(moment, nrays, seed=42):
    return dict(what=create_odim_moment_what(moment),
                data=create_data(nrays=nrays, seed=seed))
    # return dict(what=dict(attrs=create_odim_moment_what(moment)),
    #             data=create_data(nrays=nrays, seed=seed))


def create_gamic_moment(moment, nrays, a1gate, seed=42):
    mm = gamic_map[moment]
    moment = {}
    moment["data"] = np.roll(create_data(nrays=nrays, seed=seed), shift=-a1gate, axis=0)
    moment.update(mm)
    #moment["attrs"] = mm
    return moment


def create_gamic_ray_header(start, stop, nrays=360, a1gate=0, elangle=0):
    startazT, stopazT = create_ray_time(start, stop, nrays=nrays, a1gate=a1gate)
    startazA, stopazA = create_ray_azimuth(nrays=nrays)
    startelA, stopelA = create_ray_elevation(nrays=nrays, elangle=elangle)
    rh = np.zeros((nrays,), dtype=GAMIC_RAY_HEADER)
    rh["azimuth_start"] = np.roll(
        startazA, shift=(nrays - a1gate)
    )
    rh["azimuth_stop"] = np.roll(
        stopazA, shift=(nrays - a1gate)
    )
    rh["elevation_start"] = startelA
    rh["elevation_stop"] = stopelA
    rh["timestamp"] = np.roll((startazT + stopazT) / 2.0, shift=-a1gate)
    return rh


def create_data(nrays=360, seed=42):
    np.random.seed(seed)
    data = np.random.randint(0, 255, (360, 100), dtype=np.uint8)
    if nrays == 361:
        data = np.insert(data, 10, data[-1], axis=0)
    return data


def create_odim_sweep_dset(moments, start, stop, a1gate=0, elangle=0, nrays=360,
                           nbins=100, rstart=0, rscale=1000, seed=42, v=0):

    dset = dict(where=create_odim_dset_where(a1gate=a1gate, elangle=elangle,
                                             nrays=nrays, nbins=nbins,
                                             rstart=rstart, rscale=rscale),
                what=create_odim_dset_what(start, stop),
                )
    if v:
        dset.update(how=create_odim_dset_how(start, stop, nrays=nrays, a1gate=a1gate,
                                         elangle=elangle))
    for i, mom in enumerate(moments):
        dset.update({f"data{i+1}": create_odim_moment_dset(mom, nrays, seed=seed)})

    return dset


def create_gamic_sweep_dset(moments, start, stop, a1gate=0, elangle=0, nrays=360,
                           nbins=100, rstart=0, rscale=1000, seed=42):
    dset = dict(how=create_gamic_dset_how(start, elangle=elangle, nrays=nrays,
                                          nbins=nbins, rscale=rscale,
                                          ),
                )

    for i, mom in enumerate(moments):
        dset.update({f"moment_{i}": create_gamic_moment(mom, nrays, a1gate, seed=seed)})

    # add ray header
    dset.update(dict(ray_header=create_gamic_ray_header(start, stop, nrays=nrays, a1gate=a1gate, elangle=elangle)))

    return dset


def create_synthetic_odim_volume(src, v=0):
    odim = dict(where=dict(height=src.height, lon=src.lon, lat=src.lat),
                what=dict(version=src.version),
                Conventions=src.conv,
                )

    for swp in range(src.sweeps):
        dset = create_odim_sweep_dset(src.moments,
                                      src.start_time + swp * src.time_diff,
                                      src.stop_time + swp * src.time_diff,
                                      elangle=src.elevations[swp], v=v)
        odim.update({f"dataset{swp + 1}": dset})

    #data = create_synthetic_odim_dataset(odim)
    return odim


def create_synthetic_gamic_volume(src):
    gamic = dict(where=dict(height=src.height, lon=src.lon, lat=src.lat),
                 what=dict(version=src.version),
                 )

    for swp in range(src.sweeps):
        dset = create_gamic_sweep_dset(src.moments,
                                       src.start_time + swp * src.time_diff,
                                       src.stop_time + swp * src.time_diff,
                                       elangle=src.elevations[swp])
        gamic.update({f"scan{swp}": dset})

    return gamic


def create_synthetic_odim_xarray_volume(src, **kwargs):
    chunks = kwargs.pop("chunks", None)
    decode_coords = kwargs.get("decode_coords", False)
    decode_times = kwargs.get("decode_times", False)
    kwargs.pop("parallel", False)
    # site coords
    ds_list = []
    for dataset, value in src.items():
        if "dataset" not in dataset:
            continue
        ds = xr.Dataset()
        for k, v in value.items():
            if "data" in k:
                attrs = {}
                quantity = v["what"]["quantity"].item().decode()
                mapping = io.xarray.moments_mapping[quantity]
                attrs.update({key: mapping[key] for key in io.xarray.moment_attrs})
                dat = v["data"]
                attrs["scale_factor"] = v["what"]["gain"].item()
                attrs["add_offset"] = v["what"]["offset"].item()
                attrs["_FillValue"] = v["what"]["nodata"].item()
                attrs["_Undetect"] = v["what"]["undetect"].item()
                if decode_coords:
                    attrs["coordinates"] = "elevation azimuth range"
                da = xr.DataArray(dat[None, ...], dims=["time", "azimuth", "range"],
                                  attrs=attrs)
                ds = ds.assign({quantity: da})
        if decode_coords:
            ds = ds.assign_coords(src["where"])
            ds = ds.rename(
                {"height": "altitude", "lon": "longitude", "lat": "latitude"})
            ds = ds.assign_coords(dict(sweep_mode="azimuth_surveillance"))
            start_date = value["what"]["startdate"].item()
            start_time = value["what"]["starttime"].item()
            start = dateutil.parser.parse(start_date + start_time)
            end_date = value["what"]["enddate"].item()
            end_time = value["what"]["endtime"].item()
            start = dateutil.parser.parse(start_date + start_time)
            end = dateutil.parser.parse(end_date + end_time)
            da = xr.DataArray([start.replace(tzinfo=dt.timezone.utc).timestamp()], dims=["time"])
            ds = ds.assign(dict(time=da))
            ds["time"].attrs = io.xarray.time_attrs

            nrays = value["where"]["nrays"]

            if value.get("how", False):
                startazA = value["how"]["startazA"]
                stopazA = value["how"]["stopazA"]
                ds = ds.assign_coords(dict(azimuth=((startazA + stopazA) / 2.).astype("float32")))
                ds["azimuth"].attrs = io.xarray.az_attrs

                startelA = value["how"]["startelA"]
                stopelA = value["how"]["stopelA"]
                da = xr.DataArray(((startelA + stopelA) / 2.).astype("float32"), dims=["azimuth"],
                                  attrs=io.xarray.el_attrs)

                ds = ds.assign_coords(dict(elevation=da))

                startazT = value["how"]["startazT"]
                stopazT = value["how"]["stopazT"]

                da = xr.DataArray(((startazT + stopazT) / 2.), dims=["azimuth"])
                ds = ds.assign_coords(dict(rtime=da))
                ds["rtime"].attrs = io.xarray.time_attrs
            else:
                res = 360.0 / nrays
                azimuth_data = np.arange(res / 2.0, 360.0, res, dtype="float32")
                ds = ds.assign_coords(
                    dict(azimuth=azimuth_data))
                ds["azimuth"].attrs = io.xarray.az_attrs
                elangle = value["where"]["elangle"]
                elevation_data = np.ones(nrays, dtype="float32") * elangle
                da = xr.DataArray(elevation_data,
                                  dims=["azimuth"],
                                  attrs=io.xarray.el_attrs)

                ds = ds.assign_coords(dict(elevation=da))
                start = start.replace(tzinfo=dt.timezone.utc).timestamp()
                end = end.replace(tzinfo=dt.timezone.utc).timestamp()
                delta = (end - start) / nrays
                time_data = np.arange(start + delta / 2.0, end, delta)
                # todo: fix a1gate
                #time_data = np.roll(time_data)#, shift=+self.a1gate)
                da = xr.DataArray(time_data, dims=["azimuth"])
                ds = ds.assign_coords(dict(rtime=da))
                ds["rtime"].attrs = io.xarray.time_attrs

            ngates = value["where"]["nbins"]
            range_start = value["where"]["rstart"].item() * 1000.0
            bin_range = value["where"]["rscale"].item()
            cent_first = range_start + bin_range / 2.0
            range_data = np.arange(
                cent_first, range_start + bin_range * ngates, bin_range, dtype="float32"
            )
            range_attrs = io.xarray.range_attrs.copy()
            range_attrs["meters_to_center_of_first_gate"] = cent_first
            range_attrs["meters_between_gates"] = bin_range
            da = xr.DataArray(range_data, dims=["range"], attrs=range_attrs)
            ds = ds.assign_coords(dict(range=da))
            ds["range"].attrs = range_attrs

        ds = xr.decode_cf(ds, **kwargs)
        ds_list.append(ds)
    return ds_list


def create_synthetic_gamic_xarray_volume(src, **kwargs):
    chunks = kwargs.pop("chunks", None)
    decode_coords = kwargs.get("decode_coords", False)
    decode_times = kwargs.get("decode_times", False)
    kwargs.pop("parallel", False)
    # site coords
    ds_list = []
    for dataset, value in src.items():
        if "scan" not in dataset:
            continue
        ds = xr.Dataset()
        for k, v in value.items():
            if "moment" in k:
                dmax = np.iinfo(v["data"].dtype).max
                dmin = np.iinfo(v["data"].dtype).min
                minval = v["dyn_range_min"]
                maxval = v["dyn_range_max"]
                gain = (maxval - minval) / dmax
                undetect = float(dmin)
                attrs = {}
                attrs["scale_factor"] = gain
                attrs["add_offset"] = minval
                attrs["_FillValue"] = float(dmax)
                attrs["_Undetect"] = undetect

                name = v["moment"].decode().lower()
                quantity = io.xarray.GAMIC_NAMES[name]

                mapping = io.xarray.moments_mapping[quantity]
                attrs.update({key: mapping[key] for key in io.xarray.moment_attrs})
                dat = v["data"]
                if decode_coords:
                    attrs["coordinates"] = "elevation azimuth range"
                da = xr.DataArray(dat[None, ...], dims=["time", "azimuth", "range"],
                                  attrs=attrs)
                ds = ds.assign({quantity: da})
        if decode_coords:

            ds = ds.assign_coords(src["where"])
            ds = ds.rename(
                {"height": "altitude", "lon": "longitude", "lat": "latitude"})
            ds = ds.assign_coords(dict(sweep_mode="azimuth_surveillance"))
            start_date = value["how"]["timestamp"]
            start = dateutil.parser.parse(start_date)
            da = xr.DataArray([start.replace(tzinfo=dt.timezone.utc).timestamp()], dims=["time"])
            ds = ds.assign(dict(time=da))
            ds["time"].attrs = io.xarray.time_attrs

            ray_header = value["ray_header"]
            startazA = ray_header["azimuth_start"]
            stopazA = ray_header["azimuth_stop"]

            ds = ds.assign_coords(dict(azimuth=((startazA + stopazA) / 2.).astype("float32")))
            ds["azimuth"].attrs = io.xarray.az_attrs


            startelA = ray_header["elevation_start"]
            stopelA = ray_header["elevation_stop"]
            da = xr.DataArray(((startelA + stopelA) / 2.).astype("float32"), dims=["azimuth"],
                              attrs=io.xarray.el_attrs)

            ds = ds.assign_coords(dict(elevation=da))

            times = ray_header["timestamp"] / 1e6

            da = xr.DataArray(times, dims=["azimuth"])
            ds = ds.assign_coords(dict(rtime=da))
            ds["rtime"].attrs = io.xarray.time_attrs

            range_samples = value["how"]["range_samples"]
            range_step = value["how"]["range_step"]
            bin_range = range_step * range_samples
            ngates = value["how"]["bin_count"]
            range_data = np.arange(
                bin_range / 2.0, bin_range * ngates, bin_range, dtype="float32"
            )
            range_attrs = io.xarray.range_attrs.copy()
            range_attrs["meters_to_center_of_first_gate"] = bin_range / 2.0
            da = xr.DataArray(range_data, dims=["range"], attrs=range_attrs)
            ds = ds.assign_coords(dict(range=da))
            ds["range"].attrs = range_attrs

        ds = xr.decode_cf(ds, **kwargs)
        ds_list.append(ds)
    return ds_list


def create_synthetic_odim_file(tmp_local, data):
    with h5py.File(str(tmp_local), "w") as f:
        write_group(f, data)


