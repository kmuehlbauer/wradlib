# wradlib 2.0 migration guide

## Xarray readers for polar data

The xarray based radar readers for polar data have been moved to [xradar](xradar.rtfd.io)-package

### Deprecations

- wrl.io.ODIMH5
- wrl.io.CfRadial
- wrl.io.XRadVol
- wrl.io.open_odim
- wrl.io.XRadSweep
- wrl.io.XRadMoment
- wrl.io.XRadTimeSeries
- wrl.io.XRadVolume
- wrl.io.RadarVolume
- wrl.io.open_radar_dataset
- wrl.io.open_radar_mfdataset
- wrl.io.to_netcdf
- wrl.io.open_rainbow_dataset
- wrl.io.open_rainbow_mfdataset
- wrl.io.open_cfradial1_dataset
- wrl.io.open_cdradial1_mfdataset
- wrl.io.open_cfradial2_dataset
- wrl.io.open_cdradial2_mfdataset
- wrl.io.open_iris_dataset
- wrl.io.open_iris_mfdataset
- wrl.io.open_odim_dataset
- wrl.io.open_odim_mfdataset
- wrl.io.open_gamic_dataset
- wrl.io.open_gamic_mfdataset
- wrl.io.open_furuno_dataset
- wrl.io.open_furuno_mfdataset
- wrl.io.CfRadial1BackendEntrypoint
- wrl.io.CfRadial2BackendEntrypoint
- wrl.io.FurunoBackendEntrypoint
- wrl.io.GamicBackendEntrypoint
- wrl.io.OdimBackendEntrypoint
- wrl.io.RainbowBackendEntrypoint
- wrl.io.IrisBackendEntrypoint

### How can I read my data now?

#### Single sweep

```python
swp = xarray.open_dataset(filename, engine=engine, group=group)
```
`engine` would we one `BackendName``defined in [xradar](https://xradar.rtfd.io), where currently available are:

- [cfradial1](https://docs.openradarscience.org/projects/xradar/en/stable/notebooks/CfRadial1.html)
- [odim](https://docs.openradarscience.org/projects/xradar/en/stable/notebooks/ODIM_H5.html)
- [gamic](https://docs.openradarscience.org/projects/xradar/en/stable/notebooks/GAMIC.html)
- [rainbow](https://docs.openradarscience.org/projects/xradar/en/stable/notebooks/Rainbow.html)
- [iris](https://docs.openradarscience.org/projects/xradar/en/stable/notebooks/Iris.html)
- [furuno](https://docs.openradarscience.org/projects/xradar/en/stable/notebooks/Furuno.html)

`group` would be a string like `sweep_0` for first sweep, `sweep_1` for second sweep and so forth.

The above command will return an `xarray.Dataset` which is aligned with the CfRadial2/FM301 standard. Please refer to the [xradar model](https://docs.openradarscience.org/projects/xradar/en/stable/datamodel.html).

Please also refer to the [xarray.open_dataset](https://docs.xarray.dev/en/stable/generated/xarray.open_dataset.html) documentation.

#### Timeseries of sweeps

```python
ts = xarray.open_mfdataset(filelist, concat_dim=time2, engine=engine, group=group, preprocess=preprocess)
```

`preprocess` is here a function which is applied to each of the retrieved datasets to align them for stacking along the new dimension (`time2`). One use-case would be [angle reindexing](https://docs.openradarscience.org/projects/xradar/en/stable/notebooks/angle_reindexing.html).

Please also refer to the [xarray.open_dataset](https://docs.xarray.dev/en/stable/generated/xarray.open_mfdataset.html) documentation.

#### Single metadata group

The same way different metadata groups can be retrieved. Just require the wanted group with `group`-kwarg..

#### Single Volume

{{wradlib}}'s `RadarVolume` is replaced by `datatree.DataTree`.

```python
vol = xradar.open_cfradial1_datatree(filename)
```

Here, as well as above, each backend has it's own loading function:

- [xradar.open_cfradial1_datatree](https://docs.openradarscience.org/projects/xradar/en/stable/notebooks/CfRadial1.html)
- [xradar.open_odim_datatree](https://docs.openradarscience.org/projects/xradar/en/stable/notebooks/ODIM_H5.html)
- [xradar.open_gamic_datatree](https://docs.openradarscience.org/projects/xradar/en/stable/notebooks/GAMIC.html)
- [xradar.open_rainbow_datatree](https://docs.openradarscience.org/projects/xradar/en/stable/notebooks/Rainbow.html)
- [xradar.open_iris_datatree](https://docs.openradarscience.org/projects/xradar/en/stable/notebooks/Iris.html)
- [xradar.open_furuno_datatree](https://docs.openradarscience.org/projects/xradar/en/stable/notebooks/Furuno.html)
- [datatree.open_datatree](https://xarray-datatree.readthedocs.io/en/latest/generated/datatree.open_datatree.html)

#### Multiple Volumes

This is not yet available out of the box as dedicated functions (like `xarray.open_mfdataset`) but this is [worked on at xradar](https://docs.openradarscience.org/projects/xradar/en/stable/notebooks/Multi-Volume-Concatenation.html).

## IO/Xarrray

### Deprecations

- `wrl.io.radolan_to_xarray` - `wrl.io.open_radolan_dataset` or `xarray.open_dataset` with `engine="radolan"`
- `wrl.io.create_xarray_dataarray` -> `wrl.georef.create_xarray_dataarray`

## Misc

### Deprecations

- `wrl.dp.linear_despeckle` -> `wrl.util.despeckle`
- `zonalstats.DataSource` -> `wrl.io.VectorSource`
- `wrl.georef.xarray.georeference_dataset` -> `wrl.georef.polar.georeference`

## Visualization

### Deprecations

- `plot_ppi`/`plot_rhi` -> `wrl.georef.create_xarray_dataarray` and `wrl.vis.plot(da)` or xarray accessor `da.wrl.vis.plot()`
