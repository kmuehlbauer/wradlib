# wradlib 2.0 migration guide

# Xarray readers for polar data

 The xarray based radar readers for polar data have been moved to [xradar](xradar.rtfd.io)-package

## Deprecations

- wrl.io.ODIMH5
- wrl.io.CfRadial
- wrl.io.XRadVol
- wrl.open_odim
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
