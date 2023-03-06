#!/usr/bin/env python
# Copyright (c) 2011-2023, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

"""
wradlib Xarray Accessors
^^^^^^^^^^^^^^^^^^^^^^^^

Since version 2.0 wradlib makes increasing use of xarray Accessors.
Module `xarray` takes care of accessing wradlib functionality from
xarray DataArrays and Datasets.

.. currentmodule:: wradlib.xarray

.. autosummary::
   :nosignatures:
   :toctree: generated/

   {}

"""
__all__ = ["WradlibDataArrayAccessor"]
__doc__ = __doc__.format("\n   ".join(__all__))

import re

import xarray as xr

import wradlib


@xr.register_dataarray_accessor("wrl")
@xr.register_dataset_accessor("wrl")
class WradlibDataArrayAccessor:
    """DataArray Accessor for wradlib module functions"""

    __slots__ = ["_obj", "_clutter", "_dp", "_georef", "_trafo", "_vis"]

    def __init__(self, xarray_obj):
        for slot in self.__slots__:
            setattr(self, slot, None)
        self._obj = xarray_obj

    def __getattr__(self, attr):
        return getattr(self._obj, attr)

    def __repr__(self):
        return re.sub(r"<.+>", f"<{self.__class__.__name__}>", str(self._obj))

    @property
    def vis(self):
        """SubAccessor for :class:`VisMethods`."""
        if self._vis is None:
            self._vis = wradlib.vis.VisMethods(self._obj)
        return self._vis

    @property
    def clutter(self):
        """SubAccessor for :class:`DpMethods`."""
        if self._clutter is None:
            self._clutter = wradlib.clutter.ClutterMethods(self._obj)
        return self._clutter

    @property
    def dp(self):
        """SubAccessor for :class:`DpMethods`."""
        if self._dp is None:
            self._dp = wradlib.dp.DpMethods(self._obj)
        return self._dp

    @property
    def georef(self):
        """SubAccessor for :class:`DpMethods`."""
        if self._georef is None:
            self._georef = wradlib.georef.GeorefMethods(self._obj)
        return self._georef

    @property
    def trafo(self):
        """SubAccessor for :class:`DpMethods`."""
        if self._trafo is None:
            self._trafo = wradlib.trafo.TrafoMethods(self._obj)
        return self._trafo


if __name__ == "__main__":
    print("wradlib: Calling module <xarray> as main...")
