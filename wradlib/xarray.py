#!/usr/bin/env python
# Copyright (c) 2011-2023, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

"""
wradlib Xarray Accessors
^^^^^^^^^^^^^^^^^^^^^^^^

Since version 2.0 wradlib makes increasing use of xarray Accessors
Module xarray takes care of accessing wradlib functionality from
xarray DataArrays and Datasets

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
class WradlibDataArrayAccessor:
    """DataArray Accessor for wradlib module functions"""

    __slots__ = ["_obj", "_dp", "_vis"]

    def __init__(self, xarray_obj):
        for slot in self.__slots__:
            setattr(self, slot, None)
        self._obj = xarray_obj
        self._vis = wradlib.vis.VisMethods(self._obj)
        self._dp = wradlib.dp.DpMethods(self._obj)

    def __getattr__(self, attr):
        return getattr(self._obj, attr)

    def __repr__(self):
        return re.sub(r"<.+>", f"<{self.__class__.__name__}>", str(self._obj))

    @property
    def vis(self):
        """SubAccessor for :class:`VisMethods`."""
        return self._vis

    @property
    def dp(self):
        """SubAccessor for :class:`DpMethods`."""
        return self._dp


if __name__ == "__main__":
    print("wradlib: Calling module <xarray> as main...")
