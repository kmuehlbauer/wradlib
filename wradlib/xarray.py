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
__all__ = ["VisMethods", "WradlibDataArrayAccessor", "XarrayMethods"]
__doc__ = __doc__.format("\n   ".join(__all__))

import inspect
import re

import xarray as xr

import wradlib


class XarrayMethods:
    """Bind xarray methods to wradlib SubAccessor"""

    def __init__(self, xarray_obj, module):
        namespace = vars(module)
        module.__name__.split(".")[-1]
        (name for name in namespace if name[:1] != "_")
        for name in getattr(module, "__xr__"):
            func = namespace[name]
            if "xr_" in name:
                name = name[2:]
            setattr(self, name, func.__get__(xarray_obj, self.__class__))


class DpMethods(XarrayMethods):
    """wradlib xarray SubAccessor methods for DualPol."""

    def __init__(self, xarray_obj):
        self._obj = xarray_obj
        super().__init__(xarray_obj, wradlib.dp)

    def kdp_from_phidp(self, *args, **kwargs):
        return self._kdp_from_phidp(*args, **kwargs)

    kdp_from_phidp.__doc__ = wradlib.dp.xr_kdp_from_phidp.__doc__
    kdp_from_phidp.__signature__ = inspect.signature(wradlib.dp.xr_kdp_from_phidp)


class VisMethods(XarrayMethods):
    """wradlib xarray SubAccessor methods for visualization."""

    def __init__(self, xarray_obj):
        self._obj = xarray_obj
        super().__init__(xarray_obj, wradlib.vis)

    def plot(self, *args, **kwargs):
        return self._plot(*args, **kwargs)

    def pcolormesh(self, *args, **kwargs):
        kwargs.setdefault("func", "polormesh")
        return self._plot(*args, **kwargs)

    def contour(self, *args, **kwargs):
        kwargs.setdefault("func", "contour")
        return self._plot(*args, **kwargs)

    def contourf(self, *args, **kwargs):
        kwargs.setdefault("func", "contourf")
        return self._plot(*args, **kwargs)

    plot.__doc__ = wradlib.vis.xr_plot.__doc__
    plot.__signature__ = inspect.signature(wradlib.vis.xr_plot)
    pcolormesh.__doc__ = wradlib.vis.xr_plot.__doc__
    pcolormesh.__signature__ = inspect.signature(wradlib.vis.xr_plot)
    contour.__doc__ = wradlib.vis.xr_plot.__doc__
    contour.__signature__ = inspect.signature(wradlib.vis.xr_plot)
    contourf.__doc__ = wradlib.vis.xr_plot.__doc__
    contourf.__signature__ = inspect.signature(wradlib.vis.xr_plot)


@xr.register_dataarray_accessor("wrl")
class WradlibDataArrayAccessor:
    """DataArray Accessor for wradlib module functions"""

    __slots__ = ["_obj", "_dp", "_vis"]

    def __init__(self, xarray_obj):
        for slot in self.__slots__:
            setattr(self, slot, None)
        self._obj = xarray_obj
        self._vis = VisMethods(self._obj)
        self._dp = DpMethods(self._obj)

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
