#!/usr/bin/env python
# Copyright (c) 2011-2020, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

"""
wradlib Xarray Accessors
^^^^^^^^^^^^^^^^^^^^^^^^

Module xarray takes care of accessing wradlib functionality from
xarray DataArrays and Datasets

.. currentmodule:: wradlib.xarray

.. autosummary::
   :nosignatures:
   :toctree: generated/

   {}
"""
__all__ = ['DPMethods', 'ZRMethods']
__doc__ = __doc__.format('\n   '.join(__all__))

import xarray as xr

from wradlib import dp, zr, xarray


class XarrayMethods(object):
    def __init__(self, xarray_obj, module):
        self._obj = xarray_obj
        namespace = vars(module)
        modname = module.__name__.split('.')[-1]
        public = (name for name in namespace if name[:1] != "_")
        for name in getattr(module, "__all__"):
            func = namespace[name]
            setattr(self, name, func.__get__(self._obj, self.__class__))


class DPMethods(XarrayMethods):
    def __init__(self, xarray_obj):
        super(DPMethods, self).__init__(xarray_obj, dp)


class ZRMethods(XarrayMethods):
    def __init__(self, xarray_obj):
        super(ZRMethods, self).__init__(xarray_obj, zr)


@xr.register_dataarray_accessor('wrl')
class WradlibDataArrayAccessor(object):
    """DataArray Accessor for wradlib module functions
    """
    __slots__ = ['_obj', '_dp', '_zr']

    def __init__(self, xarray_obj):
        for slot in self.__slots__:
            setattr(self, slot, None)
        self._obj = xarray_obj

    def __getattr__(self, attr):
        return getattr(self._obj, attr)

    def __repr__(self):
        return re.sub(r'<.+>', '<{}>'.format(self.__class__.__name__),
                      str(self._obj))

    @property
    def dp(self):
        if self._dp is None:
            self._dp = xarray.DPMethods(self._obj)
        return self._dp

    @property
    def zr(self):
        if self._zr is None:
            self._zr = xarray.ZRMethods(self._obj)
        return self._zr


if __name__ == '__main__':
    print('wradlib: Calling module <xarray> as main...')
