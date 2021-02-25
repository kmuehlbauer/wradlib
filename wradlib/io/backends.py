#!/usr/bin/env python
# Copyright (c) 2011-2021, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

"""
wradlib backends for xarray
^^^^^^^^^^^^^^^^^^^^^^^^^^^
Reading radar data into xarray Datasets using ``xarray.open_dataset``
and ``xarray.open_mfdataset``

.. autosummary::
   :nosignatures:
   :toctree: generated/

   {}
"""
__all__ = [
    "RadolanBackendEntrypoint",
]

__doc__ = __doc__.format("\n   ".join(__all__))

import io

import numpy as np
from xarray.backends.common import AbstractDataStore, BackendArray, BackendEntrypoint
from xarray.backends.file_manager import CachingFileManager, DummyFileManager
from xarray.backends.locks import SerializableLock, ensure_lock
from xarray.backends.store import StoreBackendEntrypoint
from xarray.core.indexing import NumpyIndexingAdapter
from xarray.core.utils import Frozen, FrozenDict, close_on_error
from xarray.core.variable import Variable

from wradlib.io import radolan

RADOLAN_LOCK = SerializableLock()


class RadolanArrayWrapper(BackendArray):
    """Wraps array of RADOLAN data."""

    def __init__(self, variable_name, datastore):
        self.datastore = datastore
        self.variable_name = variable_name
        array = self.get_variable().data
        self.shape = array.shape
        self.dtype = np.dtype(array.dtype.kind + str(array.dtype.itemsize))

    def get_variable(self, needs_lock=True):
        ds = self.datastore._manager.acquire(needs_lock)
        return ds.variables[self.variable_name]

    def __getitem__(self, key):
        data = NumpyIndexingAdapter(self.get_variable().data)[key]
        # disable copy, hopefully this works
        return np.array(data, dtype=self.dtype, copy=False)


class RadolanDataStore(AbstractDataStore):
    """Implements ``xarray.AbstractDataStore`` read-only API for a RADOLAN files."""

    def __init__(self, filename_or_obj, lock=None, fillmissing=False, copy=False):
        if lock is None:
            lock = RADOLAN_LOCK
        self.lock = ensure_lock(lock)
        if isinstance(filename_or_obj, str):
            manager = CachingFileManager(
                radolan.radolan_file,
                filename_or_obj,
                lock=lock,
                kwargs=dict(fillmissing=fillmissing, copy=copy),
            )
        else:
            if isinstance(filename_or_obj, bytes):
                filename_or_obj = io.BytesIO(filename_or_obj)
            dataset = radolan.radolan_file(
                filename_or_obj, fillmissing=fillmissing, copy=copy
            )
            manager = DummyFileManager(dataset)

        self._manager = manager

        self._filename = self.ds.filename

    @property
    def ds(self):
        return self._manager.acquire()

    def open_store_variable(self, name, var):
        encoding = dict(source=self._filename)
        return Variable(
            var.dimensions, RadolanArrayWrapper(name, self), var.attributes, encoding
        )

    def get_variables(self):
        return FrozenDict(
            (k, self.open_store_variable(k, v)) for k, v in self.ds.variables.items()
        )

    def get_attrs(self):
        return Frozen(self.ds.attributes)

    def get_dimensions(self):
        return Frozen(self.ds.dimensions)

    def get_encoding(self):
        dims = self.get_dimensions()
        encoding = {"unlimited_dims": {k for k, v in dims.items() if v is None}}
        return encoding

    def close(self, **kwargs):
        self._manager.close(**kwargs)


class RadolanBackendEntrypoint(BackendEntrypoint):
    """Xarray BackendEntrypoint for RADOLAN data."""

    def open_dataset(
        self,
        filename_or_obj,
        *,
        mask_and_scale=True,
        decode_times=None,
        concat_characters=None,
        decode_coords=None,
        drop_variables=None,
        use_cftime=None,
        decode_timedelta=None,
        lock=None,
        fillmissing=False,
        copy=False,
    ):

        store = RadolanDataStore(
            filename_or_obj,
            lock=lock,
            fillmissing=fillmissing,
            copy=copy,
        )
        store_entrypoint = StoreBackendEntrypoint()
        with close_on_error(store):
            ds = store_entrypoint.open_dataset(
                store,
                mask_and_scale=mask_and_scale,
                decode_times=decode_times,
                concat_characters=concat_characters,
                decode_coords=decode_coords,
                drop_variables=drop_variables,
                use_cftime=use_cftime,
                decode_timedelta=decode_timedelta,
            )
        return ds
