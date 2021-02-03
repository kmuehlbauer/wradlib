#!/bin/bash
# Copyright (c) 2011-2021, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

set -e

pytest -n auto --verbose --doctest-modules --durations=15 --pyargs $1
