# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
from lava.magma.core.model.interfaces import AbstractPortImplementation


class AbstractCPort(AbstractPortImplementation):
    pass


class CInPort(AbstractCPort):
    pass


class COutPort(AbstractCPort):
    pass