# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
import unittest

from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.run_configs import RunConfig
from lava.proc.dense.process import Dense
from lava.proc.lif.process import LIF


class SimpleRunConfig(RunConfig):
    def __init__(self, **kwargs):
        sync_domains = kwargs.pop("sync_domains")
        super().__init__(custom_sync_domains=sync_domains)
        self.model = None
        if "model" in kwargs:
            self.model = kwargs.pop("model")

    def select(self, process, proc_models):
        if self.model is not None:
            if self.model == "sub" and isinstance(process, AbstractProcess):
                return proc_models[1]
        return proc_models[0]


class TestLifDenseLif(unittest.TestCase):
    def test_lif_dense_lif(self):
        """Crude test to check execution of PyProcModels and message
        transmission via PyPorts."""
        self.lif1 = LIF(b=4)
        self.dense = Dense(weights=2)
        self.lif2 = LIF(b=4, du=1, vth=1000)
        self.lif1.out_ports.s_out.connect(self.dense.in_ports.s_in)
        self.dense.out_ports.a_out.connect(self.lif2.in_ports.a_in)
        self.lif1.run(condition=RunSteps(num_steps=3),
                      run_cfg=SimpleRunConfig(sync_domains=[]))
        v1 = self.lif1.v.get()[0]
        v2 = self.lif2.v.get()[0]
        self.lif1.stop()
        # lif1 accumulated 4 three times and gets reset to 0
        self.assertEqual(v1, 0)
        # lif2 accumulated 4 three times (=12) and receives one spike of 2 (=14)
        self.assertEqual(v2, 14)
