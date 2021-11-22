from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var
from lava.magma.core.process.ports.ports import InPort, OutPort

class Proc(AbstractProcess):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.inp = InPort(shape=(1,))
        self.out = OutPort(shape=(1,))
        self.var = Var(shape=(1,), init=1)

from lava.magma.core.decorator import implements, requires

from lava.magma.core.resources import CPU

from lava.magma.core.model.py.model import AbstractPyProcessModel
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType

import numpy as np

@implements(proc=Proc)
@requires(CPU)
class PyProcModel(AbstractPyProcessModel):
    inp: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)
    out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)
    var: np.ndarray = LavaPyType(np.ndarray, float)

    def __init__(self, model_id, name):
        super().__init__(model_id, name)
        self.iteration = 0

    def run(self):
        if self.iteration < 2:
            val = self.inp.recv()
            print(f"Process={self.name}, "
                  f"iteration={self.iteration}, received val={val}")
            self.var[:] += val
            self.out.send(self.var)
            print(f"Process={self.name}, Var={self.var}")
            self.iteration += 1


from lava.magma.core.run_configs import RunConfig
class SelectFirstRunCfg(RunConfig):
    def select(self, process, process_models):
        return process_models[0]


from lava.magma.core.run_conditions import RunContinuous


if __name__ == "__main__":
    p1, p2, p3 = Proc(), Proc(), Proc()
    p1.out.connect(p2.inp)
    p2.out.connect(p3.inp)
    p1.run(condition=RunContinuous(), run_cfg=SelectFirstRunCfg())
    print("Running...")
    import time

    time.sleep(3)
    print("Pausing")
    p1.pause()
    print('Paused')

    print(p3.var.get())

    print('Stopping')
    p1.stop()
    print('Stopped')