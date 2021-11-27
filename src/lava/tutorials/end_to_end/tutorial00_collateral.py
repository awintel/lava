import numpy as np
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var
from lava.magma.core.process.ports.ports import InPort, OutPort
from lava.magma.core.run_conditions import RunContinuous
from lava.magma.core.decorator import implements, requires
from lava.magma.core.model.sub.model import AbstractSubProcessModel
from lava.magma.core.model.py.model import AbstractPyProcessModel
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.resources import CPU
from lava.magma.core.run_configs import RunConfig


class LeafProc(AbstractProcess):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.inp = InPort(shape=(1,))
        self.out = OutPort(shape=(1,))
        self.var = Var(shape=(1,), init=1)


@implements(proc=LeafProc)
@requires(CPU)
class PyLeafProcModel(AbstractPyProcessModel):
    inp: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)
    out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)
    var: np.ndarray = LavaPyType(np.ndarray, float)

    def __init__(self, model_id, name):
        super().__init__(model_id, name)
        self.iteration = 0

    def run(self):
        if self.iteration < 2:
            val = self.inp.recv()
            print(f"Process={self.name}: "
                  f"iter={self.iteration}, "
                  f"var + recv_val = {self.var[0]} + {val[0]} = "
                  f"{self.var[0]+val[0]}")
            self.var[:] += val
            self.out.send(self.var)
            self.iteration += 1


class CompositeProc(AbstractProcess):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.inp1 = InPort(shape=(1,))
        self.inp2 = InPort(shape=(1,))
        self.out = OutPort(shape=(1,))
        self.var = Var(shape=(1,), init=1)


@implements(proc=CompositeProc)
class CompositeSubProcModel(AbstractSubProcessModel):
    def __init__(self, proc: CompositeProc):
        super().__init__(proc)
        # Instantiate sub-Processes
        self.proc1 = LeafProc()
        self.proc2 = LeafProc()
        self.proc3 = LeafProc()
        
        # Connect sub-Processes amongst each other
        self.proc3.inp.connect_from([self.proc1.out, self.proc2.out])
        
        # Connect sub-Processes to parent Process
        proc.inp1.connect(self.proc1.inp)
        proc.inp2.connect(self.proc2.inp)
        self.proc3.out.connect(proc.out)
        
        # Expose internal Var of sub-Process
        proc.var.alias(self.proc3.var)


class LeafProcRunCfg(RunConfig):
    def select(self, process, process_models):
        return process_models[0]


class CompProcRunCfg(RunConfig):
    def select(self, process, process_models):
        selected_pm = None
        for pm in process_models:
            if issubclass(pm, AbstractSubProcessModel):
                selected_pm = pm
            elif not selected_pm and issubclass(pm, AbstractPyProcessModel):
                selected_pm = pm
        return selected_pm


def leaf_proc_demo():
    p1, p2, p3 = LeafProc(), LeafProc(), LeafProc()
    p1.out.connect(p2.inp)
    p2.out.connect(p3.inp)
    p1.run(condition=RunContinuous(), run_cfg=LeafProcRunCfg())
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


def comp_proc_demo():
    p = CompositeProc()
    p.run(condition=RunContinuous(), run_cfg=CompProcRunCfg())
    print("Running...")
    import time

    time.sleep(3)
    print("Pausing")
    p.pause()
    print('Paused')

    print(p.var.get())

    print('Stopping')
    p.stop()
    print('Stopped')


if __name__ == "__main__":
    # leaf_proc_demo()
    comp_proc_demo()
