# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
import typing as ty
from abc import ABC, abstractmethod

import numpy as np

from lava.magma.compiler.channels.pypychannel import CspSendPort, CspRecvPort
from lava.magma.core.model.model import AbstractProcessModel
from lava.magma.core.model.py.ports import AbstractPyPort, PyVarPort
from lava.magma.runtime.mgmt_token_enums import (
    enum_to_np,
    MGMT_COMMAND,
    MGMT_RESPONSE, REQ_TYPE)


class AbstractPyProcessModel(AbstractProcessModel, ABC):
    """Abstract interface for Python ProcessModels.

    Example for how variables and ports might be initialized:
        a_in: PyInPort =   LavaPyType(PyInPort.VEC_DENSE, float)
        s_out: PyInPort =  LavaPyType(PyOutPort.VEC_DENSE, bool, precision=1)
        u: np.ndarray =    LavaPyType(np.ndarray, np.int32, precision=24)
        v: np.ndarray =    LavaPyType(np.ndarray, np.int32, precision=24)
        bias: np.ndarray = LavaPyType(np.ndarray, np.int16, precision=12)
        du: int =          LavaPyType(int, np.uint16, precision=12)
    """

    def __init__(self, model_id: int, name: str):
        super().__init__(model_id, name)
        self.service_to_process_cmd: ty.Optional[CspRecvPort] = None
        self.process_to_service_ack: ty.Optional[CspSendPort] = None
        self.service_to_process_req: ty.Optional[CspRecvPort] = None
        self.process_to_service_data: ty.Optional[CspSendPort] = None
        self.service_to_process_data: ty.Optional[CspRecvPort] = None
        self.py_ports: ty.List[AbstractPyPort] = []
        self.var_ports: ty.List[PyVarPort] = []
        self.var_id_to_var_map: ty.Dict[int, ty.Any] = {}

        self._cmd_handlers = {
            MGMT_COMMAND.STOP[0]: self._stop,
            MGMT_COMMAND.PAUSE[0]: self._pause
        }
        if not hasattr(self, 'run'):
            self.run = None

    def _stop(self):
        self.process_to_service_ack.send(MGMT_RESPONSE.TERMINATED)
        self.join()

    def _pause(self):
        self.process_to_service_ack.send(MGMT_RESPONSE.PAUSED)
        # Handle get/set Var requests from runtime service
        self._handle_get_set_var()

    def _handle_get_var(self):
        """Handles the get Var command from runtime service."""
        # 1. Receive Var ID and retrieve the Var
        var_id = self.service_to_process_req.recv()[0].item()
        var_name = self.var_id_to_var_map[var_id]
        var = getattr(self, var_name)

        # 2. Send Var data
        data_port = self.process_to_service_data
        # Header corresponds to number of values
        # Data is either send once (for int) or one by one (array)
        if isinstance(var, int) or isinstance(var, np.integer):
            data_port.send(enum_to_np(1))
            data_port.send(enum_to_np(var))
        elif isinstance(var, np.ndarray):
            # FIXME: send a whole vector (also runtime_service.py)
            var_iter = np.nditer(var)
            num_items = int(np.prod(var.shape))
            data_port.send(enum_to_np(num_items))
            for value in var_iter:
                data_port.send(enum_to_np(value, np.float64))

    def _handle_set_var(self):
        """Handles the set Var command from runtime service."""
        # 1. Receive Var ID and retrieve the Var
        var_id = self.service_to_process_req.recv()[0].item()
        var_name = self.var_id_to_var_map[var_id]
        var = getattr(self, var_name)

        # 2. Receive Var data
        data_port = self.service_to_process_data
        if isinstance(var, int) or isinstance(var, np.integer):
            # First item is number of items (1) - not needed
            data_port.recv()
            # Data to set
            buffer = data_port.recv()[0]
            if isinstance(var, int):
                setattr(self, var_name, buffer.item())
            else:
                setattr(self, var_name, buffer.astype(var.dtype))
        elif isinstance(var, np.ndarray):
            # First item is number of items
            num_items = data_port.recv()[0]
            var_iter = np.nditer(var, op_flags=['readwrite'])
            # Set data one by one
            for i in var_iter:
                if num_items == 0:
                    break
                num_items -= 1
                i[...] = data_port.recv()[0]
        else:
            raise RuntimeError("Unsupported type")

    # ToDo: (AW) Could sub infinite loops in this an next method be avoided
    #  and just made part of general message handler mechanism?
    # FIXME: (PP) might not be able to perform get/set during pause
    def _handle_get_set_var(self):
        """Handles all get/set Var requests from the runtime service and calls
        the corresponding handling methods. The loop ends upon a
        new command from runtime service after all get/set Var requests have
        been handled."""
        while True:
            # Probe if there is a get/set Var request from runtime service
            if self.service_to_process_req.probe():
                # Get the type of the request
                request = self.service_to_process_req.recv()
                if self.eq_commands(request, REQ_TYPE.GET):
                    self._handle_get_var()
                elif self.eq_commands(request, REQ_TYPE.SET):
                    self._handle_set_var()
                else:
                    raise RuntimeError(f"Unknown request type {request}")

            # End if another command from runtime service arrives
            if self.service_to_process_cmd.probe():
                return

    # TODO: (PP) use select(..) to service VarPorts instead of a loop
    def _handle_var_ports(self):
        """Handles read/write requests on any VarPorts. The loop ends upon a
        new command from runtime service after all VarPort service requests have
        been handled."""
        while True:
            # Loop through read/write requests of each VarPort
            for vp in self.var_ports:
                vp.service()

            # End if another command from runtime service arrives
            if self.service_to_process_cmd.probe():
                return

    def _run(self):
        while True:
            if self.service_to_process_cmd.probe():
                cmd = self.service_to_process_cmd.recv()[0]
                if cmd in self._cmd_handlers:
                    self._cmd_handlers[cmd]()
                    if cmd == MGMT_COMMAND.STOP[0]:
                        break
                else:
                    raise ValueError(
                        f"Illegal RuntimeService command! ProcessModels of "
                        f"type {self.__class__.__qualname__} cannot handle "
                        f"command: {cmd}")
            if self.run:
                self.run()

    def __setattr__(self, key: str, value: ty.Any):
        self.__dict__[key] = value
        if isinstance(value, AbstractPyPort):
            self.py_ports.append(value)
            # Store all VarPorts for efficient RefPort -> VarPort handling
            if isinstance(value, PyVarPort):
                self.var_ports.append(value)

    def start(self):
        self.service_to_process_cmd.start()
        self.process_to_service_ack.start()
        self.service_to_process_req.start()
        self.process_to_service_data.start()
        self.service_to_process_data.start()
        for p in self.py_ports:
            p.start()
        self._run()

    def join(self):
        self.service_to_process_cmd.join()
        self.process_to_service_ack.join()
        self.service_to_process_req.join()
        self.process_to_service_data.join()
        self.service_to_process_data.join()
        for p in self.py_ports:
            p.join()

    @staticmethod
    def eq_commands(cmd, cmd_ref):
        """Returns True if the two commands are equal."""
        return cmd[0] == cmd_ref


class PyLoihiProcessModel(AbstractPyProcessModel):
    def __init__(self, model_id: int, name: str):
        super().__init__(model_id, name)
        self.current_ts = 0
        self._cmd_handlers.update({
            self.Phase.SPK[0]: self._spike,
            self.Phase.PRE_MGMT[0]: self._pre_mgmt,
            self.Phase.LRN[0]: self._lrn,
            self.Phase.POST_MGMT[0]: self._post_mgmt,
            self.Phase.HOST[0]: self._pause
        })

    class Phase:
        SPK = enum_to_np(1)
        PRE_MGMT = enum_to_np(2)
        LRN = enum_to_np(3)
        POST_MGMT = enum_to_np(4)
        HOST = enum_to_np(5)

    # FixMe: (AW) Temporary hack because of protocol difference
    def _pause(self):
        # Handle get/set Var requests from runtime service
        self._handle_get_set_var()

    def _spike(self):
        self.current_ts += 1
        self.run_spk()
        self.process_to_service_ack.send(MGMT_RESPONSE.DONE)

    def _pre_mgmt(self):
        # Enable via guard method
        if self.pre_guard():
            self.run_pre_mgmt()
        self.process_to_service_ack.send(MGMT_RESPONSE.DONE)
        # Handle VarPort requests from RefPorts
        if len(self.var_ports) > 0:
            self._handle_var_ports()

    def _lrn(self):
        # Enable via guard method
        if self.lrn_guard():
            self.run_lrn()
        self.process_to_service_ack.send(MGMT_RESPONSE.DONE)

    def _post_mgmt(self):
        # Enable via guard method
        if self.post_guard():
            self.run_post_mgmt()
        self.process_to_service_ack.send(MGMT_RESPONSE.DONE)
        # Handle VarPort requests from RefPorts
        if len(self.var_ports) > 0:
            self._handle_var_ports()

    def run_spk(self):
        pass

    def run_pre_mgmt(self):
        pass

    def run_lrn(self):
        pass

    def run_post_mgmt(self):
        pass

    def pre_guard(self):
        pass

    def lrn_guard(self):
        pass

    def post_guard(self):
        pass

    # TODO: (PP) need to handle PAUSE command
    def run_old(self):
        """Retrieves commands from the runtime service to iterate through the
        phases of Loihi and calls their corresponding methods of the
        ProcessModels. The phase is retrieved from runtime service
        (service_to_process_cmd). After calling the method of a phase of all
        ProcessModels the runtime service is informed about completion. The
        loop ends when the STOP command is received."""
        while True:
            # Probe if there is a new command from the runtime service
            if self.service_to_process_cmd.probe():
                phase = self.service_to_process_cmd.recv()
                if self.eq_commands(phase, MGMT_COMMAND.STOP):
                    self.process_to_service_ack.send(MGMT_RESPONSE.TERMINATED)
                    self.join()
                    return
                # Spiking phase - increase time step
                if self.eq_commands(phase, self.Phase.SPK):
                    self.current_ts += 1
                    self.run_spk()
                    self.process_to_service_ack.send(MGMT_RESPONSE.DONE)
                # Pre-management phase
                elif self.eq_commands(phase, self.Phase.PRE_MGMT):
                    # Enable via guard method
                    if self.pre_guard():
                        self.run_pre_mgmt()
                    self.process_to_service_ack.send(MGMT_RESPONSE.DONE)
                    # Handle VarPort requests from RefPorts
                    if len(self.var_ports) > 0:
                        self._handle_var_ports()
                # Learning phase
                elif self.eq_commands(phase, self.Phase.LRN):
                    # Enable via guard method
                    if self.lrn_guard():
                        self.run_lrn()
                    self.process_to_service_ack.send(MGMT_RESPONSE.DONE)
                # Post-management phase
                elif self.eq_commands(phase, self.Phase.POST_MGMT):
                    # Enable via guard method
                    if self.post_guard():
                        self.run_post_mgmt()
                    self.process_to_service_ack.send(MGMT_RESPONSE.DONE)
                    # Handle VarPort requests from RefPorts
                    if len(self.var_ports) > 0:
                        self._handle_var_ports()
                # Host phase - called at the last time step before STOP
                elif self.eq_commands(phase, self.Phase.HOST):
                    # Handle get/set Var requests from runtime service
                    self._handle_get_set_var()
                else:
                    raise ValueError(f"Wrong Phase Info Received : {phase}")
