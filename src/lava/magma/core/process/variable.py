# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
import typing as ty
import numpy as np

from lava.magma.core.process.interfaces import \
    AbstractProcessMember, IdGeneratorSingleton


# ToDo: (AW) Clean up class docstring
class Var(AbstractProcessMember):
    """Represents a Lava variable. A Var implements the state of a Process and
    is part of its public user interface. Vars have the following properties:

    - Vars are numeric objects: Typically vars represent ints, float data types.
    - Vars are tensor-valued: In general Vars represent multiple numeric
    values not just scalar objects with a shape.
    - Vars can be initialized with numeric objects with a dimensionality
    equal or less than specified by its shape. The initial value will be
    broadcast to the shape of the Var at compile current_ts.
    - Vars have a name: The Variable name will be assigned by the parent
    process of a Var.
    - Vars are mutable at runtime.
    - Vars are owned by a Process but shared-memory access by other Process
    is possible though should be used with caution.

    How to enable interactive Var access?
    Executable ----------
                        |
    Var -> Process -> Runtime -> RuntimeService -> ProcModel -> Var

    - Var can access Runtime via parent Process.
    - The compiler could have prepared the Executable with mapping
    information where each Var of a Process got mapped to. I.e. these can
    just be the former ExecVars. So the ExecVars are just stored inside the
    Executable.
    - Alternatively, the Executable stores a map from var_id -> ExecVar

    """

    def __init__(
            self,
            shape: ty.Tuple[int, ...],
            init: ty.Union[bool, int, list, np.ndarray] = 0,
            shareable: bool = True):
        """Initializes a new Lava variable.

        Parameters:
            :param shape: Tuple specifying the shape of variable tensor.
            :param init: Initial value assigned to Var. Compiler will broadcast
            'init' to 'shape' of Var at compile current_ts.
            :param shareable: Specifies whether Variable allows shared memory
            access by processes other than the Var's parent process.
        """
        super().__init__(shape)
        self.init = init
        self.shareable = shareable
        self.id: int = VarServer().register(self)
        self.name: str = "Unnamed variable"
        self.aliased_var: ty.Optional[Var] = None

    def alias(self, other_var: 'Var'):
        """Establishes an 'alias' relationship between this and 'other_var'.
        The other Var must be a member of a strict sub processes of this
        Var's parent process which might be instantiated within a
        SubProcessModel that implements this Var's parent process.
        Both, this and 'other_var' must have the same 'shape' and be both
        'shareable' or not.

        Parameters:
            :param other_var: The other Var that this Var is an alias for.
            Calls to Var.set(..) or Var.get() will be deferred to the aliased
            Var.
        """
        # Check compatibility of this and 'other_var'
        if not isinstance(other_var, Var):
            raise AssertionError("'other_var' must be a Var instance.")
        if self.shape != other_var.shape:
            raise AssertionError("Shapes of this and 'other_var' must "
                                 "be the same.")
        if self.shareable != other_var.shareable:
            raise AssertionError("'shareable' attribute of this and "
                                 "'other_var' must be the same.")

        # Establish 'alias' relationship
        self.aliased_var = other_var

    def validate_alias(self):
        """Validates that any aliased Var is a member of a Process that is a
        strict sub-Process of this Var's Process."""

        if self.aliased_var:
            other_var = self.aliased_var
            # Check that aliased Var has a different process and is a strict
            # sub process
            has_different_proc = self.process != other_var.process
            is_strict_sub_proc = other_var.process.is_sub_proc_of(self.process)
            # Throw exception
            if not has_different_proc or not is_strict_sub_proc:
                other_proc = other_var.process
                raise AssertionError(
                    f"The aliased Var '{other_var.name}' in process '"
                    f"'{other_proc.name}::{other_proc.__class__.__name__}' "
                    f"must be a member of a process that is a strict sub "
                    f"process of the aliasing Var's '{self.name}' in process "
                    f"'{self.process.name}::{self.process.__class__.__name__}'"
                    f".")

    def set(self, value: np.ndarray, idx: np.ndarray = None):
        """Sets value of Var. If this Var aliases another Var, then set(..) is
        delegated to aliased Var."""
        if self.aliased_var is not None:
            self.aliased_var.set(value, idx)
        else:
            if self.process.runtime:
                self.process.runtime.set_var(self.id, value, idx)
            else:
                raise ValueError(
                    "No Runtime available yet. Cannot set new 'Var' without "
                    "Runtime.")

    def get(self, idx: np.ndarray = None) -> np.ndarray:
        """Gets and returns value of Var. If this Var aliases another Var,
        then get() is delegated to aliased Var."""
        if self.aliased_var is not None:
            return self.aliased_var.get(idx)
        else:
            if self.process.runtime:
                return self.process.runtime.get_var(self.id, idx)
            else:
                return self.init

    def __repr__(self) -> str:
        rep = super().__repr__()
        return (
            rep
            + f"\n    shareable: {self.shareable}"
            + f"\n    value: {self.get()}"
        )


class VarServer(IdGeneratorSingleton):
    """VarServer singleton keeps track of all existing Vars and issues
    new globally unique Var ids."""

    instance: ty.Optional["VarServer"] = None
    is_not_initialized: bool = True

    def __new__(cls):
        if VarServer.instance is None:
            VarServer.instance = object.__new__(VarServer)
        return VarServer.instance

    def __init__(self):
        if VarServer.is_not_initialized:
            super().__init__()
            self.vars: ty.List[Var] = []
            VarServer.is_not_initialized = False

    @property
    def num_vars(self):
        """Returns number of vars created so far."""
        return len(self.vars)

    def register(self, var: Var) -> int:
        """Registers a Var with VarServer."""
        if not isinstance(var, Var):
            raise AssertionError("'var' must be a Var.")
        self.vars.append(var)
        return self.get_next_id()

    def reset_server(self):
        """Resets the VarServer to initial state."""
        self.vars = []
        self._next_id = 0
        VarServer.reset_singleton()
