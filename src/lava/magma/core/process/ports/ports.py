# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
import typing as ty
from abc import ABC, abstractmethod
import math

from lava.magma.core.process.interfaces import AbstractProcessMember
import lava.magma.core.process.ports.exceptions as pe
from lava.magma.core.process.ports.reduce_ops import AbstractReduceOp
from lava.magma.core.process.variable import Var


def to_list(obj: ty.Any) -> ty.List[ty.Any]:
    """If 'obj' is not a list, converts 'obj' into [obj]."""
    if not isinstance(obj, list):
        obj = [obj]
    return obj


def is_disjoint(a: ty.List, b: ty.List):
    """Checks that both lists are disjoint."""
    return set(a).isdisjoint(set(b))


class AbstractPort(AbstractProcessMember):
    """Abstract base class for any type of port of a Lava Process.

    Ports of a process can be connected to ports of other processes to enable
    message-based communication via channels. Sub classes of AbstractPort
    only facilitate connecting to other compatible ports. Message-passing
    itself is only handled after compilation at runtime by port
    implementations within the corresponding ProcessModel.

    Ports are tensor-valued, have a name and a parent process. In addition,
    a port may have zero or more input and output connections that contain
    references to ports that connect to this port or that this port connects
    to. Port to port connections are directional and connecting ports,
    effectively means to associate them with each other as inputs or outputs.
    These connections, imply an a-cyclic graph structure that allows the
    compiler to infer connections between processes.
    """

    def __init__(self, shape: ty.Tuple):
        super().__init__(shape)
        self.in_connections: ty.List[AbstractPort] = []
        self.out_connections: ty.List[AbstractPort] = []

    def _validate_ports(
            self,
            ports: ty.List["AbstractPort"],
            port_type: ty.Type["AbstractPort"],
            assert_same_shape: bool = True,
            assert_same_type: bool = False):
        """Checks that each port in 'ports' is of type 'port_type' and that
        shapes of each port is identical to this port's shape."""
        cls_name = port_type.__name__
        specific_cls = ports[0].__class__
        for p in ports:
            if not isinstance(p, port_type):
                raise AssertionError("'ports' must be of type {} but "
                                     "found {}.".format(cls_name, p.__class__))
            if assert_same_type:
                if not isinstance(p, specific_cls):
                    raise AssertionError(
                        "All ports must be of same type but found {} "
                        "and {}.".format(specific_cls, p.__class__)
                    )
            if assert_same_shape:
                if self.shape != p.shape:
                    raise AssertionError("Shapes {} and {} "
                                         "are incompatible."
                                         .format(self.shape, p.shape))

    def _add_inputs(self, inputs: ty.List["AbstractPort"]):
        """Adds new input connections to port. Does not allow that same
        inputs get connected more than once to port."""
        if not is_disjoint(self.in_connections, inputs):
            raise pe.DuplicateConnectionError()
        self.in_connections += inputs

    def _add_outputs(self, outputs: ty.List["AbstractPort"]):
        """Adds new output connections to port. Does not allow that same
        outputs get connected more than once to port."""
        if not is_disjoint(self.out_connections, outputs):
            raise pe.DuplicateConnectionError()
        self.out_connections += outputs

    def _connect_forward(
            self,
            ports: ty.List["AbstractPort"],
            port_type: ty.Type["AbstractPort"],
            assert_same_shape: bool = True,
            assert_same_type: bool = True):
        """Creates a forward connection from this AbstractPort to other
        ports by adding other ports to this AbstractPort's out_connection and
        by adding this AbstractIOPort to other port's in_connections."""

        self._validate_ports(
            ports, port_type, assert_same_shape, assert_same_type
        )
        # Add other ports to this port's output connections
        self._add_outputs(ports)
        # Add this port to input connections of other ports
        for p in ports:
            p._add_inputs([self])

    def _connect_backward(
            self,
            ports: ty.List["AbstractPort"],
            port_type: ty.Type["AbstractPort"],
            assert_same_shape: bool = True,
            assert_same_type: bool = True):
        """Creates a backward connection from other ports to this
        AbstractPort by adding other ports to this AbstractPort's
        in_connection and by adding this AbstractPort to other port's
        out_connections."""

        self._validate_ports(
            ports, port_type, assert_same_shape, assert_same_type
        )
        # Add other ports to this port's input connections
        self._add_inputs(ports)
        # Add this port to output connections of other ports
        for p in ports:
            p._add_outputs([self])

    def get_src_ports(self, _include_self=False) -> ty.List["AbstractPort"]:
        """Returns the list of all source ports that connect either directly
        or indirectly (through other ports) to this port."""
        if len(self.in_connections) == 0:
            if _include_self:
                return [self]
            else:
                return []
        else:
            ports = []
            for p in self.in_connections:
                ports += p.get_src_ports(True)
            return ports

    def get_dst_ports(self, _include_self=False) -> ty.List["AbstractPort"]:
        """Returns the list of all destination ports that this port connects to
        either directly or indirectly (through other ports)."""
        if len(self.out_connections) == 0:
            if _include_self:
                return [self]
            else:
                return []
        else:
            ports = []
            for p in self.out_connections:
                ports += p.get_dst_ports(True)
            return ports

    def reshape(self, new_shape: ty.Tuple) -> "ReshapePort":
        """Reshapes this port by deriving and returning a new virtual
        ReshapePort with the new shape. This implies that the resulting
        ReshapePort can only be forward connected to another port.

        Parameters
        ----------
        :param new_shape: New shape of port. Number of total elements must
        not change.
        """
        if self.size != math.prod(new_shape):
            raise pe.ReshapeError(self.shape, new_shape)

        reshape_port = ReshapePort(new_shape)
        self._connect_forward(
            [reshape_port], AbstractPort, assert_same_shape=False
        )
        return reshape_port

    def flatten(self) -> "ReshapePort":
        """Flattens this port to a (N,)-shaped port by deriving and returning
        a new virtual ReshapePort with a N equal to the total number of
        elements of this port."""
        return self.reshape((self.size,))

    def concat_with(
            self,
            ports: ty.Union["AbstractPort", ty.List["AbstractPort"]],
            axis: int) -> "ConcatPort":
        """Concatenates this port with other ports in given order along given
        axis by deriving and returning a new virtual ConcatPort. This implies
        resulting ConcatPort can only be forward connected to another port.
        All ports must have the same shape outside of the concatenation
        dimension.

        Parameters
        ----------
        :param ports: Port(s) that will be concatenated after this port.
        :param axis: Axis/dimension along which ports are concatenated.
        """
        ports = [self] + to_list(ports)
        if isinstance(self, AbstractIOPort):
            port_type = AbstractIOPort
        else:
            port_type = AbstractRVPort
        self._validate_ports(ports, port_type, assert_same_shape=False)
        return ConcatPort(ports, axis)

    def __repr__(self):
        rep = super().__repr__()
        in_conns = [f"{p.name}({p._process.name})" for p in self.in_connections]
        out_conns = [f"{p.name}({p._process.name})" for p in self.out_connections]
        return (
            rep
            + f"\n    in_connections: {in_conns}"
            + f"\n    out_connections: {out_conns}"
        )


class AbstractIOPort(AbstractPort):
    """Abstract base class for InPorts and OutPorts.
    This class needs no implementation and only serves to establish a clear
    type hierarchy needed for validating connections.
    """

    pass


class AbstractRVPort(AbstractPort):
    """Abstract base class for RefPorts and VarPorts.
    This class needs no implementation and only serves to establish a clear
    type hierarchy needed for validating connections.
    """

    pass


class AbstractSrcPort(ABC):
    """Interface for source ports such as OutPorts and RefPorts from which
    connections originate.
    This class needs no implementation and only serves to establish a clear
    type hierarchy needed for validating connections.
    """

    pass


class AbstractDstPort(ABC):
    """Interface for destination ports such as InPorts and VarPorts in which
    connections terminate.
    This class needs no implementation and only serves to establish a clear
    type hierarchy needed for validating connections.
    """

    pass


class OutPort(AbstractIOPort, AbstractSrcPort):
    """Output ports are members of a Lava Process and can be connected to
    other ports to facilitate sending of messages via channels.

    OutPorts connect to other InPorts of peer processes or to other OutPorts of
    processes that contain this OutPort's parent process as a sub process.
    Similarly, OutPorts can receive connections from other OutPorts of nested
    sub processes.
    """

    def connect(
            self, ports: ty.Union["AbstractIOPort", ty.List["AbstractIOPort"]]):
        """Connects this OutPort to other InPort(s) of another process
        or to OutPort(s) of its parent process.

        Parameters
        ----------
        :param ports: The AbstractIOPort(s) to connect to.
        """
        self._connect_forward(to_list(ports), AbstractIOPort)

    def connect_from(self, ports: ty.Union["OutPort", ty.List["OutPort"]]):
        """Connects other OutPort(s) of a nested process to this OutPort.
        OutPorts cannot receive connections from other InPorts.

        Parameters
        ----------
        :param ports: The OutPorts(s) that connect to this OutPort.
        """
        self._connect_backward(to_list(ports), OutPort)


class InPort(AbstractIOPort, AbstractDstPort):
    """Input ports are members of a Lava Process and can be connected to
    other ports to facilitate receiving of messages via channels.

    InPorts can receive connections from other OutPorts of peer processes
    or from other InPorts of processes that contain this InPort's parent
    process as a sub process. Similarly, InPorts can connect to other InPorts
    of nested sub processes.
    """

    def __init__(
            self,
            shape: ty.Tuple,
            reduce_op: ty.Optional[ty.Type[AbstractReduceOp]] = None):
        super().__init__(shape)
        self._reduce_op = reduce_op

    def connect(self, ports: ty.Union["InPort", ty.List["InPort"]]):
        """Connects this InPort to other InPort(s) of a nested process. InPorts
        cannot connect to other OutPorts.

        Parameters
        ----------
        :param ports: The InPort(s) to connect to.
        """
        self._connect_forward(to_list(ports), InPort)

    def connect_from(
            self, ports: ty.Union["AbstractIOPort", ty.List["AbstractIOPort"]]):
        """Connects other OutPort(s) to this InPort or connects other
        InPort(s) of parent process to this InPort.

        Parameters
        ----------
        :param ports: The AbstractIOPort(s) that connect to this InPort.
        """
        self._connect_backward(to_list(ports), AbstractIOPort)


class RefPort(AbstractRVPort, AbstractSrcPort):
    """RefPorts are members of a Lava Process and can be connected to
    internal Lava Vars of other processes to facilitate direct shared memory
    access to those processes.

    Shared-memory-based communication can have side-effects and should
    therefore be used with caution.

    RefPorts connect to other VarPorts of peer processes or to other RefPorts
    of processes that contain this RefPort's parent process as a sub process
    via the connect(..) method..
    Similarly, RefPorts can receive connections from other RefPorts of nested
    sub processes via the connect_from(..) method.

    Here, VarPorts only serve as a wrapper for Vars. VarPorts can be created
    statically during process definition to explicitly expose a Var for
    shared memory access (which might be safer).
    Alternatively, VarPorts can be created dynamically by connecting a
    RefPort to a Var via the connect_var(..) method."""

    def connect(
            self, ports: ty.Union["AbstractRVPort", ty.List["AbstractRVPort"]]):
        """Connects this RefPort to other VarPort(s) of another process
        or to RefPort(s) of its parent process.

        Parameters:
        -----------
        :param ports: The AbstractRVPort(s) to connect to.
        """
        for p in to_list(ports):
            if not isinstance(p, RefPort) and not isinstance(p, VarPort):
                raise TypeError(
                    "RefPorts can only be connected to RefPorts or "
                    "VarPorts: {!r}: {!r} -> {!r}: {!r}  To connect a RefPort "
                    "to a Var, use <connect_var>".format(
                        self.process.__class__.__name__, self.name,
                        p.process.__class__.__name__, p.name))
        self._connect_forward(to_list(ports), AbstractRVPort)

    def connect_from(self, ports: ty.Union["RefPort", ty.List["RefPort"]]):
        """Connects other RefPort(s) of a nested process to this RefPort.
        RefPorts cannot receive connections from other VarPorts.

        Parameters
        ----------
        :param ports: The RefPort(s) that connect to this RefPort.
        """
        for p in to_list(ports):
            if not isinstance(p, RefPort):
                raise TypeError(
                    "RefPorts can only receive connections from RefPorts: "
                    "{!r}: {!r} -> {!r}: {!r}".format(
                        self.process.__class__.__name__, self.name,
                        p.process.__class__.__name__, p.name))
        self._connect_backward(to_list(ports), RefPort)

    def connect_var(self, variables: ty.Union[Var, ty.List[Var]]):
        """Connects this RefPort to Lava Process Var(s) to facilitate shared
        memory access.

        Parameters:
        -----------
        :param variables: Var or list of Vars to connect to.
        """

        variables: ty.List[Var] = to_list(variables)
        # Check all 'variables' are actually Vars and don't have same parent
        # process as RefPort
        for v in variables:
            if not isinstance(v, Var):
                raise AssertionError(
                    "'variables' must be a Var or list of Vars but "
                    "found {}.".format(v.__class__)
                )
            if self.process is not None:
                # Only assign when parent process is already assigned
                if self.process == v.process:
                    raise AssertionError("RefPort and Var have same "
                                         "parent process.")
        var_ports = []
        var_shape = variables[0].shape
        for v in variables:
            # Check that shapes of all vars are the same
            if var_shape != v.shape:
                raise AssertionError("All 'vars' must have same shape.")
            # Create a VarPort to wrap Var
            vp = ImplicitVarPort(v)
            # Propagate name and parent process of Var to VarPort
            vp.name = "_" + v.name + "_implicit_port"
            if v.process is not None:
                # Only assign when parent process is already assigned
                vp.process = v.process
                # VarPort name could shadow existing attribute
                if hasattr(v.process, vp.name):
                    raise AssertionError(
                        "Name of implicit VarPort might conflict"
                        " with existing attribute.")
                setattr(v.process, vp.name, vp)
                v.process.var_ports.add_members({vp.name: vp})
            var_ports.append(vp)
        # Connect RefPort to VarPorts that wrap Vars
        self.connect(var_ports)

    def get_dst_vars(self) -> ty.List[Var]:
        """Returns destination Vars this RefPort is connected to."""
        return [ty.cast(VarPort, p).var for p in self.get_dst_ports()]


class VarPort(AbstractRVPort, AbstractDstPort):
    """VarPorts are members of a Lava Process and act as a wrapper for
    internal Lava Vars to facilitate connections between RefPorts and Vars
    for shared memory access from the parent process of the RefPort to
    the parent process of the Var.

    Shared-memory-based communication can have side-effects and should
    therefore be used with caution.

    VarPorts can receive connections from other RefPorts of peer processes
    or from other VarPorts of processes that contain this VarPort's parent
    process as a sub process via the connect(..) method. Similarly, VarPorts
    can connect to other VarPorts of nested sub processes via the
    connect_from(..) method.

    VarPorts can either be created in the constructor of a Process to
    explicitly expose a Var for shared memory access (which might be safer).
    Alternatively, VarPorts can be created dynamically by connecting a
    RefPort to a Var via the RefPort.connect_var(..) method."""

    def __init__(self, var: Var):
        if not isinstance(var, Var):
            raise AssertionError("'var' must be of type Var.")
        if not var.shareable:
            raise pe.VarNotSharableError(var.name)
        AbstractRVPort.__init__(self, var.shape)
        self.var = var

    def connect(self, ports: ty.Union["VarPort", ty.List["VarPort"]]):
        """Connects this VarPort to other VarPort(s) of a nested process.
        VarPorts cannot connect to other RefPorts.

        Parameters
        ----------
        :param ports: The VarPort(s) to connect to.
        """
        for p in to_list(ports):
            if not isinstance(p, VarPort):
                raise TypeError(
                    "VarPorts can only be connected to VarPorts: "
                    "{!r}: {!r} -> {!r}: {!r}".format(
                        self.process.__class__.__name__, self.name,
                        p.process.__class__.__name__, p.name))
        self._connect_forward(to_list(ports), VarPort)

    def connect_from(
            self, ports: ty.Union["AbstractRVPort", ty.List["AbstractRVPort"]]):
        """Connects other RefPort(s) to this VarPort or connects other
        VarPort(s) of parent process to this VarPort.

        Parameters
        ----------
        :param ports: The AbstractRVPort(s) that connect to this VarPort.
        """
        for p in to_list(ports):
            if not isinstance(p, RefPort) and not isinstance(p, VarPort):
                raise TypeError(
                    "VarPorts can only receive connections from RefPorts or "
                    "VarPorts: {!r}: {!r} -> {!r}: {!r}".format(
                        self.process.__class__.__name__, self.name,
                        p.process.__class__.__name__, p.name))
        self._connect_backward(to_list(ports), AbstractRVPort)

    def __repr__(self):
        rep = super().__repr__()
        var = f"{self.var.name}({self.process.name})" if self.var else "N/A"
        return (
            rep
            + f"\n    var: {var}"
        )

class ImplicitVarPort(VarPort):
    """Sub class for VarPort to identify implicitly created VarPorts when
    a RefPort connects directly to a Var."""
    pass


class AbstractVirtualPort(ABC):
    """Abstract base class interface for any type of port that merely serves
    to transforms the properties of a user-defined port.
    Needs no implementation because this class purely serves as a
    type-identifier."""

    @property
    @abstractmethod
    def _parent_port(self):
        """Must return parent port that this VirtualPort was derived from."""
        pass

    @property
    def process(self):
        """Returns parent process of parent port that this VirtualPort was
        derived from."""
        return self._parent_port.process


# ToDo: (AW) ReshapePort.connect(..) could be consolidated with
#  ConcatPort.connect(..)
class ReshapePort(AbstractPort, AbstractVirtualPort):
    """A ReshapePort is a virtual port that allows to change the shape of a
    port before connecting to another port.
    It is used by the compiler to map the indices of the underlying
    tensor-valued data array from the derived to the new shape."""

    def __init__(self, shape: ty.Tuple):
        AbstractPort.__init__(self, shape)

    @property
    def _parent_port(self) -> AbstractPort:
        return self.in_connections[0]

    def connect(self, ports: ty.Union["AbstractPort", ty.List["AbstractPort"]]):
        """Connects this ReshapePort to other port(s).

        Parameters
        ----------
        :param ports: The port(s) to connect to. Connections from an IOPort
        to a RVPort and vice versa are not allowed.
        """
        # Determine allows port_type
        if isinstance(self._parent_port, OutPort):
            # If OutPort, only allow other IO ports
            port_type = AbstractIOPort
        elif isinstance(self._parent_port, InPort):
            # If InPort, only allow other InPorts
            port_type = InPort
        elif isinstance(self._parent_port, RefPort):
            # If RefPort, only allow other Ref- or VarPorts
            port_type = AbstractRVPort
        elif isinstance(self._parent_port, VarPort):
            # If VarPort, only allow other VarPorts
            port_type = VarPort
        else:
            raise TypeError("Illegal parent port.")
        # Connect to ports
        self._connect_forward(to_list(ports), port_type)


class ConcatPort(AbstractPort, AbstractVirtualPort):
    """A ConcatPort is a virtual port that allows to concatenate multiple
    ports along given axis into a new port before connecting to another port.
    The shape of all concatenated ports outside of the concatenation
    dimension must be the same.
    It is used by the compiler to map the indices of the underlying
    tensor-valued data array from the derived to the new shape."""

    def __init__(self, ports: ty.List[AbstractPort], axis: int):
        AbstractPort.__init__(self, self._get_new_shape(ports, axis))
        self._connect_backward(
            ports, AbstractPort, assert_same_shape=False, assert_same_type=True
        )
        self.concat_axis = axis

    @staticmethod
    def _get_new_shape(ports: ty.List[AbstractPort], axis):
        """Computes shape of ConcatPort from given 'ports'."""
        # Extract shapes of given ports
        concat_shapes = [p.shape for p in ports]
        total_size = 0
        shapes_ex_axis = []
        shapes_incompatible = False
        for shape in concat_shapes:
            # Compute total size along concatenation axis
            total_size += shape[axis]
            # Extract shape dimensions other than concatenation axis
            shapes_ex_axis.append(shape[:axis] + shape[axis + 1:])
            if len(shapes_ex_axis) > 1:
                shapes_incompatible = shapes_ex_axis[-2] != shapes_ex_axis[-1]

        if shapes_incompatible:
            raise pe.ConcatShapeError(shapes_ex_axis, axis)

        # Return shape of concatenated port
        new_shape = shapes_ex_axis[0]
        return new_shape[:axis] + (total_size,) + new_shape[axis:]

    @property
    def _parent_port(self) -> AbstractPort:
        return self.in_connections[0]

    def connect(self, ports: ty.Union["AbstractPort", ty.List["AbstractPort"]]):
        """Connects this ConcatPort to other port(s)

        Parameters
        ----------
        :param ports: The port(s) to connect to. Connections from an IOPort
        to a RVPort and vice versa are not allowed.
        """
        # Determine allows port_type
        if isinstance(self._parent_port, OutPort):
            # If OutPort, only allow other IO ports
            port_type = AbstractIOPort
        elif isinstance(self._parent_port, InPort):
            # If InPort, only allow other InPorts
            port_type = InPort
        elif isinstance(self._parent_port, RefPort):
            # If RefPort, only allow other Ref- or VarPorts
            port_type = AbstractRVPort
        elif isinstance(self._parent_port, VarPort):
            # If VarPort, only allow other VarPorts
            port_type = VarPort
        else:
            raise TypeError("Illegal parent port.")
        # Connect to ports
        self._connect_forward(to_list(ports), port_type)


# ToDo: (AW) TBD...
class PermutePort(AbstractPort, AbstractVirtualPort):
    """A PermutePort is a virtual port that allows to permute the dimensions
    of a port before connecting to another port. Permutations refers to the
    change of order of the different tensor dimensions. A special case of
    permutation is transposition in 2D.
    It is used by the compiler to map the indices of the underlying
    tensor-valued data array from the derived to the new shape.

    Example:
        out_port = OutPort((2, 4, 3))
        in_port = InPort((3, 2, 4))
        out_port.permute([3, 1, 2]).connect(in_port)
    """
    pass


# ToDo: (AW) TBD...
class ReIndexPort(AbstractPort, AbstractVirtualPort):
    """A ReIndexPort is a virtual port that allows to re-index the elements
    of a port before connecting to another port. Re-indexing refers to a
    re-arrangement of the elements in a tensor without changing the shape of
    the tensor.
    It is used by the compiler to map the indices of the underlying
    tensor-valued data array from the derived to the new shape.

    Example:
        out_port = OutPort((2, 2))
        in_port = InPort((2, 2))
        out_port.reindex([3, 1, 0, 2]).connect(in_port)
    """
    pass
