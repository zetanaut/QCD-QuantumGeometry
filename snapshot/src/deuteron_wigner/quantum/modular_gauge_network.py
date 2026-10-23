"""Ordered SU(3) and shared neural-kernel building blocks for our architecture.

These are finite graph/operator controls, not an LF Hamiltonian, quantized
gauge-field state, nucleon fit, or proof of physical topological order.
U_ab transports a fundamental vector at b to a. Paths list vertices from a
to b; their matrix product acts right-to-left on the vector at b.
"""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.linalg import expm


@lru_cache(maxsize=1)
def generators():
    """Hermitian fundamental generators, tr(t_a t_b)=delta_ab/2."""
    out = []
    for i, j in ((0, 1), (0, 2), (1, 2)):
        x = np.zeros((3, 3), complex)
        x[i, j] = x[j, i] = .5
        y = np.zeros((3, 3), complex)
        y[i, j], y[j, i] = -.5j, .5j
        out.extend((x, y))
    out.extend((np.diag([1., -1., 0.])/2,
                np.diag([1., 1., -2.])/(2*np.sqrt(3))))
    result = np.array(out)
    result.setflags(write=False)
    return result


def su3(coordinates):
    c = np.asarray(coordinates, float)
    if c.shape != (8,) or not np.all(np.isfinite(c)):
        raise ValueError("Eight finite real Lie-algebra coordinates required.")
    return expm(1j*np.einsum("a,aij->ij", c, generators()))


def adjoint(u):
    t = generators()
    return (2*np.einsum("aij,bji->ab", t, u@t@u.conj().T)).real


def _validate_su3(u):
    a = np.asarray(u, complex)
    if a.shape != (3, 3) or not np.all(np.isfinite(a)):
        raise ValueError("Finite 3x3 SU(3) matrix required.")
    if np.linalg.norm(a.conj().T@a-np.eye(3)) > 2e-10 or abs(np.linalg.det(a)-1) > 2e-10:
        raise ValueError("Transport must be unitary with determinant one.")
    return a


@dataclass(frozen=True)
class Path:
    vertices: tuple
    # A physical process label is NOT flipped by Hermitian/path reversal.
    process: str = "UNSPECIFIED"

    def __post_init__(self):
        if len(self.vertices) < 1:
            raise ValueError("A path needs at least one vertex.")

    def reversed(self):
        return Path(self.vertices[::-1], self.process)


class GaugeGraph:
    def __init__(self, links):
        self.links = {}
        for (a, b), u in links.items():
            if a == b or (b, a) in self.links:
                raise ValueError("No self-edge or duplicated reverse edge.")
            self.links[a, b] = _validate_su3(u).copy()
        self.nodes = tuple(sorted({v for edge in self.links for v in edge}))

    def link(self, a, b):
        if (a, b) in self.links:
            return self.links[a, b]
        if (b, a) in self.links:
            return self.links[b, a].conj().T
        raise ValueError(f"Missing physical transport edge {(a, b)}.")

    def transport(self, path):
        if any(v not in self.nodes for v in path.vertices):
            raise ValueError("Path vertex not represented.")
        result = np.eye(3, dtype=complex)
        for a, b in zip(path.vertices[:-1], path.vertices[1:]):
            result = result@self.link(a, b)
        return result

    def transformed(self, gauges):
        if set(gauges) != set(self.nodes):
            raise ValueError("One local gauge matrix required at every node.")
        g = {n: _validate_su3(u) for n, u in gauges.items()}
        return GaugeGraph({(a, b): g[a]@u@g[b].conj().T for (a, b), u in self.links.items()})

    def loop(self, path):
        if path.vertices[0] != path.vertices[-1]:
            raise ValueError("A Wilson loop must close.")
        return np.trace(self.transport(path))/3

    def path_tangent(self, path, edge, generator):
        """dU_path/d epsilon for U_edge -> exp(i epsilon t) U_edge.

        Repeated edges and reverse occurrences all participate.
        """
        if edge not in self.links:
            raise ValueError("Differentiate a stored oriented edge.")
        pairs = list(zip(path.vertices[:-1], path.vertices[1:]))
        factors = [self.link(a, b) for a, b in pairs]
        result = np.zeros((3, 3), complex)
        for k, pair in enumerate(pairs):
            if pair == edge:
                derivative = 1j*generator@self.links[edge]
            elif pair == edge[::-1]:
                derivative = -1j*self.links[edge].conj().T@generator
            else:
                continue
            term = np.eye(3, dtype=complex)
            for j, factor in enumerate(factors):
                term = term@(derivative if j == k else factor)
            result += term
        return result

    def tree_gauge(self, tree, root):
        """Gauge fix tree links to identity while retaining every other link.

        Erasing the non-tree links afterwards changes physical information;
        it is NOT gauge fixing.
        """
        if root not in self.nodes or len(tree) != len(self.nodes)-1:
            raise ValueError("A connected spanning tree and its root required.")
        g = {root: np.eye(3, dtype=complex)}
        pending = list(tree)
        while pending:
            progressed = False
            for edge in pending[:]:
                a, b = edge
                if a in g and b not in g:
                    g[b] = g[a]@self.link(a, b)
                elif b in g and a not in g:
                    g[a] = g[b]@self.link(b, a)
                else:
                    continue
                pending.remove(edge)
                progressed = True
            if not progressed:
                raise ValueError("Edges are not a connected spanning tree.")
        if set(g) != set(self.nodes):
            raise ValueError("Tree omitted a represented node.")
        return self.transformed(g), g


def baryon_color(graph, paths, columns):
    """Epsilon contraction of three transported color columns, not a wavefunction.

    Independent of the junction gauge but generally dependent on the three
    physical paths. Spin, flavor and fermion antisymmetry are separate.
    """
    if len(paths) != 3 or len(columns) != 3 or len({p.vertices[0] for p in paths}) != 1:
        raise ValueError("Three paths sharing one baryon junction required.")
    return np.linalg.det(np.column_stack([graph.transport(p)@q for p, q in zip(paths, columns)]))


class SharedLoopKernel:
    """Two tanh feature layers with trained final coefficients, shared per loop.

    F_theta = sum_l theta . [h2(1-ReTr W_l/3)-h2(0)].
    This finite nonlinear invariant functional supplies its own Lie derivative.
    Hidden features are fixed in this bounded inference control; this is not
    a fully trained deep state model or a selected physical gauge action.
    """
    def __init__(self, first, bias_first, second, bias_second, coefficients):
        self.first = np.asarray(first, float)
        self.bias_first = np.asarray(bias_first, float)
        self.second = np.asarray(second, float)
        self.bias_second = np.asarray(bias_second, float)
        self.coefficients = np.asarray(coefficients, float)
        n = len(self.first)
        if self.first.shape != (n,) or self.bias_first.shape != (n,) or self.second.shape != (n, n) or self.bias_second.shape != (n,) or self.coefficients.shape != (n,):
            raise ValueError("Consistent finite hidden layers and coefficient vector required.")
        if not all(np.all(np.isfinite(v)) for v in (self.first, self.bias_first, self.second, self.bias_second, self.coefficients)):
            raise ValueError("Finite network parameters required.")

    def features(self, z):
        h = np.tanh(np.asarray(z)[..., None]*self.first+self.bias_first)
        result = np.tanh(h@self.second.T+self.bias_second)
        h0 = np.tanh(self.bias_first)
        return result-np.tanh(self.second@h0+self.bias_second)

    def feature_derivative(self, z):
        h = np.tanh(np.asarray(z)[..., None]*self.first+self.bias_first)
        v = np.tanh(h@self.second.T+self.bias_second)
        return (1-v*v)*(((1-h*h)*self.first)@self.second.T)

    def design(self, graph, loops):
        z = np.array([1-graph.loop(p).real for p in loops])
        return self.features(z).sum(axis=0)

    def value(self, graph, loops):
        return float(self.design(graph, loops)@self.coefficients)

    def current_design(self, graph, loops):
        result = np.zeros((len(graph.links), 8, len(self.coefficients)))
        for path in loops:
            z = 1-graph.loop(path).real
            derivative = self.feature_derivative(z)
            for i, edge in enumerate(graph.links):
                for a, t in enumerate(generators()):
                    dz = -np.trace(graph.path_tangent(path, edge, t)).real/3
                    result[i, a] += dz*derivative
        return result

    def current(self, graph, loops):
        return self.current_design(graph, loops)@self.coefficients


def gauge_divergence(graph, current):
    """Local Noether residual for a pure closed-loop functional.

    This is a classical color-response identity, not an EM form factor or a
    quantum Gauss-law state constraint.
    """
    matrices = np.einsum("ea,aij->eij", current, generators())
    residual = {v: np.zeros((3, 3), complex) for v in graph.nodes}
    for ((a, b), u), j in zip(graph.links.items(), matrices):
        residual[a] += j
        residual[b] -= u.conj().T@j@u
    return residual


def commutator_loop(first, second):
    """Closed path retaining relative non-Abelian orientation of two based loops."""
    if first.vertices[0] != first.vertices[-1] or second.vertices[0] != second.vertices[-1] or first.vertices[0] != second.vertices[0]:
        raise ValueError("Two closed loops at the same base required.")
    walks = (first.vertices, second.vertices, first.vertices[::-1], second.vertices[::-1])
    vertices = walks[0]
    for walk in walks[1:]:
        vertices += walk[1:]
    return Path(vertices, "COMMUTATOR_CONTROL")


def normalized_real_characters(u):
    """SU(3) representations 3, 6 and 8; normalized real characters.

    Each chi_R(U V^dagger)/d_R is a positive-definite group kernel;
    its real part remains a real positive-semidefinite kernel.
    """
    u = np.asarray(u, complex)
    tr = np.trace(u, axis1=-2, axis2=-1)
    tr2 = np.trace(u@u, axis1=-2, axis2=-1)
    return np.stack((tr.real/3, ((tr*tr+tr2)/2).real/6, (abs(tr)**2-1)/8), axis=-1)


def positive_character_kernel(configurations, weights):
    """A sufficient positive cross-time kernel family, not full LF dynamics.

    K_ij=exp[sum_R a_R (Re chi_R(U_i U_j^dagger)/d_R-1)], a_R>=0.
    PSD follows from character Gram kernels and the entrywise exponential
    power series (Schur products), not from checking a few eigenvalues.
    Nonnegative a_R are a sufficient parametrization, not a necessary form
    for every physical interaction or transfer kernel.
    """
    u = np.asarray([_validate_su3(x) for x in configurations])
    a = np.asarray(weights, float)
    if len(u) == 0 or a.shape != (3,) or not np.all(np.isfinite(a)) or np.any(a < 0):
        raise ValueError("Configurations and three finite nonnegative weights required.")
    relative = u[:, None]@u[None, :].conj().swapaxes(-1, -2)
    return np.exp((normalized_real_characters(relative)-1)@a)


def dress_spatial_potential(transfer, potential, time_step):
    """D K D, D=diag(exp[-dt V/2]); preserves PSD when K is PSD.

    A common potential offset only rescales the transfer. It is subtracted
    here to avoid overflow, and returned for tracking that scalar factor.
    This finite-kernel property is not a continuum reflection-positivity proof.
    """
    k, v = np.asarray(transfer, float), np.asarray(potential, float)
    if k.shape != (len(v), len(v)) or not np.all(np.isfinite(k)) or not np.all(np.isfinite(v)) or not np.isfinite(time_step) or time_step <= 0:
        raise ValueError("Finite transfer/potential and positive time step required.")
    if np.linalg.norm(k-k.T) > 1e-10:
        raise ValueError("Symmetric transfer required.")
    offset = float(np.min(v))
    d = np.exp(-time_step*(v-offset)/2)
    return d[:, None]*k*d[None, :], offset
