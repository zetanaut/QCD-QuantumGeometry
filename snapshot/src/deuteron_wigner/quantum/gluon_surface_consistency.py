"""Necessary tests of a two-point gluonic surface influence functional.

Euclidean/static candidate, NOT a QCD vacuum, light-baryon action or LF
Hamiltonian. The positive smooth kernel is a convenient finite-range test,
not a claimed physical spectral correlator. Its strength absorbs color/g^2.
Exact SU(3) identities use preserved connection utilities, not this kernel.
"""
from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy.special import erf

from .modular_gauge_network import GaugeGraph, Path, su3
from .lf_path_states import baryon_seed

PAIRS = tuple(combinations(range(4), 2))


def _positive(value, name, zero=False):
    if not np.isfinite(value) or value < 0 or (not zero and value == 0):
        raise ValueError(f"{name} must be finite and {'nonnegative' if zero else 'positive'}.")
    return float(value)


@dataclass(frozen=True)
class Patch:
    """p(u,v)=origin+u*first+v*second, 0<=u,v<=1; lengths GeV^-1.

    Six bivector components are in mu<nu order, with no extra 1/2.
    Swapping the two edges reverses orientation without changing the surface.
    """
    origin: tuple
    first: tuple
    second: tuple

    def __post_init__(self):
        for name in ('origin', 'first', 'second'):
            a = np.asarray(getattr(self, name), float)
            if a.shape != (4,) or not np.isfinite(a).all():
                raise ValueError('Finite Euclidean four-vectors required.')
            object.__setattr__(self, name, tuple(a))
        if np.linalg.norm(self.bivector()) == 0:
            raise ValueError('Patch edges must span a nonzero area.')

    def bivector(self):
        u, v = self.first, self.second
        return np.array([u[a]*v[b]-u[b]*v[a] for a, b in PAIRS])

    def translated(self, shift):
        return Patch(tuple(np.array(self.origin)+shift), self.first, self.second)

    def reversed(self):
        return Patch(self.origin, self.second, self.first)

    def quadrature(self, order):
        if isinstance(order, bool) or not isinstance(order, (int, np.integer)) or order < 2:
            raise ValueError('Integer quadrature order >=2 required.')
        x, w = np.polynomial.legendre.leggauss(order)
        x, w = (x+1)/2, w/2
        u, v = np.meshgrid(x, x, indexing='ij')
        points = (np.array(self.origin)+u.ravel()[:, None]*self.first
                  +v.ravel()[:, None]*self.second)
        flux = np.outer(np.outer(w, w).ravel(), self.bivector())
        return points, flux


def rectangle(width, time):
    width, time = _positive(width, 'width'), _positive(time, 'time')
    return Patch((0.,)*4, (width, 0., 0., 0.), (0., 0., 0., time))


def closed_box(lengths):
    """Six outward faces of a 3-volume in Euclidean axes 0,1,2."""
    lengths = np.asarray(lengths, float)
    if lengths.shape != (3,) or not np.isfinite(lengths).all() or np.any(lengths <= 0):
        raise ValueError('Three positive finite side lengths required.')
    faces = []
    for axis in range(3):
        j, k = (axis+1) % 3, (axis+2) % 3
        u, v = np.zeros(4), np.zeros(4)
        u[j], v[k] = lengths[j], lengths[k]
        for side in (0, 1):
            origin = np.zeros(4); origin[axis] = side*lengths[axis]
            faces.append(Patch(origin, u if side else v, v if side else u))
    return tuple(faces)


def surface_pairing(left, right, strength=1., length=1., order=12):
    """Integral dS_A dS'_A d exp(-|x-y|²/ell²); real, no cumulants beyond 2.

    d has units GeV^4 and ell GeV^-1; result dimensionless. This is the
    proposed scalarized correlation functional, not an exact Wilson average.
    """
    d = _positive(strength, 'strength', zero=True)
    ell = _positive(length, 'correlation length')
    if not left or not right:
        raise ValueError('Nonempty oriented surfaces required.')
    p, f = zip(*(s.quadrature(order) for s in left))
    q, h = zip(*(s.quadrature(order) for s in right))
    p, f, q, h = map(np.concatenate, (p, f, q, h))
    total = 0.
    for start in range(0, len(p), 128):
        distance2 = np.sum((p[start:start+128, None]-q[None])**2, axis=2)
        total += np.sum(np.exp(-distance2/ell**2)*(f[start:start+128]@h.T))
    return float(d*total)


def quadratic_action(surface, strength=1., length=1., order=12):
    return .5*surface_pairing(surface, surface, strength, length, order)


def interval_integral(size, length=1.):
    """Independent exact double integral exp(-(u-v)^2/ell²) on [0,L]^2."""
    size = _positive(size, 'size', zero=True)
    ell = _positive(length, 'correlation length')
    r = size/ell
    if r < 1e-3:
        return size**2*(1-r*r/6+r**4/30-r**6/168)
    return float(np.sqrt(np.pi)*ell*size*erf(r)+ell**2*np.expm1(-r*r))


def static_potential(separation, strength=1., length=1.):
    """lim_T S2(rectangle R,T)/T, GeV; soft static reference, not p/n binding."""
    d = _positive(strength, 'strength', zero=True)
    ell = _positive(length, 'correlation length')
    return .5*d*np.sqrt(np.pi)*ell*interval_integral(separation, ell)


def static_slope(separation, strength=1., length=1.):
    """dV/dR, GeV² (positive inward attraction magnitude, not signed force)."""
    r = _positive(separation, 'separation', zero=True)
    d = _positive(strength, 'strength', zero=True)
    ell = _positive(length, 'correlation length')
    return float(.5*np.pi*d*ell**2*erf(r/ell))


def box_action_exact(lengths, strength=1., length=1.):
    closed_box(lengths)  # common geometric input validation, not quadrature
    d = _positive(strength, 'strength', zero=True)
    ell = _positive(length, 'correlation length')
    integrals = [interval_integral(x, ell) for x in lengths]
    return float(d*sum(integrals[(i+1) % 3]*integrals[(i+2) % 3]
                       *(-np.expm1(-(lengths[i]/ell)**2)) for i in range(3)))


def cube_graph(seed=17, amplitude=.55):
    """Fixed random connection for exact algebra tests; not a vacuum ensemble."""
    vertices = ((0,0,0),(1,0,0),(1,1,0),(0,1,0),
                (0,0,1),(1,0,1),(1,1,1),(0,1,1))
    rng = np.random.default_rng(seed)
    edges = [(a, b) for a, b in combinations(range(8), 2)
             if sum(abs(x-y) for x, y in zip(vertices[a], vertices[b])) == 1]
    return GaugeGraph({e: su3(amplitude*rng.normal(size=8)) for e in edges})


@dataclass(frozen=True)
class BoundaryRewrite:
    prefix: Path
    face: Path
    based_face: Path
    boundary: Path


def rewrite_boundary(boundary, index, detour):
    """Exact W(old)=W(based_face) W(new), preserving auxiliary connectors.

    Replace one edge by a detour with the same endpoints. This is discrete
    Stokes algebra on arbitrary connection values, not lattice dynamics.
    """
    v, detour = boundary.vertices, tuple(detour)
    if (v[0] != v[-1] or not isinstance(index, (int, np.integer))
            or isinstance(index, bool) or not 0 <= index < len(v)-1
            or len(detour) < 2 or (detour[0], detour[-1]) != v[index:index+2]):
        raise ValueError('Closed boundary and endpoint-matched edge replacement required.')
    prefix = v[:index+1]
    face = (v[index], v[index+1])+detour[-2::-1]
    based = prefix+face[1:]+prefix[-2::-1]
    new = v[:index]+detour+v[index+2:]
    return BoundaryRewrite(Path(prefix), Path(face), Path(based), Path(new))


def cube_rewrites():
    """Bottom square versus four sides and top, all based at vertex zero."""
    boundary = Path((0,1,2,3,0)); rewrites = []
    for edge, detour in (((0,1),(0,4,5,1)), ((1,2),(1,5,6,2)),
                         ((2,3),(2,6,7,3)), ((3,0),(3,7,4,0))):
        i = next(j for j in range(len(boundary.vertices)-1)
                 if boundary.vertices[j:j+2] == edge)
        step = rewrite_boundary(boundary, i, detour)
        rewrites.append(step); boundary = step.boundary
    return tuple(rewrites)


def factored_loop(graph, edge=None, generator=None, freeze_connectors=False):
    """Exact alternative surface product and optional left link derivative.

    Derivative is a color-connection response, NOT an electromagnetic current.
    freeze_connectors is a deliberately inconsistent diagnostic.
    """
    steps = cube_rewrites()
    paths = [s.based_face for s in steps]+[steps[-1].boundary]
    factors = [graph.transport(p) for p in paths]
    total = np.eye(3, dtype=complex)
    for factor in factors:
        total = total@factor
    if edge is None:
        return total
    if generator is None:
        raise ValueError('A link generator is required for a tangent.')
    tangent = np.zeros((3,3), complex)
    for i, path in enumerate(paths):
        if freeze_connectors and i < len(steps):
            p = graph.transport(steps[i].prefix)
            derivative = p@graph.path_tangent(steps[i].face, edge, generator)@p.conj().T
        else:
            derivative = graph.path_tangent(path, edge, generator)
        term = np.eye(3, dtype=complex)
        for j, factor in enumerate(factors):
            term = term@(derivative if i == j else factor)
        tangent += term
    return total, tangent


def baryon_wilson(matrices):
    """epsilon-epsilon three-line color contraction; no light-quark dynamics."""
    if len(matrices) != 3 or any(np.shape(u) != (3,3) for u in matrices):
        raise ValueError('Three fundamental transport matrices required.')
    e = baryon_seed()
    return np.vdot(e, np.kron(matrices[0], np.kron(matrices[1], matrices[2]))@e)
