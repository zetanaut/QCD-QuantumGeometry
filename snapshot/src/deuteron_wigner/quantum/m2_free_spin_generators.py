"""Source-derived FREE light-front generators and explicit projection defects.

Symmetric p+; active transformations U(t)=exp(-itG). On flat momentum
amplitudes G=-i(v.d + div(v)/2)+A. Matrices follow source order (-,+),
NOT the usual Pauli (+,-) order. These are continuum differential actions;
cell matrices below keep their separate boundary/domain defect. Neither
this module nor transport of a supplied parent constructs interacting QCD
rotations or qualifies the existing two-Jz H2 pole as spin one half.
"""
from __future__ import annotations

from dataclasses import dataclass
from operator import index

import numpy as np

from .m2_coupled_longitudinal_jacobi import jacobi_frame, oscillator_shell


# The same helicity order as the canonical spin/flavor low bit: down, up.
SIGMA = np.array([[[0, 1], [1, 0]], [[0, 1j], [-1j, 0]],
                  [[-1, 0], [0, 1]]], dtype=complex)


def _momentum(momentum, mass):
    p = np.asarray(momentum, dtype=float)
    if (p.shape != (3,) or not np.all(np.isfinite(p)) or p[0] <= 0
            or not np.isfinite(mass) or mass < 0):
        raise ValueError("finite (p+,px,py), p+>0 and mass>=0 required")
    return p, (mass*mass + p[1]*p[1] + p[2]*p[2])/(2*p[0])


@dataclass(frozen=True)
class FreeGenerator:
    velocity: np.ndarray
    divergence: float
    spin: np.ndarray

    def apply(self, value, gradient, *, flat_measure=True):
        """Apply to a two-helicity value and its (momentum,helicity) gradient.

        Invariant dp+ d2pT/(2p+) amplitudes omit the half-density term.
        No interpolation/differentiation of finite mode samples is implied.
        """
        f, df = np.asarray(value, complex), np.asarray(gradient, complex)
        if f.shape != (2,) or df.shape != (3, 2):
            raise ValueError("value shape (2,), gradient shape (3,2) required")
        return (-1j*self.velocity@df + self.spin@f
                - (0.5j*self.divergence*f if flat_measure else 0))


def free_generator(momentum, mass, name, *, particle="quark"):
    """F1=(Bx-Jy)/sqrt2, F2=(By+Jx)/sqrt2; Ei opposite combinations.

    B is an active positive-rapidity boost. Quark A is differentiated from
    m2_canonical_emission.lf_spinor, not an assumed constituent rotation.
    Physical antiquark states have the same A: their charge-conjugated v
    spinors transform with the conjugate Wigner matrix. This is not the
    un-normal-ordered v-bilinear spin sign. Gluon A follows core.polarization with its compensating A+=0 gauge
    transformation. Only the physical MASSLESS two-helicity gluon is defined.
    """
    (p, x, y), pm = _momentum(momentum, mass)
    if particle not in ("quark", "antiquark", "gluon") or (particle == "gluon" and mass != 0):
        raise ValueError("quark, antiquark or massless two-helicity gluon required")
    z = np.zeros((2, 2), complex)
    h = SIGMA[2] * (1. if particle == "gluon" else .5)
    flip = np.zeros((2, 2, 2)) if particle == "gluon" else mass*SIGMA[:2]/2
    values = {
        "F1": (np.array([x, pm, 0.]), x/p, (-y*h-flip[1])/p),
        "F2": (np.array([y, 0., pm]), y/p, (x*h+flip[0])/p),
        "E1": (np.array([0., p, 0.]), 0., z),
        "E2": (np.array([0., 0., p]), 0., z),
        "Bz": (np.array([p, 0., 0.]), 1., z),
        "Jz": (np.array([0., -y, x]), 0., h),
    }
    combinations = {"Bx": ("E1", "F1", 1), "By": ("E2", "F2", 1),
                    "Jy": ("E1", "F1", -1), "Jx": ("F2", "E2", -1)}
    if name in combinations:
        a, b, sign = combinations[name]
        return FreeGenerator(*[(u+sign*v)/np.sqrt(2) for u, v in zip(values[a], values[b])])
    if name not in values:
        raise ValueError("unknown Lorentz generator")
    return FreeGenerator(*values[name])


def free_pauli_lubanski(momentum, mass, *, particle="quark"):
    """(W+,W-,Wx,Wy) after orbital cancellation, via Cartesian J/B.

    W0=p.J, W=E J-p cross B for the active convention above. Keep
    noncommuting matrix products when forming W squared. Massless results
    obey W=h p; no division by mass and no massive-gluon spin assignment.
    """
    p, pm = _momentum(momentum, mass)
    energy, pz = (p[0]+pm)/np.sqrt(2), (p[0]-pm)/np.sqrt(2)
    cart = np.array([p[1], p[2], pz])
    j = np.array([free_generator(p, mass, "J"+a, particle=particle).spin for a in "xyz"])
    b = np.array([free_generator(p, mass, "B"+a, particle=particle).spin for a in "xyz"])
    w0 = np.einsum("a,aij->ij", cart, j)
    cross = np.array([cart[1]*b[2]-cart[2]*b[1], cart[2]*b[0]-cart[0]*b[2],
                      cart[0]*b[1]-cart[1]*b[0]])
    w = energy*j-cross
    return np.array([(w0+w[2])/np.sqrt(2), (w0-w[2])/np.sqrt(2), w[0], w[1]])


def longitudinal_cell_operators(doubled_modes, half_length):
    """Exact Galerkin X,P+,B on phi_k=exp(i pi k x-/L)/sqrt(2L).

    Positive, distinct integer k (PBC) or half-integer k (APBC), arbitrary
    ordering. [X,P+]=i(I-B); B is the rank-one cell-face boundary, NOT an
    omitted-mode leakage estimate. X phi does not preserve the P+ domain.
    """
    inputs = tuple(doubled_modes)
    try:
        modes = tuple(index(v) for v in inputs)
    except TypeError as exc:
        raise ValueError("integer doubled modes required") from exc
    if (not modes or any(isinstance(v, (bool, np.bool_)) for v in inputs)
            or min(modes) <= 0 or len(set(modes)) != len(modes)
            or len({k % 2 for k in modes}) != 1
            or not np.isfinite(half_length) or half_length <= 0):
        raise ValueError("one positive distinct PBC/APBC lattice and L>0 required")
    k = np.asarray(modes, float)/2
    delta = k[None, :]-k[:, None]
    boundary = np.cos(np.pi*delta).astype(complex)
    coordinate = np.zeros_like(boundary)
    off = delta != 0
    coordinate[off] = -1j*half_length*boundary[off]/(np.pi*delta[off])
    return coordinate, np.diag(np.pi*k/half_length), boundary


def free_cell_galerkin(doubled_modes, half_length, cutoff, width, mass, *, particle="quark"):
    """One-particle free finite reference in source circular-HO phases.

    Order: longitudinal x circular(n+,n-) x source helicity(-,+).
    Fi=-pi X + {p-, -i d/dpi}/2 + Ai. Evaluate cubic orbital words BEFORE
    projection using an exact N+3 working shell. This is a free one-particle
    cell/reference action, not a many-body fixed-K spin operator. The cell
    boundary and omitted-shell leakage remain explicit, even at g=0.
    """
    if (isinstance(cutoff, (bool, np.bool_)) or index(cutoff) < 0
            or not np.isfinite(width) or width <= 0):
        raise ValueError("nonnegative integer cutoff and positive width required")
    free_generator([1., 0., 0.], mass, "F1", particle=particle)
    xlong, pplus, boundary = longitudinal_cell_operators(doubled_modes, half_length)
    shell = oscillator_shell(1, cutoff+3)
    lookup = {state: n for n, state in enumerate(shell)}
    annihilation = []
    for axis in range(2):
        a = np.zeros((len(shell), len(shell)), complex)
        for col, state in enumerate(shell):
            if state[axis]:
                out = list(state)
                out[axis] -= 1
                a[lookup[tuple(out)], col] = np.sqrt(state[axis])
        annihilation.append(a)
    ax = (annihilation[0]+annihilation[1])/np.sqrt(2)
    ay = 1j*(annihilation[0]-annihilation[1])/np.sqrt(2)
    momenta = [width*(a+a.conj().T)/np.sqrt(2) for a in (ax, ay)]
    derivatives = [-1j*(a-a.conj().T)/(np.sqrt(2)*width) for a in (ax, ay)]
    kinetic = mass*mass*np.eye(len(shell))+sum(k@k for k in momenta)
    n = len(oscillator_shell(1, cutoff))
    crop = lambda a: a[:n, :n]
    identity = np.eye(2)
    il, io = np.eye(len(pplus)), np.eye(n)
    inverse = np.diag(1/np.diag(pplus))
    h = SIGMA[2]*(1. if particle == "gluon" else .5)
    flip = np.zeros((2, 2, 2)) if particle == "gluon" else mass*SIGMA[:2]/2
    f = []
    for axis, k in enumerate(momenta):
        d = derivatives[axis]
        orbital = (-np.kron(xlong, crop(k))
                   +np.kron(inverse/4, crop(kinetic@d+d@kinetic)))
        spin = ((-np.kron(crop(momenta[1]), h)-np.kron(io, flip[1])) if axis == 0
                else (np.kron(crop(momenta[0]), h)+np.kron(io, flip[0])))
        f.append(np.kron(orbital, identity)+np.kron(inverse, spin))
    return {"F1": f[0], "F2": f[1], "Pplus": np.kron(pplus, np.eye(2*n)),
            "Pminus": np.kron(np.kron(inverse/2, crop(kinetic)), identity),
            "Px": np.kron(np.kron(il, crop(momenta[0])), identity),
            "Py": np.kron(np.kron(il, crop(momenta[1])), identity),
            "Jz": np.kron(il, np.kron(np.diag([a-b for a, b in shell[:n]]), identity)
                           +np.kron(io, h)),
            "cell_boundary_x": np.kron(np.kron(boundary, crop(momenta[0])), identity),
            "cell_boundary_y": np.kron(np.kron(boundary, crop(momenta[1])), identity)}


def free_jacobi_tangent(modes, coordinates, total_pplus, masses, axis):
    """Free F_axis tangent at an actual source Jacobi grid point.

    Includes differentiation of the mass-weighted Jacobi frame as x changes.
    Returned fraction velocity is a continuum tangent, NOT a finite-grid
    interpolation rule. Rows of coordinates are Q1,...,Q(N-1), total PT.
    """
    frame = jacobi_frame(modes)
    physical = frame.from_jacobi(coordinates)
    masses = np.asarray(masses, float)
    if (masses.shape != frame.fractions.shape or not np.all(np.isfinite(masses))
            or np.any(masses < 0) or not np.isfinite(total_pplus) or total_pplus <= 0
            or isinstance(axis, (bool, np.bool_)) or axis not in (0, 1)):
        raise ValueError("one nonnegative mass per slot, P+>0, axis 0/1 required")
    x = frame.fractions
    pp = total_pplus*x
    pminus = (masses*masses + np.sum(physical*physical, axis=1))/(2*pp)
    pp_dot = float(np.sum(physical[:, axis]))
    dx = (physical[:, axis]-x*pp_dot)/total_pplus
    dp = np.zeros_like(physical)
    dp[:, axis] = pminus
    df = np.zeros_like(frame.forward)
    # Each non-CM row has one positive daughter and one negative daughter.
    for n, row in enumerate(frame.forward[:-1]):
        a, b = row > 0, row < 0
        xa, xb, da, db = x[a].sum(), x[b].sum(), dx[a].sum(), dx[b].sum()
        df[n, a] = row[a]*(db/xb-da/xa-(da+db)/(xa+xb))/2
        df[n, b] = row[b]*(da/xa-db/xb-(da+db)/(xa+xb))/2
    return pp_dot, dx, df@physical + frame.forward@dp


def projected_commutator(isometry, first, second):
    """Transport supplied parent generators and expose exact leakage.

    For H2 pass U=Omega(E) N(E)^(-1/2), at ONE common anchor. Raw matched
    products require N^-1 between operators. This routine does not supply
    absent interacting generators, repair an algebra or estimate all omitted
    Fock/continuum physics. Parent operators must be explicit Hermitian matrices.
    """
    u, a, b = (np.asarray(v, complex) for v in (isometry, first, second))
    if (u.ndim != 2 or not 0 < u.shape[1] <= u.shape[0]
            or a.shape != (u.shape[0], u.shape[0]) or b.shape != a.shape
            or not all(np.all(np.isfinite(v)) for v in (u, a, b))
            or not np.allclose(u.conj().T@u, np.eye(u.shape[1]), rtol=0, atol=2e-11)
            or not np.allclose(a, a.conj().T, rtol=0, atol=2e-11)
            or not np.allclose(b, b.conj().T, rtol=0, atol=2e-11)):
        raise ValueError("finite isometry and compatible Hermitian parent operators required")
    ar, br = u.conj().T@a@u, u.conj().T@b@u
    qa, qb = a@u-u@ar, b@u-u@br
    direct = ar@br-br@ar
    parent = u.conj().T@(a@b-b@a)@u
    leakage = -qa.conj().T@qb+qb.conj().T@qa
    return {"commutator": direct, "parent_commutator": parent, "leakage": leakage,
            "first_leakage_norm": float(np.linalg.norm(qa, 2)),
            "second_leakage_norm": float(np.linalg.norm(qb, 2)),
            "identity_residual": float(np.linalg.norm(direct-parent-leakage))}


def moving_frame_connection(isometry, directional_derivative, parent_spin):
    """Transport -i v.d+A through an explicitly moving orthonormal frame U.

    directional_derivative is v.dU, INCLUDING the wave, pole and norm
    derivatives appropriate to the application. Returns U^dag A U-iU^dag dU
    and the orthogonal action (I-UU^dag)(A U-i dU). This is a product-rule
    identity, not an implementation of absent H2/interaction derivatives.
    The common flat half-density scalar remains unchanged.
    """
    u, du, a = (np.asarray(v, complex) for v in (isometry, directional_derivative, parent_spin))
    projected_commutator(u, a, a)  # same finite/Hermitian/isometry validation
    if (du.shape != u.shape or not np.all(np.isfinite(du))
            or not np.allclose(du.conj().T@u+u.conj().T@du, 0., rtol=0, atol=2e-11)):
        raise ValueError("finite derivative preserving U^dag U=I required")
    action = a@u-1j*du
    effective = u.conj().T@action
    return effective, action-u@effective
