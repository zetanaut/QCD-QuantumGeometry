"""Necessary order-g² covariance control before finite-cell/Fock projection.

Distinguishable-quark elastic t-channel amplitude, per g² and T1^a T2^a.
There is no identical-flavor exchange or annihilation diagram in this
channel. Reconstruct the propagating term from the ACTUAL cubic kernels and
the instantaneous term from the ordered Gauss density. Compare against the
independent covariant current contraction on energy-conserving external
legs. This is a colored partonic perturbative diagnostic, not a nucleon or
deuteron prediction. Failure here cannot be repaired by a finite-state
spin relabeling. It does not by itself specify the required counterterms.
"""
from math import sqrt
import numpy as np

from deuteron_wigner.bridge.modes.core import GAMMA, GAMMA_PLUS, GAMMA_MINUS
from .m2_canonical_emission import lf_spinor
from .m2_cubic_lorentz_ward import _momentum, _minus, emission_minus_vertex, source_regulator_lorentz
from .m2_quartic_spin_density import instantaneous_gluon_density


def elastic_distinct_quark_tree(momenta, masses, *, regulator=None):
    """All four external spin indices, order (out1,out2;in1,in2).

    momenta=(out1,out2,in1,in2), masses=(m1,m2). All external physical
    plus momenta are positive and all four momentum components conserved.
    The exchanged gluon is the source positive-energy light-front mode
    with k+=(abs q+) and kT=sign(q+) qT, not a covariant off-shell field
    inserted into the Gaussian. The source regulator acts on all three
    physical legs at each cubic vertex and all four at the Gauss vertex.

    Returned matrices are dimensionless covariant-external-normalization
    amplitudes. Intermediate C3*C3/Delta E and C4 use the canonical flat
    field normalization; the four external sqrt(2p+) factors are removed
    once at the end. q+=0 and q²=0 require other domains/limits, not zeros.
    """
    p = np.array([_momentum(v) for v in momenta])
    m = np.asarray(masses, float)
    if p.shape != (4, 3) or m.shape != (2,) or not np.all(np.isfinite(m)) or np.any(m < 0):
        raise ValueError("four physical endpoints and two finite nonnegative masses required")
    ms = np.tile(m, 2)
    minus = np.array([_minus(q, mass) for q, mass in zip(p, ms)])
    if (not np.allclose(p[:2].sum(axis=0), p[2:].sum(axis=0), rtol=0, atol=1e-12)
            or not np.isclose(minus[:2].sum(), minus[2:].sum(), rtol=0, atol=1e-12)):
        raise ValueError("on-shell four-momentum conservation required")
    q, qm = p[2]-p[0], minus[2]-minus[0]
    qsq = 2*q[0]*qm-q[1:]@q[1:]
    if abs(q[0]) < 1e-12 or abs(qsq) < 1e-12:
        raise ValueError("nonzero exchange plus and invariant transfer required")
    k = np.sign(q[0])*q
    propagating = np.zeros((4, 4), complex)
    for h in (-1, 1):
        if q[0] > 0:
            a = emission_minus_vertex(p[0], k, p[2], m[0], h)
            b = emission_minus_vertex(p[3], k, p[1], m[1], h).conj().T
        else:
            a = emission_minus_vertex(p[2], k, p[0], m[0], h).conj().T
            b = emission_minus_vertex(p[1], k, p[3], m[1], h)
        propagating += np.kron(a, b)/(qsq/(2*k[0]))
    spinors = [np.column_stack([lf_spinor(*leg, mass, h) for h in (-1, 1)]) for leg, mass in zip(p, ms)]
    gammas = (GAMMA_PLUS, GAMMA_MINUS, GAMMA[1], GAMMA[2])
    currents = [np.array([spinors[j].conj().T@GAMMA[0]@gamma@spinors[j+2] for gamma in gammas]) for j in (0, 1)]
    j1, j2 = currents
    norm = sqrt(np.prod(2*p[:, 0]))
    jproduct = np.kron(j1[0], j2[0])/norm
    instantaneous = (instantaneous_gluon_density(q[0], -q[0], jproduct)["canonical"]
                     +instantaneous_gluon_density(-q[0], q[0], jproduct)["canonical"])
    covariant = -(np.kron(j1[0], j2[1])+np.kron(j1[1], j2[0])
                  -np.kron(j1[2], j2[2])-np.kron(j1[3], j2[3]))/qsq
    weights = [1., 1., 1.]
    if regulator is not None:
        for j in (0, 1):
            weights[j] = source_regulator_lorentz([p[j], k, p[j+2]], [m[j], 0., m[j]], "F1", regulator)[0]
        weights[2] = source_regulator_lorentz(p, ms, "F1", regulator)[0]
    propagating *= norm
    instantaneous *= norm
    product_weight = weights[0]*weights[1]
    return {"propagating": propagating, "instantaneous": instantaneous,
            "bare_total": propagating+instantaneous, "covariant": covariant,
            "regulated_total": product_weight*propagating+weights[2]*instantaneous,
            "covariant_tensor_weight": product_weight,
            "light_front_direction_term": (weights[2]-product_weight)*instantaneous,
            "cubic_weights": tuple(weights[:2]), "quartic_weight": weights[2],
            "exchange_GeV2": float(qsq), "positive_exchange_mode": k,
            "energy_denominator_GeV": float(qsq/(2*k[0])),
            "external_current_conservation_residual": max(float(np.linalg.norm(q[0]*j[1]+qm*j[0]-q[1]*j[2]-q[2]*j[3])) for j in currents)}
