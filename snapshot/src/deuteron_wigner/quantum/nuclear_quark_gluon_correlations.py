"""Nuclear scalar/trace matching and resolved LO unpolarized PDF controls.

Not a binding action, a physical PDF fit, a vacuum solver or TMD evolution.
X is a fraction of the WHOLE nuclear P+, with support [0,1]. Species order
is (q_0, qbar_0, ..., q_{nf-1}, qbar_{nf-1}, g). Remaining array axes are
untouched spectator/source/target-spin indices, not parton-helicity channels.
Scalar anomaly matrix elements are NOT gluon momentum fractions or densities.
"""
from dataclasses import dataclass

import numpy as np
from scipy.integrate import quad_vec
from scipy.linalg import expm

from ..bridge.r0.core import PlusDistributionTerm

CF, CA, TR = 4.0 / 3.0, 3.0, 0.5


def _nf(nf):
    if isinstance(nf, (bool, np.bool_)) or not isinstance(nf, (int, np.integer)) or not 1 <= nf <= 6:
        raise ValueError("One to six fixed active quark flavors required.")
    return int(nf)


def _array(value, nf):
    a = np.asarray(value)
    if a.ndim < 1 or a.shape[0] != 2 * nf + 1 or not np.isfinite(a).all():
        raise ValueError("Finite resolved quark/antiquark/gluon array required.")
    return a


def nonsinglet_moment(n):
    """Integral z**(n-1) P_qq(z) dz; alpha_s/(2 pi) is NOT included."""
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError("Positive integer Mellin index required.")
    harmonic = sum(1.0 / j for j in range(1, n + 1))
    return CF * (1.5 - 2 * harmonic + 1.0 / (n * (n + 1)))


def moment_generator(n, nf):
    """Independent analytic LO Mellin matrix, full flavor/antiquark content.

    n>=2 because the unrestricted gluon number moment has a small-z pole.
    n=2 column sums vanish: total momentum is conserved, not each species.
    """
    nf = _nf(nf)
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 2:
        raise ValueError("Singlet Mellin index must be an integer >=2.")
    size = 2 * nf + 1
    m = np.zeros((size, size))
    for j in range(size - 1):
        m[j, j] = nonsinglet_moment(n)
        m[j, -1] = TR * (1 / n - 2 / (n + 1) + 2 / (n + 2))
        m[-1, j] = CF * (2 / (n - 1) - 2 / n + 1 / (n + 1))
    harmonic = sum(1.0 / j for j in range(1, n))
    beta0 = 11 - 2 * nf / 3
    m[-1, -1] = 2 * CA * (-harmonic + 1 / (n - 1) - 2 / n
                         + 1 / (n + 1) - 1 / (n + 2)) + beta0 / 2
    return m


def kernel_moment_by_quadrature(n, nf):
    """Independent distribution integration, reusing the preserved D0 action."""
    nf = _nf(nf)
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 2:
        raise ValueError("Singlet Mellin index must be an integer >=2.")
    weight = lambda z: z ** (n - 1)
    plus = PlusDistributionTerm(1.0).action(weight)
    regular, _ = quad_vec(
        lambda z: weight(z) * np.array([-(1 + z), z*z + (1-z)**2,
                                       2/z - 2 + z, 1/z - 2 + z - z*z]),
        0, 1, epsabs=2e-12, epsrel=2e-12)
    m = np.zeros((2 * nf + 1, 2 * nf + 1))
    for j in range(2 * nf):
        m[j, j] = CF * (2 * plus + regular[0] + 1.5)
        m[j, -1] = TR * regular[1]
        m[-1, j] = CF * regular[2]
    m[-1, -1] = 2 * CA * (plus + regular[3]) + (11 - 2 * nf / 3) / 2
    return m


def lo_x_action(pdf, x, nf):
    """Actual (P_LO convolution f)(x), without alpha_s/(2 pi).

    Smooth callable PDFs, 0<x<1; no endpoint delta-valued input PDFs.
    The z=1 plus subtraction and its finite lower-limit term are explicit.
    This is an infinitesimal evolution action, NOT a finite-step x-space solve.
    """
    nf = _nf(nf)
    if not np.isfinite(x) or not 0 < x < 1:
        raise ValueError("An interior whole-target momentum fraction required.")
    at_x = _array(pdf(x), nf)

    def sampled(z):
        value = _array(pdf(x / z), nf)
        if value.shape != at_x.shape:
            raise ValueError("PDF spectator axes changed with x.")
        return value / z

    plus, _ = quad_vec(lambda z: (sampled(z) - at_x) / (1 - z), x, 1,
                       epsabs=2e-10, epsrel=2e-10)
    plus += at_x * np.log1p(-x)

    def regular(z):
        f = sampled(z)
        out = np.empty_like(f, dtype=np.result_type(f, float))
        out[:-1] = -CF * (1 + z) * f[:-1] + TR * (z*z + (1-z)**2) * f[-1]
        out[-1] = CF * (2/z - 2 + z) * np.sum(f[:-1], axis=0)
        out[-1] += 2 * CA * (1/z - 2 + z - z*z) * f[-1]
        return out

    out, _ = quad_vec(regular, x, 1, epsabs=2e-10, epsrel=2e-10)
    out[:-1] += CF * (2 * plus[:-1] + 1.5 * at_x[:-1])
    out[-1] += 2 * CA * plus[-1] + (11 - 2 * nf / 3) / 2 * at_x[-1]
    return out


def evolve_moment(values, n, nf, alpha_at_mu0, mu0, mu):
    """Fixed-nf LO running and resummed Mellin transport; not a physical fit.

    All scales GeV, alpha is alpha_s. No heavy-flavor thresholds, small-x
    resummation or rapidity evolution are implied. Caller qualifies the
    perturbative scale domain; a finite denominator is not that qualification.
    """
    nf = _nf(nf)
    a = _array(values, nf)
    if not np.isfinite((alpha_at_mu0, mu0, mu)).all() or min(alpha_at_mu0, mu0, mu) <= 0:
        raise ValueError("Positive finite coupling and scales required.")
    beta0 = 11 - 2 * nf / 3
    denominator = 1 + alpha_at_mu0 * beta0 / (4 * np.pi) * 2 * np.log(mu / mu0)
    if denominator <= 0:
        raise ValueError("Evolution crosses the one-loop Landau pole.")
    alpha = alpha_at_mu0 / denominator
    evolution_time = 2 / beta0 * np.log(denominator)
    matrix = expm(evolution_time * moment_generator(n, nf))
    return np.tensordot(matrix, a, axes=(1, 0)), float(alpha)


def embed_nucleon_pdfs(pdfs, plus_fractions, x, nf):
    """One resolved nuclear momentum configuration, not a nuclear wave solve.

    Nucleon y_a=P_a+/P_A+ sum to one in this impulse reference. Each f_a
    uses its OWN nucleon fraction. F_A(X)=sum theta(y_a-X) f_a(X/y_a)/y_a.
    A nuclear wavefunction average and connected operators remain separate.
    """
    nf = _nf(nf)
    fractions = np.asarray(plus_fractions, float)
    if (fractions.ndim != 1 or len(pdfs) != len(fractions) or len(pdfs) == 0
            or not np.isfinite(fractions).all() or np.any(fractions <= 0)
            or not np.isclose(fractions.sum(), 1, rtol=0, atol=2e-13)):
        raise ValueError("One positive normalized plus fraction per nucleon required.")
    if not np.isfinite(x) or not 0 < x < 1:
        raise ValueError("An interior whole-target momentum fraction required.")
    template = _array(pdfs[0](0.5), nf)
    out = np.zeros_like(template, dtype=np.result_type(template, float))
    for pdf, y in zip(pdfs, fractions):
        if x < y:
            sample = _array(pdf(x / y), nf)
            if sample.shape != out.shape:
                raise ValueError("All nucleons must use common resolved spectator axes.")
            out = out + sample / y
    return out


@dataclass(frozen=True)
class ScalarBindingMatch:
    mass: float
    sigma: np.ndarray
    anomaly: float
    connected_sigma: np.ndarray
    connected_anomaly: float


def binding_scalar_match(a, z, proton_mass, neutron_mass, proton_sigma,
                         neutron_sigma, binding, binding_mass_response, gamma_mass):
    """Necessary forward scalar matching in one fixed QCD scheme/active theory.

    sigma_f=m_f dM/dm_f; binding_mass_response_f=m_f dB/dm_f.
    gamma_mass=-d ln(m_f)/d ln(mu); Θ=β/(2g)[G²]+sum(1+gamma_f)[m_f qbar_f q_f].
    Forward, vacuum-disconnected-subtracted matrix elements / (2 M) are used.
    'anomaly' is this SCALAR operator contribution, not trace(T_g), gluon
    energy, PDF momentum, a positive occupation or separately observable mass.
    Inputs must come from a common qualified action for a physical application.
    """
    if (isinstance(a, bool) or isinstance(z, bool)
            or not isinstance(a, (int, np.integer)) or not isinstance(z, (int, np.integer))
            or a < 1 or not 0 <= z <= a):
        raise ValueError("Integer A>=1 and 0<=Z<=A required.")
    p, n, b, gamma = map(lambda v: np.asarray(v, float),
                        (proton_sigma, neutron_sigma, binding_mass_response, gamma_mass))
    if (p.ndim != 1 or p.size == 0 or n.shape != p.shape or b.shape != p.shape
            or gamma.shape != p.shape or not all(np.isfinite(v).all() for v in (p,n,b,gamma))):
        raise ValueError("Common finite flavor-resolved sigma and gamma arrays required.")
    if not np.isfinite((proton_mass, neutron_mass, binding)).all() or min(proton_mass, neutron_mass) <= 0:
        raise ValueError("Finite binding and positive nucleon masses required.")
    mass = z * proton_mass + (a-z) * neutron_mass - binding
    if mass <= 0:
        raise ValueError("Positive total target mass required.")
    sigma = z*p + (a-z)*n - b
    anomaly = mass - np.dot(1+gamma, sigma)
    return ScalarBindingMatch(float(mass), sigma, float(anomaly), -b.copy(),
                              float(-binding + np.dot(1+gamma, b)))


def moment_null_shape(x):
    """Signed shape with zero number AND momentum moments; not a physical PDF.

    Its N=3 Mellin moment is 1/420, showing that two conserved moments cannot
    identify the x-dependent correlation. It is not a chosen residual ansatz.
    """
    x = np.asarray(x)
    return x * (1-x) * (1-5*x+5*x*x)
