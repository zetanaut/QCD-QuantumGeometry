"""Finite causal response and matching controls, NOT a production QCD action.

Mass squared and z are GeV^2. Residues are dimensionless for the normalized
response C(z) (GeV^-2). Physical current dimensions must be restored using
declared source factors. Positive physical spectral matrices, signed matched
remainders and colored interaction kernels are different objects.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class SpectralResponse:
    """C(z)=sum_n R_n/(s_n-z), a finite, unsubtracted dispersion reference.

    `physical_response` requires PSD residues for compatible Hermitian
    color-singlet sources in a positive-metric space. `signed_remainder`
    does not. These roles classify spectral positivity, not a certificate
    of local QFT: crossing/locality and allowed quantum-number pairings
    need separate qualification. Neither constructs interacting QCD.
    No positivity condition on a colored gluon/quark kernel is inferred.
    """
    mass_squared: np.ndarray
    residues: np.ndarray
    role: str

    def __post_init__(self):
        s, r = np.asarray(self.mass_squared, float), np.asarray(self.residues, complex)
        if s.ndim != 1 or not len(s) or r.ndim != 3 or r.shape[0] != len(s) or r.shape[1] != r.shape[2]:
            raise ValueError("Nonempty masses and one square residue matrix per pole required.")
        if not np.isfinite(s).all() or not np.isfinite(r).all() or np.any(s < 0):
            raise ValueError("Finite residues and nonnegative mass squared required.")
        if not np.allclose(r, r.conj().transpose(0, 2, 1), atol=1e-13, rtol=0):
            raise ValueError("Hermitian spectral residues required.")
        if self.role not in ("physical_response", "signed_remainder"):
            raise ValueError("Explicit physical-response or signed-remainder role required.")
        if self.role == "physical_response" and np.min(np.linalg.eigvalsh(r)) < -1e-12:
            raise ValueError("Physical-response spectral residues must be PSD.")
        s, r = s.copy(), r.copy()
        s.flags.writeable = r.flags.writeable = False
        object.__setattr__(self, "mass_squared", s)
        object.__setattr__(self, "residues", r)

    def analytic(self, z):
        z = np.asarray(z, complex)
        if not np.isfinite(z).all():
            raise ValueError("Finite invariant argument required.")
        denominators = self.mass_squared-z[..., None]
        if np.any(denominators == 0):
            raise ValueError("Use a boundary limit, not evaluation on an exact pole.")
        return np.einsum("...n,nab->...ab", 1/denominators, self.residues)

    def euclidean(self, q_squared):
        q = np.asarray(q_squared, float)
        if np.any(q < 0) or not np.isfinite(q).all():
            raise ValueError("Nonnegative finite Euclidean Q^2 required.")
        return self.analytic(-q)

    def retarded(self, omega, spatial_squared):
        """Upper-half omega continuation, Fourier convention exp(-i omega t)."""
        omega = np.asarray(omega, complex)
        if np.any(omega.imag <= 0) or not np.isfinite(spatial_squared) or spatial_squared < 0:
            raise ValueError("Im omega > 0 and nonnegative spatial momentum squared required.")
        return self.analytic(omega**2-spatial_squared)

    def moment(self, order):
        if isinstance(order, bool) or not isinstance(order, int) or order < 0:
            raise ValueError("Nonnegative integer spectral moment required.")
        return np.einsum("n,nab->ab", self.mass_squared**order, self.residues)

    def time_response(self, times, spatial_squared=0., *, euclidean=False):
        """Fixed-spatial-momentum retarded or Euclidean free-mode response.

        Retarded response is zero for t<0; Euclidean response is exp(-E|t|)/2E.
        These are exact finite-pole controls, not interacting multipoint
        unitarity or Lorentz covariance of the quark functional.
        """
        t = np.asarray(times, float)
        if not np.isfinite(t).all() or not np.isfinite(spatial_squared) or spatial_squared < 0:
            raise ValueError("Finite times and nonnegative spatial momentum squared required.")
        e = np.sqrt(self.mass_squared+spatial_squared)
        if np.any(e == 0):
            raise ValueError("Separate treatment needed for the zero-energy mode.")
        kernel = (np.exp(-abs(t[..., None])*e)/(2*e) if euclidean else
                  np.where(t[..., None] >= 0, np.sin(t[..., None]*e)/e, 0.))
        return np.einsum("...n,nab->...ab", kernel, self.residues)

    def reflection_matrix(self, times, spatial_squared=0.):
        """Matrix K_(ia,jb)=C_E(t_i+t_j)_ab for positive Euclidean times."""
        t = np.asarray(times, float)
        if t.ndim != 1 or np.any(t <= 0):
            raise ValueError("Strictly positive reflection-test times required.")
        c = self.time_response(t[:, None]+t[None, :], spatial_squared, euclidean=True)
        return c.transpose(0, 2, 1, 3).reshape(len(t)*c.shape[-1], -1)

    def subtracted_remainder(self, q_squared, reference_squared, order):
        """C_E(Q^2) minus its first `order` Taylor terms about Q0^2.

        This is the subtraction-independent remainder, not a full physical
        correlator with unknown contact/subtraction constants set to zero.
        """
        q, q0 = np.asarray(q_squared, float), float(reference_squared)
        if not np.isfinite(q).all() or np.any(q < 0) or not np.isfinite(q0) or q0 < 0:
            raise ValueError("Finite nonnegative subtraction arguments required.")
        if isinstance(order, bool) or not isinstance(order, int) or order < 1:
            raise ValueError("At least one explicit subtraction required.")
        s = self.mass_squared
        if np.any(s+q0 == 0) or np.any(s+q[..., None] == 0):
            raise ValueError("Subtraction cannot be evaluated on a massless pole.")
        k = (-(q-q0)[..., None])**order / ((s+q0)**order*(s+q[..., None]))
        return np.einsum("...n,nab->...ab", k, self.residues)

    def minus(self, retained):
        if self.residues.shape[1:] != retained.residues.shape[1:]:
            raise ValueError("Full and retained source coordinates must agree.")
        # Do not clip the difference to a positive cone. Shared source units,
        # renormalization and operator definitions must also agree physically.
        return SpectralResponse(np.r_[self.mass_squared, retained.mass_squared],
                                np.concatenate([self.residues, -retained.residues]),
                                "signed_remainder")


def pcac_pole(pion_mass, decay_constant, pseudoscalar_residue, current_mass, source_scale):
    """Normalized (P, div A / mu) pole from explicitly supplied PCAC inputs.

    Source overlaps a=(G_pi/mu^2, f_pi*m_pi^2/mu^3) are dimensionless;
    a_A=(2m/mu)*a_P. This transports a supplied pole, never generates one.
    Restoring mu^4 gives the dimension-GeV^2 correlator for (P,div A/mu).
    The Ward contact polynomial and transverse axial current are NOT given.
    """
    values = [pion_mass, decay_constant, pseudoscalar_residue, current_mass, source_scale]
    if not np.isfinite(values).all() or min(values) <= 0:
        raise ValueError("Positive explicit finite-mass pole and source normalization required.")
    lhs, rhs = decay_constant*pion_mass**2, 2*current_mass*pseudoscalar_residue
    if not np.isclose(lhs, rhs, rtol=1e-10, atol=1e-14):
        raise ValueError("Supplied residues do not satisfy finite-mass PCAC.")
    a = np.array([pseudoscalar_residue/source_scale**2, lhs/source_scale**3])
    return SpectralResponse(np.array([pion_mass**2]), np.array([np.outer(a, a)]), "physical_response")


def gaussian_response(mass_matrix, sources, z, transform=None, *, transform_sources=True):
    """Exact real-field quadratic source response B^T(M^2-z I)^-1 B.

    A -> T(z)^T A T(z), B -> T(z)^T B. Transpose, NOT Hermitian
    conjugation at complex z, preserves analyticity of real-field kernels.
    This is a source-equivalence control, not a QCD bound-state solver.
    """
    m, b = np.asarray(mass_matrix), np.asarray(sources)
    if np.ndim(z) != 0 or not np.isfinite(z):
        raise ValueError("One finite invariant argument required.")
    if m.ndim != 2 or m.shape[0] != m.shape[1] or b.ndim != 2 or b.shape[0] != len(m):
        raise ValueError("Compatible square mass matrix and source columns required.")
    if not np.isrealobj(m) or not np.isrealobj(b) or not np.isfinite(m).all() or not np.isfinite(b).all():
        raise ValueError("Finite real-field mass matrix and source columns required.")
    if not np.allclose(m, m.T, atol=1e-13, rtol=0) or np.min(np.linalg.eigvalsh(m)) <= 0:
        raise ValueError("Positive symmetric mass-squared matrix required.")
    a = m-z*np.eye(len(m))
    if transform is not None:
        t = np.asarray(transform)
        if t.shape != m.shape or not np.isfinite(t).all() or np.linalg.matrix_rank(t) < len(m):
            raise ValueError("Finite invertible field transformation required.")
        a = t.T@a@t
        if transform_sources:
            b = t.T@b
    return b.T@np.linalg.solve(a, b)


def spectral_null_experiment():
    """Two positive spectra with identical finite spacelike data and UV moments.

    Fixed poles; residues are diagnostic parameters, NOT a reconstruction
    of physical QCD data. Exact finite nullspace differs from instability
    of unique analytic continuation with complete exact continuum data.
    """
    s = np.linspace(.25, 4., 9)
    q = np.array([.2, .6, 1.4, 3.])
    constraints = np.vstack([1/(q[:, None]+s), np.ones(len(s)), s])
    _, singular, vh = np.linalg.svd(constraints, full_matrices=True)
    rank = int(np.linalg.matrix_rank(constraints))
    null = vh[rank:]
    target = np.zeros(len(s)); target[4] = 1
    v = null.T@(null@target)
    v /= np.max(abs(v))
    weights = np.full(len(s), 1/len(s))
    change = .8*np.min(weights)*v
    left, right = weights+change, weights-change
    response = lambda w: SpectralResponse(s, w[:, None, None], "physical_response")
    a, b = response(left), response(right)
    time = np.linspace(.2, 8., 65)
    z = s[4]+.025j  # resolved invariant-mass spectral bin, not an epsilon limit
    spectrum_a, spectrum_b = a.analytic(z)[0, 0].imag, b.analytic(z)[0, 0].imag
    return dict(mass_squared=s, spacelike_q_squared=q, residues_a=left, residues_b=right,
                constraint_rank=rank, nullity=len(s)-rank, singular_values=singular,
                constraint_max_error=float(np.max(abs(constraints@(left-right)))),
                smeared_spectral_relative_difference=float(abs(spectrum_a-spectrum_b)/max(abs(spectrum_a), abs(spectrum_b))),
                spectral_bin_z=z,
                euclidean_time_relative_difference=float(np.linalg.norm(a.time_response(time, euclidean=True)-b.time_response(time, euclidean=True))/np.linalg.norm(a.time_response(time, euclidean=True))))


def resolving_probes(mass_squared, spacelike_q_squared, width):
    """Greedy resolved spectral bins for a KNOWN finite-pole family only.

    Starts with the spacelike points and M0/M1. Unit row scaling is a
    diagnostic conditioning convention, not a measurement covariance or
    a statistically optimal physical experimental design.
    """
    s, q = np.asarray(mass_squared, float), np.asarray(spacelike_q_squared, float)
    if s.ndim != 1 or q.ndim != 1 or not len(s) or not np.isfinite(s).all() or not np.isfinite(q).all() or np.any(s <= 0) or np.any(q < 0) or not np.isfinite(width) or width <= 0:
        raise ValueError("Positive pole masses, nonnegative spacelike points and bin width required.")
    rows = np.vstack([1/(q[:, None]+s), np.ones(len(s)), s])
    candidates = width/((s[:, None]-s[None, :])**2+width**2)
    rows /= np.linalg.norm(rows, axis=1)[:, None]
    candidates /= np.linalg.norm(candidates, axis=1)[:, None]
    selected = []
    while np.linalg.matrix_rank(rows) < len(s):
        rank = np.linalg.matrix_rank(rows)
        null = np.linalg.svd(rows, full_matrices=True)[2][rank:]
        scores = np.linalg.norm(candidates@null.T, axis=1)
        scores[selected] = -1
        index = int(np.argmax(scores))
        if scores[index] < 1e-12:
            raise ValueError("Candidate spectral bins do not resolve the remaining directions.")
        selected.append(index)
        rows = np.vstack([rows, candidates[index]])
    return rows, np.array(selected, int)
