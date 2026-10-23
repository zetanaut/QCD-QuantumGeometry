"""Local two-multiplet chiral common-action pilot at TREE level only.

This is a coupled linear-sigma/quark-meson effective reference, not a
QCD-derived collective sector, production model or momentum-dependent
quantum quark dressing. Canonical quark/gluon utilities remain separate.
Fields/masses are GeV; potential GeV^4; source c and kappa GeV^2.
"""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import least_squares, brentq, minimize

from .canonical_lf_fields import GAMMA
from .chiral_soft_matching import G5, square, vector

HBARC = .1973269804  # GeV fm, only a unit conversion


@dataclass(frozen=True)
class ChiralGraphAction:
    """V=sum_i lambda_i/4 (phi_i^2-a_i^2)^2
            +kappa/2 (phi_1-phi_2)^2 -m sum_i c_i sigma_i.

    phi_i=(sigma_i,pi_i^1,pi_i^2,pi_i^3); canonical positive local kinetic
    terms. Quarks have bare mass m plus sum_i g_i(sigma_i+i gamma5 tau.pi_i).
    External scalar/pseudoscalar sources act on BOTH quarks and c_i phi_i.
    The c_i are effective source coefficients, not inferred QCD Z factors.
    """
    quartic: np.ndarray
    radial_squared: np.ndarray
    coupling: float
    scalar_source: np.ndarray
    yukawa: np.ndarray

    def __post_init__(self):
        for name in ("quartic", "radial_squared", "scalar_source", "yukawa"):
            x = np.asarray(getattr(self, name), float)
            if x.shape != (2,) or not np.isfinite(x).all():
                raise ValueError("Two finite coefficients per resolved multiplet required.")
            x = x.copy(); x.flags.writeable = False
            object.__setattr__(self, name, x)
        if np.any(self.quartic <= 0) or np.any(self.scalar_source <= 0) or not np.isfinite(self.coupling) or self.coupling <= 0:
            raise ValueError("Positive quartics, aligned source coefficients and inter-node coupling required.")

    def potential(self, fields, current_mass, *, pseudoscalar_source=None):
        f = np.asarray(fields, float)
        if f.shape != (2, 4) or not np.isfinite(f).all() or not np.isfinite(current_mass) or current_mass < 0:
            raise ValueError("Finite O(4) fields and nonnegative mass source required.")
        p = np.zeros(3) if pseudoscalar_source is None else np.asarray(pseudoscalar_source, float)
        if p.shape != (3,) or not np.isfinite(p).all():
            raise ValueError("Three finite external pseudoscalar source components required.")
        r = np.sum(f*f, axis=1)
        return float(np.sum(self.quartic/4*(r-self.radial_squared)**2)
                     +self.coupling/2*np.sum((f[0]-f[1])**2)
                     -self.scalar_source@(current_mass*f[:, 0]+f[:, 1:]@p))

    def matrices(self, vacuum):
        v = np.asarray(vacuum, float)
        if v.shape != (2,) or not np.isfinite(v).all():
            raise ValueError("Two finite vacuum components required.")
        hp = np.diag(self.quartic*(v*v-self.radial_squared))
        hp += self.coupling*np.array([[1., -1.], [-1., 1.]])
        hs = hp+np.diag(2*self.quartic*v*v)
        return hp, hs

    def gap(self, vacuum, current_mass):
        return self.matrices(vacuum)[0]@vacuum-current_mass*self.scalar_source

    def solve(self, current_mass, start=(.05, .08)):
        if not np.isfinite(current_mass) or current_mass < 0:
            raise ValueError("Nonnegative finite mass source required.")
        initial = np.asarray(start, float)
        if initial.shape != (2,) or not np.isfinite(initial).all() or np.any(initial <= 0):
            raise ValueError("Two positive initial radial amplitudes required.")
        # The potential is strictly convex in x_i=v_i^2 after alignment:
        # -sqrt(x1*x2) and -m*c_i*sqrt(x_i) are convex and lambda_i>0.
        # Minimize the potential first; minimizing gap^2 alone can instead
        # stick at the small-field boundary. Refine its stationary point.
        def radial(x):
            v = np.sqrt(x)
            f = np.zeros((2, 4)); f[:, 0] = v
            return self.potential(f, current_mass), self.gap(v, current_mass)/(2*v)
        minimum = minimize(radial, initial**2, jac=True, method="L-BFGS-B",
                           bounds=[(1e-18, None)]*2,
                           options=dict(ftol=1e-15, gtol=1e-12, maxiter=500))
        fit = least_squares(lambda v: self.gap(v, current_mass), np.sqrt(minimum.x),
                            jac=lambda v: self.matrices(v)[1], bounds=(1e-10, np.inf),
                            xtol=1e-14, ftol=1e-14, gtol=1e-14, max_nfev=400)
        hp, hs = self.matrices(fit.x)
        if not fit.success or np.linalg.norm(self.gap(fit.x, current_mass)) > 2e-11 or np.min(np.linalg.eigvalsh(hs)) <= 0:
            raise ValueError("No stable broken-vacuum solution in this run.")
        masses, modes = np.linalg.eigh(hp)
        for i in range(2):
            if fit.x@modes[:, i] < 0: modes[:, i] *= -1
        if masses[0] < -1e-10:
            raise ValueError("Unstable pion curvature.")
        return dict(vacuum=fit.x, pion_mass_squared=masses, pion_modes=modes,
                    scalar_mass_squared=np.linalg.eigvalsh(hs),
                    decay_constants=fit.x@modes,
                    pseudoscalar_residues=self.scalar_source@modes/2,
                    quark_mass=float(current_mass+self.yukawa@fit.x),
                    gap_norm=float(np.linalg.norm(self.gap(fit.x, current_mass))))

    def scalar_vertex(self, vacuum, pion_mode):
        """sigma_i pi_light pi_light vertex from the THIRD potential derivative."""
        return 2*self.quartic*np.asarray(vacuum)*np.asarray(pion_mode)**2

    def scalar_form_factor(self, vacuum, pion_mode, q_squared):
        """F_S(Q^2)=c^T(H_sigma+Q^2)^-1 g_sigma-pi-pi, GeV.

        O_S=-delta L/dm=qbar q-sum c_i sigma_i; the two minus signs from
        induced sigma response give the positive displayed scalar vertex.
        At tree level this equals d m_pi^2/dm at Q^2=0.
        """
        q = float(q_squared)
        if not np.isfinite(q) or q < 0:
            raise ValueError("Finite spacelike Q^2 required.")
        hs = self.matrices(vacuum)[1]
        return float(self.scalar_source@np.linalg.solve(hs+q*np.eye(2), self.scalar_vertex(vacuum, pion_mode)))

    def scalar_radius_squared(self, vacuum, pion_mode):
        hs = self.matrices(vacuum)[1]
        x = np.linalg.solve(hs, self.scalar_vertex(vacuum, pion_mode))
        return float(6*HBARC**2*(self.scalar_source@np.linalg.solve(hs, x))/(self.scalar_source@x))

    def quark_scalar_vertex(self, vacuum, q_squared):
        """Common mass-source vertex, equals d(m+g.v)/dm at zero transfer."""
        q = float(q_squared)
        if not np.isfinite(q) or q < 0:
            raise ValueError("Finite spacelike Q^2 required.")
        return float(1+self.yukawa@np.linalg.solve(self.matrices(vacuum)[1]+q*np.eye(2), self.scalar_source))

    def quark_sources(self, vacuum, current_mass, outgoing, incoming):
        """Generated tree axial and weighted pseudoscalar vertices, tau/2 stripped.

        Includes both collective pion modes and bare quark source. Same
        phase convention as the preserved chiral source controls.
        """
        p1, p0 = vector(outgoing), vector(incoming)
        q = p1-p0; hp = self.matrices(vacuum)[0]
        if np.linalg.norm(self.gap(vacuum, current_mass)) > 2e-10:
            raise ValueError("Stationary vacuum required for the axial identity.")
        r = np.linalg.inv(hp-square(q)*np.eye(2))
        axial = np.array([g@G5 for g in GAMMA])+2*(self.yukawa@r@vacuum)*np.einsum("m,ij->mij", q, G5)
        pseudoscalar = 2*current_mass*(1+self.yukawa@r@self.scalar_source)*G5
        return axial, pseudoscalar

    def quark_mass_matrix(self, fields, current_mass, *, pseudoscalar_source=None):
        """Resolved Dirac x flavor x color Yukawa/bare mass operator, 24x24."""
        f = np.asarray(fields, float)
        if f.shape != (2, 4) or not np.isfinite(f).all():
            raise ValueError("Two finite O(4) multiplets required.")
        tau = np.array([[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]])
        result = (current_mass+self.yukawa@f[:, 0])*np.eye(24, dtype=complex)
        for i in range(2):
            for a in range(3):
                result += self.yukawa[i]*f[i, a+1]*np.kron(np.kron(1j*G5, tau[a]), np.eye(3))
        if pseudoscalar_source is not None:
            p = np.asarray(pseudoscalar_source, float)
            if p.shape != (3,) or not np.isfinite(p).all():
                raise ValueError("Three finite external pseudoscalar source components required.")
            for a in range(3):
                result += p[a]*np.kron(np.kron(1j*G5, tau[a]), np.eye(3))
        return result


def scalar_elimination(hessian, left_source, right_source, z):
    """Exact two-node Schur/source elimination, not an H2/Fock truncation.

    Response l^T A^-1 r = l2*r2/A22 +
      (l1-l2*A21/A22)(r1-A12*r2/A22)/(A11-A12*A21/A22).
    Both source endpoints and the direct source-source term are necessary.
    """
    a = np.asarray(hessian)-z*np.eye(2)
    l, r = np.asarray(left_source), np.asarray(right_source)
    direct = l[1]*r[1]/a[1, 1]
    effective = a[0, 0]-a[0, 1]*a[1, 0]/a[1, 1]
    return direct+(l[0]-l[1]*a[1, 0]/a[1, 1])*(r[0]-a[0, 1]*r[1]/a[1, 1])/effective


def gauged_kinetic(fields, derivatives, vector_source, axial_source):
    """Lorentz-covariant O(4) kinetic term with full v/a external connections.

    D_mu sigma=d_sigma-a.pi; D_mu pi=d_pi+a*sigma+v cross pi.
    One- and two-source vertices follow from this same quadratic form.
    """
    f, d, v, a = map(np.asarray, (fields, derivatives, vector_source, axial_source))
    if f.shape != (2, 4) or d.shape != (4, 2, 4) or v.shape != (4, 3) or a.shape != (4, 3):
        raise ValueError("Two O(4) fields and four-component external connections required.")
    ds = d[:, :, 0]-np.einsum("ma,ia->mi", a, f[:, 1:])
    dpi = d[:, :, 1:]+a[:, None, :]*f[None, :, :1]+np.cross(v[:, None, :], f[None, :, 1:])
    return .5*np.array([1., -1., -1., -1.])@(np.sum(ds*ds, axis=1)+np.sum(dpi*dpi, axis=(1, 2)))


def minimal_vector_current(outgoing, incoming, outgoing_mode, incoming_mode):
    """Pion EM vertex from canonical kinetic terms; charge e stripped.

    Constant mixing preserves unit charge but cannot generate a radius.
    This is an exact tree-family statement, not a fitted zero for missing
    physical loop or higher-derivative/source interactions.
    """
    return (vector(outgoing)+vector(incoming))*np.dot(outgoing_mode, incoming_mode)


def calibrated_action(pion_mass, decay_constant, scalar_radius_squared_fm2,
                      current_mass, angle, quartic_ratio, coupling, yukawa):
    """Inverse calibration then independent vacuum solve; not a prediction.

    Fits m_pi,f_pi and one normalized scalar slope. Angle, quartic ratio,
    kappa and Yukawa remain explicit sensitivity directions. No raw scalar
    normalization or quark-mass scheme matching is inferred from the slope.
    """
    if min(pion_mass, decay_constant, scalar_radius_squared_fm2, current_mass, quartic_ratio, coupling) <= 0 or not 0 < angle < np.pi/2:
        raise ValueError("Positive targets and interior mixing angle required.")
    u = np.array([np.cos(angle), np.sin(angle)])
    v = decay_constant*u
    c = pion_mass**2*v/current_mass
    def build(scale):
        lam = scale*np.array([quartic_ratio, 1/quartic_ratio])
        a2 = v*v-(pion_mass**2-coupling*(v-v[::-1])/v)/lam
        return ChiralGraphAction(lam, a2, coupling, c, np.asarray(yukawa))
    scale = brentq(lambda x: build(x).scalar_radius_squared(v, u)-scalar_radius_squared_fm2, .01, 2000., xtol=1e-12)
    return build(scale)
