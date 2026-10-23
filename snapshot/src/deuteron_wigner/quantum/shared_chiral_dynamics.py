"""Nonproduction two-flavor chiral quark determinant, leading color order.

One common parity-even, covariantly Pauli--Villars-subtracted action generates
the gap, pion inverse propagator and axial/pseudoscalar source residues.
This is an NJL-type reference, NOT QCD matching, confinement, an LF Hamiltonian,
a nucleon model or a full electromagnetic/transverse-current calculation.
Masses/momenta are GeV; s is Minkowski invariant mass squared. Integrals are
Euclidean, analytically continued only below the unconfined qqbar threshold.
"""
from dataclasses import dataclass
from functools import cached_property

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.optimize import brentq
from scipy.special import xlogy
from .canonical_lf_fields import GAMMA
from .chiral_soft_matching import G5, square, vector


@dataclass(frozen=True)
class CovariantPV:
    """Two subtractions c=(1,-2,1), squared shifts (0,Lambda^2,2 Lambda^2).

    Apply to the common D^dagger D determinant before its variations. There
    is no IR cutoff to remove the physical quark threshold by prescription.
    """
    cutoff: float
    quadrature: int = 96

    def __post_init__(self):
        if not np.isfinite(self.cutoff) or self.cutoff <= 0:
            raise ValueError("Positive finite cutoff in GeV required.")
        if isinstance(self.quadrature, bool) or not isinstance(self.quadrature, int) or self.quadrature < 16:
            raise ValueError("At least 16 quadrature nodes required.")

    @cached_property
    def nodes(self):
        x, w = leggauss(self.quadrature)
        return (x+1)/2, w/2

    def i1(self, mass_squared):
        """Regulated tadpole int d^4p/(2pi)^4 1/(p^2+M^2), GeV^2."""
        if not np.isfinite(mass_squared) or mass_squared < 0:
            raise ValueError("Finite nonnegative squared mass required.")
        x = mass_squared/self.cutoff**2
        return self.cutoff**2 * (xlogy(x, x)-2*xlogy(x+1, x+1)+xlogy(x+2, x+2))/(16*np.pi**2)

    def bubble(self, mass, s, *, derivative=False):
        """I2(s), dimensionless; d I2/ds has units GeV^-2.

        Positive-mass, real below-threshold continuation only. No principal
        value, quark confinement or imaginary part is supplied silently.
        """
        if not np.isfinite([mass, s]).all() or mass <= 0 or s >= 4*mass**2:
            raise ValueError("Positive mass and s < 4 M^2 required.")
        x, w = self.nodes
        beta = x*(1-x)
        a, l = mass**2-beta*s, self.cutoff**2
        value = (beta*2*l*l/(a*(a+l)*(a+2*l)) if derivative
                 else np.log1p(l*l/(a*(a+2*l))))
        return float(w @ value/(16*np.pi**2))

    def log_difference(self, mass_squared, reference_squared):
        """PV int log ratio, GeV^4; field-independent divergence cancels."""
        if not np.isfinite([mass_squared, reference_squared]).all() or min(mass_squared, reference_squared) < 0:
            raise ValueError("Finite nonnegative squared masses required.")
        def f(z):
            x = z/self.cutoff**2 + np.arange(3)
            return np.array([1., -2., 1.]) @ xlogy(x*x, x)
        return self.cutoff**4*(f(mass_squared)-f(reference_squared))/(32*np.pi**2)

    def direct_euclidean_integrals(self, mass, q_squared, *, routing=0.5, nodes=120):
        """Independent radial/angular 4D quadrature, no Feynman parameter.

        Propagators carry p+eta q and p-(1-eta)q. Stable combined rational
        subtractions are integrated over the entire radial half-line.
        Returns (I1, I2(-Q^2)). Negative Q^2 is allowed only with symmetric
        routing and -Q^2 < 4 M^2: q4=i sqrt(-Q^2) has no contour crossing.
        Routing dependence is a negative-control risk, not a new parameter.
        """
        if not np.isfinite([mass, q_squared, routing]).all() or mass <= 0 or not 0 <= routing <= 1:
            raise ValueError("Positive mass, finite Q^2, routing in [0,1] required.")
        if q_squared < 0 and (routing != .5 or -q_squared >= 4*mass*mass):
            raise ValueError("Timelike continuation requires symmetric routing below threshold.")
        if not isinstance(nodes, int) or isinstance(nodes, bool) or nodes < 16:
            raise ValueError("At least 16 direct quadrature nodes required.")
        u, w = leggauss(nodes)
        u, w = (u+1)/2, w/2
        r = self.cutoff*np.tan(np.pi*u/2)
        dr = self.cutoff*np.pi/2/np.cos(np.pi*u/2)**2
        theta = np.arange(1, nodes+1)*np.pi/(nodes+1)
        z, wz = np.cos(theta), np.pi/(nodes+1)*np.sin(theta)**2
        q, l = np.sqrt(complex(q_squared)), self.cutoff**2
        radial = r[:, None]
        a = radial**2 + mass**2 + routing**2*q_squared + 2*radial*routing*q*z
        b = radial**2 + mass**2 + (1-routing)**2*q_squared - 2*radial*(1-routing)*q*z
        pa, pb = a*(a+l)*(a+2*l), b*(b+l)*(b+2*l)
        # Sum_j c_j/[(a+j*l)(b+j*l)], including the coincident a=b case.
        integrand = 2*l*l*(a*a+a*b+b*b+3*l*(a+b)+2*l*l)/(pa*pb)
        i2 = np.sum(w*r**3*dr*(integrand @ wz))/(4*np.pi**3)
        t = r*r+mass**2
        i1 = np.sum(w*r**3*dr*2*l*l/(t*(t+l)*(t+2*l)))/(8*np.pi**2)
        if abs(i2.imag) > 1e-12:
            raise ValueError("Unexpected imaginary part below threshold.")
        return float(i1), float(i2.real)


@dataclass(frozen=True)
class RadialInteraction:
    """U(rho)=a*rho + lambda/2*(rho-rho_reference)^2, rho=(sigma^2+pi^2)/2.

    a in GeV^2, lambda dimensionless, rho_reference in GeV^2. lambda=0
    corresponds to G=1/(2a) in G[(qbar q)^2+(qbar i gamma5 tau q)^2].
    Nonnegative lambda keeps the added invariant bounded below. Parameters
    are explicit; this small family is not a production neural functional.
    """
    a: float
    quartic: float = 0.
    rho_reference: float = 0.

    def __post_init__(self):
        if not np.isfinite([self.a, self.quartic, self.rho_reference]).all() or self.a <= 0 or self.quartic < 0 or self.rho_reference < 0:
            raise ValueError("Positive a and nonnegative quartic/reference required.")

    def value(self, rho):
        return self.a*rho + self.quartic/2*(rho-self.rho_reference)**2

    def slope(self, rho):
        return self.a+self.quartic*(rho-self.rho_reference)


@dataclass(frozen=True)
class PionSolution:
    mass: float
    mass_squared: float
    quark_mass: float
    norm: float
    quark_coupling: float
    decay_constant: float
    pseudoscalar_residue: float
    condensate_per_flavor: float
    scalar_curvature: float


@dataclass(frozen=True)
class SharedChiralAction:
    regulator: CovariantPV
    interaction: RadialInteraction
    current_mass: float
    colors: int

    def __post_init__(self):
        if not np.isfinite(self.current_mass) or self.current_mass < 0:
            raise ValueError("Nonnegative current quark mass required.")
        if isinstance(self.colors, bool) or not isinstance(self.colors, int) or self.colors < 1:
            raise ValueError("Positive integer number of colors required.")

    def potential(self, sigma, pion=0.):
        """Constant-field effective potential relative to the zero-field loop.

        The auxiliary fields have no inserted pion kinetic term or mass.
        The loop is -2*Nc*Nf Tr log with Nf=2, spin determinant included.
        """
        rho = (sigma*sigma+pion*pion)/2
        m2 = (self.current_mass+sigma)**2+pion*pion
        return self.interaction.value(rho)-4*self.colors*self.regulator.log_difference(m2, self.current_mass**2)

    def gap(self, mass):
        """dV/dsigma, GeV^3; M=m+sigma, not a supplied constituent input."""
        if not np.isfinite(mass) or mass < self.current_mass:
            raise ValueError("Mass must be finite and at least the current mass.")
        sigma = mass-self.current_mass
        return sigma*self.interaction.slope(sigma*sigma/2)-8*self.colors*mass*self.regulator.i1(mass*mass)

    def solve_mass(self, upper_mass):
        """Solve the positive vacuum branch in a user-declared bracket.

        For this nonnegative-quartic family there is one positive stationary
        branch (besides the symmetric chiral point). This is not a general
        neural-action vacuum solver.
        """
        if not np.isfinite(upper_mass) or upper_mass <= self.current_mass or self.gap(upper_mass) <= 0:
            raise ValueError("Upper mass must enclose the positive stable branch.")
        if self.current_mass == 0:
            if self.interaction.slope(0) >= 8*self.colors*self.regulator.i1(0):
                return 0.
            return brentq(lambda m: self.interaction.slope(m*m/2)-8*self.colors*self.regulator.i1(m*m),
                          0., upper_mass, xtol=1e-14)
        return brentq(self.gap, self.current_mass, upper_mass, xtol=1e-14)

    def pion_inverse(self, mass, s):
        """K_pi(s)=U'-8Nc I1-4Nc s I2(s), GeV^2, BEFORE using the gap."""
        sigma = mass-self.current_mass
        return (self.interaction.slope(sigma*sigma/2)-8*self.colors*self.regulator.i1(mass*mass)
                -4*self.colors*s*self.regulator.bubble(mass, s))

    def pion_norm(self, mass, s):
        """-d K_pi/ds; positive residue normalization at the pion pole."""
        return 4*self.colors*(self.regulator.bubble(mass, s)+s*self.regulator.bubble(mass, s, derivative=True))

    def scalar_inverse(self, mass, s):
        sigma = mass-self.current_mass
        return (self.pion_inverse(mass, s)+sigma*sigma*self.interaction.quartic
                +16*self.colors*mass*mass*self.regulator.bubble(mass, s))

    def axial_mixing(self, mass, s):
        """Coefficient of pion--A_mu loop, with A_mu=qbar gamma_mu gamma5 tau/2 q."""
        return 4*self.colors*mass*self.regulator.bubble(mass, s)

    def pseudoscalar_mixing(self, mass, s):
        """P=qbar i gamma5 tau/2 q, BEFORE using the pion pole equation."""
        return 4*self.colors*self.regulator.i1(mass*mass)+2*self.colors*s*self.regulator.bubble(mass, s)

    def solve_pion(self, mass):
        """Bound pole and source residues from independently evaluated loops."""
        if mass <= 0 or abs(self.gap(mass)) > 1e-10:
            raise ValueError("A nonzero self-consistent stationary mass is required.")
        if self.scalar_inverse(mass, 0) <= 0:
            raise ValueError("Vacuum is not radially stable.")
        k0 = self.pion_inverse(mass, 0.)
        if self.current_mass == 0:
            if abs(k0) > 1e-10:
                raise ValueError("Goldstone identity failed.")
            s = 0.
        else:
            endpoint = 4*mass*mass*(1-1e-8)
            if k0 <= 0 or self.pion_inverse(mass, endpoint) >= 0:
                raise ValueError("No real bound pion below the unconfined qqbar threshold.")
            s = brentq(lambda t: self.pion_inverse(mass, t), 0., endpoint, xtol=1e-14)
        norm = self.pion_norm(mass, s)
        if norm <= 0 or not np.isfinite(norm):
            raise ValueError("Nonpositive or nonfinite pion norm.")
        g = 1/np.sqrt(norm)
        return PionSolution(np.sqrt(s), s, mass, norm, g,
                            g*self.axial_mixing(mass, s),
                            g*self.pseudoscalar_mixing(mass, s),
                            -4*self.colors*mass*self.regulator.i1(mass*mass),
                            self.scalar_inverse(mass, 0))

    def normalized_inverse(self, solution, s):
        """K/Z_pole has unit pole slope. Off-pole dynamics are NOT discarded."""
        return self.pion_inverse(solution.quark_mass, s)/solution.norm

    def quark_source_vertices(self, mass, outgoing, incoming):
        """Resummed reference-model axial and weighted pseudoscalar vertices.

        Strip tau/2 and use the Minkowski phases of chiral_soft_matching.
        The bare vertices plus the generated pion-channel loop responses
        give Gamma_A=gamma gamma5+2 q R_A/K gamma5 and
        Gamma_P=(1+2 R_P/K) gamma5. They are not inserted Ward ansatze.
        Only this leading-color contact-family approximation is covered;
        QCD transverse amplitudes and other channels are not supplied.
        """
        if mass <= 0 or abs(self.gap(mass)) > 1e-10:
            raise ValueError("A nonzero stationary mass is required.")
        p1, p0 = vector(outgoing), vector(incoming)
        q = p1-p0
        s = square(q)
        k = self.pion_inverse(mass, s)
        if abs(k) < 1e-12:
            raise ValueError("On-pole current requires a residue/distribution prescription.")
        axial = np.array([g @ G5 for g in GAMMA])
        axial += 2*self.axial_mixing(mass,s)/k*np.einsum("m,ij->mij", q, G5)
        weighted_p = 2*self.current_mass*(1+2*self.pseudoscalar_mixing(mass,s)/k)*G5
        return axial, weighted_p

    def pole_subtracted_axial_exchange(self, solution, s):
        """R_A(s)^2/K(s) - f_pi^2/(m_pi^2-s), away from the pole.

        Only the scalar coefficient of longitudinal pion exchange is covered,
        NOT the full AA correlator (which also needs source contact/regular
        terms). This difference is nonzero: it must not be counted twice.
        """
        delta = solution.mass_squared-s
        if abs(delta) < 1e-9:
            raise ValueError("Use a separately derived coincident-pole limit.")
        mass = solution.quark_mass
        return self.axial_mixing(mass, s)**2/self.pion_inverse(mass, s)-solution.decay_constant**2/delta
