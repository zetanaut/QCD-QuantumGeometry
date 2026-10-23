"""Single order-g² QQ insertion: order-g⁴ H, norm and two-energy good current.

The current uses the FULL raw W-induced wave tail. Compressing W into the
old free-source range suffices for this H insertion but generally loses
finite-transfer current terms already at the SAME order. This class neither
resums W nor claims all g⁴ paths, annihilation, six-body, or physical matching.
"""
import numpy as np
from scipy import sparse

from .m2_excluded_fock_matching import intermediate_density_action


class ExcludedInteractionInsertion:
    physical = False

    def __init__(self, reference, *, sector_operators, label):
        if not isinstance(label, str) or not label.strip():
            raise ValueError("explicit operator scope label required")
        expected = {s.source.space.sector for s in reference.sectors}
        if set(sector_operators) != expected:
            raise ValueError("explicit operator for each represented sector required")
        self.reference, self.label = reference, label
        self.operators = {}
        self.sources = {}
        self.energies = {}
        for sector in reference.sectors:
            space = sector.source.space
            supplied = sector_operators[space.sector]
            if not isinstance(supplied, tuple) or len(supplied) != 2 or supplied[0] is not space:
                raise ValueError("QQ operator must bind this exact raw coordinate space")
            w = sparse.csr_matrix(supplied[1], dtype=complex)
            if w.shape != (space.dimension, space.dimension) or not np.all(np.isfinite(w.data)):
                raise ValueError("finite QQ operator on exact raw coordinates required")
            diff = w-w.getH()
            if sparse.linalg.norm(diff) > 2e-12*max(1., sparse.linalg.norm(w)):
                raise ValueError("Hermitian QQ insertion required")
            self.operators[space.sector] = w
            self.sources[space.sector] = sector.lift(sector.coupling_by_power[1])
            p = reference.pilot
            self.energies[space.sector] = space.free_energies(
                p.width_GeV, p.quark_mass_GeV, p.gluon_mass_GeV)

    def wave_coefficients(self, sector_name, energy):
        """Q wave g A1 + g³ A3 from this insertion alone; raw Q coordinates."""
        d = float(energy)-self.energies[sector_name]
        if not np.all(np.isfinite(d)) or np.min(abs(d)) < 1e-10:
            raise ValueError("finite off-pole energy required")
        a1 = self.sources[sector_name]/d[:, None]
        a3 = (self.operators[sector_name]@a1)/d[:, None]
        return a1, a3

    def hamiltonian_coefficient(self, energy):
        """V1† R W R V1, units GeV²; multiply by g^4 exactly once."""
        return sum(self.sources[s].conj().T@self.wave_coefficients(s, energy)[1]
                   for s in self.sources)

    def norm_coefficient(self, energy):
        """-d_E H4 = A1† A3 + A3† A1; not required to be positive by itself."""
        result = np.zeros_like(self.reference.pilot.matrix)
        for s in self.sources:
            a1, a3 = self.wave_coefficients(s, energy)
            result += a1.conj().T@a3+a3.conj().T@a1
        return result

    def current_coefficient(self, energy_out, energy_in, transfer, *, charge):
        """A3_f† rho_Q A1_i + A1_f† rho_Q A3_i at order g^4."""
        result = np.zeros_like(self.reference.pilot.matrix)
        for sector in self.reference.sectors:
            space = sector.source.space
            a1f, a3f = self.wave_coefficients(space.sector, energy_out)
            a1i, a3i = self.wave_coefficients(space.sector, energy_in)
            acted = intermediate_density_action(space, np.column_stack((a1i, a3i)), transfer,
                b_GeV=self.reference.pilot.width_GeV, charge=charge)
            n = a1i.shape[1]
            result += a3f.conj().T@acted[:, :n]+a1f.conj().T@acted[:, n:]
        return result

    def free_source_leakage(self, energy):
        """Actual single-W wave tail outside the preceding free-source image."""
        return {s.source.space.sector: float(np.linalg.norm(
            (a3 := self.wave_coefficients(s.source.space.sector, energy)[1])
            - s.lift(s.project(a3)))) for s in self.reference.sectors}
