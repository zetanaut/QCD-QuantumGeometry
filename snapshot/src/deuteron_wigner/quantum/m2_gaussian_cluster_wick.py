"""Ordered Wick Hamiltonian coefficients from the SAME active-cluster kernel.

These are not constant-trace substitutions and not a complete matched
current model. Every term keeps its original field ordering, color factor
and signed constrained denominator; the common quartic kernel is integrated
on its contracted legs before the longitudinal sum.
"""
from dataclasses import dataclass
from math import pi
import numpy as np
from .m2_gaussian_field_regulator import GaussianClusterRegulator
from .m2_canonical_quark_self import CF, _integer, quark_wick_sums
from .m2_canonical_gluon_self import CA, TF, gluon_wick_sums
from .m2_wick_one_body import WickOneBody


def cluster_wick_summands(species, mode, *, regulator, quark_flavors, reference_shape=None):
    """Mode-resolved IF/IG/A4 terms in GeV²; multiply K2/pi for M²/g².

    The envelope applies to participating positive physical fields, not
    signed inverse-derivative transfer modes. No k<p restriction is added.
    Exact zero transfer is omitted ONLY from the IG direct term.
    """
    if not isinstance(regulator, GaussianClusterRegulator):
        raise ValueError("explicit common active-cluster regulator required")
    reference_shape=regulator.shape_for(reference_shape)
    shape_factor=1.
    if reference_shape is not None:
        # Integral d²k of Q^m exp(-a Q) = m!/a^m times the base trace.
        ratio=1-reference_shape.a_reference/reference_shape.a
        shape_factor=sum(ratio**m for m in range(reference_shape.degree+1))
    p = _integer(mode, "external field mode")
    args = dict(boson_modes=regulator.boson_modes, fermion_modes=regulator.fermion_modes)
    # Reuse existing parity/flavor validation, not their summed coefficients.
    gluon_wick_sums(2, **args, quark_flavors=quark_flavors)
    if not regulator.contains(species, p):
        raise ValueError("external field must belong to the declared envelope")
    k = 2*np.arange(1, regulator.boson_modes+1, dtype=float)
    r = 2*np.arange(regulator.fermion_modes, dtype=float)+1
    # Start from the bare Gaussian trace. A bound regulator's public trace
    # is already shaped; using it here would count the shape weight twice.
    tb = np.array([regulator.width_GeV**2*x/(4*pi*p) for x in k])
    tf = np.array([regulator.width_GeV**2*x/(4*pi*p) for x in r])
    tb*=shape_factor;tf*=shape_factor
    if species in ("q", "qbar"):
        quark_wick_sums(p, **args)
        constrained = p-k if species == "q" else p+k
        direct = np.zeros(len(r))
        nonzero = r != p
        direct[nonzero] = 1/(p-r[nonzero])**2
        return {"IF":(k, CF*tb/(k*constrained)),
                "IG":(r, CF*tf*(direct-1/(p+r)**2))}
    direct = np.zeros(len(k))
    nonzero = k != p
    direct[nonzero] = (p+k[nonzero])**2/(4*p*k[nonzero]*(p-k[nonzero])**2)
    pair = (p-k)**2/(4*p*k*(p+k)**2)
    return {"IF":(r, TF*quark_flavors/p*tf*(1/(p-r)-1/(p+r))),
            "IG":(k, CA*tb*(direct+pair)),
            "A4":(k, CA/(2*p)*tb/k)}


@dataclass(frozen=True, kw_only=True)
class GaussianClusterWick:
    """Explicit H subtraction object; deliberately not accepted as WickOneBody.

    This Hamiltonian-only layer does not assemble its gauged current.
    The represented finite H2 assembly is in m2_gaussian_cluster_four_current;
    it requires explicitly bound current and projector prescriptions.
    The bare WickMatchedFourthOrderH2 cannot silently consume this object.
    """
    regulator: GaussianClusterRegulator
    quark_flavors: int
    normal_order_subtraction: bool
    finite_quark_mass2_per_gs2_GeV2: float
    finite_gluon_mass2_per_gs2_GeV2: float

    def __post_init__(self):
        cluster_wick_summands("g", 2, regulator=self.regulator, quark_flavors=self.quark_flavors)
        if not isinstance(self.normal_order_subtraction, (bool, np.bool_)):
            raise ValueError("explicit Boolean normal-order subtraction required")
        if not np.all(np.isfinite([self.finite_quark_mass2_per_gs2_GeV2,
                                  self.finite_gluon_mass2_per_gs2_GeV2])):
            raise ValueError("explicit finite signed kinetic mass² coefficients required")

    def mass_components(self, species, mode, *, total_k2):
        """Ordered, subtraction and finite kinetic terms, separately M²/g²."""
        p, total = _integer(mode, "field mode"), _integer(total_k2, "total_k2")
        if p >= total:
            raise ValueError("external constituent must have 0<p2<K2")
        # A finite Fock basis can retain free spectator modes beyond the
        # interaction field envelope. Their quadratic Wick descendant is
        # exactly zero, not an undefined trace or a removed free state.
        if self.regulator.contains(species, p):
            pieces = cluster_wick_summands(species, p, regulator=self.regulator,
                                           quark_flavors=self.quark_flavors)
            ordered = total/pi*sum(np.sum(values) for _, values in pieces.values())
        else:
            ordered = 0.
        finite = (self.finite_gluon_mass2_per_gs2_GeV2 if species == "g" else
                  self.finite_quark_mass2_per_gs2_GeV2)*total/p
        return float(ordered), float(-ordered if self.normal_order_subtraction else 0.), float(finite)

    # The shared exact projection uses only mass_components, not the old
    # constant coincidence trace or its canonical-current implementation.
    retained_mass_coefficient = WickOneBody.retained_mass_coefficient
    shell_mass_diagonal = WickOneBody.shell_mass_diagonal

    @property
    def complete_regulated_H2_matching(self):
        return False
