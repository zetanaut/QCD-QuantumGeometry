"""Explicit finite-representative H2 four-current through common order g_s^4.

This combines physical free/contact/paired-Wick currents, exact oscillator
projector products, the gapped Delta+=0 spectral envelope and the proved
total-isospin source-domain cancellation. It selects no physical finite
conditions and does not establish regulator independence or covariance.
"""
from dataclasses import dataclass
import numpy as np
from .m2_gaussian_cluster_h2_current import assemble_cluster_h2_current
from .m2_gaussian_cluster_smooth_minus_h2 import assemble_cluster_h2_smooth_minus_projector
from .m2_gaussian_cluster_retained_projector import exact_retained_transverse_projector_coefficients
from .m2_gaussian_cluster_Q_projector import (
    cluster_free_Q_projector_h2_coefficients,cluster_cubic_Q_projector_h2_coefficients)
from .m2_gaussian_cluster_quartic_Q_projector import cluster_quartic_source_projector_h2_coefficients
from .m2_gaussian_cluster_quartic_Q_pair_projector import cluster_quartic_pair_projector_h2_coefficients
from .m2_total_isospin import require_half_isospin_H2_snapshot


@dataclass(frozen=True)
class ClusterH2FourCurrent:
    physical_numerator: object
    smooth_minus: object
    isospin_prescription: str
    isospin_residuals: tuple
    contributions: dict

    all_represented_four_components_through_g4=True
    retained_isospin_H2_pullback_closed=True
    raw_isospin_minus_supplied=False
    zero_plus_transfer_field_envelope_bound=True
    # Completion refers only to the represented finite H2 model, at Delta+=0
    # and common raw order g_s^4 with the explicitly bound prescriptions.
    # Independent reference and P/Q/K sensitivity controls are saved in
    # docs/next_level/m2_wick_regulator_matching_result.md. They do not
    # establish cutoff convergence, full QCD matching or physical conditions.
    complete_H2_matched_current=True
    matching_scope="finite H2 g4; Delta+=0; explicit Gaussian/current/isospin prescriptions"
    regulator_independence_established=False
    physical_finite_conditions_selected=False
    physical=False

    @property
    def coefficients(self):
        """Order0..4 arrays in (+,-,x,y) order, divided by 2P+."""
        return {n:sum(v[n] for v in self.contributions.values()) for n in range(5)}

    def current(self,*,normalized):
        if not isinstance(normalized,(bool,np.bool_)):
            raise ValueError("explicit Boolean two-endpoint normalization choice required")
        f,i=self.physical_numerator.outgoing,self.physical_numerator.incoming
        value=sum(i.owner.pilot.coupling**n*v for n,v in self.coefficients.items())
        if normalized:value=np.einsum("ab,mbc,cd->mad",f.whitener,value,i.whitener,optimize=True)
        return value


def assemble_cluster_h2_four_current(outgoing,incoming,transfer,*,charge,
        loop_flavor_charges,prescription,quadrature_order,minus_context,
        field_envelope_prescription,isospin_prescription):
    """Bind every finite representative explicitly; no default physical current.

    Both energies, spin endpoints and norms belong to the same H2 definition.
    The source-domain isospin zero is used only after checking actual waves;
    transverse raw isospin terms remain in their original source builders.
    The smooth-minus context also specifies the common mean transverse boost.
    """
    from .m2_smooth_longitudinal_projector import SmoothMinusProjectorContext
    from .m2_finite_cell_envelope_current import SPECTRAL_PRESCRIPTION
    if not isinstance(minus_context,SmoothMinusProjectorContext):
        raise ValueError("explicit independently derived smooth-minus context required")
    if field_envelope_prescription!=SPECTRAL_PRESCRIPTION:
        raise ValueError("explicit gapped spectral field-envelope prescription required")
    residuals=tuple(require_half_isospin_H2_snapshot(s,prescription=isospin_prescription)
                    for s in (outgoing,incoming))
    args=dict(charge=charge,prescription=prescription,quadrature_order=quadrature_order)
    physical=assemble_cluster_h2_current(outgoing,incoming,transfer,
        loop_flavor_charges=loop_flavor_charges,
        mean_total_transverse_GeV=minus_context.mean_total_transverse_GeV,**args)
    minus=assemble_cluster_h2_smooth_minus_projector(outgoing,incoming,transfer,
        minus_context=minus_context,field_envelope_prescription=field_envelope_prescription,**args)
    contributions=dict(physical.contributions)
    transverse={"retained":exact_retained_transverse_projector_coefficients(
        outgoing.owner.pilot,incoming.owner.pilot,transfer,**args)}
    for name,builder in (("free_Q",cluster_free_Q_projector_h2_coefficients),
                         ("cubic_Q",cluster_cubic_Q_projector_h2_coefficients),
                         ("quartic_source",cluster_quartic_source_projector_h2_coefficients),
                         ("quartic_pair",cluster_quartic_pair_projector_h2_coefficients)):
        transverse[name]=builder(outgoing,incoming,transfer,**args)
    shape=physical.coefficients[0].shape
    for name,block in minus.contributions.items():
        values={n:np.zeros(shape,complex) for n in range(5)}
        for n in range(5):
            values[n][1]=block[n][0]
            values[n][2:]=transverse[name].get(n,0.)
            if not np.all(np.isfinite(values[n])):
                raise ArithmeticError("nonfinite combined projector coefficient")
            values[n].setflags(write=False)
        contributions["projector_"+name]=values
    return ClusterH2FourCurrent(physical,minus,isospin_prescription,residuals,contributions)
