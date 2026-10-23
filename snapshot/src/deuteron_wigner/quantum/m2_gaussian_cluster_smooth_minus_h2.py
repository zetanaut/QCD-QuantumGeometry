"""All smooth oscillator-minus P/Q returns through the common H2 order g4.

The finite-cell FIELD envelope is explicitly bound in its gapped spectral
representative at Delta+=0. Its linear response is then exactly zero.
Retained-isospin minus is NOT supplied; this assembly is not a complete
four-current and must not be used to select physical finite conditions.
"""
from dataclasses import dataclass
import numpy as np
from .m2_gaussian_cluster_h2 import GaussianClusterSnapshot
from .m2_gaussian_cluster_h2_current import _endpoints
from .m2_gaussian_cluster_retained_projector import exact_retained_smooth_minus_projector_coefficients
from .m2_gaussian_cluster_Q_projector import (
    cluster_free_Q_projector_h2_coefficients,cluster_cubic_Q_projector_h2_coefficients)
from .m2_gaussian_cluster_quartic_Q_projector import cluster_quartic_source_projector_h2_coefficients
from .m2_gaussian_cluster_quartic_Q_pair_projector import cluster_quartic_pair_projector_h2_coefficients
from .m2_smooth_longitudinal_projector import SmoothMinusProjectorContext
from .m2_finite_cell_envelope_current import SPECTRAL_PRESCRIPTION


@dataclass(frozen=True)
class ClusterH2SmoothMinusAssembly:
    outgoing: GaussianClusterSnapshot
    incoming: GaussianClusterSnapshot
    transfer_GeV: tuple
    minus_context: SmoothMinusProjectorContext
    charge: str
    prescription: str
    field_envelope_prescription: str
    quadrature_order: int
    contributions: dict

    all_smooth_oscillator_H2_words_through_g4=True
    zero_plus_transfer_field_envelope_bound=True
    retained_isospin_minus_supplied=False
    complete_H2_matched_current=False
    physical=False

    @property
    def coefficients(self):
        """One minus component, shape (1,Pf,Pi), with orders0..4 separated."""
        return {n:sum(block[n] for block in self.contributions.values()) for n in range(5)}

    def current(self,*,normalized):
        if not isinstance(normalized,(bool,np.bool_)):
            raise ValueError("explicit Boolean two-endpoint normalization choice required")
        g=self.incoming.owner.pilot.coupling
        value=sum(g**n*a for n,a in self.coefficients.items())
        if normalized:
            value=np.einsum("ab,mbc,cd->mad",self.outgoing.whitener,value,self.incoming.whitener,optimize=True)
        return value


def assemble_cluster_h2_smooth_minus_projector(outgoing,incoming,transfer,*,charge,
        prescription,quadrature_order,minus_context,field_envelope_prescription):
    """Exact P/free-Q/cubic-Q/quartic-source/quartic-pair oscillator assembly.

    Both wave endpoints are retained through g4 before applying either
    norm map. No independent higher-order subset is imported. Canonical,
    field-Gaussian, paired Wick and retained-isospin currents are separate.
    """
    if not isinstance(minus_context,SmoothMinusProjectorContext):
        raise ValueError("explicit independently derived smooth-minus context required")
    if field_envelope_prescription!=SPECTRAL_PRESCRIPTION:
        raise ValueError("explicit gapped spectral field-envelope prescription required")
    delta,_,order=_endpoints(outgoing,incoming,transfer,minus_context.mean_total_transverse_GeV,
                              prescription,quadrature_order)
    args=dict(charge=charge,prescription=prescription,quadrature_order=order,minus_context=minus_context)
    contributions={"retained":exact_retained_smooth_minus_projector_coefficients(
        outgoing.owner.pilot,incoming.owner.pilot,delta,**args)}
    for name,builder in (("free_Q",cluster_free_Q_projector_h2_coefficients),
                         ("cubic_Q",cluster_cubic_Q_projector_h2_coefficients),
                         ("quartic_source",cluster_quartic_source_projector_h2_coefficients),
                         ("quartic_pair",cluster_quartic_pair_projector_h2_coefficients)):
        contributions[name]=builder(outgoing,incoming,delta,**args)
    shape=(1,len(outgoing.owner.pilot.matrix),len(incoming.owner.pilot.matrix))
    for block in contributions.values():
        for n in range(5):
            if n not in block:block[n]=np.zeros(shape,complex)
            if block[n].shape!=shape or not np.all(np.isfinite(block[n])):
                raise ArithmeticError("inconsistent smooth-minus H2 coefficient")
            block[n].setflags(write=False)
    return ClusterH2SmoothMinusAssembly(outgoing,incoming,delta,minus_context,charge,
        prescription,field_envelope_prescription,order,contributions)
