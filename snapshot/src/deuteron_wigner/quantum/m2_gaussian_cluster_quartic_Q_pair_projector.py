"""Physical Q5Q5 quartic oscillator-projector boundary, per g_s².

The physical pair-annihilation extension is matched to the auxiliary
Hamiltonian on its actual H2 source waves, not arbitrary raw isospin
columns. No auxiliary leg is gauged. Neutral gluon H keeps spectators.
An explicit context selects the independently derived smooth-minus family.
"""
from functools import lru_cache
from itertools import product
from math import pi,sqrt
import numpy as np
from .m2_gaussian_cluster_quartic_pair_actions import (
    ClusterFiveQuarticCurrent,SPECIES,_spins,_charged_terms,
    CHARGED_COMPONENTS,NEUTRAL_COMPONENTS)
from .m2_gaussian_cluster_projector_kernel import pair_hamiltonian_orbital
from .m2_gaussian_cluster_interactions import _frame
from .m2_excluded_fock_kinematics import longitudinal_states,color_source
from .m2_excluded_current_scattering import current_pair_color,canonical_gluon_current_factor
from .m2_excluded_four_gluon import quartic_spin_color
from .m2_excluded_pair_current import adjoint_pair_current_factor
from .m2_six_body_cubic import SixBodyCubicAction,five_statistics_project
from .m2_six_body_transverse_contact import _block,_add
from .m2_canonical_good_current import parton_charge
from .m2_smooth_longitudinal_projector import exact_projector_components,projector_component_count


def neutral_pair_hamiltonian_terms(total_k2,*,charge):
    """Independent gg-number, A4 and gg-annihilation H inventory.

    Same ordered color kernels and Bose factors as the original H. The
    zero physical field-current charge is NOT used to discard these terms.
    """
    sector="qqqgg";spins,dims=_spins(sector);allspins=np.arange(len(spins))
    pair=(3,4);c=color_source("g_to_gg");paircolor=c@c.conj().T
    for inc in longitudinal_states(sector,total_k2):
        for k1 in range(2,sum(inc[3:]),2):
            out=inc[:3]+(k1,sum(inc[3:])-k1);dk=k1-inc[3]
            if dk:
                h=2*total_k2/(pi*dk**2)*current_pair_color(sector,3,4)
                for j in pair:h=h*canonical_gluon_current_factor(out[j],inc[j])
                yield from _charged_terms("gg_number",sector,sector,inc,out,pair,h,np.zeros_like(h),
                                          allspins,allspins,charge)
            for ho1,ho2,hi1,hi2 in product((-1,1),repeat=4):
                selected=np.flatnonzero((2*spins[:,3]-1==hi1)&(2*spins[:,4]-1==hi2))
                changed=spins[selected].copy();changed[:,3]=(ho1+1)//2;changed[:,4]=(ho2+1)//2
                mapped=np.ravel_multi_index(changed.T,dims)
                h=total_k2/(16*pi*sqrt(np.prod(inc[3:]+out[3:])))*quartic_spin_color(ho1,ho2,hi1,hi2)
                yield from _charged_terms("A4",sector,sector,inc,out,pair,h,np.zeros_like(h),
                                          selected,mapped,charge)
                h=2*total_k2/(pi*sum(inc[3:])**2)*paircolor
                h=h*adjoint_pair_current_factor(*out[3:],ho1,ho2)*np.conj(
                    adjoint_pair_current_factor(*inc[3:],hi1,hi2))
                yield from _charged_terms("gg_annihilation",sector,sector,inc,out,pair,h,np.zeros_like(h),
                                          selected,mapped,charge)


@lru_cache(maxsize=4096)
def _pair_boundary(si,so,inc,out,pair,ci,co,b,reg,pp,delta,order,side,q,minus_context=None):
    h=pair_hamiltonian_orbital(inc,out,pair,species=SPECIES[si],out_species=SPECIES[so],
        b_GeV=b,regulator=reg,total_transverse_GeV=(0.,0.))
    fi,fo=_frame(inc,SPECIES[si]),_frame(out,SPECIES[so])
    ei,ef=np.zeros(5),np.zeros(5)
    (ef if side=="f" else ei)[q]=1.
    value=exact_projector_components(h,minus_context=minus_context,out_cutoff=co,in_cutoff=ci,
        out_weights=fo.forward[:-1].T,in_weights=fi.forward[:-1].T,
        out_charges=ef,in_charges=ei,transfer_GeV=delta,total_pplus_GeV=pp,quadrature_order=order)
    value.setflags(write=False)
    return value


class ClusterFiveQuarticProjector(ClusterFiveQuarticCurrent):
    """Physical raw-Q5 transverse or explicit smooth-minus H2 action."""
    minus_projector_supplied=False
    exact_neutral_current_components=()  # These H blocks do have projector currents.

    def __init__(self,*args,minus_context=None,**kwargs):
        self.ncomponents=projector_component_count(minus_context)
        self.minus_context=minus_context
        self.minus_projector_supplied=minus_context is not None
        super().__init__(*args,**kwargs)
        if self.mean!=(0.,0.):raise ValueError("projector currently uses explicit mean PT zero")
        self.terms+=tuple(neutral_pair_hamiltonian_terms(
            self.incoming["qqqgg"].total_k2,charge=self.charge))

    @classmethod
    def from_models(cls,outgoing,incoming,transfer,*,charge,prescription,quadrature_order,minus_context=None):
        # Base construction validates the full physical H2 definitions. The
        # smooth context changes only the projector tangent, never H or terms.
        obj=super().from_models(outgoing,incoming,transfer,charge=charge,
            mean_total_transverse_GeV=(0.,0.),prescription=prescription,quadrature_order=quadrature_order)
        obj.ncomponents=projector_component_count(minus_context)
        obj.minus_context=minus_context
        obj.minus_projector_supplied=minus_context is not None
        return obj

    def apply(self,columns,*,component,components=None):
        if isinstance(component,(bool,np.bool_)) or component not in range(self.ncomponents):
            raise ValueError("component outside transverse x/y or explicitly selected smooth-minus family")
        if components is not None and (not components or not set(components)<=set(CHARGED_COMPONENTS+NEUTRAL_COMPONENTS)):
            raise ValueError("explicit nonempty physical quartic component set required")
        count=SixBodyCubicAction._validate(self.incoming,columns)
        source={n:five_statistics_project(s,columns[n]) for n,s in self.incoming.items()}
        result={n:np.zeros((s.dimension,count),complex) for n,s in self.outgoing.items()}
        for t in self.terms:
            if components is not None and t.component not in components:continue
            si,so=self.incoming[t.incoming_sector],self.outgoing[t.outgoing_sector]
            li,lo=si.longitudinal.index(t.incoming),so.longitudinal.index(t.outgoing)
            block=_block(si,li,t.selected,source[si.sector])
            for side,s,indices in (("i",si,t.selected),("f",so,t.mapped)):
                for q,kind in enumerate(SPECIES[s.sector]):
                    if kind=="g":continue
                    weight=parton_charge(s.spins[indices,q],kind,self.charge)
                    if not np.any(weight):continue
                    orb=_pair_boundary(si.sector,so.sector,t.incoming,t.outgoing,t.pair,si.cutoff,so.cutoff,
                        self.b,self.regulator,self.P,self.delta,self.order,side,q,self.minus_context)[component]
                    value=np.einsum("oi,ab,ibsn,s->oasn",orb,t.hamiltonian,block,weight,optimize=True)
                    _add(so,lo,t.mapped,value,result[so.sector])
        return {n:five_statistics_project(s,result[n]) for n,s in self.outgoing.items()}

    def adjoint(self,columns,*,component,components=None):
        reverse=ClusterFiveQuarticProjector(self.incoming,self.outgoing,tuple(-x for x in self.delta),
            regulator=self.regulator,b_GeV=self.b,total_pplus_GeV=self.P,charge=self.charge,
            mean_total_transverse_GeV=(0.,0.),prescription=self.prescription,quadrature_order=self.order,
            minus_context=self.minus_context)
        return reverse.apply(columns,component=component,components=components)


def cluster_quartic_pair_projector_h2_coefficients(outgoing,incoming,transfer,*,charge,prescription,quadrature_order,
                                                 minus_context=None):
    """A1f† B55 A1i at g4, transverse or explicit smooth minus; no Wick duplication."""
    from .m2_gaussian_cluster_h2_current import _endpoints
    delta,_,order=_endpoints(outgoing,incoming,transfer,(0.,0.),prescription,quadrature_order)
    action=ClusterFiveQuarticProjector.from_models(outgoing.owner,incoming.owner,delta,
        charge=charge,prescription=prescription,quadrature_order=order,minus_context=minus_context)
    af,ai=outgoing.five_wave_coefficients[0],incoming.five_wave_coefficients[0]
    result=[]
    for j in range(projector_component_count(minus_context)):
        value=action.apply(ai,component=j)
        result.append(sum(af[n].conj().T@value[n] for n in af))
    return {4:np.array(result)}
