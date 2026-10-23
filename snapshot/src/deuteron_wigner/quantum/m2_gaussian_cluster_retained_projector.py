"""Exact retained P-projector currents of the common Hamiltonian through gs².

Every Gaussian H tail is integrated before the oscillator projection.
Charged spectator recoil is retained. Separate entry points provide the
transverse and independently derived smooth-minus oscillator components.
Neither entry point silently supplies an isospin tangent or full H2 matching.
"""
from functools import lru_cache
from itertools import product
import numpy as np
from deuteron_wigner.microscopic.h0.color import emitted_gluon_color_amplitudes
from .m2_canonical_good_current import parton_charge
from .m2_coupled_longitudinal_jacobi import emission_routes,longitudinal_basis,jacobi_frame
from .m2_spin_flavor_s3_completion import SPIN_FLAVOR_BASIS
from .m2_source_vectors import source_tensor
from .m2_gaussian_cluster_h2 import GaussianClusterPilot,require_same_cluster_definition
from .m2_gaussian_cluster_excluded_current import _definition
from .m2_gaussian_cluster_quartic_pair_actions import physical_pair_terms,SPECIES,_spins
from .m2_gaussian_cluster_projector_kernel import (
    cubic_hamiltonian_orbital,pair_hamiltonian_orbital)
from .m2_smooth_longitudinal_projector import (
    exact_projector_components,projector_component_count,retained_free_projector_components)


@lru_cache(maxsize=2048)
def _cubic_boundary(route,hs,cutoff,b,mass,reg,pp,delta,order,particle,minus_context=None):
    h=cubic_hamiltonian_orbital(route,hs,b_GeV=b,mass_GeV=mass,regulator=reg,
        total_pplus_GeV=pp,total_transverse_GeV=(0.,0.))
    ei,ef=np.zeros(3),np.zeros(4);ei[particle]=ef[particle]=1.
    fi,fo=jacobi_frame(route.incoming),jacobi_frame(route.outgoing)
    v=exact_projector_components(h,minus_context=minus_context,out_cutoff=cutoff,in_cutoff=cutoff,
        out_weights=fo.forward[:-1].T,in_weights=fi.forward[:-1].T,
        out_charges=ef,in_charges=ei,transfer_GeV=delta,total_pplus_GeV=pp,quadrature_order=order)
    v.setflags(write=False);return v


@lru_cache(maxsize=2048)
def _pair_boundary(sector,inc,out,pair,cutoff,b,reg,pp,delta,order,particle,minus_context=None):
    h=pair_hamiltonian_orbital(inc,out,pair,species=SPECIES[sector],out_species=SPECIES[sector],
        b_GeV=b,regulator=reg,total_transverse_GeV=(0.,0.))
    e=np.zeros(len(inc));e[particle]=1.
    fi,fo=jacobi_frame(inc),jacobi_frame(out)
    v=exact_projector_components(h,minus_context=minus_context,out_cutoff=cutoff,in_cutoff=cutoff,
        out_weights=fo.forward[:-1].T,in_weights=fi.forward[:-1].T,
        out_charges=e,in_charges=e,transfer_GeV=delta,total_pplus_GeV=pp,quadrature_order=order)
    v.setflags(write=False);return v


def _retained_projector_coefficients(outgoing,incoming,transfer,*,
        charge,prescription,quadrature_order,minus_context):
    """Actual P coefficients at gs^0,gs^1,gs^2, including all charged quarks.

    Wick/finite kinetic terms commute with this oscillator projector and
    contribute no P boundary at Delta+=0. They are not omitted from H.
    """
    if not all(isinstance(p,GaussianClusterPilot) for p in (outgoing,incoming)):
        raise ValueError("both actual common-cluster retained pilots required")
    require_same_cluster_definition(outgoing,incoming)
    for key in ("total_k2","cutoff","target","width_GeV","quark_mass_GeV","gluon_mass_GeV","coupling"):
        if getattr(outgoing,key)!=getattr(incoming,key):
            raise ValueError("retained projector endpoints disagree on "+key)
    p=incoming
    delta,_,order,b,mass,pp=_definition(p.field_regulator,transfer,(0.,0.),prescription,
        quadrature_order,p.width_GeV,p.quark_mass_GeV,p.total_pplus_GeV)
    nF,nI=len(outgoing.matrix),len(p.matrix)
    nm=projector_component_count(minus_context)
    result={n:np.zeros((nm,nF,nI),complex) for n in (0,1,2)}
    result[0]=retained_free_projector_components(outgoing,incoming,delta,
        total_pplus_GeV=pp,charge=charge,quadrature_order=order,minus_context=minus_context)
    sf=np.array(SPIN_FLAVOR_BASIS);lookup={tuple(s):j for j,s in enumerate(sf)}
    def emission(f,i,d):
        bra=source_tensor(f,"qqqg",np.eye(len(f.matrix)))
        ket=source_tensor(i,"qqq",np.eye(len(i.matrix)))
        index_i,index_f=({m:j for j,m in enumerate(longitudinal_basis(s,p.total_k2))} for s in ("qqq","qqqg"))
        value=np.zeros((nm,len(f.matrix),len(i.matrix)),complex)
        for route in emission_routes(p.total_k2):
            color=emitted_gluon_color_amplitudes(route.emitter)[:,None]
            for hf,hg,hi in product((-1,1),repeat=3):
                selected=np.flatnonzero(2*(sf[:,route.emitter]%2)-1==hi)
                changed=sf[selected].copy();changed[:,route.emitter]=2*(changed[:,route.emitter]//2)+(hf+1)//2
                mapped=np.array([lookup[tuple(s)] for s in changed])
                left=np.take(bra[index_f[route.outgoing],:,:,:,(hg+1)//2],mapped,axis=2)
                right=np.take(ket[index_i[route.incoming]],selected,axis=2)
                for q in range(3):
                    orb=_cubic_boundary(route,(hf,hg,hi),p.cutoff,b,mass,p.field_regulator,pp,d,order,q,minus_context)
                    charges=parton_charge(sf[selected,q],"q",charge)
                    value+=np.einsum("oasf,moi,ab,ibsn,s->mfn",left.conj(),orb,color,right,charges,optimize=True)
        return value
    result[1]=emission(outgoing,incoming,delta)+emission(incoming,outgoing,tuple(-x for x in delta)).conj().transpose(0,2,1)
    for sector in ("qqq","qqqg"):
        uf,ui=[source_tensor(owner,sector,np.eye(len(owner.matrix))) for owner in (outgoing,incoming)]
        if sector=="qqqg":
            uf=uf.reshape(*uf.shape[:3],128,uf.shape[-1]);ui=ui.reshape(*ui.shape[:3],128,ui.shape[-1])
        slots={m:j for j,m in enumerate(longitudinal_basis(sector,p.total_k2))}
        spins,_=_spins(sector)
        for term in physical_pair_terms(sector,p.total_k2,total_pplus_GeV=pp,charge=charge,include_conversion=False):
            left=np.take(uf[slots[term.outgoing]],term.mapped,axis=2)
            right=np.take(ui[slots[term.incoming]],term.selected,axis=2)
            for q in range(3):
                orb=_pair_boundary(sector,term.incoming,term.outgoing,term.pair,p.cutoff,b,
                    p.field_regulator,pp,delta,order,q,minus_context)
                charges=parton_charge(spins[term.selected,q],"q",charge)
                result[2]+=np.einsum("oasf,moi,ab,ibsn,s->mfn",left.conj(),orb,term.hamiltonian,right,charges,optimize=True)
    return result


def exact_retained_transverse_projector_coefficients(outgoing,incoming,transfer,*,
        charge,prescription,quadrature_order):
    """Original two transverse components; unchanged explicit prescription."""
    return _retained_projector_coefficients(outgoing,incoming,transfer,charge=charge,
        prescription=prescription,quadrature_order=quadrature_order,minus_context=None)


def exact_retained_smooth_minus_projector_coefficients(outgoing,incoming,transfer,*,
        charge,prescription,quadrature_order,minus_context):
    """One smooth oscillator minus component, shape (1,Pf,Pi), through g2.

    This does not include a retained-isospin minus partner. Wick and finite
    kinetic H terms remain present but have no oscillator boundary.
    """
    if minus_context is None:raise ValueError("explicit smooth-minus context required")
    return _retained_projector_coefficients(outgoing,incoming,transfer,charge=charge,
        prescription=prescription,quadrature_order=quadrature_order,minus_context=minus_context)
