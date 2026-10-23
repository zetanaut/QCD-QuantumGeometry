"""Canonical order-g_s qqq <-> qqqg transverse electromagnetic contact.

Derived from the same eliminated quark component as the Hamiltonian.
This is J_T/(2P+), per g_s; no M^2 conversion factor belongs here.
It is not a complete physical current or a choice of Galerkin gauging.
"""
from functools import lru_cache
from math import pi,sqrt

import numpy as np
from scipy import sparse

from deuteron_wigner.bridge.modes.core import polarization
from deuteron_wigner.microscopic.h0.color import emitted_gluon_color_amplitudes
from .m2_canonical_good_current import parton_charge
from .m2_constrained_fermion_current import one_gluon_current_spin,discrete_gluon_field_factor
from .m2_coupled_longitudinal_jacobi import (
    emission_coordinate_map,emission_routes,jacobi_frame,longitudinal_basis,
    oscillator_rotation,oscillator_shell,
)
from .m2_coupled_pilot import _displacement
from .m2_spin_flavor_s3_completion import SPIN_FLAVOR_BASIS


@lru_cache(maxsize=1024)
def contact_route_orbital(route,in_cutoff,out_cutoff,b_GeV,transfer):
    """Exact local coincidence and displaced-parent overlap, units GeV.

    Relative m=0 circular radial modes all have integral b/sqrt(pi).
    This includes sqrt(x_q*x_g/x_parent), exactly as the canonical emission
    measure. Parent overlaps are continuum D(s), not exponentiated finite
    ladders. A photon hits the parent at the local quark/gluon vertex.
    """
    b=float(b_GeV);delta=np.asarray(transfer,float)
    if b<=0 or not np.isfinite(b) or delta.shape!=(2,) or not np.all(np.isfinite(delta)):
        raise ValueError("finite transfer and positive width required")
    incoming=oscillator_shell(2,in_cutoff);outgoing=oscillator_shell(3,out_cutoff)
    parent=jacobi_frame(route.incoming);daughter=jacobi_frame(route.outgoing)
    xq,xg=daughter.fractions[[route.emitter,3]]
    weight=sqrt(xq*xg/(xq+xg))*b/sqrt(pi)
    shift=parent.forward[:-1,route.emitter,None]*delta[None,:]
    plus=(shift[:,0]-1j*shift[:,1])/(2*b)
    alpha=np.r_[plus,plus.conj()]
    adapted=np.zeros((len(outgoing),len(incoming)),complex)
    for row,o in enumerate(outgoing):
        if o[2]!=o[5]:
            continue  # exact angular rule for a momentum-independent contact
        po=(o[0],o[1],o[3],o[4])
        for col,i in enumerate(incoming):
            adapted[row,col]=weight*np.prod([_displacement(a,c,z) for a,c,z in zip(po,i,alpha)])
    result=oscillator_rotation(emission_coordinate_map(route),out_cutoff).T@adapted
    result.setflags(write=False)
    return result


def emission_transverse_contact(total_k2,in_cutoff,out_cutoff,transfer,*,
                                b_GeV,total_pplus_GeV,gluon_helicity,charge):
    """Raw qqq -> qqqg contact, two sparse matrices per g_s.

    Flattened order is longitudinal, orbital, color, spin/flavor. Separate
    external cutoffs are useful for testing the Galerkin boundary. There is
    one emitted gluon: no additional Bose factor. Sum all three emitters
    coherently before symmetry projection.
    """
    delta=tuple(float(v) for v in transfer)
    pplus=float(total_pplus_GeV)
    if (len(delta)!=2 or not np.all(np.isfinite(delta))
            or not np.isfinite(pplus) or pplus<=0 or gluon_helicity not in (-1,1)):
        raise ValueError("finite transfer, positive P+ and physical gluon helicity required")
    q,g=(longitudinal_basis(s,total_k2) for s in ("qqq","qqqg"))
    nq,ng=len(oscillator_shell(2,in_cutoff)),len(oscillator_shell(3,out_cutoff))
    qi,gi=({s:i for i,s in enumerate(states)} for states in (q,g))
    sf=np.array(SPIN_FLAVOR_BASIS)
    entries=[([],[],[]),([],[],[])]
    for route in emission_routes(total_k2):
        f=jacobi_frame(route.outgoing)
        pq=f.fractions[route.emitter]*pplus
        pg=f.fractions[3]*pplus;parent=pq+pg
        eps=polarization(pg,0.,0.,gluon_helicity).conj()[2:]
        # J_T[1] is independent of transverse momenta and mass; these zeros
        # evaluate that exact algebraic result, not a massless model choice.
        spin=one_gluon_current_spin((pq,0.,0.),(parent,0.,0.),mass_GeV=0.,
            polarization=eps,right_plus=pq,left_plus=parent)[2:]
        if np.any(spin[:,0,1]!=0) or np.any(spin[:,1,0]!=0):
            raise ArithmeticError("unexpected helicity flip in the transverse contact")
        spin*=discrete_gluon_field_factor(route.outgoing[3])
        orbital=contact_route_orbital(route,in_cutoff,out_cutoff,b_GeV,delta)
        color=emitted_gluon_color_amplitudes(route.emitter)[:,None]
        charges=parton_charge(sf[:,route.emitter],"q",charge)
        for j in range(2):
            diagonal=charges*spin[j,sf[:,route.emitter]%2,sf[:,route.emitter]%2]
            block=sparse.kron(orbital,sparse.kron(color,sparse.diags(diagonal)),format="coo")
            rows,cols,values=entries[j]
            rows.extend(gi[route.outgoing]*ng*128+block.row)
            cols.extend(qi[route.incoming]*nq*64+block.col)
            values.extend(block.data)
    shape=(len(g)*ng*128,len(q)*nq*64)
    return tuple(sparse.coo_matrix((values,(rows,cols)),shape=shape).tocsr()
                 for rows,cols,values in entries)


def transverse_contact_between(outgoing,incoming,transfer,*,total_pplus_GeV,charge):
    """Retained number-changing contact, per g_s, with both endpoints.

    Absorption is the independently assembled reversed-transfer emission
    adjoint. This does not insert the free one-body current, loops, current
    Wick descendants, excluded-Fock matching or a projector-gauging current.
    """
    for name in ("total_k2","cutoff","target","width_GeV","quark_mass_GeV",
                 "gluon_mass_GeV","coupling"):
        if getattr(outgoing,name)!=getattr(incoming,name):
            raise ValueError("current endpoints disagree on "+name)
    delta=np.asarray(transfer,float)
    if delta.shape!=(2,) or not np.all(np.isfinite(delta)):
        raise ValueError("finite transverse transfer required")
    p=incoming
    nf=[u.shape[1] for u in outgoing.bases];ni=[u.shape[1] for u in incoming.bases]
    result=np.zeros((2,sum(nf),sum(ni)),complex)
    of=np.cumsum([0]+nf);oi=np.cumsum([0]+ni)
    for sector,hg in ((1,-1),(2,1)):
        def raw(d):
            return emission_transverse_contact(p.total_k2,p.cutoff,p.cutoff,d,
                b_GeV=p.width_GeV,total_pplus_GeV=total_pplus_GeV,
                gluon_helicity=hg,charge=charge)
        forward,backward=raw(delta),raw(-delta)
        for j in range(2):
            result[j,of[sector]:of[sector+1],:oi[1]]=(
                outgoing.bases[sector].getH()@forward[j]@incoming.bases[0]).toarray()
            result[j,:of[1],oi[sector]:oi[sector+1]]=(
                outgoing.bases[0].getH()@backward[j].getH()@incoming.bases[sector]).toarray()
    return result
