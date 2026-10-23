"""Physical four-leg common-cluster currents before any Wick contraction.

Each ordered quartic Hamiltonian coefficient and canonical current remain
separate. There is no electromagnetic gauging of a spin-zero auxiliary
factorization leg. This is an explicit endpoint-product representative,
not physical finite matching, a gauged envelope or all H2 current words.
"""
from math import pi, sqrt
import numpy as np
from deuteron_wigner.bridge.modes.core import polarization
from .m2_gaussian_field_regulator import _array
from .m2_gaussian_cluster_interactions import _frame, pair_coordinate_map
from .m2_coupled_pair_action import _pair
from .m2_gaussian_cluster_current import cluster_current_integrals, ClusterCurrentIntegral
from .m2_canonical_instantaneous_fermion import transverse_spin_factor
from .m2_constrained_fermion_current import two_gluon_current_spin, discrete_gluon_field_factor


def pair_current_geometry(incoming,outgoing,pair,*,species,out_species,
                          transfer_GeV,mean_total_transverse_GeV):
    """Two outgoing then two incoming physical legs with shared-vertex recoil.

    One independent relative coordinate on each side is integrated. The
    other pair-adapted coordinates preserve the physical spectators. The
    bra and ket remain in their distinct global Jacobi frames.
    """
    incoming,outgoing,si,so = tuple(incoming),tuple(outgoing),tuple(species),tuple(out_species)
    fi,fo = _frame(incoming,si),_frame(outgoing,so)
    p,_ = _pair(pair,len(incoming))
    if len(incoming)!=len(outgoing) or sum(incoming)!=sum(outgoing):
        raise ValueError("common total K2 and particle count required")
    if any((incoming[j],si[j])!=(outgoing[j],so[j]) for j in range(len(incoming)) if j not in p):
        raise ValueError("physical spectators must remain unchanged")
    delta = _array(transfer_GeV,(2,),"transverse transfer")
    mean = _array(mean_total_transverse_GeV,(2,),"mean transverse momentum")
    d = len(incoming)-1
    ri,ro = pair_coordinate_map(incoming,si,p),pair_coordinate_map(outgoing,so,p)
    adapted_out = np.column_stack((np.eye(d),np.zeros(d)))
    adapted_in = adapted_out.copy();adapted_in[0] = 0.;adapted_in[0,-1] = 1.
    lo,li = ro.T@adapted_out,ri.T@adapted_in
    recoil = np.zeros((d,2))
    recoil[1:] = -(ri@fi.forward[:-1])[1:,p[0],None]*delta
    in_shift = ri.T@recoil
    maps = np.vstack((fo.inverse[list(p),:-1]@lo,fi.inverse[list(p),:-1]@li))
    shifts = np.vstack((fo.fractions[list(p),None]*(mean+delta/2),
        fi.inverse[list(p),:-1]@in_shift+fi.fractions[list(p),None]*(mean-delta/2)))
    x = sum(fi.fractions[list(p)])
    jacobian = sqrt(np.prod(fi.fractions[list(p)])*np.prod(fo.fractions[list(p)]))/x
    return dict(total_k2=sum(incoming),modes=tuple(outgoing[j] for j in p)+tuple(incoming[j] for j in p),
        species=tuple(so[j] for j in p)+tuple(si[j] for j in p),out_map=lo,in_map=li,
        in_shift=in_shift,physical_field_maps=maps,physical_field_shifts=shifts,jacobian=jacobian)


def cluster_pair_current_orbitals(incoming,outgoing,pair,in_cutoff,out_cutoff,b_GeV,*,
        species,out_species,charges,regulator,total_pplus_GeV,transfer_GeV,
        mean_total_transverse_GeV,hamiltonian_coefficient,canonical_current_coefficient,
        prescription,quadrature_order,reference_shape=None):
    """Color/Fock-stripped constant quartic kernel, per its stated order in gs.

    Supply four physical charges in outgoing-pair/incoming-pair order.
    The M² coefficient is charge stripped; the canonical J/(2P+) coefficient
    includes the charge and its signed constrained denominators. This does
    not derive a canonical minus current by differentiating a transverse Ward
    identity. Each ordered color/spin coefficient must be passed separately.
    """
    geometry = pair_current_geometry(incoming,outgoing,pair,species=species,out_species=out_species,
        transfer_GeV=transfer_GeV,mean_total_transverse_GeV=mean_total_transverse_GeV)
    d = geometry["out_map"].shape[1]
    result = cluster_current_integrals(regulator=regulator,modes=geometry["modes"],
        species=geometry["species"],outgoing_legs=(True,True,False,False),charges=charges,
        transfer_GeV=transfer_GeV,total_k2=geometry["total_k2"],total_pplus_GeV=total_pplus_GeV,
        prescription=prescription,quadrature_order=quadrature_order,out_cutoff=out_cutoff,
        in_cutoff=in_cutoff,out_map=geometry["out_map"],in_map=geometry["in_map"],
        in_shift=geometry["in_shift"],b_GeV=b_GeV,physical_field_maps=geometry["physical_field_maps"],
        physical_field_shifts=geometry["physical_field_shifts"],
        hamiltonian_constant=hamiltonian_coefficient,hamiltonian_linear=np.zeros((2,d)),
        hamiltonian_leg_derivatives=np.zeros((4,2)),
        canonical_current_constant=canonical_current_coefficient,canonical_current_linear=np.zeros((4,2,d)),
        reference_shape=reference_shape)
    return ClusterCurrentIntegral(geometry["jacobian"]*result.canonical,
                                  geometry["jacobian"]*result.regulator,prescription,result.reference_shape)


def ordered_qg_coefficients(modes,*,total_k2,total_pplus_GeV,quark_helicity,
                            gluon_helicity,crossed,flavor_charge):
    """Same retained IF Hamiltonian and canonical quadratic-current ordering.

    Legs: q_out,g_out,q_in,g_in. The physical ordered color T_out T_in
    (or its reverse) remains external. The quark and gluon helicities here
    are unchanged, the exact retained qg spin selection. The crossed signed
    r2=p_q,in-k_g,out is not replaced by its absolute value.
    """
    if not isinstance(crossed,(bool,np.bool_)):
        raise ValueError("explicit Boolean constrained-field ordering required")
    w = tuple(modes)
    if len(w)!=4 or any(isinstance(n,(bool,np.bool_)) or not isinstance(n,(int,np.integer))
                       or n<=0 or n%2!=(j%2==0) for j,n in enumerate(w)):
        raise ValueError("positive physical q_out,g_out,q_in,g_in doubled modes required")
    pp,charge = float(total_pplus_GeV),float(flavor_charge)
    if (sum(w[:2])!=sum(w[2:]) or not np.all(np.isfinite([pp,charge])) or pp<=0
            or isinstance(total_k2,(bool,np.bool_)) or not isinstance(total_k2,(int,np.integer))
            or total_k2<sum(w[:2])):
        raise ValueError("conserved plus modes, positive total P+ and finite charge required")
    r2 = w[2]-w[1] if crossed else w[2]+w[3]
    spin = transverse_spin_factor(quark_helicity,gluon_helicity,crossed=bool(crossed))
    h = total_k2/(2*pi*sqrt(w[1]*w[3])*r2)*spin
    eps = polarization(pp,0.,0.,gluon_helicity)[2:]
    left,right = (eps,eps.conj()) if crossed else (eps.conj(),eps)
    current = two_gluon_current_spin(left_polarization=left,right_polarization=right,
        left_plus=pp*r2/total_k2,right_plus=pp*r2/total_k2)[:,(quark_helicity+1)//2,(quark_helicity+1)//2]
    current *= charge*discrete_gluon_field_factor(w[1])*discrete_gluon_field_factor(w[3])
    return complex(h),current
