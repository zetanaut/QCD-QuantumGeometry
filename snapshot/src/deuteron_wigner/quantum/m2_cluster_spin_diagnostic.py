"""Finite elastic pole and spin diagnostics, not physical covariance repair.

The previous unequal-anchor quantum references remain transition benchmarks.
Elastic diagnostics solve H_eff(e) u=e u and retain u^dagger N(e) u=1.
"""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import brentq
from scipy import sparse

from .m2_gaussian_cluster_h2 import GaussianClusterH2, require_same_cluster_definition
from .m2_spin_resolved_current import transverse_parity_map
from .m2_orientation_current_diagnostic import orientation_current_structures


@dataclass(frozen=True)
class FiniteElasticPole:
    snapshot: object
    raw_state: np.ndarray
    register_state: np.ndarray
    energy_GeV2: float
    residual_GeV2: float
    gap_GeV2: float
    bracket: tuple
    physical = False


def finite_elastic_pole(model, *, bracket_GeV2, tolerance_GeV2=2e-13):
    """Lowest finite-model branch in an explicit pole-free energy bracket.

    No physical mass is supplied or fitted. The interval must lie strictly
    below every represented free Q5/Q6 propagation threshold. This identifies
    a local branch, not the lowest solution over every possible energy domain.
    """
    if not isinstance(model, GaussianClusterH2):
        raise TypeError("explicit common-regulated H2 model required")
    bracket=np.asarray(bracket_GeV2,float)
    if bracket.shape!=(2,) or not np.all(np.isfinite(bracket)) or not 0<bracket[0]<bracket[1]:
        raise ValueError("ordered positive finite pole bracket required")
    if not np.isfinite(tolerance_GeV2) or tolerance_GeV2<=0:
        raise ValueError("positive residual tolerance required")
    propagators=(*model.six.propagation5.values(),*model.six.propagation6.values())
    threshold=min(p.minimum_energy_GeV2 for p in propagators)
    if bracket[1]>=threshold-1e-10:
        raise ValueError("elastic bracket must lie below all represented free Q poles")
    # Retain the explicit sign change on the lowest finite-H eigenbranch.
    def residual(e):
        s=model.snapshot(e)
        return float(np.linalg.eigvalsh(s.effective_hamiltonian)[0]-e)
    left,right=map(residual,bracket)
    if not left>0>right:
        raise ValueError("bracket must enclose a decreasing lowest-branch crossing")
    e=float(brentq(residual,*bracket,xtol=5e-15,rtol=1e-14))
    s=model.snapshot(e)
    values,vectors=np.linalg.eigh(s.effective_hamiltonian)
    u=vectors[:,0]
    u=u/np.sqrt(np.vdot(u,s.metric@u).real)
    v=np.linalg.solve(s.whitener,u)
    r=float(np.linalg.norm(s.effective_hamiltonian@u-e*u))
    lr=float(np.linalg.norm(s.matrix@v-e*v))
    if max(r,lr)>tolerance_GeV2 or not np.isclose(np.vdot(v,v),1,atol=3e-13,rtol=0):
        raise ArithmeticError("finite elastic pole or norm failed")
    gap=float(np.linalg.eigvalsh(s.matrix)[1]-e)
    if gap<=1e-10:raise ValueError("isolated spin-fiber ground required; use subspace diagnostics")
    return FiniteElasticPole(s,u,v,e,r,gap,tuple(bracket))


def elastic_spin_pair(plus_model, minus_model, *, bracket_GeV2, intrinsic_parity):
    """Independently solve both fibers; source parity fixes their relative phase."""
    if isinstance(intrinsic_parity,(bool,np.bool_)) or intrinsic_parity not in (-1,1):
        raise ValueError("explicit +/- intrinsic-parity diagnostic required")
    if (plus_model.pilot.jz_twice,minus_model.pilot.jz_twice)!=(1,-1):
        raise ValueError("ordered plus/minus spin fibers required")
    require_same_cluster_definition(plus_model.pilot,minus_model.pilot)
    poles=[finite_elastic_pole(m,bracket_GeV2=bracket_GeV2) for m in (plus_model,minus_model)]
    if abs(poles[0].energy_GeV2-poles[1].energy_GeV2)>2e-11:
        raise ArithmeticError("spin-fiber mass splitting; elastic spin-half packet invalid")
    e=poles[0].energy_GeV2
    snapshots=[poles[0].snapshot,minus_model.snapshot(e)]
    t=transverse_parity_map(minus_model.pilot,plus_model.pilot)
    errors={}
    for name in ('effective_hamiltonian','metric','matrix'):
        error=np.linalg.norm(getattr(snapshots[1],name)@t-t@getattr(snapshots[0],name))
        errors[name]=float(error)
        if error>3e-11:raise ArithmeticError("source transverse parity failed: "+name)
    raw=[poles[0].raw_state,1j*intrinsic_parity*t@poles[0].raw_state]
    states=[np.linalg.solve(s.whitener,u) for s,u in zip(snapshots,raw)]
    return dict(energy_GeV2=e,snapshots=snapshots,raw_states=raw,states=states,
        parity_map=t,parity_errors=errors,independent_poles=poles,intrinsic_parity=intrinsic_parity)


def rotate_current_blocks(blocks, angle, *, jz_out_twice, jz_in_twice):
    """Kinematic z rotation; last three axes are (mu,out,in), mu=(+,-,x,y)."""
    a=np.asarray(blocks,complex)
    if a.ndim<3 or a.shape[-3]!=4 or not np.all(np.isfinite(a)) or not np.isfinite(angle):
        raise ValueError("finite four-current blocks and angle required")
    result=a.copy();c,s=np.cos(angle),np.sin(angle)
    result[...,2,:,:]=c*a[...,2,:,:]-s*a[...,3,:,:]
    result[...,3,:,:]=s*a[...,2,:,:]+c*a[...,3,:,:]
    return result*np.exp(.5j*(jz_in_twice-jz_out_twice)*angle)


def complete_elastic_spin_amplitudes(pp,mp,transfer):
    """Exact parity/z-rotation/reversal completion, NOT a dynamical-spin fit.

    Requires equal-energy elastic states in the source positive-parity phase,
    zero mean P_T, and separately established kinematic current identities.
    Input last axis is (+,-,x,y). All leading family/order axes survive.
    This map's use must be labeled; it does not independently test its inputs'
    symmetry, and must not be used on unequal-energy transitions by default.
    """
    a,b=np.asarray(pp,complex),np.asarray(mp,complex)
    d=np.asarray(transfer,float)
    if a.shape!=b.shape or a.ndim<1 or a.shape[-1]!=4 or d.shape!=(2,) or not np.all(np.isfinite(d)) or np.linalg.norm(d)==0:
        raise ValueError("equal four-component amplitude shapes and nonzero transfer required")
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("finite spin amplitudes required")
    angle=2*np.arctan2(d[1],d[0])-np.pi
    c,s=np.cos(angle),np.sin(angle)
    result=np.zeros((*a.shape,2,2),complex)
    result[...,0,0]=a;result[...,1,0]=b
    result[...,0,1]=b.conj()*np.array([-1,-1,1,1])
    result[...,1,1]=a
    # Px reverses x, then Rz brings Px Delta back to Delta.
    result[...,2,1,1]=-c*a[...,2]-s*a[...,3]
    result[...,3,1,1]=-s*a[...,2]+c*a[...,3]
    return result


def elastic_orientation_analysis(current, transfer, *, energy_GeV2, total_pplus_GeV,
                                 helicity_basis=None):
    """Orthogonal diagnostic residuals; coefficients never modify the current.

    Two-structure residual is independent of the choice of B-tensor convention.
    Five-structure coefficients and their out-of-span residual are both kept.
    Raw-order packets may be analyzed linearly at fixed external kinematics.
    """
    a=np.asarray(current,complex)
    if a.shape!=(4,2,2) or not np.all(np.isfinite(a)) or not np.isfinite(energy_GeV2) or energy_GeV2<=0:
        raise ValueError("finite elastic helicity packet and positive mass squared required")
    basis=orientation_current_structures(transfer,mass_GeV=np.sqrt(energy_GeV2),
                                       total_pplus_GeV=total_pplus_GeV)
    if helicity_basis is not None:
        u=np.asarray(helicity_basis,complex)
        if u.shape!=(2,2) or not np.all(np.isfinite(u)) or not np.allclose(u.conj().T@u,np.eye(2),atol=2e-13,rtol=0):
            raise ValueError("explicit unitary spin-coordinate map required")
        basis=np.einsum('ab,kmbc,cd->kmad',u.conj().T,basis,u,optimize=True)
    axes=np.column_stack([b.ravel() for b in basis])
    fits={}
    for n in (2,5):
        coeff,_,rank,sv=np.linalg.lstsq(axes[:,:n],a.ravel(),rcond=None)
        if rank!=n:raise ValueError("current structures are unresolved")
        residual=a-np.einsum('k,kmab->mab',coeff,basis[:n])
        fits[str(n)]=dict(coefficients=coeff,residual=residual,
            residual_norm=float(np.linalg.norm(residual)),condition=float(sv[0]/sv[-1]))
    return dict(two= fits['2'],five=fits['5'],basis=basis,
        continuity=np.einsum('j,jab->ab',transfer,a[2:]),physical=False,covariance_repaired=False)


@dataclass(frozen=True)
class FreeSpinFiber:
    """Only H0 and source coordinates; no fabricated interaction coefficients."""
    total_k2: int
    cutoff: int
    target: str
    jz_twice: int
    width_GeV: float
    quark_mass_GeV: float
    gluon_mass_GeV: float
    bases: tuple
    h0: np.ndarray
    coupling: float = 0.
    physical: bool = False

    @property
    def matrix(self):return self.h0


def free_spin_fiber(*,total_k2,cutoff,target,jz_twice,b_GeV,quark_mass_GeV,gluon_mass_GeV):
    """Source-owned g=0 carrier, avoiding evaluation of unused interactions."""
    from .m2_coupled_pilot import symmetry_basis,free_sector
    labels=(('qqq',0),('qqqg',-1),('qqqg',1))
    bases=tuple(symmetry_basis(s,total_k2,cutoff,target,jz_twice,h) for s,h in labels)
    h=sparse.block_diag([u.getH()@free_sector(s,total_k2,cutoff,b_GeV,
        quark_mass_GeV,gluon_mass_GeV)@u for (s,_),u in zip(labels,bases)]).toarray()
    return FreeSpinFiber(total_k2,cutoff,target,jz_twice,b_GeV,quark_mass_GeV,gluon_mass_GeV,bases,h)


def free_current_parts(outgoing,incoming,transfer,*,total_pplus_GeV,charge,
                       quadrature_order,minus_context):
    """Exact g0 physical and retained-projector pieces, before any state choice.

    Q-wave returns start at g2; Wick/contact terms at g1 or g2. This is the
    whole represented g0 current, not a shortcut for nonzero coupling.
    """
    from .m2_canonical_good_current import good_current_density_between
    from .m2_free_minus_current import free_minus_current_between
    from .m2_transverse_current_projection import free_transverse_current_between
    from .m2_smooth_longitudinal_projector import retained_free_projector_components
    if outgoing.coupling!=0 or incoming.coupling!=0:
        raise ValueError("free diagnostic requires explicitly zero coupling")
    f=np.zeros((4,len(outgoing.matrix),len(incoming.matrix)),complex)
    p=np.zeros_like(f)
    args=dict(total_pplus_GeV=total_pplus_GeV,charge=charge,
              mean_total_transverse=minus_context.mean_total_transverse_GeV)
    f[0]=good_current_density_between(outgoing,incoming,transfer,charge=charge)
    f[1]=free_minus_current_between(outgoing,incoming,transfer,**args)
    f[2:]=free_transverse_current_between(outgoing,incoming,transfer,**args)
    args=dict(total_pplus_GeV=total_pplus_GeV,charge=charge,quadrature_order=quadrature_order)
    p[1]=retained_free_projector_components(outgoing,incoming,transfer,
                                           minus_context=minus_context,**args)[0]
    p[2:]=retained_free_projector_components(outgoing,incoming,transfer,**args)
    return f,p
