"""Analytic paired finite-kinetic-mass responses of the common H2 four-current.

Differentiate the explicit mu_q² or mu_g² coefficient per g_s², keeping the
vertex mass, regulator, Fock spaces and energies fixed. Only H orders2/4,
N order4 and current orders2/4 change. No finite condition is selected.
"""
from dataclasses import dataclass
import numpy as np
from .m2_gaussian_cluster_h2 import GaussianClusterSnapshot
from .m2_gaussian_cluster_h2_current import _endpoints
from .m2_wick_one_body import WickOneBody
from .m2_canonical_good_current import parton_charge
from .m2_canonical_quark_self import _integer
from .m2_wick_current_projection import retained_wick_minus_between,shell_wick_minus_between
from .m2_shell_current import shell_density_between
from .m2_shell_minus_current import shell_minus_current_between
from .m2_shell_transverse_current import shell_transverse_current_between
from .m2_gaussian_cluster_excluded_current import build_cluster_excluded_cubic_current
from .m2_gaussian_cluster_Q_projector import (shell_free_transverse_projector,
    shell_free_smooth_minus_projector,cluster_excluded_cubic_projector,
    cluster_excluded_cubic_orbital_projector)
from .m2_total_isospin import require_half_isospin_H2_snapshot
from .m2_smooth_longitudinal_projector import SmoothMinusProjectorContext
from .m2_finite_cell_envelope_current import SPECTRAL_PRESCRIPTION


@dataclass(frozen=True)
class UnitFiniteKineticMass:
    parameter: str

    def __post_init__(self):
        if self.parameter not in ("quark","gluon"):
            raise ValueError("explicit quark or gluon finite kinetic-mass parameter required")

    def mass_components(self,species,mode,*,total_k2):
        mode,total_k2=_integer(mode,"constituent mode"),_integer(total_k2,"total K2")
        if species not in ("q","qbar","g") or mode>=total_k2:
            raise ValueError("physical constituent species and 0<p2<K2 required")
        selected=(species=="g")==(self.parameter=="gluon")
        return 0.,0.,float(total_k2/mode) if selected else 0.

    retained_mass_coefficient=WickOneBody.retained_mass_coefficient
    shell_mass_diagonal=WickOneBody.shell_mass_diagonal

    def minus_particle_weight(self,species,mode,spins,*,total_k2,total_pplus_GeV,
                              charge,loop_flavor_charges):
        self.mass_components(species,mode,total_k2=total_k2)
        if not np.isfinite(total_pplus_GeV) or total_pplus_GeV<=0:
            raise ValueError("positive finite total P+ required")
        e=parton_charge(spins,species,charge)
        if self.parameter=="gluon" or species=="g":return np.zeros_like(e,dtype=float)
        return e*total_k2**2/(2*mode**2*total_pplus_GeV**2)


def inverse_sqrt_frechet(metric,direction):
    """Exact Hermitian inverse-square-root derivative, regular at degeneracy.

    The divided difference is -1/(sqrt(x)sqrt(y)(sqrt(x)+sqrt(y))).
    This includes off-diagonal norm motion; differentiating eigenvalues
    alone is incorrect when metric and direction do not commute.
    """
    n,d=np.asarray(metric,complex),np.asarray(direction,complex)
    if n.ndim!=2 or n.shape[0]!=n.shape[1] or d.shape!=n.shape or not np.all(np.isfinite([n,d])):
        raise ValueError("finite square metric and matching direction required")
    if not np.allclose(n,n.conj().T,atol=1e-13,rtol=1e-13) or not np.allclose(d,d.conj().T,atol=1e-13,rtol=1e-13):
        raise ValueError("Hermitian norm and direction required")
    values,u=np.linalg.eigh(n)
    if np.min(values)<=0:raise ValueError("positive metric required; no eigenvalue clipping")
    roots=np.sqrt(values)
    divided=-1/(roots[:,None]*roots[None,:]*(roots[:,None]+roots[None,:]))
    return u@(divided*(u.conj().T@d@u))@u.conj().T


@dataclass(frozen=True)
class FiniteMassSnapshotResponse:
    snapshot: GaussianClusterSnapshot
    parameter: str
    H_coefficients: dict
    metric_coefficient_g4: np.ndarray
    C3_direction: dict
    physical=False

    @property
    def effective_hamiltonian(self):
        g=self.snapshot.owner.pilot.coupling
        return sum(g**n*v for n,v in self.H_coefficients.items())

    @property
    def metric(self):
        return self.snapshot.owner.pilot.coupling**4*self.metric_coefficient_g4

    @property
    def whitener(self):
        return inverse_sqrt_frechet(self.snapshot.metric,self.metric)

    @property
    def matrix(self):
        s=self.snapshot;dw=self.whitener
        a=s.effective_hamiltonian+s.energy_GeV2*(s.metric-np.eye(len(s.metric)))
        da=self.effective_hamiltonian+s.energy_GeV2*self.metric
        return dw@a@s.whitener+s.whitener@da@s.whitener+s.whitener@a@dw


def finite_mass_snapshot_response(snapshot,*,parameter):
    """D_PP at g²; V1† R5 D55 A1 at g4; its norm/backflow partners."""
    if not isinstance(snapshot,GaussianClusterSnapshot):
        raise ValueError("actual common-cluster H2 snapshot required")
    unit=UnitFiniteKineticMass(parameter);m=snapshot.owner
    a1=snapshot.five_wave_coefficients[0]
    dc={n:m.qq.propagation[n].resolve(snapshot.energy_GeV2,
        unit.shell_mass_diagonal(s)[:,None]*a1[n]) for n,s in m.spaces5.items()}
    h2=unit.retained_mass_coefficient(m.pilot)
    h4=sum(m.qq.sources_by_power[1][n].conj().T@dc[n] for n in dc)
    norm=sum(a1[n].conj().T@dc[n]+dc[n].conj().T@a1[n] for n in dc)
    return FiniteMassSnapshotResponse(snapshot,parameter,{2:h2,4:h4},norm,dc)


def finite_mass_current_coefficients(outgoing,incoming,transfer,*,parameter,charge,
        loop_flavor_charges,prescription,quadrature_order,minus_context,
        field_envelope_prescription,isospin_prescription):
    """Analytic raw-current derivative, not differencing two assembled currents.

    Direct finite J-minus acts in PP at g2 and A1f†J55 A1i at g4.
    dC3=R5 D55 A1 changes the free-Q and cubic-source returns at g4,
    including their exact oscillator partners. All other words are fixed.
    """
    unit=UnitFiniteKineticMass(parameter)
    if not isinstance(minus_context,SmoothMinusProjectorContext):
        raise ValueError("explicit smooth-minus context required")
    if field_envelope_prescription!=SPECTRAL_PRESCRIPTION:
        raise ValueError("explicit gapped field-envelope prescription required")
    delta,mean,order=_endpoints(outgoing,incoming,transfer,minus_context.mean_total_transverse_GeV,
                               prescription,quadrature_order)
    expected=parton_charge(np.array([0,2]),"q",charge)
    if not np.array_equal(np.asarray(loop_flavor_charges),expected):
        raise ValueError("same two physical loop charges required")
    for s in (outgoing,incoming):require_half_isospin_H2_snapshot(s,prescription=isospin_prescription)
    rf,ri=[finite_mass_snapshot_response(s,parameter=parameter) for s in (outgoing,incoming)]
    f,i=outgoing.owner,incoming.owner;p=i.pilot
    af,ai=outgoing.five_wave_coefficients[0],incoming.five_wave_coefficients[0]
    shape=(4,len(f.pilot.matrix),len(p.matrix));values={n:np.zeros(shape,complex) for n in range(5)}
    direct=dict(operators=unit,total_pplus_GeV=p.total_pplus_GeV,charge=charge,loop_flavor_charges=loop_flavor_charges)
    values[2][1]=retained_wick_minus_between(f.pilot,p,delta,**direct)
    args=dict(charge=charge,prescription=prescription,quadrature_order=order)
    fieldargs=dict(args,mean_total_transverse_GeV=mean)
    freeargs=dict(b_GeV=p.width_GeV,total_pplus_GeV=p.total_pplus_GeV,charge=charge)
    for name,ins in i.spaces5.items():
        outs=f.spaces5[name];nc=len(p.matrix)
        joined=np.column_stack((ai[name],ri.C3_direction[name]))
        wick=shell_wick_minus_between(outs,ins,ai[name],delta,b_GeV=p.width_GeV,**direct)
        values[4][1]+=af[name].conj().T@wick
        for mu in range(4):
            if mu==0:
                applied=shell_density_between(outs,ins,joined,delta,b_GeV=p.width_GeV,charge=charge)
            elif mu==1:
                applied=shell_minus_current_between(outs,ins,joined,delta,mass_GeV=p.quark_mass_GeV,
                    mean_total_transverse=mean,**freeargs)
                applied+=shell_free_smooth_minus_projector(outs,ins,joined,delta,
                    quadrature_order=order,minus_context=minus_context,**freeargs)
            else:
                applied=shell_transverse_current_between(outs,ins,joined,delta,component=mu-2,
                    mean_total_transverse=mean,**freeargs)
                applied+=shell_free_transverse_projector(outs,ins,joined,delta,component=mu-2,
                    quadrature_order=order,**freeargs)
            values[4][mu]+=rf.C3_direction[name].conj().T@applied[:,:nc]+af[name].conj().T@applied[:,nc:]
        forward=build_cluster_excluded_cubic_current(p,outs,delta,**fieldargs)
        backward=build_cluster_excluded_cubic_current(f.pilot,ins,-np.asarray(delta),**fieldargs)
        values[4][:,:,forward.retained_columns]+=np.einsum("ab,mbc->mac",rf.C3_direction[name].conj().T,forward.total)
        values[4][:,backward.retained_columns,:]+=np.einsum("mba,bc->mac",backward.total.conj(),ri.C3_direction[name])
        kf=cluster_excluded_cubic_projector(p,outs,delta,**args)
        kb=cluster_excluded_cubic_projector(f.pilot,ins,-np.asarray(delta),**args)
        values[4][2:]+=np.einsum("ab,mbc->mac",rf.C3_direction[name].conj().T,kf)
        values[4][2:]+=np.einsum("mba,bc->mac",kb.conj(),ri.C3_direction[name])
        kf=cluster_excluded_cubic_orbital_projector(p,outs,delta,minus_context=minus_context,**args)[0]
        kb=cluster_excluded_cubic_orbital_projector(f.pilot,ins,-np.asarray(delta),minus_context=minus_context,**args)[0]
        values[4][1]+=rf.C3_direction[name].conj().T@kf+kb.conj().T@ri.C3_direction[name]
    for v in values.values():v.setflags(write=False)
    return values


def normalized_finite_mass_current_derivative(assembly,*,parameter):
    """Differentiate both original norm maps, including noncommuting motion."""
    from .m2_gaussian_cluster_four_current import ClusterH2FourCurrent
    if not isinstance(assembly,ClusterH2FourCurrent):raise ValueError("actual bound four-current assembly required")
    base=assembly.physical_numerator
    f,i=base.outgoing,base.incoming
    coefficients=finite_mass_current_coefficients(f,i,base.transfer_GeV,parameter=parameter,
        charge=base.charge,loop_flavor_charges=base.loop_flavor_charges,
        prescription=base.prescription,quadrature_order=base.quadrature_order,
        minus_context=assembly.smooth_minus.minus_context,
        field_envelope_prescription=assembly.smooth_minus.field_envelope_prescription,
        isospin_prescription=assembly.isospin_prescription)
    rf,ri=[finite_mass_snapshot_response(s,parameter=parameter) for s in (f,i)]
    raw=assembly.current(normalized=False)
    derivative=sum(i.owner.pilot.coupling**n*v for n,v in coefficients.items())
    product=lambda a,j,b:np.einsum("ab,mbc,cd->mad",a,j,b,optimize=True)
    return product(rf.whitener,raw,i.whitener)+product(f.whitener,derivative,i.whitener)+product(f.whitener,raw,ri.whitener)
