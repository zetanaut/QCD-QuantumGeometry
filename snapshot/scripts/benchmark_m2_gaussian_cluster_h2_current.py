"""Diagnose the full physical numerator plus finite-parent projector Ward."""
import json
import numpy as np
from deuteron_wigner.quantum.m2_gaussian_cluster_h2_current import (
    assemble_cluster_h2_current,cluster_transverse_projector_h2_coefficients)
from deuteron_wigner.quantum.m2_h2_projector_current import H2ProjectorCurrent
from deuteron_wigner.quantum.m2_canonical_good_current import good_current_density_between
from deuteron_wigner.quantum.m2_gaussian_cluster_h2 import build_cluster_pilot,GaussianClusterH2
from deuteron_wigner.quantum.m2_gaussian_cluster_wick import GaussianClusterWick
from deuteron_wigner.quantum.m2_gaussian_field_regulator import GaussianClusterRegulator
from deuteron_wigner.quantum.m2_interacting_q_loop_regulator import InteractingQLoopRegulatorCurrent
from deuteron_wigner.quantum.m2_gaussian_cluster_retained_projector import exact_retained_transverse_projector_coefficients
from deuteron_wigner.quantum.m2_gaussian_cluster_Q_projector import (
    cluster_free_Q_projector_h2_coefficients,cluster_cubic_Q_projector_h2_coefficients)
from deuteron_wigner.quantum.m2_gaussian_cluster_quartic_Q_projector import cluster_quartic_source_projector_h2_coefficients
from deuteron_wigner.quantum.m2_gaussian_cluster_quartic_Q_pair_projector import cluster_quartic_pair_projector_h2_coefficients

# Explicit exploratory control, not physical calibration or a production default.
def pilot(cutoff=0):
    reg=GaussianClusterRegulator(width_GeV=.8,boson_modes=8,fermion_modes=8)
    wick=GaussianClusterWick(regulator=reg,quark_flavors=2,normal_order_subtraction=True,
        finite_quark_mass2_per_gs2_GeV2=.011,finite_gluon_mass2_per_gs2_GeV2=-.007)
    return build_cluster_pilot(total_k2=9,cutoff=cutoff,target="PROTON",jz_twice=1,
        b_GeV=.4,quark_mass_GeV=.35,gluon_mass_GeV=0.,coupling=.3,regulator=reg,
        total_pplus_GeV=1.3,wick_operators=wick)

def scalar_vectors(count,seed):
    rng=np.random.default_rng(seed)
    x=rng.normal(size=(count,1))+1j*rng.normal(size=(count,1))
    return x/np.linalg.norm(x)
from deuteron_wigner.quantum.m2_gaussian_cluster_current import PRESCRIPTION

D=np.array([.12,.16])
inner=GaussianClusterH2(pilot(),excluded_cutoff=0,six_cutoff=0)
f,i=inner.snapshot(1.39),inner.snapshot(1.47)
print("construct physical current",flush=True)
a=assemble_cluster_h2_current(f,i,D,charge="electromagnetic",loop_flavor_charges=(2/3,-1/3),
    mean_total_transverse_GeV=(0.,0.),prescription=PRESCRIPTION,quadrature_order=8)
print("construct finite-parent projector",flush=True)
outer=GaussianClusterH2(pilot(),excluded_cutoff=1,six_cutoff=1)
q=InteractingQLoopRegulatorCurrent(outer,outer,inner_cutoffs={n:0 for n in (*outer.spaces5,*outer.spaces6)},
    energy_out=1.39,energy_in=1.47,transfer=D,charge="electromagnetic",total_pplus_GeV=1.3,quadrature_order=16)
c=H2ProjectorCurrent(q,parent_outgoing=pilot(1),parent_incoming=pilot(1),quadrature_order=16)
vf,vi=scalar_vectors(len(f.owner.pilot.matrix),321),scalar_vectors(len(i.owner.pilot.matrix),987)
b=cluster_transverse_projector_h2_coefficients(f,i,c,outgoing_vectors=vf,incoming_vectors=vi)
print("exact retained projector; preceding PP-only control",flush=True)
exactP=exact_retained_transverse_projector_coefficients(f.owner.pilot,i.owner.pilot,D,
    charge="electromagnetic",prescription=PRESCRIPTION,quadrature_order=8)
print("exact free/cubic Q projector; preceding partial-Q control",flush=True)
exactQ0=cluster_free_Q_projector_h2_coefficients(f,i,D,
    charge="electromagnetic",prescription=PRESCRIPTION,quadrature_order=8)
exactQ1=cluster_cubic_Q_projector_h2_coefficients(f,i,D,
    charge="electromagnetic",prescription=PRESCRIPTION,quadrature_order=8)
print("exact quartic Q source/pair projectors; independent minus/envelope remain open",flush=True)
exactQS=cluster_quartic_source_projector_h2_coefficients(f,i,D,
    charge="electromagnetic",prescription=PRESCRIPTION,quadrature_order=8)
exactQ2=cluster_quartic_pair_projector_h2_coefficients(f,i,D,
    charge="electromagnetic",prescription=PRESCRIPTION,quadrature_order=8)
print("form independent Hamiltonian Ward comparison",flush=True)
rho=good_current_density_between(f.owner.pilot,i.owner.pilot,D,charge="electromagnetic")
hs={}
for endpoint,snapshot in (("outgoing",f),("incoming",i)):
    p=snapshot.owner.pilot
    h={0:p.h0,1:p.emission_per_gs,2:p.cross_pairs_per_gs2+p.instantaneous_fermion_per_gs2+p.wick_mass_per_gs2,
       3:np.zeros_like(p.matrix),4:np.zeros_like(p.matrix)}
    for n,v in q.return_coefficients(endpoint=endpoint).items():h[n]=h[n]+v
    hs[endpoint]=h
values=[]
for n,v in a.coefficients.items():
    expected=hs["outgoing"][n]@rho-rho@hs["incoming"][n]
    if n:expected+=(f.energy_GeV2-i.energy_GeV2)*v[0]
    rhs=(vf.conj().T@expected@vi)[0,0]
    field=(2*1.3*vf.conj().T@np.einsum("j,jab->ab",D,v[2:])@vi)[0,0]
    terms={k:(2*1.3*np.einsum("j,jab->ab",D,w[n]))[0,0] for k,w in b.items()}
    total=field+sum(terms.values())
    exact_term=(2*1.3*vf.conj().T@np.einsum("j,jab->ab",D,exactP[n])@vi)[0,0] if n in exactP else 0.
    only_exact_P=field+exact_term-rhs
    exact_Q=exactQ0.get(n,0.)+exactQ1.get(n,0.)
    q_term=(2*1.3*vf.conj().T@np.einsum("j,jab->ab",D,exact_Q)@vi)[0,0] if n>=2 else 0.
    with_free_cubic_Q=only_exact_P+q_term
    quartic=exactQS.get(n,0.)+exactQ2.get(n,0.)
    q2_term=(2*1.3*vf.conj().T@np.einsum("j,jab->ab",D,quartic)@vi)[0,0] if n>=3 else 0.
    with_all_Q=with_free_cubic_Q+q2_term
    pair=lambda z:[float(z.real),float(z.imag)]
    values.append(dict(power=n,rhs=pair(rhs),physical=pair(field),
        projector={k:pair(x) for k,x in terms.items()},residual=pair(total-rhs),
        residual_abs=float(abs(total-rhs)),
        residual_with_exact_PP_only=pair(only_exact_P),residual_with_exact_PP_only_abs=float(abs(only_exact_P)),
        residual_with_exact_PP_free_cubic_Q=pair(with_free_cubic_Q),
        residual_with_exact_PP_free_cubic_Q_abs=float(abs(with_free_cubic_Q)),
        residual_with_all_exact_transverse_projectors=pair(with_all_Q),
        residual_with_all_exact_transverse_projectors_abs=float(abs(with_all_Q))))
print(json.dumps(dict(physical=False,full_H2_matching_complete=False,regulator_width_GeV=.8,
    finite_quark_mass2_per_gs2_GeV2=.011,finite_gluon_mass2_per_gs2_GeV2=-.007,
    parent_P=1,parent_Q5=1,parent_Q6=1,inner_P=0,inner_Q5=0,inner_Q6=0,values=values),indent=2),flush=True)
