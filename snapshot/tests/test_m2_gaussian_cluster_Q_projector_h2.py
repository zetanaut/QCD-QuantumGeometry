"""Common-order H2 Ward checks including exact free/cubic Q boundaries."""
import numpy as np
import pytest
from deuteron_wigner.quantum.m2_gaussian_cluster_h2_current import assemble_cluster_h2_current
from deuteron_wigner.quantum.m2_gaussian_cluster_Q_projector import (
    cluster_free_Q_projector_h2_coefficients,cluster_cubic_Q_projector_h2_coefficients,
    cluster_excluded_cubic_projector,ClusterSixCubicProjector,shell_free_transverse_projector)
from deuteron_wigner.quantum.m2_gaussian_cluster_retained_projector import exact_retained_transverse_projector_coefficients
from deuteron_wigner.quantum.m2_canonical_good_current import good_current_density_between
from test_m2_gaussian_cluster_h2 import model
from test_m2_gaussian_cluster_cubic_h2_current import independent_waves
from test_m2_gaussian_cluster_Q_projector import D,PP,ARGS


def H_coefficients(m,e):
    a,b=independent_waves(m,e);p=m.pilot
    v,w=m.qq.sources_by_power[1],m.qq.sources_by_power[2]
    h={0:p.h0,1:p.emission_per_gs,
       2:p.cross_pairs_per_gs2+p.instantaneous_fermion_per_gs2+p.wick_mass_per_gs2}
    h[2]=h[2]+sum(v[n].conj().T@a[0][n] for n in v)
    h[3]=sum(v[n].conj().T@a[1][n]+w[n].conj().T@a[0][n] for n in v)
    h[4]=sum(v[n].conj().T@a[2][n]+w[n].conj().T@a[1][n] for n in v)
    h[4]+=sum(m.six.W[n].conj().T@b[n] for n in b)
    np.testing.assert_allclose(sum(p.coupling**n*x for n,x in h.items()),
                               m.snapshot(e).effective_hamiltonian,atol=3e-14)
    return h


@pytest.mark.parametrize("target,jz",[("PROTON",1),("NEUTRON",-1)])
def test_actual_common_H2_full_matrix_Ward_through_g2_and_remaining_quartic_Q_defect(target,jz):
    f,i=model(target=target,jz=jz).snapshot(1.39),model(target=target).snapshot(1.47)
    a=assemble_cluster_h2_current(f,i,D,loop_flavor_charges=(2/3,-1/3),
        mean_total_transverse_GeV=(0.,0.),**ARGS)
    kp=exact_retained_transverse_projector_coefficients(f.owner.pilot,i.owner.pilot,D,**ARGS)
    k0=cluster_free_Q_projector_h2_coefficients(f,i,D,**ARGS)
    k1=cluster_cubic_Q_projector_h2_coefficients(f,i,D,**ARGS)
    hf,hi=H_coefficients(f.owner,f.energy_GeV2),H_coefficients(i.owner,i.energy_GeV2)
    rho=good_current_density_between(f.owner.pilot,i.owner.pilot,D,charge="electromagnetic")
    residuals=[]
    for n,v in a.coefficients.items():
        current=v[2:]+kp.get(n,0.)+k0.get(n,0.)+k1.get(n,0.)
        expected=hf[n]@rho-rho@hi[n]
        if n:expected+=(f.energy_GeV2-i.energy_GeV2)*v[0]
        residue=2*PP*np.einsum("j,jab->ab",D,current)-expected
        residuals.append(float(np.linalg.norm(residue)))
        if n<=2:
            np.testing.assert_allclose(residue,0.,atol=3e-14)
            np.testing.assert_allclose(f.whitener@residue@i.whitener,0.,atol=3e-14)
    print(target,jz,"full matrix Ward norms with exact PP/free/cubic-Q only",residuals,flush=True)
    # Complete Q quartic/source-projector words remain a separate requirement.
    assert not a.complete_H2_matched_current


def test_exact_cubic_Q_H2_powers_against_independent_full_wave_polynomial():
    f,i=model(target="PROTON").snapshot(1.39),model(target="PROTON").snapshot(1.47)
    af,bf=independent_waves(f.owner,f.energy_GeV2);ai,bi=independent_waves(i.owner,i.energy_GeV2)
    result=cluster_cubic_Q_projector_h2_coefficients(f,i,D,**ARGS)
    plus={n:cluster_excluded_cubic_projector(i.owner.pilot,s,D,**ARGS) for n,s in f.owner.spaces5.items()}
    minus={n:cluster_excluded_cubic_projector(f.owner.pilot,s,-D,**ARGS) for n,s in i.owner.spaces5.items()}
    kf=ClusterSixCubicProjector.from_models(f.owner,i.owner,D,**ARGS)
    kb=ClusterSixCubicProjector.from_models(i.owner,f.owner,-D,**ARGS)
    actf={j:kf.apply(ai[0],component=j) for j in range(2)}
    actb={j:kb.apply(af[0],component=j) for j in range(2)}
    points=np.exp(2j*np.pi*np.arange(7)/7);samples=[]
    for z in points:
        v=np.zeros_like(result[2])
        for n in plus:
            wf=sum(z**(q+1)*a[n].conj().T for q,a in enumerate(af))
            wi=sum(z**(q+1)*a[n] for q,a in enumerate(ai))
            v+=z*(np.einsum("ab,jbc->jac",wf,plus[n])+np.einsum("jba,bc->jac",minus[n].conj(),wi))
        for j in range(2):
            v[j]+=z**4*sum(bf[n].conj().T@actf[j][n]+actb[j][n].conj().T@bi[n] for n in bi)
        samples.append(v)
    for n in (2,3,4):
        reference=sum(v/z**n for z,v in zip(points,samples))/len(points)
        np.testing.assert_allclose(result[n],reference,atol=8e-16)
