"""All exact quartic Q boundaries in the independent two-energy H2 Ward test."""
import numpy as np
import pytest
from deuteron_wigner.quantum.m2_gaussian_cluster_h2_current import assemble_cluster_h2_current
from deuteron_wigner.quantum.m2_gaussian_cluster_Q_projector import (
    cluster_free_Q_projector_h2_coefficients,cluster_cubic_Q_projector_h2_coefficients)
from deuteron_wigner.quantum.m2_gaussian_cluster_quartic_Q_projector import (
    cluster_quartic_source_projector_h2_coefficients,cluster_quartic_source_projector)
from deuteron_wigner.quantum.m2_gaussian_cluster_quartic_Q_pair_projector import (
    cluster_quartic_pair_projector_h2_coefficients,ClusterFiveQuarticProjector)
from deuteron_wigner.quantum.m2_gaussian_cluster_retained_projector import exact_retained_transverse_projector_coefficients
from deuteron_wigner.quantum.m2_canonical_good_current import good_current_density_between
from test_m2_gaussian_cluster_Q_projector_h2 import H_coefficients
from test_m2_gaussian_cluster_Q_projector import D,PP,ARGS
from test_m2_gaussian_cluster_h2 import model
from test_m2_gaussian_cluster_cubic_h2_current import independent_waves
from deuteron_wigner.quantum.m2_six_body_quartic import COMPONENTS


@pytest.mark.parametrize("target,jz,cutoff",[("PROTON",1,0),("NEUTRON",-1,0),("PROTON",1,1)])
def test_complete_transverse_H2_full_matrix_two_energy_Ward_through_g4(target,jz,cutoff):
    f=model(q5=cutoff,q6=cutoff,target=target,jz=jz).snapshot(1.39)
    i=model(q5=cutoff,q6=cutoff,target=target).snapshot(1.47)
    a=assemble_cluster_h2_current(f,i,D,loop_flavor_charges=(2/3,-1/3),
                                 mean_total_transverse_GeV=(0.,0.),**ARGS)
    kp=exact_retained_transverse_projector_coefficients(f.owner.pilot,i.owner.pilot,D,**ARGS)
    k0=cluster_free_Q_projector_h2_coefficients(f,i,D,**ARGS)
    k1=cluster_cubic_Q_projector_h2_coefficients(f,i,D,**ARGS)
    ks=cluster_quartic_source_projector_h2_coefficients(f,i,D,**ARGS)
    k2=cluster_quartic_pair_projector_h2_coefficients(f,i,D,**ARGS)
    hf,hi=H_coefficients(f.owner,f.energy_GeV2),H_coefficients(i.owner,i.energy_GeV2)
    rho=good_current_density_between(f.owner.pilot,i.owner.pilot,D,charge="electromagnetic")
    residuals=[]
    for n,v in a.coefficients.items():
        current=v[2:]+sum(k.get(n,0.) for k in (kp,k0,k1,ks,k2))
        expected=hf[n]@rho-rho@hi[n]
        if n:expected+=(f.energy_GeV2-i.energy_GeV2)*v[0]
        residue=2*PP*np.einsum("j,jab->ab",D,current)-expected
        residuals.append(float(np.linalg.norm(residue)))
        print(target,jz,"Q shell",cutoff,"power",n,"Ward Frobenius",residuals[-1],flush=True)
        np.testing.assert_allclose(residue,0.,atol=3e-14)
        np.testing.assert_allclose(f.whitener@residue@i.whitener,0.,atol=3e-14)
    # Transverse closure cannot supply an independent J-minus partner.
    assert not a.complete_H2_matched_current


def test_quartic_Q_words_match_independent_full_wave_polynomial_without_later_order_subsets():
    f,i=model().snapshot(1.39),model().snapshot(1.47)
    af,bf=independent_waves(f.owner,f.energy_GeV2);ai,bi=independent_waves(i.owner,i.energy_GeV2)
    source=cluster_quartic_source_projector_h2_coefficients(f,i,D,**ARGS)
    pair=cluster_quartic_pair_projector_h2_coefficients(f,i,D,**ARGS)
    forward=[];backward=[]
    for n,spaces_f,spaces_i,components in ((5,f.owner.spaces5,i.owner.spaces5,COMPONENTS[:3]),
                                           (6,f.owner.spaces6,i.owner.spaces6,COMPONENTS)):
        for c in components:
            sea=c in ("valence_sea","gluon_IF_pair","gluon_IG_pair")
            name=("qqqq-qbar" if sea else "qqqgg") if n==5 else ("qqqq-qbar-g" if sea else "qqqggg")
            forward.append((n,name,cluster_quartic_source_projector(i.owner.pilot,spaces_f[name],D,component=c,**ARGS)))
            backward.append(cluster_quartic_source_projector(f.owner.pilot,spaces_i[name],-D,component=c,**ARGS))
    action=ClusterFiveQuarticProjector.from_models(f.owner,i.owner,D,**ARGS)
    joined={n:np.column_stack([a[n] for a in ai]) for n in ai[0]}
    applied=[action.apply(joined,component=j) for j in range(2)];nc=len(i.owner.pilot.matrix)
    points=np.exp(2j*np.pi*np.arange(9)/9);s_samples=[];p_samples=[]
    for z in points:
        sf=np.zeros_like(source[3]);pf=np.zeros_like(pair[4])
        for (n,name,kf),kb in zip(forward,backward):
            if n==5:
                left=sum(z**(q+1)*a[name].conj().T for q,a in enumerate(af))
                right=sum(z**(q+1)*a[name] for q,a in enumerate(ai))
            else:left=z*z*bf[name].conj().T;right=z*z*bi[name]
            sf+=z*z*(np.einsum("ab,jbc->jac",left,kf)+np.einsum("jba,bc->jac",kb.conj(),right))
        for j in range(2):
            for name in ai[0]:
                left=sum(z**(q+1)*a[name].conj().T for q,a in enumerate(af))
                right=sum(z**(q+1)*applied[j][name][:,q*nc:(q+1)*nc] for q in range(3))
                pf[j]+=z*z*left@right
        s_samples.append(sf);p_samples.append(pf)
    def coefficient(samples,n):return sum(v/z**n for z,v in zip(points,samples))/len(points)
    for n in (3,4):np.testing.assert_allclose(source[n],coefficient(s_samples,n),atol=2e-15)
    np.testing.assert_allclose(pair[4],coefficient(p_samples,4),atol=2e-15)
    assert np.linalg.norm(coefficient(s_samples,5))>1e-9
    assert np.linalg.norm(coefficient(p_samples,6))>1e-9
    assert set(source)=={3,4} and set(pair)=={4}
    # Both original metrics act on the selected coefficient sum only.
    g=i.owner.pilot.coupling
    selected=g**3*source[3]+g**4*(source[4]+pair[4])
    reference=g**3*coefficient(s_samples,3)+g**4*(coefficient(s_samples,4)+coefficient(p_samples,4))
    for a,b in zip(selected,reference):
        np.testing.assert_allclose(f.whitener@a@i.whitener,f.whitener@b@i.whitener,atol=2e-15)
