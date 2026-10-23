"""Full retained-matrix H2 minus words versus independent raw-wave polynomials."""
import numpy as np
import pytest
from deuteron_wigner.quantum.m2_gaussian_cluster_smooth_minus_h2 import assemble_cluster_h2_smooth_minus_projector
from deuteron_wigner.quantum.m2_finite_cell_envelope_current import SPECTRAL_PRESCRIPTION
from deuteron_wigner.quantum.m2_gaussian_cluster_retained_projector import exact_retained_smooth_minus_projector_coefficients
from deuteron_wigner.quantum.m2_gaussian_cluster_Q_projector import (
    shell_free_smooth_minus_projector,cluster_excluded_cubic_orbital_projector,ClusterSixCubicProjector)
from deuteron_wigner.quantum.m2_gaussian_cluster_quartic_Q_projector import cluster_quartic_source_smooth_minus_projector
from deuteron_wigner.quantum.m2_gaussian_cluster_quartic_Q_pair_projector import ClusterFiveQuarticProjector
from deuteron_wigner.quantum.m2_six_body_quartic import COMPONENTS
from test_m2_gaussian_cluster_h2 import model
from test_m2_gaussian_cluster_cubic_h2_current import independent_waves
from test_m2_gaussian_cluster_smooth_minus_actions import ZERO
from test_m2_gaussian_cluster_Q_projector import D,PP,B,ARGS


def assembled(f,i,delta=D):
    return assemble_cluster_h2_smooth_minus_projector(f,i,delta,minus_context=ZERO,
        field_envelope_prescription=SPECTRAL_PRESCRIPTION,**ARGS)


@pytest.mark.parametrize("target,jz",[("PROTON",1),("NEUTRON",-1)])
def test_full_smooth_minus_H2_all_orders_reverse_both_energies_and_both_norms(target,jz):
    f=model(target=target,jz=jz).snapshot(1.39)
    i=model(target=target).snapshot(1.47)
    a,b=assembled(f,i),assembled(i,f,-D)
    assert set(a.contributions)=={"retained","free_Q","cubic_Q","quartic_source","quartic_pair"}
    for family,block in a.contributions.items():
        for n,v in block.items():
            np.testing.assert_allclose(v,b.contributions[family][n].conj().transpose(0,2,1),atol=3e-14)
    raw=a.current(normalized=False)
    np.testing.assert_allclose(a.current(normalized=True)[0],f.whitener@raw[0]@i.whitener,atol=3e-14)
    print(target,jz,"smooth-minus coefficient norms",[np.linalg.norm(a.coefficients[n]) for n in range(5)],flush=True)
    assert np.linalg.norm(raw)>1e-6
    assert a.all_smooth_oscillator_H2_words_through_g4
    assert a.zero_plus_transfer_field_envelope_bound
    assert not a.retained_isospin_minus_supplied
    assert not a.complete_H2_matched_current and not a.physical
    with pytest.raises(ValueError):a.current(normalized=None)


def test_every_smooth_minus_H2_family_matches_independent_degree_eight_full_wave_polynomial():
    f,i=model().snapshot(1.39),model().snapshot(1.47)
    a=assembled(f,i)
    af,bf=independent_waves(f.owner,f.energy_GeV2)
    ai,bi=independent_waves(i.owner,i.energy_GeV2)
    nc=len(i.owner.pilot.matrix)
    shape=(1,len(f.owner.pilot.matrix),nc)
    terms={name:[] for name in a.contributions}
    def add(name,power,value):terms[name].append((power,np.asarray(value).reshape(shape)))
    args=dict(ARGS,minus_context=ZERO)
    retained=exact_retained_smooth_minus_projector_coefficients(f.owner.pilot,i.owner.pilot,D,**args)
    for n,v in retained.items():add("retained",n,v)
    joined_i={n:np.column_stack([v[n] for v in ai]) for n in ai[0]}
    joined_f={n:np.column_stack([v[n] for v in af]) for n in af[0]}
    free_args=dict(b_GeV=B,total_pplus_GeV=PP,charge="electromagnetic",quadrature_order=8,minus_context=ZERO)
    for name,space in i.owner.spaces5.items():
        values=shell_free_smooth_minus_projector(f.owner.spaces5[name],space,joined_i[name],D,**free_args)
        for left in range(3):
            for right in range(3):
                add("free_Q",left+right+2,af[left][name].conj().T@values[:,right*nc:(right+1)*nc])
        kf=cluster_excluded_cubic_orbital_projector(i.owner.pilot,f.owner.spaces5[name],D,**args)
        kb=cluster_excluded_cubic_orbital_projector(f.owner.pilot,space,-D,**args)
        for q in range(3):
            add("cubic_Q",q+2,af[q][name].conj().T@kf[0]+kb[0].conj().T@ai[q][name])
    for name,space in i.owner.spaces6.items():
        values=shell_free_smooth_minus_projector(f.owner.spaces6[name],space,bi[name],D,**free_args)
        add("free_Q",4,bf[name].conj().T@values)
    forward=ClusterSixCubicProjector.from_models(f.owner,i.owner,D,**args)
    backward=ClusterSixCubicProjector.from_models(i.owner,f.owner,-D,**args)
    jf,jb=forward.apply(joined_i,component=0),backward.apply(joined_f,component=0)
    for name in bi:
        for q in range(3):
            add("cubic_Q",q+4,bf[name].conj().T@jf[name][:,q*nc:(q+1)*nc]
                +jb[name][:,q*nc:(q+1)*nc].conj().T@bi[name])
    for count,spaces_f,spaces_i,components in (
        (5,f.owner.spaces5,i.owner.spaces5,COMPONENTS[:3]),
        (6,f.owner.spaces6,i.owner.spaces6,COMPONENTS)):
        for component in components:
            sea=component in ("valence_sea","gluon_IF_pair","gluon_IG_pair")
            name=("qqqq-qbar" if sea else "qqqgg") if count==5 else ("qqqq-qbar-g" if sea else "qqqggg")
            kf=cluster_quartic_source_smooth_minus_projector(i.owner.pilot,spaces_f[name],D,component=component,**args)[0]
            kb=cluster_quartic_source_smooth_minus_projector(f.owner.pilot,spaces_i[name],-D,component=component,**args)[0]
            if count==5:
                for q in range(3):add("quartic_source",q+3,af[q][name].conj().T@kf+kb.conj().T@ai[q][name])
            else:add("quartic_source",4,bf[name].conj().T@kf+kb.conj().T@bi[name])
    pair=ClusterFiveQuarticProjector.from_models(f.owner,i.owner,D,**args)
    applied=pair.apply(joined_i,component=0)
    for name in ai[0]:
        for left in range(3):
            for right in range(3):
                add("quartic_pair",left+right+4,af[left][name].conj().T@applied[name][:,right*nc:(right+1)*nc])
    points=np.exp(2j*np.pi*np.arange(9)/9)
    reference={}
    for name,words in terms.items():
        samples=[sum(z**power*v for power,v in words) for z in points]
        reference[name]={n:sum(v/z**n for z,v in zip(points,samples))/9 for n in range(9)}
        for n in range(5):
            np.testing.assert_allclose(a.contributions[name][n],reference[name][n],atol=4e-15)
    assert np.linalg.norm(reference["quartic_source"][5])>1e-12
    assert np.linalg.norm(reference["quartic_pair"][6])>1e-12
    g=i.owner.pilot.coupling
    selected=sum(g**n*reference[name][n] for name in terms for n in range(5))
    np.testing.assert_allclose(a.current(normalized=False),selected,atol=4e-15)
    np.testing.assert_allclose(a.current(normalized=True)[0],f.whitener@selected[0]@i.whitener,atol=4e-15)


def test_full_minus_assembly_rejects_unidentified_context_and_envelope():
    f,i=model().snapshot(1.39),model().snapshot(1.47)
    for changes in (dict(minus_context=None),dict(field_envelope_prescription=None)):
        args=dict(ARGS,minus_context=ZERO,field_envelope_prescription=SPECTRAL_PRESCRIPTION)
        with pytest.raises(ValueError):assemble_cluster_h2_smooth_minus_projector(f,i,D,**{**args,**changes})


def test_Q_shell_one_full_minus_matrix_reversal_and_both_norms():
    m=model(q5=1,q6=1)
    f,i=m.snapshot(1.39),m.snapshot(1.47)
    a,b=assembled(f,i),assembled(i,f,-D)
    for name,block in a.contributions.items():
        for n,v in block.items():
            np.testing.assert_allclose(v,b.contributions[name][n].conj().transpose(0,2,1),atol=4e-14)
    raw=a.current(normalized=False)[0]
    np.testing.assert_allclose(a.current(normalized=True)[0],f.whitener@raw@i.whitener,atol=4e-14)
    norms=[float(np.linalg.norm(a.coefficients[n])) for n in range(5)]
    assert all(np.isfinite(norms)) and all(v>1e-10 for v in norms)
    print("PROTON Q1 smooth-minus norms",norms,flush=True)
