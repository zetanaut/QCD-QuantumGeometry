"""Quartic source projector versus the original H, not a fitted Ward defect."""
import numpy as np
import pytest
from deuteron_wigner.quantum.m2_gaussian_cluster_quartic_Q_projector import (
    cluster_quartic_source_projector,_apply_source_raw)
from deuteron_wigner.quantum.m2_gaussian_cluster_quartic_source_current import build_cluster_quartic_source_current
from deuteron_wigner.quantum.m2_gaussian_cluster_h2 import _source
from deuteron_wigner.quantum.m2_source_vectors import source_tensor
from deuteron_wigner.quantum.m2_canonical_good_current import good_current_density_between
from deuteron_wigner.quantum.m2_shell_current import shell_density_between
from test_m2_gaussian_cluster_h2 import model
from test_m2_gaussian_cluster_Q_projector import D,PP,ARGS
from test_m2_gaussian_cluster_quartic_source_actions import FAMILIES,space


def original(p,c,n,cutoff):
    kind={"valence_IF":"IF","valence_IG":"IG","valence_sea":"sea"}[c] if n==5 else c
    return _source(p,kind,cutoff).values


@pytest.mark.parametrize("n,c",FAMILIES)
def test_every_full_quartic_source_Ward_and_raw_H_equivalence(n,c):
    p=model().pilot;s=space(n,c)
    v=source_tensor(p,"qqq" if n==5 else "qqqg",np.eye(len(p.matrix)))
    h=original(p,c,n,0)
    np.testing.assert_allclose(_apply_source_raw(p,s,v,component=c),h,atol=4e-15)
    k=cluster_quartic_source_projector(p,s,D,component=c,**ARGS)
    a=build_cluster_quartic_source_current(p,s,D,component=c,mean_total_transverse_GeV=(0.,0.),**ARGS)
    physical=np.zeros_like(k);physical[:,:,a.retained_columns]=a.total[2:]
    rho=good_current_density_between(p,p,D,charge="electromagnetic")
    expected=h@rho-shell_density_between(s,s,h,D,b_GeV=p.width_GeV,charge="electromagnetic")
    np.testing.assert_allclose(2*PP*np.einsum("j,jab->ab",D,k+physical),expected,atol=2e-14)
    for a in k:np.testing.assert_allclose(s.statistics_project(a),a,atol=2e-15)
    print(n,c,"H/projector norms",np.linalg.norm(h),np.linalg.norm(k),flush=True)


@pytest.mark.parametrize("n,c",[(5,"valence_IF"),(5,"valence_sea"),(6,"gluon_A4"),(6,"gluon_IG_pair")])
def test_neutron_opposite_spin_shell_one_raw_Ward_and_flavor_resolution(n,c):
    i=model(target="NEUTRON").pilot;f=model(target="NEUTRON",jz=-1).pilot
    si,sf=space(n,c,"NEUTRON",1,1),space(n,c,"NEUTRON",-1,1)
    hi,hf=original(i,c,n,1),original(f,c,n,1)
    k=cluster_quartic_source_projector(i,sf,D,component=c,**ARGS)
    a=build_cluster_quartic_source_current(i,sf,D,component=c,mean_total_transverse_GeV=(0.,0.),**ARGS)
    physical=np.zeros_like(k);physical[:,:,a.retained_columns]=a.total[2:]
    rho=good_current_density_between(f,i,D,charge="electromagnetic")
    expected=hf@rho-shell_density_between(sf,si,hi,D,b_GeV=i.width_GeV,charge="electromagnetic")
    np.testing.assert_allclose(2*PP*np.einsum("j,jab->ab",D,k+physical),expected,atol=2e-14)
    up,down=[cluster_quartic_source_projector(i,sf,D,component=c,**dict(ARGS,charge=q))
             for q in ("up_number","down_number")]
    np.testing.assert_allclose(k,(2*up-down)/3,atol=3e-15)
    refined=cluster_quartic_source_projector(i,sf,D,component=c,**dict(ARGS,quadrature_order=12))
    np.testing.assert_allclose(k,refined,atol=3e-15)
    assert np.linalg.norm(k)>1e-8
    if c=="gluon_A4":np.testing.assert_array_equal(physical,0.)


def test_source_projector_rejects_wrong_endpoint_or_unidentified_definition():
    p=model().pilot;s=space(5,"valence_IF")
    with pytest.raises(ValueError,match="actual"):
        cluster_quartic_source_projector(object(),s,D,component="valence_IF",**ARGS)
    with pytest.raises(ValueError,match="matching"):
        cluster_quartic_source_projector(p,s,D,component="valence_sea",**ARGS)
    with pytest.raises(ValueError):
        cluster_quartic_source_projector(p,s,D,component="valence_IF",**dict(ARGS,prescription="unknown"))
