"""Independent total-isospin algebra and the strong H2 source-domain test."""
from itertools import product
import numpy as np
import pytest
from scipy import sparse
from deuteron_wigner.quantum.m2_total_isospin import (
    SPECIES,spin_total_isospin_squared,total_isospin_squared_apply)
from deuteron_wigner.quantum.m2_isospin_projector_current import retained_isospin_projector
from deuteron_wigner.quantum.m2_source_vectors import source_tensor
from deuteron_wigner.quantum.m2_gaussian_cluster_Q_projector import _apply_excluded_raw
from deuteron_wigner.quantum.m2_gaussian_cluster_quartic_Q_projector import _apply_source_raw
from test_m2_gaussian_cluster_h2 import model
from test_m2_gaussian_cluster_Q_projector import raw
from test_m2_gaussian_cluster_quartic_source_actions import FAMILIES,space


@pytest.mark.parametrize("sector",list(SPECIES))
def test_Casimir_equals_independent_quark_and_antiquark_Pauli_generators(sector):
    species=SPECIES[sector];dims=[2 if s=="g" else 4 for s in species]
    pauli=[np.array([[0,1],[1,0]])/2,np.array([[0,-1j],[1j,0]])/2,np.diag([1,-1])/2]
    generators=[]
    for t in pauli:
        total=sparse.csr_matrix((np.prod(dims),)*2,dtype=complex)
        for a,kind in enumerate(species):
            if kind=="g":continue
            local=np.kron(t if kind=="q" else -t.T,np.eye(2))
            factors=[sparse.csr_matrix(local) if k==a else sparse.eye(d) for k,d in enumerate(dims)]
            value=factors[0]
            for f in factors[1:]:value=sparse.kron(value,f,format="csr")
            total+=value
        generators.append(total)
    expected=sum(t@t for t in generators)
    residue=spin_total_isospin_squared(sector)-expected
    np.testing.assert_allclose(residue.data,0.,atol=1e-15)


@pytest.mark.parametrize("target,cutoff",[("PROTON",0),("NEUTRON",0),("PROTON",1),("NEUTRON",1)])
def test_actual_H2_wave_coefficients_and_strong_Q_actions_preserve_total_half_isospin(target,cutoff):
    m=model(q5=cutoff,q6=cutoff,target=target)
    a=m.snapshot(1.39)
    residuals=[]
    for waves in a.five_wave_coefficients:
        for n,s in m.spaces5.items():
            r=total_isospin_squared_apply(s,waves[n])-.75*waves[n]
            residuals.append(np.linalg.norm(r))
            np.testing.assert_allclose(r,0.,atol=3e-13)
    for n,s in m.spaces6.items():
        r=total_isospin_squared_apply(s,a.six_wave_coefficient[n])-.75*a.six_wave_coefficient[n]
        residuals.append(np.linalg.norm(r))
        np.testing.assert_allclose(r,0.,atol=3e-13)
    x=raw(m.spaces5);cx={n:total_isospin_squared_apply(s,x[n]) for n,s in m.spaces5.items()}
    hx,hcx=m.qq.action.apply(x),m.qq.action.apply(cx)
    for n,s in m.spaces5.items():
        np.testing.assert_allclose(total_isospin_squared_apply(s,hx[n]),hcx[n],atol=3e-13)
    ux,ucx=m.six.U.apply(x),m.six.U.apply(cx)
    for n,s in m.spaces6.items():
        np.testing.assert_allclose(total_isospin_squared_apply(s,ux[n]),ucx[n],atol=3e-13)
    print(target,"Q",cutoff,"max total-I wave residual",max(residuals),flush=True)


@pytest.mark.parametrize("kind,component",[("cubic","qqqgg"),("cubic","qqqq-qbar")]+FAMILIES)
def test_raw_complement_sources_have_total_three_halves_and_zero_H2_pullback(kind,component):
    m=model();p=m.pilot
    sector="qqqg" if kind=="cubic" or kind==6 else "qqq"
    template=source_tensor(p,sector,np.eye(len(p.matrix)))
    rng=np.random.default_rng(830)
    v=rng.normal(size=(*template.shape[:-1],2))+1j*rng.normal(size=(*template.shape[:-1],2))
    iso=retained_isospin_projector()
    if sector=="qqq":outside=v-np.einsum("st,loatk->loask",iso,v,optimize=True)
    else:outside=v-np.einsum("st,loathk->loashk",iso,v,optimize=True)
    if kind=="cubic":
        s=m.spaces5[component];image=_apply_excluded_raw(p,s,outside)
    else:
        s=space(kind,component);image=_apply_source_raw(p,s,outside,component=component)
    np.testing.assert_allclose(total_isospin_squared_apply(s,image),3.75*image,atol=3e-13)
    snapshot=m.snapshot(1.39)
    waves=(snapshot.five_wave_coefficients if kind=="cubic" or kind==5 else [snapshot.six_wave_coefficient])
    for a in waves:np.testing.assert_allclose(a[s.sector].conj().T@image,0.,atol=3e-13)
    if (kind,component) not in ((6,"valence_IG"),(6,"gluon_IG_ggg")):
        assert np.linalg.norm(image)>1e-6
