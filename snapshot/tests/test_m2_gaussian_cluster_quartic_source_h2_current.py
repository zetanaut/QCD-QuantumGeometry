"""Quartic P→Q5/Q6 current returns at precisely the declared H2 powers."""
import numpy as np
import pytest
from deuteron_wigner.quantum import m2_gaussian_cluster_quartic_source_current as current
from test_m2_gaussian_cluster_h2 import model
from test_m2_gaussian_cluster_cubic_h2_current import independent_waves
from test_m2_gaussian_cluster_quartic_source_actions import FAMILIES,space
from test_m2_gaussian_cluster_quartic_source_current import D,MEAN
from deuteron_wigner.quantum.m2_gaussian_cluster_current import PRESCRIPTION

ARGS = dict(charge="electromagnetic",mean_total_transverse_GeV=MEAN,
            prescription=PRESCRIPTION,quadrature_order=8)


def polynomial_reference(f,i):
    """Full Neumann wave recurrence; DFT separates omitted fifth-order terms."""
    af,a6f = independent_waves(f.owner,f.energy_GeV2)
    ai,a6i = independent_waves(i.owner,i.energy_GeV2)
    ni,nf = len(i.owner.pilot.matrix),len(f.owner.pilot.matrix)
    blocks = []
    for n,c in FAMILIES:
        name = space(n,c).sector
        fs = f.owner.spaces5[name] if n==5 else f.owner.spaces6[name]
        ins = i.owner.spaces5[name] if n==5 else i.owner.spaces6[name]
        jf = current.build_cluster_quartic_source_current(i.owner.pilot,fs,D,component=c,**ARGS)
        jb = current.build_cluster_quartic_source_current(f.owner.pilot,ins,tuple(-x for x in D),component=c,**ARGS)
        blocks.append((n,name,jf,jb))
    points = np.exp(2j*np.pi*np.arange(8)/8)
    results = {}
    for part in ("canonical","regulator"):
        samples = []
        for z in points:
            value = np.zeros((4,nf,ni),complex)
            for n,name,jf,jb in blocks:
                forward = np.zeros((4,jf.outgoing_space.dimension,ni),complex)
                backward = np.zeros((4,jb.outgoing_space.dimension,nf),complex)
                forward[:,:,jf.retained_columns] = getattr(jf,part)
                backward[:,:,jb.retained_columns] = getattr(jb,part)
                bra = (sum(z**(k+1)*a[name].conj().T for k,a in enumerate(af)) if n==5
                       else z**2*a6f[name].conj().T)
                ket = sum(z**(k+1)*a[name] for k,a in enumerate(ai)) if n==5 else z**2*a6i[name]
                value += z**2*(np.einsum("ab,mbc->mac",bra,forward)+np.einsum("mba,bc->mac",backward.conj(),ket))
            samples.append(value)
        results[part] = {n:sum(v/z**n for z,v in zip(points,samples))/8 for n in range(7)}
    return results


@pytest.mark.parametrize("target,jz",[("PROTON",1),("NEUTRON",1),("NEUTRON",-1)])
def test_actual_source_H2_words_two_energy_spin_endpoints_order_separation_and_both_norms(target,jz,monkeypatch):
    i = model(target=target).snapshot(1.47)
    f = model(target=target,jz=jz).snapshot(1.39)
    a = current.cluster_quartic_source_h2_coefficients(f,i,D,**ARGS)
    ref = polynomial_reference(f,i)
    assert set(a)=={3,4}
    for part in ("canonical","regulator"):
        for n in (3,4):
            np.testing.assert_allclose(getattr(a[n],part),ref[part][n],atol=5e-15)
        for n in (0,1,2,6):
            np.testing.assert_allclose(ref[part][n],0.,atol=5e-16)
        if jz==1:
            assert np.linalg.norm(ref[part][5])>1e-10
        else:
            # A transverse-constant quartic source preserves total spin;
            # with P/Q ground orbitals no internal orbital can absorb ΔJz.
            # Nonzero opposite-Jz shell1 actions are tested independently.
            for n in (3,4,5):
                np.testing.assert_allclose(ref[part][n],0.,atol=5e-16)
    back = current.cluster_quartic_source_h2_coefficients(i,f,tuple(-x for x in D),**ARGS)
    for n in a:
        for part in ("canonical","regulator"):
            np.testing.assert_allclose(getattr(a[n],part),getattr(back[n],part).conj().transpose(0,2,1),atol=5e-15)
    monkeypatch.setattr(current,"cluster_quartic_source_h2_coefficients",lambda *args,**kwargs:a)
    for normalized in (False,True):
        value = current.cluster_quartic_source_h2_current_between(f,i,D,normalized=normalized,**ARGS)
        for part in ("canonical","regulator"):
            expected = sum(.3**n*getattr(c,part) for n,c in a.items())
            if normalized:expected = np.einsum("ab,mbc,cd->mad",f.whitener,expected,i.whitener)
            np.testing.assert_allclose(getattr(value,part),expected,atol=3e-16)
        assert not value.physical and not value.longitudinal_envelope_gauged
    with pytest.raises(ValueError,match="four-current"):
        i.current(D,charge="electromagnetic")


def test_full_H2_source_binding_guards():
    a,b = model().snapshot(1.4),model(width=.4).snapshot(1.4)
    with pytest.raises(ValueError,match="identical cluster"):
        current.cluster_quartic_source_h2_coefficients(a,b,D,**ARGS)
    with pytest.raises(ValueError,match="both actual"):
        current.cluster_quartic_source_h2_coefficients(a.owner,b.owner,D,**ARGS)
    with pytest.raises(ValueError,match="Boolean"):
        current.cluster_quartic_source_h2_current_between(a,a,D,normalized="auto",**ARGS)
    larger = model(q5=1).snapshot(1.4)
    with pytest.raises(ValueError,match="same declared Q-sector"):
        current.cluster_quartic_source_h2_coefficients(larger,a,D,**ARGS)
