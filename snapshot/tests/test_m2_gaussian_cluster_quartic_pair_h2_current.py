"""Quartic pair H2 returns: independently reconstructed waves and powers."""
import numpy as np
import pytest
from deuteron_wigner.quantum import m2_gaussian_cluster_quartic_pair_actions as current
from test_m2_gaussian_cluster_h2 import model
from test_m2_gaussian_cluster_cubic_h2_current import independent_waves
from test_m2_gaussian_cluster_quartic_pair_actions import ARGS,DELTA,action,spaces,raw


def polynomial_reference(f,i):
    """Degree-eight full-wave product, DFT without conjugating formal g."""
    af,_=independent_waves(f.owner,f.energy_GeV2)
    ai,_=independent_waves(i.owner,i.energy_GeV2)
    pp=current.cluster_quartic_pair_current_between(f.owner.pilot,i.owner.pilot,DELTA,**ARGS)
    operator=current.ClusterFiveQuarticCurrent.from_models(f.owner,i.owner,DELTA,**ARGS)
    ni=len(i.owner.pilot.matrix)
    joined={n:np.column_stack([a[n] for a in ai]) for n in ai[0]}
    points=np.exp(2j*np.pi*np.arange(9)/9)
    results={}
    for part in ("canonical","regulator"):
        acted={mu:operator.apply(joined,component=mu,part=part) for mu in range(4)}
        samples=[]
        for z in points:
            value=z**2*getattr(pp,part).copy()
            for mu in range(4):
                for n in ai[0]:
                    bra=sum(z**(k+1)*a[n].conj().T for k,a in enumerate(af))
                    ket=sum(z**(k+1)*acted[mu][n][:,k*ni:(k+1)*ni] for k in range(3))
                    value[mu]+=z**2*bra@ket
            samples.append(value)
        results[part]={n:sum(v/z**n for z,v in zip(points,samples))/9 for n in range(9)}
    return results


@pytest.mark.parametrize("target,jz",[("PROTON",1),("NEUTRON",1),("NEUTRON",-1)])
def test_pair_H2_words_full_wave_polynomial_both_endpoints_and_normalization(target,jz,monkeypatch):
    i=model(target=target).snapshot(1.47)
    f=model(target=target,jz=jz).snapshot(1.39)
    coeff=current.cluster_quartic_pair_h2_coefficients(f,i,DELTA,**ARGS)
    reference=polynomial_reference(f,i)
    assert set(coeff)=={2,4}
    for part in ("canonical","regulator"):
        for n in (2,4):
            np.testing.assert_allclose(getattr(coeff[n],part),reference[part][n],atol=6e-15)
        for n in (0,1,3):
            np.testing.assert_allclose(reference[part][n],0.,atol=5e-16)
        if jz==1:
            assert np.linalg.norm(reference[part][4])>1e-9
            assert sum(np.linalg.norm(reference[part][n]) for n in range(5,9))>1e-10
        else:
            # Spin-diagonal pair contacts cannot bridge different Jz when
            # both Q5 orbital spaces contain only the ground orbital.
            for n in range(9):
                np.testing.assert_allclose(reference[part][n],0.,atol=5e-16)
    reverse=current.cluster_quartic_pair_h2_coefficients(i,f,tuple(-x for x in DELTA),**ARGS)
    for n in coeff:
        for part in ("canonical","regulator"):
            np.testing.assert_allclose(getattr(coeff[n],part),
                getattr(reverse[n],part).conj().transpose(0,2,1),atol=5e-15)
    monkeypatch.setattr(current,"cluster_quartic_pair_h2_coefficients",lambda *a,**k:coeff)
    for normalized in (False,True):
        got=current.cluster_quartic_pair_h2_current_between(f,i,DELTA,normalized=normalized,**ARGS)
        for part in ("canonical","regulator"):
            expected=sum(.3**n*getattr(v,part) for n,v in coeff.items())
            if normalized:
                expected=np.einsum("ab,mbc,cd->mad",f.whitener,expected,i.whitener)
            np.testing.assert_allclose(getattr(got,part),expected,atol=5e-16)
        assert not got.physical and not got.complete_gauged_contractions
        np.testing.assert_array_equal(got.total[0],0.)
    with pytest.raises(ValueError,match="four-current"):
        i.current(DELTA,charge="electromagnetic")


def test_raw_pair_rectangular_action_equals_complete_larger_ground_columns():
    small,out=spaces(0,"NEUTRON",1),spaces(1,"NEUTRON",-1)
    large=spaces(1,"NEUTRON",1);x=raw(small,999)
    padded={}
    for n,s in large.items():
        rows=np.flatnonzero(s.coordinates[:,1]==0)
        np.testing.assert_array_equal(s.coordinates[rows],small[n].coordinates)
        padded[n]=np.zeros((s.dimension,1),complex);padded[n][rows]=x[n]
    rectangular,complete=action(small,out),action(large,out)
    for part in ("canonical","regulator"):
        for mu in (1,2,3):
            a=rectangular.apply(x,component=mu,part=part)
            b=complete.apply(padded,component=mu,part=part)
            for n in out:np.testing.assert_allclose(a[n],b[n],atol=2e-15)


def test_H2_pair_binding_requires_common_definitions_and_both_resolutions():
    a,b=model(),model(width=.4)
    with pytest.raises(ValueError,match="identical cluster"):
        current.ClusterFiveQuarticCurrent.from_models(a,b,DELTA,**ARGS)
    with pytest.raises(ValueError,match="both actual"):
        current.ClusterFiveQuarticCurrent.from_models(a.pilot,b.pilot,DELTA,**ARGS)
    with pytest.raises(ValueError,match="both actual"):
        current.cluster_quartic_pair_h2_coefficients(a,b,DELTA,**ARGS)
    ss=a.snapshot(1.4)
    with pytest.raises(ValueError,match="Boolean"):
        current.cluster_quartic_pair_h2_current_between(ss,ss,DELTA,normalized="auto",**ARGS)
    larger=model(q5=1).snapshot(1.4)
    with pytest.raises(ValueError,match="same declared Q-sector"):
        current.cluster_quartic_pair_h2_coefficients(larger,ss,DELTA,**ARGS)


def test_Q5_shell_one_opposite_spin_H2_return_is_not_the_ground_orbital_zero():
    i=model(q5=1,target="NEUTRON").snapshot(1.47)
    f=model(q5=1,target="NEUTRON",jz=-1).snapshot(1.39)
    coeff=current.cluster_quartic_pair_h2_coefficients(f,i,DELTA,**ARGS)
    reverse=current.cluster_quartic_pair_h2_coefficients(i,f,tuple(-x for x in DELTA),**ARGS)
    np.testing.assert_allclose(coeff[2].total,0.,atol=4e-16)
    assert np.linalg.norm(coeff[4].regulator)>1e-9
    for part in ("canonical","regulator"):
        np.testing.assert_allclose(getattr(coeff[4],part),
            getattr(reverse[4],part).conj().transpose(0,2,1),atol=6e-15)
