"""Complete cubic H2 current words: independent polynomial wave-map extraction."""
import numpy as np
import pytest
from deuteron_wigner.quantum import m2_gaussian_cluster_excluded_current as current
from deuteron_wigner.quantum.m2_gaussian_cluster_current_actions import cluster_cubic_current_between
from test_m2_gaussian_cluster_h2 import model
from test_m2_gaussian_cluster_excluded_current import ARGS,DELTA,MEAN,make,raw,spaces


def independent_waves(owner,energy):
    """Full free-Q recurrence, independent of the snapshot's named returns."""
    props = {**owner.qq.propagation,**owner.six.propagation6}
    s1 = {**owner.qq.sources_by_power[1],**{n:np.zeros_like(v) for n,v in owner.six.W.items()}}
    s2 = {**owner.qq.sources_by_power[2],**owner.six.W}
    resolve = lambda x:{n:props[n].resolve(energy,a) for n,a in x.items()}
    def u(x):
        return {**owner.six.U.adjoint({n:x[n] for n in owner.spaces6}),
                **owner.six.U.apply({n:x[n] for n in owner.spaces5})}
    def w(x):
        return {**owner.qq.action.apply({n:x[n] for n in owner.spaces5}),
                **{n:np.zeros_like(x[n]) for n in owner.spaces6}}
    a1 = resolve(s1);ua1 = u(a1)
    a2 = resolve({n:s2[n]+ua1[n] for n in s2})
    ua2,wa1 = u(a2),w(a1)
    a3 = resolve({n:ua2[n]+wa1[n] for n in ua2})
    return ([{n:a[n] for n in owner.spaces5} for a in (a1,a2,a3)],
            {n:a2[n] for n in owner.spaces6})


def polynomial_reference(f,i,part):
    """Sample full, untruncated wave products; extract powers by an exact DFT.

    The formal variable multiplies bra coefficients without conjugation.
    The degree-six polynomial includes the g5/g6 terms which the declared
    H2 result must not import as an isolated higher-order subset.
    """
    pf,pi = f.owner.pilot,i.owner.pilot
    af,a6f = independent_waves(f.owner,f.energy_GeV2)
    ai,a6i = independent_waves(i.owner,i.energy_GeV2)
    pp = getattr(cluster_cubic_current_between(pf,pi,DELTA,**ARGS),part)
    pfive,bfive = {},{}
    reverse = tuple(-x for x in DELTA)
    for n,s in i.owner.spaces5.items():
        forward = current.build_cluster_excluded_cubic_current(pi,f.owner.spaces5[n],DELTA,**ARGS)
        backward = current.build_cluster_excluded_cubic_current(pf,s,reverse,**ARGS)
        jf = np.zeros((4,f.owner.spaces5[n].dimension,len(pi.matrix)),complex)
        jb = np.zeros((4,s.dimension,len(pf.matrix)),complex)
        jf[:,:,forward.retained_columns] = getattr(forward,part)
        jb[:,:,backward.retained_columns] = getattr(backward,part)
        pfive[n],bfive[n] = jf,jb
    forward = current.ClusterSixCubicCurrent.from_models(f.owner,i.owner,DELTA,**ARGS)
    backward = current.ClusterSixCubicCurrent.from_models(i.owner,f.owner,reverse,**ARGS)
    ni,nf = len(pi.matrix),len(pf.matrix)
    joined_i = {n:np.column_stack([a[n] for a in ai]) for n in ai[0]}
    joined_f = {n:np.column_stack([a[n] for a in af]) for n in af[0]}
    applied_i,applied_f = {},{}
    for mu in range(4):
        applied_i[mu] = forward.apply(joined_i,component=mu,part=part)
        applied_f[mu] = backward.apply(joined_f,component=mu,part=part)
    samples = []
    points = np.exp(2j*np.pi*np.arange(8)/8)
    for z in points:
        value = z*pp.copy()
        for n in ai[0]:
            bra = sum(z**(k+1)*a[n].conj().T for k,a in enumerate(af))
            ket = sum(z**(k+1)*a[n] for k,a in enumerate(ai))
            value += z*np.einsum("ab,mbc->mac",bra,pfive[n])
            value += z*np.einsum("mba,bc->mac",bfive[n].conj(),ket)
        for mu in range(4):
            for n in a6i:
                ket = sum(z**(k+1)*applied_i[mu][n][:,k*ni:(k+1)*ni] for k in range(3))
                bra = sum(z**(k+1)*applied_f[mu][n][:,k*nf:(k+1)*nf].conj().T for k in range(3))
                value[mu] += z**3*(a6f[n].conj().T@ket+bra@a6i[n])
        samples.append(value)
    return {n:sum(v/z**n for z,v in zip(points,samples))/8 for n in range(7)}


@pytest.mark.parametrize("target,jz",[("PROTON",1),("NEUTRON",-1)])
def test_all_common_cubic_current_words_against_independent_full_wave_polynomial(target,jz,monkeypatch):
    incoming = model(0,0,target).snapshot(1.47)
    outgoing = model(0,0,target,jz=jz).snapshot(1.39)
    coefficients = current.cluster_cubic_h2_coefficients(outgoing,incoming,DELTA,**ARGS)
    assert set(coefficients)=={1,2,3,4}
    for part in ("canonical","regulator"):
        reference = polynomial_reference(outgoing,incoming,part)
        for n in coefficients:
            np.testing.assert_allclose(getattr(coefficients[n],part),reference[n],atol=6e-15)
        np.testing.assert_allclose(reference[0],0.,atol=3e-16)
        assert sum(np.linalg.norm(reference[n]) for n in (5,6))>1e-10
    for n in coefficients:
        assert np.linalg.norm(coefficients[n].total)>1e-8
        np.testing.assert_array_equal(coefficients[n].total[0],0.)
    reverse = current.cluster_cubic_h2_coefficients(incoming,outgoing,tuple(-x for x in DELTA),**ARGS)
    for n in coefficients:
        for part in ("canonical","regulator"):
            np.testing.assert_allclose(getattr(coefficients[n],part),
                getattr(reverse[n],part).conj().transpose(0,2,1),atol=4e-15)
    # The actual computed coefficients are reused to isolate and verify
    # the wrapper's two-endpoint normalization, without rebuilding currents.
    monkeypatch.setattr(current,"cluster_cubic_h2_coefficients",lambda *a,**k:coefficients)
    normed = current.cluster_cubic_h2_current_between(outgoing,incoming,DELTA,normalized=True,**ARGS)
    unnormed = current.cluster_cubic_h2_current_between(outgoing,incoming,DELTA,normalized=False,**ARGS)
    for part in ("canonical","regulator"):
        expected = sum(.3**n*getattr(a,part) for n,a in coefficients.items())
        np.testing.assert_array_equal(getattr(unnormed,part),expected)
        np.testing.assert_allclose(getattr(normed,part),
            np.einsum("ab,mbc,cd->mad",outgoing.whitener,expected,incoming.whitener),atol=4e-16)
    assert not normed.physical and not normed.complete_gauged_contractions
    with pytest.raises(ValueError,match="four-current"):
        incoming.current(DELTA,charge="electromagnetic")


def test_Q5_Q6_flavor_and_both_boost_laws_on_complete_action():
    inc,out = spaces(0,"PROTON",-1);x = raw(inc)
    beta = np.array([.04,-.07])
    a,u,d = [make(inc,out,charge=q) for q in ("electromagnetic","up_number","down_number")]
    long = make(inc,out,pp=2.6)
    shift = make(inc,out,mean=tuple(np.asarray(MEAN)+1.3*beta))
    for part in ("canonical","regulator"):
        jt = [a.apply(x,component=m,part=part) for m in (2,3)]
        for mu in (1,2,3):
            base = a.apply(x,component=mu,part=part)
            up,down = u.apply(x,component=mu,part=part),d.apply(x,component=mu,part=part)
            boosted,moved = long.apply(x,component=mu,part=part),shift.apply(x,component=mu,part=part)
            for n in out:
                np.testing.assert_allclose(base[n],(2*up[n]-down[n])/3,atol=2e-16)
                np.testing.assert_allclose(boosted[n],base[n]/(4 if mu==1 else 2),atol=2e-16)
                expected = base[n]+sum(beta[j]*jt[j][n] for j in (0,1)) if mu==1 else base[n]
                np.testing.assert_allclose(moved[n],expected,atol=4e-16)


def test_full_H2_binding_rejects_different_regulators_and_resolution():
    a,b = model(),model(width=.4)
    with pytest.raises(ValueError,match="identical cluster"):
        current.ClusterSixCubicCurrent.from_models(a,b,DELTA,**ARGS)
    with pytest.raises(ValueError,match="both actual"):
        current.ClusterSixCubicCurrent.from_models(a.pilot,b.pilot,DELTA,**ARGS)
    with pytest.raises(ValueError,match="Boolean"):
        current.cluster_cubic_h2_current_between(a.snapshot(1.4),a.snapshot(1.4),DELTA,normalized="auto",**ARGS)
    higher = model(1,0)
    with pytest.raises(ValueError,match="same declared Q-sector"):
        current.cluster_cubic_h2_coefficients(higher.snapshot(1.4),a.snapshot(1.4),DELTA,**ARGS)


def test_rectangular_raw_action_equals_complete_larger_incoming_ground_columns():
    small,out = spaces(0,"PROTON",-1)
    large,larger_out = spaces(1,"PROTON",-1)
    x = raw(small,799)
    padded = {}
    for name,sp in large.items():
        rows = np.flatnonzero(sp.coordinates[:,1]==0)
        assert len(rows)==small[name].dimension
        np.testing.assert_array_equal(sp.coordinates[rows],small[name].coordinates)
        padded[name] = np.zeros((sp.dimension,1),complex)
        padded[name][rows] = x[name]
    rectangular,complete = make(small,larger_out),make(large,larger_out)
    for part in ("canonical","regulator"):
        for mu in (1,2,3):
            a,b = rectangular.apply(x,component=mu,part=part),complete.apply(padded,component=mu,part=part)
            for name in larger_out:
                np.testing.assert_allclose(a[name],b[name],atol=3e-16)
