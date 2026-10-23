from itertools import permutations,product
from math import factorial,sqrt
import numpy as np
import pytest

from deuteron_wigner.quantum.m2_excluded_fock_action import IntermediateSpace
from deuteron_wigner.quantum.m2_six_body_cubic import SixBodyCubicAction,five_statistics_project


def test_two_to_three_boson_factors_against_independent_CCR():
    d=3;rng=np.random.default_rng(1453)
    a=rng.normal(size=(d,d))+1j*rng.normal(size=(d,d))
    tensor=(a+a.T)/2;tensor/=np.linalg.norm(tensor)
    raw=rng.normal(size=(d,d,d))+1j*rng.normal(size=(d,d,d))
    vertex=(raw+raw.swapaxes(0,1))/2
    emitted=rng.normal(size=d)+1j*rng.normal(size=d)

    def occupations(n):
        return [x for x in product(range(n+1),repeat=d) if sum(x)==n]
    def from_tensor(t):
        n=t.ndim
        return {occ:sqrt(factorial(n)/np.prod([factorial(k) for k in occ]))
                *t[tuple(i for i,k in enumerate(occ) for _ in range(k))]
                for occ in occupations(n)}
    def ladder(state,mode,creation):
        result={}
        for occ,amplitude in state.items():
            number=occ[mode]
            if not creation and not number:continue
            out=list(occ);out[mode]+=1 if creation else -1;out=tuple(out)
            result[out]=result.get(out,0)+amplitude*sqrt(number+1 if creation else number)
        return result
    initial=from_tensor(tensor)
    expected={x:0j for x in occupations(3)}
    for i,j,k in product(range(d),repeat=3):
        acted=ladder(ladder(ladder(initial,k,False),j,True),i,True)
        for occ,value in acted.items():expected[occ]+=.5*vertex[i,j,k]*value
    distinguished=np.einsum("ijk,ks->ijs",vertex,tensor)
    projected=sum(distinguished.transpose(p) for p in permutations(range(3)))/6
    actual=from_tensor(sqrt(3)*projected)
    np.testing.assert_allclose(list(actual.values()),list(expected.values()),atol=8e-15)

    expected={x:0j for x in occupations(3)}
    for k in range(d):
        for occ,value in ladder(initial,k,True).items():expected[occ]+=emitted[k]*value
    distinguished=tensor[:,:,None]*emitted
    projected=sum(distinguished.transpose(p) for p in permutations(range(3)))/6
    actual=from_tensor(sqrt(3)*projected)
    np.testing.assert_allclose(list(actual.values()),list(expected.values()),atol=8e-15)
    # One annihilator on two normalized identical bosons supplies sqrt(2).
    for k in range(d):
        actual=from_tensor(sqrt(2)*tensor[:,k])
        expected=ladder(initial,k,False)
        np.testing.assert_allclose(list(actual.values()),[expected.get(x,0) for x in actual],atol=2e-15)


@pytest.mark.parametrize("cutoff",[0,1])
def test_actual_cubic_action_adjoint_statistics_and_nonzero_sectors(cutoff):
    spaces={n:IntermediateSpace(n,9,cutoff,"PROTON",1) for n in ("qqqgg","qqqq-qbar")}
    action=SixBodyCubicAction(spaces,out_cutoff=cutoff,b_GeV=.4,mass_GeV=.35)
    rng=np.random.default_rng(2394)
    def random(spaces):
        return {n:(rng.normal(size=(s.dimension,1))+1j*rng.normal(size=(s.dimension,1)))/sqrt(s.dimension)
                for n,s in spaces.items()}
    x,z=random(spaces),random(action.outgoing)
    y,back=action.apply(x),action.adjoint(z)
    np.testing.assert_allclose(sum(np.vdot(z[n],y[n]) for n in y),
                               sum(np.vdot(back[n],x[n]) for n in x),atol=3e-17)
    for name,space in action.outgoing.items():
        assert np.linalg.norm(y[name])>1e-3
        np.testing.assert_allclose(space.statistics_project(y[name]),y[name],atol=5e-16)
    for name,space in spaces.items():
        np.testing.assert_allclose(five_statistics_project(space,back[name]),back[name],atol=3e-16)
    # The contact-free three-gluon vertex is linear in transverse momentum:
    # it vanishes in the all-ground HO block, but not the complete first shell.
    w=action.components["qqqgg:g_to_gg"][2]
    assert (w.nnz==0) if cutoff==0 else (np.linalg.norm(w.data)>1e-3)
    assert not action.direct_quartic_Q6P_supplied


def test_actual_six_cubic_sources_interfere_coherently():
    spaces={n:IntermediateSpace(n,9,0,"PROTON",1) for n in ("qqqgg","qqqq-qbar")}
    action=SixBodyCubicAction(spaces,out_cutoff=0,b_GeV=.4,mass_GeV=.35)
    rng=np.random.default_rng(732)
    x={n:five_statistics_project(s,rng.normal(size=(s.dimension,1)).astype(complex)) for n,s in spaces.items()}
    z={n:np.zeros_like(v) for n,v in x.items()}
    separated=[]
    for name in x:
        columns=dict(z);columns[name]=x[name]
        separated.append(action.apply(columns))
    coherent=action.apply(x)
    for name in coherent:
        np.testing.assert_allclose(coherent[name],sum(c[name] for c in separated),atol=8e-15)
    # Both five-body channels reach sea+g and must be summed as amplitudes.
    assert all(np.linalg.norm(c["qqqq-qbar-g"])>1e-3 for c in separated)


def test_invalid_cubic_coordinate_binding():
    s=IntermediateSpace("qqqgg",9,0,"PROTON",1)
    with pytest.raises(ValueError):
        SixBodyCubicAction({"qqqgg":s},out_cutoff=0,b_GeV=.4,mass_GeV=.35)
