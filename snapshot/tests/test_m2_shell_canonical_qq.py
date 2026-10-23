from functools import lru_cache
import numpy as np
import pytest
from scipy.sparse.linalg import LinearOperator,gmres

from deuteron_wigner.quantum.m2_coupled_pilot import build_coupled_pilot
from deuteron_wigner.quantum.m2_order_resolved_reference import build_order_resolved_reference
from deuteron_wigner.quantum.m2_shell_fock_reference import build_shell_fock_reference
from deuteron_wigner.quantum.m2_canonical_qq import CanonicalQQSourceAction,CanonicalQQInsertion
from deuteron_wigner.quantum.m2_shell_canonical_qq import FactorizedCanonicalQQSourceAction,ShellCanonicalQQInsertion
from deuteron_wigner.quantum.m2_shell_current import shell_density_between


def build(cutoff,*,raw=True):
    p=build_coupled_pilot(total_k2=9,cutoff=0,target="PROTON",jz_twice=1,b_GeV=.4,
        quark_mass_GeV=.35,gluon_mass_GeV=0.,coupling=.3)
    r=(build_shell_fock_reference if raw else build_order_resolved_reference)(p,out_cutoff=cutoff)
    action=(FactorizedCanonicalQQSourceAction if raw else CanonicalQQSourceAction)(r)
    return (ShellCanonicalQQInsertion if raw else CanonicalQQInsertion)(r,action=action)


@pytest.fixture(scope="module")
def obj():
    value=build(2)
    # Reuse identical raw solutions across H/norm/current assertions. No source
    # column or tested operator is removed; this cache belongs only to this test.
    value.order_waves=lru_cache(maxsize=8)(value.order_waves)
    return value


@pytest.mark.parametrize("power",[4,5,6])
def test_raw_zero_shell_reproduces_assembled_control_all_orders(power):
    old,new=build(0,raw=False),build(0)
    for norm in (False,True):
        np.testing.assert_allclose(new.return_power_coefficient(power,1.4,norm=norm),
                                   old.return_power_coefficient(power,1.4,norm=norm),atol=3e-17)
    np.testing.assert_allclose(new.current_power_coefficient(power,1.4,1.1,(.12,.16),charge="electromagnetic"),
        old.current_power_coefficient(power,1.4,1.1,(.12,.16),charge="electromagnetic"),atol=3e-17)


@pytest.mark.parametrize("power",[4,5,6])
def test_second_shell_each_order_energy_derivative_charge_and_reversal(obj,power):
    h=obj.return_power_coefficient(power,1.4)
    n=obj.return_power_coefficient(power,1.4,norm=True)
    np.testing.assert_allclose(h,h.conj().T,atol=3e-17)
    derivative=(obj.return_power_coefficient(power,1.40001)-obj.return_power_coefficient(power,1.39999))/2e-5
    np.testing.assert_allclose(n,-derivative,atol=3e-12)
    for charge,total in (("baryon",1),("up_number",2),("down_number",1)):
        np.testing.assert_allclose(obj.current_power_coefficient(power,1.4,1.4,(0,0),charge=charge),
                                   total*n,atol=3e-17)
    x=obj.current_power_coefficient(power,1.4,1.1,(.12,.16),charge="electromagnetic")
    y=obj.current_power_coefficient(power,1.1,1.4,(-.12,-.16),charge="electromagnetic")
    np.testing.assert_allclose(x,y.conj().T,atol=3e-17)
    assert np.linalg.norm(h)>1e-8


def test_component_sum_and_raw_tail_not_projected_back_to_source(obj):
    a,b=obj.order_waves(1.4,1)
    np.testing.assert_allclose(sum(obj.action.component_returns(a).values()),
                               obj.return_power_coefficient(4,1.4),atol=3e-17)
    # A generic raw W tail need not remain in the original cubic source image.
    for name in a:
        u,s,_=np.linalg.svd(obj.sources[name],full_matrices=False)
        u=u[:,s>1e-12]
        assert np.linalg.norm(b[name]-u@(u.conj().T@b[name]))>1e-5
    assert not obj.physical
    with pytest.raises(ValueError):obj.leading_waves(1.4,True)
    with pytest.raises(ValueError):obj.return_power_coefficient(3,1.4)


def test_second_shell_single_insertion_against_independent_interacting_Q_solve(obj):
    names=list(obj.action.spaces);ss=obj.action.spaces
    cuts=np.cumsum([0]+[ss[n].dimension for n in names]);dimension=cuts[-1]
    slices={n:slice(cuts[k],cuts[k+1]) for k,n in enumerate(names)}
    def unpack(x):return {n:x[slices[n],None] for n in names}
    def pack(x):return np.concatenate([x[n].ravel() for n in names])
    rng=np.random.default_rng(512)
    v=rng.normal(size=len(obj.reference.pilot.matrix))+1j*rng.normal(size=len(obj.reference.pilot.matrix))
    v/=np.linalg.norm(v);g=.3
    source=pack({n:sum(g**p*obj.sources_by_power[p][n]@v for p in (1,2)) for n in names})
    def resolve(e,lam):
        def action(x):
            columns=unpack(x);w=obj.action.apply(columns)
            return pack({n:e*columns[n]-obj.propagation[n].apply(columns[n])-lam*w[n] for n in names})
        op=LinearOperator((dimension,dimension),matvec=action,dtype=complex)
        pre=LinearOperator((dimension,dimension),dtype=complex,
            matvec=lambda x:pack({n:obj.propagation[n].resolve(e,a) for n,a in unpack(x).items()}))
        wave,info=gmres(op,source,M=pre,rtol=2e-13,atol=1e-16)
        assert info==0
        assert np.linalg.norm(action(wave)-source)<2e-13
        return wave
    def density(x):
        return pack({n:shell_density_between(ss[n],ss[n],a,(.12,.16),b_GeV=.4,charge="electromagnetic")
                     for n,a in unpack(x).items()})
    h=sum(g**(p-2)*obj.return_power_coefficient(p,1.4) for p in (4,5,6))
    norm=sum(g**(p-2)*obj.return_power_coefficient(p,1.4,norm=True) for p in (4,5,6))
    current=sum(g**(p-2)*obj.current_power_coefficient(p,1.1,1.4,(.12,.16),charge="electromagnetic")
                for p in (4,5,6))
    expected=np.array([np.vdot(v,m@v) for m in (h,norm,current)])
    errors=[]
    for step in (.04,.02):
        values=[]
        for lam in (-step,step):
            wi,wf=resolve(1.4,lam),resolve(1.1,lam)
            values.append(np.array([np.vdot(source,wi),np.vdot(wi,wi),np.vdot(wf,density(wi))]))
        errors.append(abs((values[1]-values[0])/(2*step)-expected))
    assert np.all((errors[0]/errors[1]>3.8)&(errors[0]/errors[1]<4.2))
    assert max(errors[1])<1e-3*max(abs(expected))
