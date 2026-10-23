import numpy as np
import pytest
from scipy.sparse.linalg import LinearOperator,gmres

from deuteron_wigner.quantum.m2_coupled_pilot import build_coupled_pilot
from deuteron_wigner.quantum.m2_shell_fock_reference import build_shell_fock_reference
from deuteron_wigner.quantum.m2_order_resolved_reference import build_order_resolved_reference
from deuteron_wigner.quantum.m2_canonical_six_body import CanonicalSixBodyExcursion
from deuteron_wigner.quantum.m2_shell_current import shell_density_between


def pilot():
    return build_coupled_pilot(total_k2=9,cutoff=0,target="PROTON",jz_twice=1,b_GeV=.4,
        quark_mass_GeV=.35,gluon_mass_GeV=0.,coupling=.3)


def test_raw_source_reference_recovers_legacy_complete_excursion():
    p=pilot()
    old=CanonicalSixBodyExcursion(build_order_resolved_reference(p,out_cutoff=0),out_cutoff=0)
    raw=CanonicalSixBodyExcursion(build_shell_fock_reference(p,out_cutoff=0),out_cutoff=0)
    for e in (1.4,1.1):
        np.testing.assert_allclose(raw.hamiltonian_coefficient(e),old.hamiltonian_coefficient(e),atol=3e-18)
        np.testing.assert_allclose(raw.norm_coefficient(e),old.norm_coefficient(e),atol=3e-18)
    np.testing.assert_allclose(raw.current_coefficient(1.4,1.1,(.12,.16),charge="electromagnetic"),
                               old.current_coefficient(1.4,1.1,(.12,.16),charge="electromagnetic"),atol=3e-18)


@pytest.fixture(scope="module")
def obj():
    return CanonicalSixBodyExcursion(build_shell_fock_reference(pilot(),out_cutoff=2),out_cutoff=2)


def test_second_shell_coherence_norm_derivative_and_two_energy_current(obj):
    h=obj.hamiltonian_coefficient(1.4);norm=obj.norm_coefficient(1.4)
    np.testing.assert_allclose(sum(obj.return_components(1.4).values()),h,atol=2e-17)
    np.testing.assert_allclose(h,h.conj().T,atol=1e-17)
    derivative=(obj.hamiltonian_coefficient(1.4+1e-5)-obj.hamiltonian_coefficient(1.4-1e-5))/2e-5
    np.testing.assert_allclose(norm,-derivative,atol=2e-12)
    np.testing.assert_allclose(obj.current_coefficient(1.4,1.4,(0,0),charge="baryon"),norm,atol=3e-18)
    forward=obj.current_coefficient(1.4,1.1,(.12,.16),charge="electromagnetic")
    reverse=obj.current_coefficient(1.1,1.4,(-.12,-.16),charge="electromagnetic")
    np.testing.assert_allclose(forward,reverse.conj().T,atol=1e-17)
    for name in ("energies5","energies6"):
        with pytest.raises(ValueError,match="non-diagonal"):
            getattr(obj,name)


def test_second_shell_independent_interacting_resolvent_order_remainder(obj):
    spaces={**obj.spaces5,**obj.spaces6};offset=0;slices={}
    propagation={**obj.propagation5,**obj.propagation6}
    for n,s in spaces.items():slices[n]=slice(offset,offset+s.dimension);offset+=s.dimension
    rng=np.random.default_rng(817)
    v=rng.normal(size=len(obj.reference.pilot.matrix))+1j*rng.normal(size=len(obj.reference.pilot.matrix))
    v/=np.linalg.norm(v)
    def pack(columns):return np.concatenate([columns[n].ravel() for n in spaces])
    def unpack(value,names):return {n:value[slices[n],None] for n in names}
    def density(value,delta):
        return pack({n:shell_density_between(s,s,value[slices[n],None],delta,b_GeV=.4,charge="electromagnetic")
                     for n,s in spaces.items()})
    def exact(g,e):
        rhs=pack({**{n:g*a@v for n,a in obj.V.items()},**{n:g*g*a@v for n,a in obj.W.items()}})
        def action(value):
            diagonal={n:e*value[slices[n],None]-p.apply(value[slices[n],None])
                      for n,p in propagation.items()}
            crossed={**obj.U.adjoint(unpack(value,obj.spaces6)),
                     **obj.U.apply(unpack(value,obj.spaces5))}
            return pack({n:diagonal[n]-g*crossed[n] for n in spaces})
        op=LinearOperator((offset,offset),matvec=action,dtype=complex)
        pre=LinearOperator((offset,offset),dtype=complex,
            matvec=lambda x:pack({n:p.resolve(e,x[slices[n],None]) for n,p in propagation.items()}))
        wave,info=gmres(op,rhs,M=pre,rtol=2e-13,atol=1e-16)
        assert info==0
        assert np.linalg.norm(action(wave)-rhs)<2e-13
        return rhs,wave
    ei,ef=1.4,1.1;delta=(.12,.16)
    ai=obj.waves(ei)["five_leading"];af=obj.waves(ef)["five_leading"]
    li=pack({**{n:a@v for n,a in ai.items()},**{n:np.zeros(s.dimension) for n,s in obj.spaces6.items()}})
    lf=pack({**{n:a@v for n,a in af.items()},**{n:np.zeros(s.dimension) for n,s in obj.spaces6.items()}})
    wanted=np.array([np.vdot(v,obj.hamiltonian_coefficient(ei)@v),
                     np.vdot(v,obj.norm_coefficient(ei)@v),
                     np.vdot(v,obj.current_coefficient(ef,ei,delta,charge="electromagnetic")@v)])
    errors=[]
    for g in (.08,.04):
        rhs,wi=exact(g,ei);_,wf=exact(g,ef)
        leading_h=sum(np.vdot(obj.V[n]@v,ai[n]@v) for n in obj.V)
        actual=np.array([(np.vdot(rhs,wi)-g*g*leading_h)/g**4,
             (np.vdot(wi,wi)-g*g*np.vdot(li,li))/g**4,
             (np.vdot(wf,density(wi,delta))-g*g*np.vdot(lf,density(li,delta)))/g**4])
        errors.append(abs(actual-wanted))
    assert np.all((errors[0]/errors[1]>3.85)&(errors[0]/errors[1]<4.15))
    assert max(errors[1])<.02*max(abs(wanted))
