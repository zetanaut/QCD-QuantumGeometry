import numpy as np
import pytest
from scipy.sparse.linalg import LinearOperator,gmres

from deuteron_wigner.quantum.m2_coupled_pilot import build_coupled_pilot
from deuteron_wigner.quantum.m2_order_resolved_reference import build_order_resolved_reference
from deuteron_wigner.quantum.m2_canonical_six_body import CanonicalSixBodyExcursion
from deuteron_wigner.quantum.m2_excluded_fock_matching import intermediate_density_action
from deuteron_wigner.quantum.m2_six_body_space import density_action


@pytest.fixture(scope="module")
def obj():
    p=build_coupled_pilot(total_k2=9,cutoff=0,target="PROTON",jz_twice=1,b_GeV=.4,
        quark_mass_GeV=.35,gluon_mass_GeV=0.,coupling=.3)
    return CanonicalSixBodyExcursion(build_order_resolved_reference(p,out_cutoff=0),out_cutoff=0)


def test_actual_direct_iterated_interference_and_norm_derivative(obj):
    parts=obj.return_components(1.4);h=obj.hamiltonian_coefficient(1.4)
    np.testing.assert_allclose(sum(parts.values()),h,atol=5e-18)
    assert np.linalg.norm(parts["cross"])>1e-6
    np.testing.assert_allclose(h,h.conj().T,atol=5e-18)
    eps=1e-5
    derivative=(obj.hamiltonian_coefficient(1.4+eps)-obj.hamiltonian_coefficient(1.4-eps))/(2*eps)
    np.testing.assert_allclose(obj.norm_coefficient(1.4),-derivative,atol=3e-13)
    a=obj.waves(1.4)
    six=sum(x.conj().T@x for x in a["six_leading"].values())
    assert np.linalg.norm(obj.norm_coefficient(1.4)-six)>1e-5


@pytest.mark.parametrize("charge,q",[("baryon",1),("electromagnetic",1),("up_number",2),("down_number",1)])
def test_actual_zero_transfer_charge_and_norm(obj,charge,q):
    np.testing.assert_allclose(obj.current_coefficient(1.4,1.4,(0,0),charge=charge),
                               q*obj.norm_coefficient(1.4),atol=2e-18)


def test_actual_two_energy_current_reversal(obj):
    x=obj.current_coefficient(1.4,1.1,(.12,.16),charge="electromagnetic")
    y=obj.current_coefficient(1.1,1.4,(-.12,-.16),charge="electromagnetic")
    np.testing.assert_allclose(x,y.conj().T,atol=5e-18)


def test_against_independent_raw_matrix_free_Q5_Q6_resolvent(obj):
    spaces={**obj.spaces5,**obj.spaces6};offset=0;slices={}
    for n,s in spaces.items():slices[n]=slice(offset,offset+s.dimension);offset+=s.dimension
    energies=np.concatenate([obj.energies5.get(n,obj.energies6.get(n)) for n in spaces])
    rng=np.random.default_rng(889)
    v=rng.normal(size=len(obj.reference.pilot.matrix))+1j*rng.normal(size=len(obj.reference.pilot.matrix))
    v/=np.linalg.norm(v)
    def pack(columns):return np.concatenate([columns[n].ravel() for n in spaces])
    def unpack(value,names):return {n:value[slices[n],None] for n in names}
    def density(value,delta):
        return pack({**{n:intermediate_density_action(s,value[slices[n],None],delta,b_GeV=.4,charge="electromagnetic")
                         for n,s in obj.spaces5.items()},
                     **{n:density_action(s,value[slices[n],None],delta,b_GeV=.4,charge="electromagnetic")
                         for n,s in obj.spaces6.items()}})
    def exact(g,e):
        rhs=pack({**{n:g*a@v for n,a in obj.V.items()},**{n:g*g*a@v for n,a in obj.W.items()}})
        def action(value):
            u6=obj.U.apply(unpack(value,obj.spaces5))
            u5=obj.U.adjoint(unpack(value,obj.spaces6))
            return (e-energies)*value-g*pack({**u5,**u6})
        op=LinearOperator((offset,offset),matvec=action,dtype=complex)
        pre=LinearOperator((offset,offset),matvec=lambda x:x/(e-energies),dtype=complex)
        wave,info=gmres(op,rhs,M=pre,rtol=2e-13,atol=1e-16)
        assert info==0
        assert np.linalg.norm(action(wave)-rhs)<1e-13
        return rhs,wave
    ei,ef=1.4,1.1;delta=(.12,.16)
    aif=obj.waves(ei)["five_leading"];aff=obj.waves(ef)["five_leading"]
    leading_i=pack({**{n:a@v for n,a in aif.items()},
                    **{n:np.zeros(s.dimension) for n,s in obj.spaces6.items()}})
    leading_f=pack({**{n:a@v for n,a in aff.items()},
                    **{n:np.zeros(s.dimension) for n,s in obj.spaces6.items()}})
    wanted=[np.vdot(v,obj.hamiltonian_coefficient(ei)@v),
            np.vdot(v,obj.norm_coefficient(ei)@v),
            np.vdot(v,obj.current_coefficient(ef,ei,delta,charge="electromagnetic")@v)]
    errors=[]
    for g in (.08,.04):
        rhs,wi=exact(g,ei);_,wf=exact(g,ef)
        leading_H=sum(np.vdot(obj.V[n]@v,aif[n]@v) for n in obj.V)
        actual=[(np.vdot(rhs,wi)-g*g*leading_H)/g**4,
                (np.vdot(wi,wi)-g*g*np.vdot(leading_i,leading_i))/g**4,
                (np.vdot(wf,density(wi,delta))-g*g*np.vdot(leading_f,density(leading_i,delta)))/g**4]
        errors.append(np.abs(np.array(actual)-wanted))
    assert np.all((errors[0]/errors[1]>3.9)&(errors[0]/errors[1]<4.1))
    assert max(errors[1])<1e-6


def test_first_complete_shell_and_neutron_charge():
    p=build_coupled_pilot(total_k2=9,cutoff=0,target="NEUTRON",jz_twice=1,b_GeV=.4,
        quark_mass_GeV=.35,gluon_mass_GeV=0.,coupling=.3)
    obj=CanonicalSixBodyExcursion(build_order_resolved_reference(p,out_cutoff=1),out_cutoff=1)
    np.testing.assert_allclose(obj.current_coefficient(1.4,1.4,(0,0),charge="electromagnetic"),0,atol=2e-18)
    assert np.linalg.norm(obj.hamiltonian_coefficient(1.4))>1e-4
    assert not obj.complete_gs4_QCD and not obj.physical
