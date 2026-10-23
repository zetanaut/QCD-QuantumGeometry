"""Q5/Q6 packed-current reference, reversed endpoints and sea/statistics signs."""
import numpy as np
import pytest

from deuteron_wigner.quantum.m2_excluded_fock_action import IntermediateSpace
from deuteron_wigner.quantum.m2_six_body_space import SixBodySpace
from deuteron_wigner.quantum.m2_excluded_fock_kinematics import SPECIES as FIVE,frame as frame5
from deuteron_wigner.quantum.m2_six_body_kinematics import SPECIES as SIX,frame as frame6
from deuteron_wigner.quantum.m2_canonical_good_current import parton_charge
from deuteron_wigner.quantum.m2_shell_current import shell_density_between
from deuteron_wigner.quantum.m2_transverse_current_projection import displacement_and_moments
from deuteron_wigner.quantum.m2_shell_transverse_current import shell_transverse_current_between

SECTORS=("qqqgg","qqqq-qbar","qqqggg","qqqq-qbar-g")


def space(sector,cutoff,jz):
    return (IntermediateSpace if sector in FIVE else SixBodySpace)(sector,9,cutoff,"NEUTRON",jz)


def full_tensor_reference(f,s,a,delta,j):
    """No packed Lz routing: assemble every spin coefficient, then gather."""
    species,frame=(FIVE[s.sector],frame5) if s.sector in FIVE else (SIX[s.sector],frame6)
    n=len(s.orbitals);d=len(species)-1;result=np.zeros((f.dimension,a.shape[1]),complex)
    for li,modes in enumerate(s.longitudinal):
        fr=frame(s.sector,modes)
        kernel=np.zeros((n,n,s.nspin),complex)
        for active,parton in enumerate(species):
            if parton=="g":continue
            x=fr.fractions[active]
            shift=fr.forward[:-1,active,None]*delta[None,:]
            rho,m=displacement_and_moments(d,s.cutoff,shift,b_GeV=.4)
            intrinsic=sum(fr.inverse[active,k]*m[k,j] for k in range(d))
            conv=intrinsic+(1-x)*delta[j]/2*rho
            spin=2*(s.spins[:,active]%2)-1
            q=parton_charge(s.spins[:,active],parton,"electromagnetic")
            kernel+=(conv[:,:,None]+.5j*[-delta[1],delta[0]][j]*rho[:,:,None]*spin)*q/(x*1.3)
        rows=np.flatnonzero(s.coordinates[:,0]==li)
        orb,color,spin=s.coordinates[rows,1:].T
        raw=np.zeros((n,s.ncolor,s.nspin,a.shape[1]),complex);raw[orb,color,spin]=a[rows]
        values=np.einsum("ois,icsk->ocsk",kernel,raw,optimize=True)
        rows=np.flatnonzero(f.coordinates[:,0]==li)
        orb,color,spin=f.coordinates[rows,1:].T
        result[rows]=values[orb,color,spin]
    return result


@pytest.mark.parametrize("sector",SECTORS)
@pytest.mark.parametrize("jz",[-1,1])
def test_exact_packed_action_against_full_spin_orbital_tensor(sector,jz):
    s=space(sector,1,1);f=space(sector,1,jz)
    rng=np.random.default_rng(678)
    a=(rng.normal(size=(s.dimension,1))+1j*rng.normal(size=(s.dimension,1)))/np.sqrt(s.dimension)
    delta=np.array([.12,.16])
    for j in (0,1):
        actual=shell_transverse_current_between(f,s,a,delta,component=j,b_GeV=.4,
            total_pplus_GeV=1.3,charge="electromagnetic")
        np.testing.assert_allclose(actual,full_tensor_reference(f,s,a,delta,j),atol=6e-17)


@pytest.mark.parametrize("sector",SECTORS)
@pytest.mark.parametrize("jz",[-1,1])
def test_second_shell_reversal_longitudinal_and_transverse_boosts(sector,jz):
    s=space(sector,2,1);f=space(sector,2,jz)
    rng=np.random.default_rng(862)
    a=(rng.normal(size=(s.dimension,1))+1j*rng.normal(size=(s.dimension,1)))/np.sqrt(s.dimension)
    b=(rng.normal(size=(f.dimension,1))+1j*rng.normal(size=(f.dimension,1)))/np.sqrt(f.dimension)
    delta=np.array([.12,.16])
    def action(out,inc,v,d,mean=(0,0),p=1.3):
        return shell_transverse_current_between(out,inc,v,d,component=1,b_GeV=.4,
            total_pplus_GeV=p,charge="electromagnetic",mean_total_transverse=mean)
    value=action(f,s,a,delta)
    reverse=action(s,f,b,-delta)
    # The six-body dot product sums hundreds of thousands of amplitudes.
    roundoff=32*np.finfo(float).eps*max(np.linalg.norm(b)*np.linalg.norm(value),
                                      np.linalg.norm(reverse)*np.linalg.norm(a))
    assert np.vdot(b,value)==pytest.approx(np.vdot(reverse,a),abs=roundoff,rel=0.)
    rho=shell_density_between(f,s,a,delta,b_GeV=.4,charge="electromagnetic")
    np.testing.assert_allclose(action(f,s,a,delta,mean=(.03,-.07))-value,-.07/1.3*rho,atol=3e-17)
    np.testing.assert_allclose(action(f,s,a,delta,p=2.6),value/2,atol=2e-17)


@pytest.mark.parametrize("sector",SECTORS)
def test_identical_particle_statistics_preserved(sector):
    s=space(sector,1,1)
    rng=np.random.default_rng(864)
    a=s.statistics_project((rng.normal(size=(s.dimension,1))
                           +1j*rng.normal(size=(s.dimension,1)))/np.sqrt(s.dimension))
    value=shell_transverse_current_between(s,s,a,(.12,.16),component=0,b_GeV=.4,
            total_pplus_GeV=1.3,charge="electromagnetic")
    np.testing.assert_allclose(s.statistics_project(value),value,atol=2e-16)


def test_explicit_component_and_matching_endpoints_required():
    s=space("qqqgg",0,1)
    for component in (None,True,2):
        with pytest.raises(ValueError):
            shell_transverse_current_between(s,s,np.zeros((s.dimension,1)),(0,0),
                component=component,b_GeV=.4,total_pplus_GeV=1.,charge="baryon")
