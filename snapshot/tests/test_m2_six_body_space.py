import numpy as np
import pytest
from deuteron_wigner.quantum.m2_six_body_space import SixBodySpace,density_action

@pytest.mark.parametrize("sector,target,cutoff",[
    ("qqqggg","PROTON",0),("qqqggg","NEUTRON",1),
    ("qqqq-qbar-g","PROTON",0),("qqqq-qbar-g","NEUTRON",1)])
def test_statistics_free_energy_and_all_charges(sector,target,cutoff):
    s=SixBodySpace(sector,9,cutoff,target,1)
    rng=np.random.default_rng(893)
    v=(rng.normal(size=(s.dimension,1))+1j*rng.normal(size=(s.dimension,1)))/np.sqrt(s.dimension)
    a=s.statistics_project(v)
    assert np.linalg.norm(a)>1e-3
    np.testing.assert_allclose(s.statistics_project(a),a,atol=8e-16)
    np.testing.assert_allclose(s.permute(a,(1,0,2,3,4,5)),-a,atol=5e-16)
    if sector=="qqqggg":
        np.testing.assert_allclose(s.permute(a,(0,1,2,4,3,5)),a,atol=5e-16)
    else:
        np.testing.assert_allclose(s.permute(a,(3,1,2,0,4,5)),-a,atol=5e-16)
    en=s.free_energies(.4,.35,0.)
    np.testing.assert_allclose(s.statistics_project(en[:,None]*a),en[:,None]*a,atol=2e-15)
    for charge,q in (("baryon",1),("electromagnetic",int(target=="PROTON")),
                     ("up_number",2 if target=="PROTON" else 1),("down_number",1 if target=="PROTON" else 2)):
        np.testing.assert_allclose(density_action(s,a,(0,0),b_GeV=.4,charge=charge),q*a,atol=1e-15)
    rho=density_action(s,a,(.12,.16),b_GeV=.4,charge="electromagnetic")
    np.testing.assert_allclose(s.statistics_project(rho),rho,atol=8e-16)
    z=density_action(s,v,(-.12,-.16),b_GeV=.4,charge="electromagnetic")
    np.testing.assert_allclose(v.conj().T@rho,z.conj().T@a,atol=8e-16)

def test_invalid_six_body_scope():
    with pytest.raises(ValueError):SixBodySpace("qqqggg",9,-1,"PROTON",1)
    with pytest.raises(ValueError):SixBodySpace("qqqggg",9,True,"PROTON",1)
    with pytest.raises(ValueError):SixBodySpace("qqqggg",9,0,"PROTON",True)

@pytest.mark.parametrize("sector",["qqqggg","qqqq-qbar-g"])
def test_opposite_helicity_density_reversal_and_nonzero_first_shell(sector):
    from deuteron_wigner.quantum.m2_six_body_space import density_action_between
    up=SixBodySpace(sector,9,1,"PROTON",1)
    down=SixBodySpace(sector,9,1,"PROTON",-1)
    rng=np.random.default_rng(493)
    def state(s):
        return s.statistics_project((rng.normal(size=(s.dimension,1))+1j*rng.normal(size=(s.dimension,1)))/np.sqrt(s.dimension))
    u,v=state(up),state(down)
    x=density_action_between(down,up,u,(.12,.16),b_GeV=.4,charge="electromagnetic")
    y=density_action_between(up,down,v,(-.12,-.16),b_GeV=.4,charge="electromagnetic")
    assert np.linalg.norm(x)>1e-3
    np.testing.assert_allclose(v.conj().T@x,y.conj().T@u,atol=8e-17)
    np.testing.assert_allclose(down.statistics_project(x),x,atol=5e-16)
    np.testing.assert_allclose(density_action_between(down,up,u,(0,0),b_GeV=.4,charge="electromagnetic"),0.,atol=0.)


def test_current_rejects_different_six_body_regulators():
    from deuteron_wigner.quantum.m2_six_body_space import density_action_between
    up=SixBodySpace("qqqggg",9,0,"PROTON",1)
    down=SixBodySpace("qqqggg",11,0,"PROTON",-1)
    with pytest.raises(ValueError):
        density_action_between(down,up,np.zeros((up.dimension,1)),(0,0),b_GeV=.4,charge="electromagnetic")

@pytest.mark.parametrize("sector,p",[
    ("qqqggg",(0,1,2,5,3,4)),
    ("qqqq-qbar-g",(3,0,2,1,4,5)),
])
def test_exact_block_permutation_against_unpacked_reference(sector,p):
    from deuteron_wigner.quantum.m2_six_body_kinematics import color_permutation,permutation_frame
    from deuteron_wigner.quantum.m2_coupled_longitudinal_jacobi import oscillator_rotation
    s=SixBodySpace(sector,11,1,"PROTON",1)
    rng=np.random.default_rng(581)
    v=rng.normal(size=(s.dimension,2))+1j*rng.normal(size=(s.dimension,2))
    result=np.zeros_like(v);color=color_permutation(sector,p)
    spin_lookup=np.arange(s.nspin).reshape(s.spin_dims).transpose(p).ravel()
    long_lookup={state:i for i,state in enumerate(s.longitudinal)}
    # Original uncompressed transformation is an independent numerical
    # reference for the exact quantum-number block implementation.
    for li,state in enumerate(s.longitudinal):
        target,r=permutation_frame(sector,state,p)
        rows=np.flatnonzero(s.coordinates[:,0]==li)
        block=np.zeros((len(s.orbitals),s.ncolor,s.nspin,2),complex)
        orb,col,spin=s.coordinates[rows,1:].T
        block[orb,col,spin]=v[rows]
        block=np.einsum("oi,ab,ibsk->oask",oscillator_rotation(r,s.cutoff),color,block,optimize=True)
        s.add(result,long_lookup[target],np.arange(s.nspin),block[:,:,spin_lookup])
    np.testing.assert_allclose(s.permute(v,p),result,atol=8e-15)
