"""Full tensor enumeration, boosts, statistics and raw second-shell adjoints."""
import numpy as np
import pytest
from deuteron_wigner.quantum.m2_canonical_good_current import parton_charge
from deuteron_wigner.quantum.m2_free_minus_current import projected_free_minus
from deuteron_wigner.quantum.m2_shell_minus_current import shell_minus_current_between
from deuteron_wigner.quantum.m2_shell_transverse_current import shell_transverse_current_between
from deuteron_wigner.quantum.m2_shell_current import shell_density_between
from test_m2_shell_transverse_current import SECTORS,space,FIVE,SIX,frame5,frame6


@pytest.mark.parametrize("sector",SECTORS)
@pytest.mark.parametrize("jz",[-1,1])
def test_raw_minus_against_full_spin_coordinate_enumeration(sector,jz):
    i=space(sector,1,1);f=space(sector,1,jz);rng=np.random.default_rng(753)
    a=(rng.normal(size=(i.dimension,1))+1j*rng.normal(size=(i.dimension,1)))/sqrt_dimension(i)
    delta=np.array([.12,.16]);actual=shell_minus_current_between(f,i,a,delta,b_GeV=.4,
        total_pplus_GeV=1.3,mass_GeV=.35,charge="electromagnetic")
    species,frame=(FIVE[sector],frame5) if sector in FIVE else (SIX[sector],frame6)
    n=len(i.orbitals);expected=np.zeros_like(actual)
    for li,modes in enumerate(i.longitudinal):
        fr=frame(sector,modes)
        source=np.zeros((n,i.ncolor,i.nspin,1),complex);image=np.zeros_like(source)
        rows=np.flatnonzero(i.coordinates[:,0]==li)
        source[tuple(i.coordinates[rows,1:].T)]=a[rows]
        for active,parton in enumerate(species):
            if parton=="g":continue
            x=fr.fractions[active]
            kernel=projected_free_minus(len(species)-1,1,fr.forward[:-1,active,None]*delta,
                fr.inverse[active,:-1],b_GeV=.4,pplus_GeV=x*1.3,
                outgoing_offset=(1-x/2)*delta,incoming_offset=-x*delta/2,
                mass_GeV=.35,species=parton)
            # Independently enumerate all physical spin labels, not grouped
            # packed lookup blocks. The plane/polar kernel has separate tests.
            for si,spin in enumerate(i.spins):
                charge=float(parton_charge(spin[active],parton,"electromagnetic"))
                for hf in (0,1):
                    out=spin.copy();out[active]=2*(out[active]//2)+hf
                    sf=np.ravel_multi_index(out,i.spin_dims)
                    image[:,:,sf]+=charge*np.einsum("oi,ick->ock",kernel[hf,spin[active]%2],source[:,:,si])
        rows=np.flatnonzero(f.coordinates[:,0]==li)
        expected[rows]=image[tuple(f.coordinates[rows,1:].T)]
    np.testing.assert_allclose(actual,expected,atol=3e-16)


def sqrt_dimension(s):return np.sqrt(s.dimension)


@pytest.mark.parametrize("sector",SECTORS)
def test_second_shell_reversed_transfer_and_LF_boosts(sector):
    i=space(sector,2,1);f=space(sector,2,-1);rng=np.random.default_rng(754)
    a=(rng.normal(size=(i.dimension,1))+1j*rng.normal(size=(i.dimension,1)))/sqrt_dimension(i)
    v=(rng.normal(size=(f.dimension,1))+1j*rng.normal(size=(f.dimension,1)))/sqrt_dimension(f)
    delta=np.array([.12,.16]);boost=np.array([.04,-.07]);pp=1.3
    def j(out,inc,a,d,mean=(0.,0.),p=pp):
        return shell_minus_current_between(out,inc,a,d,b_GeV=.4,total_pplus_GeV=p,
                                          mass_GeV=.35,charge="electromagnetic",mean_total_transverse=mean)
    actual=j(f,i,a,delta);reverse=j(i,f,v,-delta)
    np.testing.assert_allclose(np.vdot(v,actual),np.vdot(reverse,a),atol=3e-16)
    np.testing.assert_allclose(j(f,i,a,delta,p=2*pp),actual/4,atol=0.)
    jt=[shell_transverse_current_between(f,i,a,delta,component=k,b_GeV=.4,
            total_pplus_GeV=pp,charge="electromagnetic") for k in (0,1)]
    rho=shell_density_between(f,i,a,delta,b_GeV=.4,charge="electromagnetic")
    expected=sum(boost[k]/pp*jt[k] for k in (0,1))+np.dot(boost,boost)/(2*pp**2)*rho
    np.testing.assert_allclose(j(f,i,a,delta,mean=boost)-actual,expected,atol=5e-16)


@pytest.mark.parametrize("sector",SECTORS)
def test_identical_statistics_and_flavor_remain_resolved(sector):
    s=space(sector,1,1);rng=np.random.default_rng(755)
    a=s.statistics_project((rng.normal(size=(s.dimension,1))+1j*rng.normal(size=(s.dimension,1)))/sqrt_dimension(s))
    def j(charge):
        return shell_minus_current_between(s,s,a,(.12,.16),b_GeV=.4,total_pplus_GeV=1.3,
                                          mass_GeV=.35,charge=charge)
    value=j("electromagnetic")
    np.testing.assert_allclose(s.statistics_project(value),value,atol=7e-16)
    np.testing.assert_allclose((2*j("up_number")-j("down_number"))/3,value,atol=5e-16)


def test_invalid_current_input():
    s=space("qqqgg",0,1)
    with pytest.raises(ValueError):
        shell_minus_current_between(s,s,np.zeros((s.dimension,1)),(0,0),b_GeV=.4,
                                    total_pplus_GeV=1.3,mass_GeV=-1.,charge="baryon")
