from itertools import product
import numpy as np
import pytest

from deuteron_wigner.bridge.modes.core import GAMMA,GAMMA_PLUS,GAMMA_MINUS,polarization
from deuteron_wigner.quantum.m2_excluded_fock_vertices import antiquark_spinor
from deuteron_wigner.quantum.m2_antiquark_contact_current import antiquark_one_gluon_current_spin
from deuteron_wigner.quantum.m2_constrained_fermion_current import discrete_gluon_field_factor


@pytest.mark.parametrize("mass",[0.,.35,1.2])
@pytest.mark.parametrize("scale",[.31,1.,2.7])
@pytest.mark.parametrize("hg",[-1,1])
def test_crossed_contact_full_Dirac_insertions_and_reversed_transfer(mass,scale,hg):
    pf=np.array([.7*scale,.13,-.2]);pi=np.array([1.3*scale,-.08,.24])
    vf=np.column_stack([antiquark_spinor(*pf,mass,h) for h in (-1,1)])
    vi=np.column_stack([antiquark_spinor(*pi,mass,h) for h in (-1,1)])
    eps=-np.array([hg,-1j])/np.sqrt(2)
    right,left=-pi[0],-pf[0]
    value=antiquark_one_gluon_current_spin(pf,pi,mass_GeV=mass,
        polarization=eps,right_plus=right,left_plus=left)
    ge=GAMMA[1]*eps[0]+GAMMA[2]*eps[1]
    expected=-np.array([(vi.conj().T@GAMMA[0]@(
        g@GAMMA_PLUS@ge/(2*right)+ge@GAMMA_PLUS@g/(2*left))@vf).T
        for g in (GAMMA_PLUS,GAMMA_MINUS,GAMMA[1],GAMMA[2])])/(2*np.sqrt(pf[0]*pi[0]))
    np.testing.assert_allclose(value,expected,atol=8e-14)
    reverse=antiquark_one_gluon_current_spin(pi,pf,mass_GeV=mass,
        polarization=eps.conj(),right_plus=left,left_plus=right)
    np.testing.assert_allclose(value,reverse.conj().transpose(0,2,1),atol=4e-14)
    np.testing.assert_array_equal(value[0],0.)


@pytest.mark.parametrize("hg",[-1,1])
@pytest.mark.parametrize("mass",[0.,.35])
@pytest.mark.parametrize("P",[.7,2.3])
def test_every_six_body_antiquark_emission_route_Ward_identity(hg,mass,P):
    from deuteron_wigner.quantum.m2_six_body_vertices import split_routes,cubic_mass_kernel
    from deuteron_wigner.quantum.m2_six_body_kinematics import frame
    for route in split_routes(9,"qqqq-qbar","anti_to_antig"):
        x,y=frame(route.sector,route.outgoing).fractions[list(route.daughters)]
        X=x+y;scale=np.sqrt(x*y/X)
        pq=np.array([.11,-.18]);pg=np.array([.21,.08]);delta=np.array([.12,.16])
        q1=(pq-x/X*(pq+pg))/scale
        q2=(pq-delta-x/X*(pq+pg-delta))/scale
        eps=polarization(y*P,*pg,hg).conj()[2:]
        current=antiquark_one_gluon_current_spin((x*P,*pq),(X*P,*(pq+pg-delta)),
            mass_GeV=mass,polarization=eps,right_plus=-X*P,left_plus=-x*P)[2:]
        current*=discrete_gluon_field_factor(route.outgoing[5])
        va=np.array([[cubic_mass_kernel(route,q1,mass,(ho,hg,hi),
            total_pplus=P,parent_transverse=pq+pg) for hi in (-1,1)] for ho in (-1,1)])
        vb=np.array([[cubic_mass_kernel(route,q2,mass,(ho,hg,hi),
            total_pplus=P,parent_transverse=pq+pg-delta) for hi in (-1,1)] for ho in (-1,1)])
        roundoff=16*np.finfo(float).eps*max(1.,np.linalg.norm(va),np.linalg.norm(vb))
        # Physical e_qbar and Tbar are stripped from BOTH sides; no extra minus.
        np.testing.assert_allclose(va-vb,2*P*np.einsum("j,joi->oi",delta,current),
                                   atol=roundoff,rtol=0.)


def test_charge_color_factorization_has_only_one_common_CAR_minus():
    from deuteron_wigner.microscopic.h0.color import representation_generators
    charge=2/3
    for t in representation_generators("3"):
        # Kernel contains minus crossed spin; external factors are physical.
        np.testing.assert_allclose((-charge)*(-t.T)*(-1),-charge*t.T,atol=0.)


def test_antiquark_contact_boost_scales_all_components_correctly():
    pf=np.array([.7,.13,-.2]);pi=np.array([1.3,-.08,.24]);eps=np.array([1.,1j]);scale=2.7
    def kernel(f,i):
        return antiquark_one_gluon_current_spin(f,i,mass_GeV=.35,polarization=eps,
            right_plus=-i[0],left_plus=-f[0])
    base=kernel(pf,pi);pf[0]*=scale;pi[0]*=scale
    np.testing.assert_allclose(kernel(pf,pi),np.array([1,1/scale**2,1/scale,1/scale])[:,None,None]*base,
                               atol=3e-15)
