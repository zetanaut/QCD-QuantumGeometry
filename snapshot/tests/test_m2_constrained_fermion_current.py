from itertools import product
import numpy as np
import pytest

from deuteron_wigner.bridge.modes.core import GAMMA,GAMMA_PLUS,GAMMA_MINUS,LAMBDA_PLUS
from deuteron_wigner.quantum.m2_canonical_emission import lf_spinor
from deuteron_wigner.quantum.m2_excluded_fock_vertices import antiquark_spinor
from deuteron_wigner.quantum.m2_constrained_fermion_current import (
    GOOD_AXES,ANTI_AXES,bad_field_kernel,free_fermion_current,
    one_gluon_current_spin,two_gluon_current_spin,discrete_gluon_field_factor,
    pair_creation_current_spin,
)

CURRENT_GAMMA=(GAMMA_PLUS,GAMMA_MINUS,GAMMA[1],GAMMA[2])


def spinors(p,m,species):
    return np.column_stack([(lf_spinor if species=="q" else antiquark_spinor)(*p,m,h) for h in (-1,1)])


@pytest.mark.parametrize("species",["q","qbar"])
@pytest.mark.parametrize("mass",[0.,.35,1.2])
@pytest.mark.parametrize("frame",[(1.,0.),(1.7,.23),(.31,-.17)])
def test_free_constraint_full_Dirac_bilinear_reversal_and_Ward_identity(species,mass,frame):
    scale,skew=frame
    pf=np.array([scale*(1.1+skew),.13,-.2]);pi=np.array([scale*1.1,-.08,.24])
    vf,vi=spinors(pf,mass,species),spinors(pi,mass,species)
    axes=GOOD_AXES if species=="q" else ANTI_AXES
    sign=1 if species=="q" else -1
    np.testing.assert_allclose(LAMBDA_PLUS@vf/np.sqrt(np.sqrt(2)*pf[0]),axes,atol=5e-16)
    np.testing.assert_allclose((np.eye(4)+bad_field_kernel(sign*pf,mass))@axes,
                               vf/np.sqrt(np.sqrt(2)*pf[0]),atol=2e-15)
    value=free_fermion_current(pf,pi,mass_GeV=mass,species=species)
    if species=="q":
        direct=np.array([vf.conj().T@GAMMA[0]@g@vi for g in CURRENT_GAMMA])
    else:
        direct=np.array([(vi.conj().T@GAMMA[0]@g@vf).T for g in CURRENT_GAMMA])
    direct/=2*np.sqrt(pf[0]*pi[0])
    np.testing.assert_allclose(value,direct,atol=3e-15)
    np.testing.assert_allclose(value[0],np.eye(2),atol=5e-16)
    reverse=free_fermion_current(pi,pf,mass_GeV=mass,species=species)
    np.testing.assert_allclose(value,reverse.conj().transpose(0,2,1),atol=3e-15)
    qplus=pf[0]-pi[0]
    qminus=(mass**2+pf[1:]@pf[1:])/(2*pf[0])-(mass**2+pi[1:]@pi[1:])/(2*pi[0])
    ward=qplus*value[1]+qminus*value[0]-sum((pf[j]-pi[j])*value[j+1] for j in (1,2))
    np.testing.assert_allclose(ward,0.,atol=3e-15)


@pytest.mark.parametrize("right,left",list(product((.7,-.7),(1.3,-1.3))))
@pytest.mark.parametrize("mass",[0.,.35])
@pytest.mark.parametrize("helicity",[-1,1])
def test_one_gluon_current_from_independent_instantaneous_Dirac_insertions(right,left,mass,helicity):
    pf=(.7,.13,-.2);pi=(1.3,-.08,.24)
    vf,vi=spinors(pf,mass,"q"),spinors(pi,mass,"q")
    eps=-np.array([helicity,-1j])/np.sqrt(2)  # created physical polarization
    ge=GAMMA[1]*eps[0]+GAMMA[2]*eps[1]
    value=one_gluon_current_spin(pf,pi,mass_GeV=mass,polarization=eps,
                                 right_plus=right,left_plus=left)
    direct=np.array([vf.conj().T@GAMMA[0]@(
        g@GAMMA_PLUS@ge/(2*right)+ge@GAMMA_PLUS@g/(2*left))@vi
        for g in CURRENT_GAMMA])/(2*np.sqrt(pf[0]*pi[0]))
    np.testing.assert_allclose(value,direct,atol=3e-15)
    np.testing.assert_allclose(value[0],0.,atol=0.)
    reversed_=one_gluon_current_spin(pi,pf,mass_GeV=mass,polarization=eps.conj(),
                                      right_plus=left,left_plus=right)
    np.testing.assert_allclose(value,reversed_.conj().transpose(0,2,1),atol=3e-15)
    assert np.linalg.norm(value[2:])>.1


@pytest.mark.parametrize("left_h,right_h",list(product((-1,1),repeat=2)))
def test_ordered_two_gluon_current_against_double_instantaneous_Dirac_insertions(left_h,right_h):
    pf=(.7,.13,-.2);pi=(1.3,-.08,.24);left,right=-.9,1.1
    vf,vi=spinors(pf,.35,"q"),spinors(pi,.35,"q")
    el=-np.array([left_h,-1j])/np.sqrt(2);er=-np.array([right_h,1j])/np.sqrt(2)
    gl=GAMMA[1]*el[0]+GAMMA[2]*el[1];gr=GAMMA[1]*er[0]+GAMMA[2]*er[1]
    value=two_gluon_current_spin(left_polarization=el,right_polarization=er,
                                 left_plus=left,right_plus=right)
    direct=np.array([vf.conj().T@GAMMA[0]@gl@GAMMA_PLUS@g@GAMMA_PLUS@gr@vi
        for g in CURRENT_GAMMA])/(8*left*right*np.sqrt(pf[0]*pi[0]))
    np.testing.assert_allclose(value,direct,atol=3e-15)
    np.testing.assert_allclose(value[[0,2,3]],0.,atol=0.)


def test_longitudinal_boost_weights_and_no_mass_conversion_in_current():
    pf=np.array([.7,.13,-.2]);pi=np.array([1.3,-.08,.24]);scale=2.7
    vf=pf.copy();vi=pi.copy();vf[0]*=scale;vi[0]*=scale
    weights=np.array([1,1/scale**2,1/scale,1/scale])
    for species in ("q","qbar"):
        a=free_fermion_current(pf,pi,mass_GeV=.35,species=species)
        b=free_fermion_current(vf,vi,mass_GeV=.35,species=species)
        np.testing.assert_allclose(b,weights[:,None,None]*a,atol=3e-15)
    eps=np.array([1,1j])/np.sqrt(2)
    a=one_gluon_current_spin(pf,pi,mass_GeV=.35,polarization=eps,right_plus=.7,left_plus=1.3)
    b=one_gluon_current_spin(vf,vi,mass_GeV=.35,polarization=eps,right_plus=.7*scale,left_plus=1.3*scale)
    np.testing.assert_allclose(b,weights[:,None,None]*a,atol=3e-15)
    for mode in (2,4,12):
        for total in (.7,2.1):
            ell=np.pi*9/total;kplus=mode*total/9
            assert discrete_gluon_field_factor(mode)==pytest.approx(1/np.sqrt(2*ell*kplus),abs=2e-16)


def test_signed_denominators_are_not_removed_or_replaced_by_absolute_values():
    with pytest.raises(ValueError):bad_field_kernel((0.,.1,.2),.35)
    with pytest.raises(ValueError):free_fermion_current((-.7,.1,.2),(1.,0.,0.),mass_GeV=.35,species="q")
    with pytest.raises(ValueError):one_gluon_current_spin((1.,0.,0.),(1.,0.,0.),mass_GeV=.35,
        polarization=(1.,0.),right_plus=0.,left_plus=1.)
    for mode in (0,1,-2,True):
        with pytest.raises(ValueError):discrete_gluon_field_factor(mode)
    a=two_gluon_current_spin(left_polarization=(1.,0.),right_polarization=(1.,0.),left_plus=-.7,right_plus=1.)
    b=two_gluon_current_spin(left_polarization=(1.,0.),right_polarization=(1.,0.),left_plus=.7,right_plus=1.)
    np.testing.assert_allclose(a,-b,atol=0.)


@pytest.mark.parametrize("mass",[0.,.35,1.2])
@pytest.mark.parametrize("scale",[.7,2.1])
@pytest.mark.parametrize("hg",[-1,1])
def test_crossed_pair_contact_against_independent_quark_antiquark_Dirac_insertions(mass,scale,hg):
    pq=(.7*scale,.13,-.2);pb=(1.3*scale,-.08,.24)
    uq,vb=spinors(pq,mass,"q"),spinors(pb,mass,"qbar")
    eps=-np.array([hg,1j])/np.sqrt(2)
    right,left=pq[0],-pb[0]
    value=pair_creation_current_spin(pq,pb,mass_GeV=mass,polarization=eps,
                                     right_plus=right,left_plus=left)
    ge=GAMMA[1]*eps[0]+GAMMA[2]*eps[1]
    expected=np.array([uq.conj().T@GAMMA[0]@(
        g@GAMMA_PLUS@ge/(2*right)+ge@GAMMA_PLUS@g/(2*left))@vb
        for g in CURRENT_GAMMA])/(2*np.sqrt(pq[0]*pb[0]))
    np.testing.assert_allclose(value,expected,atol=5e-15)
    np.testing.assert_allclose(value[0],0.,atol=0.)


@pytest.mark.parametrize("hg",[-1,1])
@pytest.mark.parametrize("mass",[0.,.35])
@pytest.mark.parametrize("pplus",[.7,2.3])
def test_pair_contact_Ward_identity_with_opposite_physical_daughter_charges(hg,mass,pplus):
    from deuteron_wigner.bridge.modes.core import polarization
    from deuteron_wigner.quantum.m2_excluded_fock_kinematics import split_routes,frame
    from deuteron_wigner.quantum.m2_excluded_fock_vertices import cubic_mass_kernel
    for route in split_routes(9,"g_to_pair"):
        x,y=frame(route.sector,route.outgoing).fractions[list(route.daughters)]
        total=x+y;factor=np.sqrt(x*y/total)
        pq=np.array([.11,-.18]);pb=np.array([.21,.08]);delta=np.array([.12,.16])
        pg=pq+pb-delta
        eps=polarization(total*pplus,*pg,hg)[2:]
        current=pair_creation_current_spin((x*pplus,*pq),(y*pplus,*pb),
            mass_GeV=mass,polarization=eps,right_plus=x*pplus,left_plus=-y*pplus)[2:]
        current*=discrete_gluon_field_factor(route.incoming[3])
        # [V,rho]=V(pq,pbar-Delta)-V(pq-Delta,pbar): gluon charge is zero,
        # while daughter charges are +Q and -Q. Divide out Q only once.
        qa=(pq-x/total*pg)/factor;qb=(pq-delta-x/total*pg)/factor
        va=np.array([[cubic_mass_kernel(route,qa,mass,(h,hb,hg),
                total_pplus=pplus,parent_transverse=pg)
            for hb in (-1,1)] for h in (-1,1)])
        vb=np.array([[cubic_mass_kernel(route,qb,mass,(h,hb,hg),
                total_pplus=pplus,parent_transverse=pg)
            for hb in (-1,1)] for h in (-1,1)])
        # Subtracting two nonzero mass-flip amplitudes cancels analytically.
        # Bound roundoff by the operands, not the vanishing difference.
        roundoff=16*np.finfo(float).eps*max(1.,np.linalg.norm(va),np.linalg.norm(vb))
        np.testing.assert_allclose(va-vb,2*pplus*np.einsum("j,joi->oi",delta,current),
                                   atol=roundoff,rtol=0.)
