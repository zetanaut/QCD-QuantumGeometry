from itertools import product
from math import pi,sqrt
import numpy as np
import pytest
from deuteron_wigner.bridge.modes.core import GAMMA,GAMMA_PLUS
from deuteron_wigner.quantum.m2_canonical_emission import lf_spinor
from deuteron_wigner.quantum.m2_excluded_fock_vertices import antiquark_spinor
from deuteron_wigner.quantum.m2_excluded_fock_kinematics import color_source
from deuteron_wigner.quantum.m2_excluded_pair_conversion import (
    conversion_ordered_color,conversion_spin,instantaneous_conversion_kernel,
    conversion_contact,FiveBodyPairConversion,
)
from test_m2_excluded_pair_current import build
from test_m2_excluded_fermion_scattering import direct_contact


def test_IF_conversion_color_commutator_and_relative_IG_phase():
    direct,cross=conversion_ordered_color(),conversion_ordered_color(True)
    current=color_source("g_to_pair")@color_source("g_to_gg").conj().T
    np.testing.assert_allclose(direct-cross,1j*current,atol=3e-15)


@pytest.mark.parametrize("outgoing",[(1,1,1,1,5),(1,1,1,3,3),(1,1,1,5,1)])
def test_full_external_spinors_finite_cell_and_both_signed_constrained_lines(outgoing):
    incoming=(1,1,1,2,4)
    for pp,mass in ((.7,.16),(3.2,.72)):
        ell=pi*9/pp
        p1,p2=pp*np.array(incoming[3:])/9
        pq,pa=pp*np.array(outgoing[3:])/9
        for hq,ha,h1,h2 in product((-1,1),repeat=4):
            u=lf_spinor(pq,.21,-.12,mass,hq)/sqrt(2*pq)
            v=antiquark_spinor(pa,-.13,.34,mass,ha)/sqrt(2*pa)
            eps=[-np.array([h,1j])/sqrt(2) for h in (h1,h2)]
            a,b=[sum(GAMMA[i+1]*e[i] for i in range(2)) for e in eps]
            s1=u.conj()@GAMMA[0]@a@GAMMA_PLUS@b@v
            s2=u.conj()@GAMMA[0]@b@GAMMA_PLUS@a@v
            assert s1==pytest.approx(conversion_spin(hq,ha,h1,h2),abs=3e-15)
            factor=2*pp/ell*.5/(2*sqrt(p1*p2))
            expected=factor*(s1*conversion_ordered_color()/(p2-pa)+
                             s2*conversion_ordered_color(True)/(p1-pa))
            actual=instantaneous_conversion_kernel(incoming,outgoing,(hq,ha,h1,h2))
            np.testing.assert_allclose(actual,expected,atol=4e-15)


def test_cross_species_contact_against_independent_coordinate_quadrature():
    incoming,outgoing=(1,1,1,2,4),(1,1,1,5,1)
    direct=direct_contact("qqqgg",incoming,outgoing,(3,4),.4,out_sector="qqqq-qbar")
    np.testing.assert_allclose(conversion_contact(incoming,outgoing,.4,1),direct,atol=4e-15)


@pytest.mark.parametrize("cutoff",[0,1])
def test_direct_IG_conversion_matches_independent_coupled_pair_current_factorization(cutoff):
    obj=build(cutoff);f=obj.factorization
    direct=FiveBodyPairConversion(f.spaces,b_GeV=.4,channel="IG")
    for power in (1,2):
        values={s.source.space.sector:s.lift(s.coupling_by_power[power]) for s in obj.reference.sectors}
        actual=direct.apply(values)
        for name,other in (("qqqgg","qqqq-qbar"),("qqqq-qbar","qqqgg")):
            expected=f.matrices[name]@f.auxiliary.weight@f.matrices[other].conj().T@values[other]
            np.testing.assert_allclose(actual[name],expected,atol=4e-14)


def test_IF_conversion_exact_adjoint_and_both_species_statistics():
    obj=build(0);spaces=obj.factorization.spaces
    conversion=FiveBodyPairConversion(spaces,b_GeV=.4,channel="IF")
    rng=np.random.default_rng(449)
    v={name:rng.normal(size=(s.dimension,2))+1j*rng.normal(size=(s.dimension,2))
       for name,s in spaces.items()}
    u={name:rng.normal(size=(s.dimension,2))+1j*rng.normal(size=(s.dimension,2))
       for name,s in spaces.items()}
    av,au=conversion.apply(v),conversion.apply(u)
    left=sum(np.vdot(u[n],av[n]) for n in spaces)
    right=sum(np.vdot(au[n],v[n]) for n in spaces)
    assert left==pytest.approx(right,abs=4e-14)
    assert sum(np.linalg.norm(a) for a in av.values())>1e-3
    for n,s in spaces.items():
        np.testing.assert_allclose(s.statistics_project(av[n]),av[n],atol=3e-14)
        np.testing.assert_allclose(s.permute(av[n],(1,0,2,3,4)),-av[n],atol=3e-14)

