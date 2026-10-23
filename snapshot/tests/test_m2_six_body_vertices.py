from itertools import product
from math import pi,sqrt
import numpy as np
import pytest
from deuteron_wigner.bridge.modes.core import GAMMA,polarization
from deuteron_wigner.pilot.color import structure_constants
from deuteron_wigner.microscopic.h0.color import representation_generators
from deuteron_wigner.quantum.m2_canonical_emission import lf_spinor
from deuteron_wigner.quantum.m2_excluded_fock_vertices import antiquark_spinor,_slash_epsilon
from deuteron_wigner.quantum.m2_excluded_fock_kinematics import color_basis as five_color,frame as five_frame
from deuteron_wigner.quantum.m2_six_body_kinematics import color_basis,frame
from deuteron_wigner.quantum.m2_six_body_vertices import (
    split_routes,split_frame,color_source,daughter_momenta,cubic_mass_kernel,cubic_orbital,pair_color,
)

CHANNELS=[("qqqgg","q_to_qg"),("qqqgg","g_to_gg"),("qqqgg","g_to_pair"),
          ("qqqq-qbar","q_to_qg"),("qqqq-qbar","anti_to_antig")]

@pytest.mark.parametrize("sector,channel",CHANNELS)
def test_six_particle_routes_measures_boosts_and_affine_vertex(sector,channel):
    routes=split_routes(11,sector,channel)
    assert routes
    for route in routes:
        r=split_frame(route)
        np.testing.assert_allclose(r@r.T,np.eye(5),atol=1e-15)
        a,b=five_frame(sector,route.incoming),frame(route.sector,route.outgoing)
        i,j=route.daughters
        ratio=b.fractions[i]*b.fractions[j]/(b.fractions[i]+b.fractions[j])
        assert b.transverse_measure_factor/a.transverse_measure_factor==pytest.approx(ratio,rel=5e-15)
    route=routes[-1]
    for hs in product((-1,1),repeat=3):
        values=[cubic_mass_kernel(route,(.13,-.21),.35,hs,total_pplus=pp,parent_transverse=pt)
                for pp,pt in ((.9,(0,0)),(3.,(.41,-.23)),(6.2,(-.17,.34)))]
        np.testing.assert_allclose(values,[values[0]]*3,atol=5e-15)
        zero=cubic_mass_kernel(route,(0,0),.35,hs)
        x=cubic_mass_kernel(route,(1,0),.35,hs)-zero
        y=cubic_mass_kernel(route,(0,1),.35,hs)-zero
        assert values[0]==pytest.approx(zero+.13*x-.21*y,abs=3e-15)
        assert cubic_orbital(route,0,1,.4,.35,hs).shape==(11,1)


def test_antiquark_vector_crossing_with_independent_CAR_sign():
    route=split_routes(9,"qqqq-qbar","anti_to_antig")[-1]
    for hs in product((-1,1),repeat=3):
        h1,hg,h0=hs
        left,glue,parent=daughter_momenta(route,(.13,-.21),2.7,(.34,.16))
        vi,vf=antiquark_spinor(*parent,.35,h0),antiquark_spinor(*left,.35,h1)
        uf,ui=lf_spinor(*left,.35,h1),lf_spinor(*parent,.35,h0)
        gamma=_slash_epsilon(polarization(*glue,hg).conj())
        # Charge conjugation reverses vector endpoints without a second CAR sign.
        np.testing.assert_allclose(vi.conj()@GAMMA[0]@gamma@vf,
                                   uf.conj()@GAMMA[0]@gamma@ui,atol=5e-15)
        field=2*2.7/sqrt(pi*9/2.7*8*left[0]*glue[0]*parent[0])
        expected=field*(vi.conj()@GAMMA[0]@gamma@vf)
        # Normal ordering the original -psi_bar gamma A psi dd^dag term
        # gives +vbar_in gamma v_out T^T, equal to kernel*(-T^T).
        assert -cubic_mass_kernel(route,(.13,-.21),.35,hs,total_pplus=2.7,
                                 parent_transverse=(.34,.16))==pytest.approx(expected,abs=3e-15)


def test_three_gluon_and_pair_raw_color_phases():
    route=split_routes(9,"qqqgg","g_to_gg")[0]
    source=five_color("qqqgg").reshape(3,3,3,8,8,6)
    raw=np.einsum("abc,ijkdcn->ijkdabn",structure_constants(),source).reshape(13824,6)
    np.testing.assert_allclose(color_basis("qqqggg")@color_source(route),raw,atol=2e-14)
    rawpair=np.einsum("bIJ,ijkabn->ijkIJan",np.array(representation_generators("3")),source).reshape(1944,6)
    np.testing.assert_allclose(color_basis("qqqq-qbar-g")@pair_color(),rawpair,atol=2e-14)
