from itertools import combinations,permutations,product
from math import factorial,pi,sqrt
import numpy as np
import pytest

from deuteron_wigner.bridge.modes.core import GAMMA,GAMMA_PLUS
from deuteron_wigner.microscopic.h0.color import ColorSingletBasis,representation_generators
from deuteron_wigner.pilot.color import levi_civita3,structure_constants
from deuteron_wigner.quantum.m2_canonical_emission import lf_spinor
from deuteron_wigner.quantum.m2_excluded_fock_vertices import antiquark_spinor
from deuteron_wigner.quantum.m2_excluded_four_gluon import distinguished_quartic_tensor
from deuteron_wigner.quantum.m2_coupled_pilot import build_coupled_pilot
from deuteron_wigner.quantum.m2_excluded_fock_kinematics import frame as parent_frame
from deuteron_wigner.quantum.m2_six_body_kinematics import frame,color_basis
from deuteron_wigner.quantum.m2_six_body_quartic import (
    COMPONENTS,direct_routes,double_split_frame,contact_orbital,mass_kernel,
    A4_ordered_color,gluon_pair_ordered_color,gluon_pair_current_color,
    pair_scattering_spin,build_direct_source,
)


def test_A4_one_to_three_against_independent_normal_ordered_occupation_polynomial():
    f=levi_civita3().real
    labels=[x for x in product(range(4),repeat=6) if sum(x)==3]
    lookup={s:i for i,s in enumerate(labels)}
    direct=np.zeros((len(labels),6),complex)
    for adj,b,c,d,e,ip,jp in product(range(3),range(3),range(3),range(3),range(3),range(2),range(2)):
        coef=.25*f[adj,b,c]*f[adj,d,e]
        if coef==0:continue
        slots=(2*b+ip,2*c+jp,2*d+ip,2*e+jp)
        for annihilator in range(4):
            initial=slots[annihilator];created=[slots[i] for i in range(4) if i!=annihilator]
            s=[0]*6;amplitude=coef
            for k in created:
                s[k]+=1;amplitude*=sqrt(s[k])
            direct[lookup[tuple(s)],initial]+=amplitude
    raw=np.zeros((3,2,3,2,3,2,3,2),complex)
    for pols in product(range(2),repeat=4):
        raw[:,pols[0],:,pols[1],:,pols[2],:,pols[3]]=distinguished_quartic_tensor(f,np.eye(2)[list(pols)])/(4*sqrt(6))
    ordered=raw.reshape(216,6)
    actual=np.zeros_like(direct)
    for row,occ in enumerate(labels):
        slots=tuple(k for k,n in enumerate(occ) for _ in range(n))
        actual[row]=sqrt(6/np.prod([factorial(n) for n in occ]))*ordered[np.ravel_multi_index(slots,(6,6,6))]
    np.testing.assert_allclose(actual,direct,atol=8e-15)


def test_two_creation_factor_on_one_existing_boson():
    # Pdag=.5 Mij a_i^dag a_j^dag; independent ladder amplitudes.
    d=3;rng=np.random.default_rng(856)
    z=rng.normal(size=(d,d))+1j*rng.normal(size=(d,d));m=(z+z.T)/2
    v=rng.normal(size=d)+1j*rng.normal(size=d)
    labels=[x for x in product(range(4),repeat=d) if sum(x)==3]
    direct={x:0j for x in labels}
    for i,j,k in product(range(d),repeat=3):
        occ=[0]*d;occ[k]=1;amp=.5*m[i,j]*v[k]
        for a in (i,j):occ[a]+=1;amp*=sqrt(occ[a])
        direct[tuple(occ)]+=amp
    raw=v[:,None,None]*m[None,:,:]
    out=sqrt(6)/2*sum(raw.transpose(p) for p in permutations(range(3)))/6
    for occ in labels:
        slots=tuple(k for k,n in enumerate(occ) for _ in range(n))
        amplitude=sqrt(6/np.prod([factorial(n) for n in occ]))*out[slots]
        assert amplitude==pytest.approx(direct[occ],abs=5e-15)


@pytest.mark.parametrize("component",COMPONENTS)
def test_all_direct_routes_double_jacobi_measure_and_width(component):
    for route in direct_routes(11,component):
        r=double_split_frame(route)
        np.testing.assert_allclose(r@r.T,np.eye(5),atol=1e-15)
        a,b=parent_frame("qqqg",route.incoming),frame(route.sector,route.outgoing)
        x,y,z=b.fractions[list(route.daughters)]
        assert b.transverse_measure_factor/a.transverse_measure_factor==pytest.approx(x*y*z/(x+y+z),rel=4e-15)
        low=contact_orbital(route,0,0,.4)
        high=contact_orbital(route,0,1,.4)
        assert high.shape==(11,1)
        np.testing.assert_allclose(high[0],low[0],atol=5e-17)
        np.testing.assert_allclose(high[1:],0.,atol=0.)
        np.testing.assert_allclose(contact_orbital(route,0,1,.8),4*high,atol=0.)


def test_A4_complete_color_projection_and_normalized_field_factor():
    source=ColorSingletBasis.construct("qqqg").tensors.reshape(3,3,3,8,2)
    u=color_basis("qqqggg")
    route=direct_routes(9,"gluon_A4")[0]
    for hs in product((-1,1),repeat=4):
        eps=[-np.array([h,1j])/sqrt(2) for h in hs]
        eps[:3]=[a.conj() for a in eps[:3]]
        tensor=distinguished_quartic_tensor(structure_constants(),eps)
        raw=np.einsum("abcd,ijkdn->ijkabcn",tensor,source).reshape(13824,2)
        k=np.array(route.outgoing[3:]+(route.incoming[3],))
        for length in (.3,2.7):
            pp=pi*9/(2*length);mom=pi*k/(2*length)
            factor=2*pp/(2*length)*.25/np.sqrt(np.prod(2*mom))/sqrt(6)
            np.testing.assert_allclose(mass_kernel(route,hs),factor*u.conj().T@raw,atol=3e-15)
        if sum(hs[:3])!=hs[3]:np.testing.assert_allclose(mass_kernel(route,hs),0.,atol=1e-15)


def test_gluon_pair_current_color_relative_sign():
    np.testing.assert_allclose(gluon_pair_ordered_color()-gluon_pair_ordered_color(True),
                               -gluon_pair_current_color(),atol=8e-15)


def test_gluon_pair_IF_crossing_signed_propagators_and_field_normalization():
    for route in direct_routes(11,"gluon_IF_pair"):
        kq,ka,kg=route.outgoing[3:];kin=route.incoming[3]
        for hq,ha,hg,hi in product((-1,1),repeat=4):
            direct,reverse=pair_scattering_spin(hq,ha,hg,hi)
            for pp,mass in ((.7,.15),(3.8,.62)):
                K=sum(route.incoming);cell=pi*K/pp
                pq,pa,pg,pin=pp*np.array([kq,ka,kg,kin])/K
                u=lf_spinor(pq,.12,-.18,mass,hq)
                v=antiquark_spinor(pa,-.17,.23,mass,ha)
                eo=(-np.array([hg,1j])/sqrt(2)).conj();ei=-np.array([hi,1j])/sqrt(2)
                go,gi=[sum(GAMMA[i+1]*e[i] for i in range(2)) for e in (eo,ei)]
                s1=u.conj()@GAMMA[0]@go@GAMMA_PLUS@gi@v
                s2=u.conj()@GAMMA[0]@gi@GAMMA_PLUS@go@v
                np.testing.assert_allclose([s1,s2],sqrt(4*pq*pa)*np.array([direct,reverse]),atol=5e-15)
                factor=2*pp/cell/2/np.sqrt(16*pq*pa*pg*pin)
                expected=2*factor*(s1*gluon_pair_ordered_color()/(pin-pa)
                                     +s2*gluon_pair_ordered_color(True)/(-pg-pa))
                np.testing.assert_allclose(mass_kernel(route,(hq,ha,hg,hi)),expected,atol=4e-15)


@pytest.mark.parametrize("component",COMPONENTS)
def test_actual_direct_source_symmetry_and_exact_minimum_grid_zero(component):
    p=build_coupled_pilot(total_k2=9,cutoff=0,target="PROTON",jz_twice=1,b_GeV=.4,
        quark_mass_GeV=.35,gluon_mass_GeV=0.,coupling=.3)
    s=build_direct_source(p,out_cutoff=0,component=component)
    a=s.matrix_per_gs2
    np.testing.assert_allclose(s.space.statistics_project(a),a,atol=4e-16)
    np.testing.assert_allclose(s.space.permute(a,(1,0,2,3,4,5)),-a,atol=4e-16)
    if component in ("valence_IG","gluon_IG_ggg"):
        np.testing.assert_array_equal(a,0.)
        assert any(np.linalg.norm(mass_kernel(r,h))>1e-3
                   for r in direct_routes(11,component) for h in product((-1,1),repeat=4))
    else:
        assert np.linalg.norm(a)>1e-3
    assert np.array_equal(s.retained_columns,np.arange(p.bases[0].shape[1],len(p.matrix)))


def test_invalid_direct_kernel_inputs():
    with pytest.raises(ValueError):direct_routes(9,"missing")
    with pytest.raises(ValueError):mass_kernel(direct_routes(9,"valence_IF")[0],(True,1,1,-1))

def test_gluon_current_three_gluon_source_against_full_four_field_assignments(monkeypatch):
    from deuteron_wigner.quantum import m2_six_body_quartic as q
    p=build_coupled_pilot(total_k2=11,cutoff=0,target="PROTON",jz_twice=1,b_GeV=.4,
        quark_mass_GeV=.35,gluon_mass_GeV=0.,coupling=.3)
    expected=build_direct_source(p,out_cutoff=0,component="gluon_IG_ggg").matrix_per_gs2
    assert np.linalg.norm(expected)>1e-3
    def full_fields(route,hs):
        k=np.array(route.outgoing[3:]+(route.incoming[3],),float)
        signed=k*np.array([1,1,1,-1])
        eps=[-np.array([h,1j])/sqrt(2) for h in hs]
        eps[:3]=[e.conj() for e in eps[:3]]
        result=np.zeros((22,2),complex)
        for order in permutations(range(4)):
            transfer=signed[order[0]]+signed[order[1]]
            if transfer==0:continue
            # -1/2 (1/i d J)(1/i d J), J=f A d A.
            # Each differentiated field supplies i*signed_k. The second
            # current transfer is minus the first; sum all24 assignments.
            factor=-.5*signed[order[1]]*signed[order[3]]/transfer**2
            spin=np.dot(eps[order[0]],eps[order[1]])*np.dot(eps[order[2]],eps[order[3]])
            result+=factor*spin*A4_ordered_color(order)
        # All four field normalizations; three outgoing normalized bosons.
        return sum(route.incoming)/(2*pi*sqrt(6*np.prod(k)))*result
    monkeypatch.setattr(q,"mass_kernel",full_fields)
    actual=build_direct_source(p,out_cutoff=0,component="gluon_IG_ggg").matrix_per_gs2
    np.testing.assert_allclose(actual,expected,atol=3e-16)


def test_gluon_to_gluon_pair_current_independent_finite_cell_normalization():
    for route in direct_routes(11,"gluon_IG_pair"):
        kq,ka,kg=route.outgoing[3:];kin=route.incoming[3];K=sum(route.incoming)
        for hs in product((-1,1),repeat=4):
            hq,ha,hg,hi=hs
            for pp,mass in ((.7,.15),(3.8,.62)):
                pq,pa,pg,pin=pp*np.array([kq,ka,kg,kin])/K
                cell=pi*K/pp
                u=lf_spinor(pq,.13,-.24,mass,hq);v=antiquark_spinor(pa,-.11,.17,mass,ha)
                jq=u.conj()@GAMMA[0]@GAMMA_PLUS@v
                jg=(pin+pg)*int(hg==hi)*gluon_pair_current_color()
                fields=np.sqrt(16*pq*pa*pg*pin)
                # Two currents, positive pair transfer, normalized four-quark creation.
                expected=2*2*pp/cell*jq*jg/fields/(pq+pa)**2
                np.testing.assert_allclose(mass_kernel(route,hs),expected,atol=4e-15)
