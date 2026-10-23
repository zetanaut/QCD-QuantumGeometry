"""Gauge before Wick: direct loop integrals, signed CAR and H/J pairing."""
from collections import defaultdict,Counter
from dataclasses import replace
from itertools import product
from math import pi,factorial
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import roots_hermite
from deuteron_wigner.quantum.m2_gaussian_cluster_wick_current import (
    _gaussian_moment,cluster_contraction_paths,ordered_cluster_wick_terms,ClusterWickCurrentWeights)
from deuteron_wigner.quantum.m2_gaussian_cluster_wick import cluster_wick_summands
from deuteron_wigner.quantum.m2_gaussian_cluster_current import PRESCRIPTION
from deuteron_wigner.quantum.m2_canonical_quark_self import CF
from deuteron_wigner.quantum.m2_current_wick_self import ordered_wick_coefficient
from test_m2_gaussian_cluster_wick import regulator,operators
from test_m2_canonical_quark_self import _annihilators


@pytest.mark.parametrize("n",[0,1,2])
@pytest.mark.parametrize("a",[0.,1e-12,.3,5.,1e4])
def test_exact_path_moments_are_regular_at_zero_and_large_argument(n,a):
    # Rescale large-a quadrature so it does not miss the narrow endpoint.
    s = max(1.,np.sqrt(a))
    expected = quad(lambda u:(u/s)**n*np.exp(-a*(u/s)**2)/s,0.,min(s,12.),epsabs=2e-14)[0]
    assert _gaussian_moment(n,a)==pytest.approx(expected,abs=3e-15,rel=3e-13)


def direct_paths(p,k,delta,mean,reg,reference_shape=None):
    """Differentiate each original physical leg BEFORE numerical loop integration."""
    delta,mean = np.asarray(delta),np.asarray(mean)
    pf,pi_ = mean+delta/2,mean-delta/2
    w = np.array([p,k,p,k]);S = sum(w);L = reg.width_GeV**2
    nodes,weights = roots_hermite(22);ids = np.array(list(product(range(22),repeat=2)))
    z = nodes[ids]
    loop = k/p*mean+np.sqrt(L*k/p)*z
    base = np.stack((np.broadcast_to(pf,loop.shape),loop,np.broadcast_to(pi_,loop.shape),loop),axis=1)
    measure = np.prod(weights[ids],axis=1)*np.exp(np.sum(z*z,axis=1))*L*k/p/(2*pi)**2
    ts,tws = np.polynomial.legendre.leggauss(24)
    scalar = np.zeros(2);transverse = np.zeros((2,2))
    for a,sign in enumerate((-1,-1,1,1)):
        for t,weight in zip((ts+1)/2,tws/2):
            legs = base.copy();legs[:,a] += sign*t*delta
            weighted = np.sum(legs*legs/w[None,:,None],axis=(1,2))
            Q = (S*weighted-np.sum(np.sum(legs,axis=1)**2,axis=1))/2
            kernel = measure*np.exp(-Q/(2*L))
            if reference_shape is not None:
                s=reference_shape;z=(s.a-s.a_reference)*Q
                # Independent finite polynomial derivative, not the path
                # integral or polynomial-moment implementation under test.
                ratio=1-s.a_reference/s.a
                kernel*=sum(z**m/factorial(m) for m in range(s.degree+1))-ratio*sum(z**m/factorial(m) for m in range(s.degree))
            dw = (weighted-S*np.sum(legs[:,a]**2,axis=1)/w[a]**2)/2
            dt = S*legs[:,a]/w[a]-np.sum(legs,axis=1)
            scalar[a%2] += weight*np.sum(kernel*(-dw/(2*L)))
            transverse[a%2] += weight*np.einsum("n,nj->j",kernel,-dt/(2*L))
    return scalar,transverse


@pytest.mark.parametrize("p,k",[(1.,2.),(5.,2.),(2.,7.)])
@pytest.mark.parametrize("delta",[(0.,0.),(.12,.16),(.8,-.6)])
@pytest.mark.parametrize("width",[.6,1.1])
def test_external_and_internal_paths_against_uncontracted_four_leg_quadrature(p,k,delta,width):
    reg = regulator(width)
    a = cluster_contraction_paths(regulator=reg,external_mode=p,loop_mode=k,transfer_GeV=delta)
    direct,transverse = direct_paths(p,k,delta,(.23,-.17),reg)
    np.testing.assert_allclose([a.external_mode_path_GeV2,a.loop_mode_path_GeV2],direct,atol=7e-14,rtol=3e-11)
    np.testing.assert_allclose(transverse,0.,atol=5e-14)
    np.testing.assert_allclose(a.base_trace_GeV2,reg.contracted_kernel_trace(external_mode=p,loop_mode=k,
        outgoing_transverse_GeV=np.array([.23,-.17])+np.asarray(delta)/2,
        incoming_transverse_GeV=np.array([.23,-.17])-np.asarray(delta)/2),atol=3e-16)
    b = cluster_contraction_paths(regulator=reg,external_mode=p,loop_mode=k,transfer_GeV=-np.asarray(delta))
    assert a==b


def test_external_path_is_not_a_derivative_of_the_already_contracted_base_trace():
    reg = regulator();p,k,delta = 3.,2.,np.array([.6,-.4])
    a = cluster_contraction_paths(regulator=reg,external_mode=p,loop_mode=k,transfer_GeV=delta)
    old = reg.contracted_kernel_derivatives(external_mode=p,loop_mode=k,
        outgoing_transverse_GeV=delta/2,incoming_transverse_GeV=-delta/2)
    assert abs(a.external_mode_path_GeV2-old["external_mode"])>1e-4
    assert abs(a.loop_mode_path_GeV2-old["loop_mode"])>1e-4
    # Nor can the external derivative be integrated along one common
    # contracted external-mode path: the two individual means differ.
    L,D = reg.width_GeV**2,float(delta@delta)
    I = L*k/(4*pi*p);aa=(p+k)*D/(4*p*L)
    naive = I*(-_gaussian_moment(0,aa)/p+k*D/(4*L*p*p)*_gaussian_moment(2,aa))
    assert abs(a.external_mode_path_GeV2-naive)>1e-4


def terms(species,p,*,delta=(.12,.16),reg=None,pp=1.3,e=None,loops=(2/3,-1/3)):
    return ordered_cluster_wick_terms(species,p,regulator=regulator() if reg is None else reg,
        quark_flavors=len(loops),total_k2=9,total_pplus_GeV=pp,transfer_GeV=delta,
        external_charge=(0. if species=="g" else (2/3 if species=="q" else -2/3)) if e is None else e,
        loop_flavor_charges=loops,prescription=PRESCRIPTION)


@pytest.mark.parametrize("species,p",[("q",1),("q",5),("qbar",3),("g",2),("g",6)])
@pytest.mark.parametrize("width",[.3,.8,1.7])
def test_every_ordered_current_term_has_its_exact_Hamiltonian_trace_partner(species,p,width):
    reg = regulator(width)
    got = terms(species,p,reg=reg)
    h = cluster_wick_summands(species,p,regulator=reg,quark_flavors=2)
    for name,(modes,values) in h.items():
        for k,value in zip(modes,values):
            actual = sum(t.hamiltonian_per_gs2_GeV2 for t in got if t.source==name and t.loop_mode==k)
            assert actual==pytest.approx(9/pi*value,abs=3e-15)
    assert all(t.canonical_minus==0 for t in got if t.source!="IF")
    if species=="g":
        assert {t.loop_flavor for t in got if t.source=="IF"}=={0,1}
        assert all(t.loop_species=="qbar" for t in got if t.source=="IF")
        assert all(t.external_leg_minus==0 for t in got)
        assert all(t.minus==0 for t in got if t.source!="IF")


@pytest.mark.parametrize("species,p",[("q",1),("qbar",3),("g",2)])
def test_canonical_squared_denominator_part_matches_independent_legacy_field_algebra_per_mode(species,p):
    reg = regulator();got = terms(species,p)
    for k in sorted({t.loop_mode for t in got if t.source=="IF"}):
        # Isolate one term by a difference of the Fock-verified legacy sum;
        # insert its own common Gaussian trace only after that isolation.
        n = k//2 if species!="g" else (k+1)//2
        def legacy(count):
            if count==0:return 0.
            args = dict(boson_modes=8,fermion_modes=8,total_k2=9,total_pplus_GeV=1.3,
                coincident_trace_GeV2=1.,loop_flavor_charges=(2/3,-1/3))
            args["boson_modes" if species!="g" else "fermion_modes"]=count
            # Quark/antiquark external modes must stay inside the fermion envelope.
            return ordered_wick_coefficient(species,p,**args)
        trace = reg.contracted_kernel_trace(external_mode=p,loop_mode=k,
            outgoing_transverse_GeV=(.06,.08),incoming_transverse_GeV=(-.06,-.08))
        charge = 1 if species=="g" else (2/3 if species=="q" else -2/3)
        expected = charge*(legacy(n)-legacy(n-1))*trace
        assert sum(t.canonical_minus for t in got if t.source=="IF" and t.loop_mode==k)==pytest.approx(expected,abs=3e-15)


@pytest.mark.parametrize("n",[1,2,3])
@pytest.mark.parametrize("kind",["q","qbar"])
def test_IG_regulator_current_charge_routing_from_full_unreordered_CAR_product(n,kind):
    ann = _annihilators(2*n);b,d = ann[:n],ann[n:]
    bilinear = defaultdict(list)
    modes = list(range(1,2*n,2))
    for i,p in enumerate(modes):
        for j,r in enumerate(modes):
            bilinear[p-r].append((1.,b[i].T@b[j],(("q",p),("q",r))))
            bilinear[p-r].append((-1.,d[i].T@d[j],(("qbar",p),("qbar",r))))
            bilinear[p+r].append((1.,b[i].T@d[j].T,(("q",p),("qbar",r))))
            bilinear[-p-r].append((1.,d[j]@b[i],(("qbar",r),("q",p))))
    reg = replace(regulator(),fermion_modes=n)
    L = reg.width_GeV**2
    for i,p in enumerate(modes):
        occupied = 1<<(i+(n if kind=="qbar" else 0))
        external = (kind,p)
        H,J = 0.,0.
        for w,left in bilinear.items():
            if w==0:continue
            for c,a,at in left:
                for e,bop,bt in bilinear[-w]:
                    full = a@bop
                    connected = float(full[occupied,occupied]-full[0,0])
                    if connected==0:continue
                    tags = Counter(at+bt)
                    assert tags[external]==2
                    del tags[external]
                    assert len(tags)==1 and next(iter(tags.values()))==2
                    loop_kind,k = next(iter(tags))
                    trace = L*k/(4*pi*p)
                    weight = .5*c*e*connected/w**2
                    eq = 1. if kind=="q" else -1.
                    el = 1. if loop_kind=="q" else -1.
                    H += weight*trace
                    J += weight*trace*(eq/p-el/k)
        got = [t for t in terms(kind,p,reg=reg,delta=(0.,0.),e=eq) if t.source=="IG"]
        assert sum(t.hamiltonian_per_gs2_GeV2 for t in got)==pytest.approx(2*CF*9/pi*H,abs=4e-15)
        assert sum(t.regulator_minus for t in got)==pytest.approx(CF*9**2/(pi*1.3**2)*J,abs=4e-15)


@pytest.mark.parametrize("species,p",[("q",3),("qbar",3),("g",2)])
def test_zero_transfer_IF_current_is_correct_physical_charge_derivative_with_common_trace(species,p):
    reg = regulator();e = 0. if species=="g" else (2/3 if species=="q" else -2/3)
    k = np.arange(1.,16.,2.) if species=="g" else np.arange(2.,17.,2.)
    def energy(shift):
        if species=="g":
            r = k+shift
            value = .5/p*np.sum(reg.width_GeV**2*r/(4*pi*p)*(1/(p-r)-1/(p+r)))
            return (2/3-1/3)*9/(2*pi*1.3)*value
        q = p+shift
        r = q-k if species=="q" else q+k
        return e*CF*9/(2*pi*1.3)*np.sum(reg.width_GeV**2*k/(4*pi*q)/(k*r))
    eps = 1e-4
    derivative = (energy(-2*eps)-8*energy(-eps)+8*energy(eps)-energy(2*eps))/(12*eps)*9/1.3
    expected = derivative if species=="g" else -derivative
    got = [t for t in terms(species,p,delta=(0.,0.)) if t.source=="IF"]
    assert sum(t.minus for t in got)==pytest.approx(expected,abs=3e-11,rel=3e-10)
    assert abs(sum(t.regulator_minus for t in got))>1e-5


def test_exact_normal_order_subtraction_and_finite_kinetic_partner_are_separate():
    for width in (.3,.8,1.7):
        op = operators(regulator(width))
        for species,p in (("q",1),("qbar",3),("g",2)):
            values = {}
            for component in ("canonical","regulator","ordered","subtraction","finite","total"):
                w = ClusterWickCurrentWeights(operators=op,transfer_GeV=(.12,.16),
                    prescription=PRESCRIPTION,component=component)
                values[component] = w.minus_particle_weight(species,p,np.arange(4),total_k2=9,
                    total_pplus_GeV=1.3,charge="electromagnetic",loop_flavor_charges=(2/3,-1/3))
            np.testing.assert_allclose(values["canonical"]+values["regulator"],values["ordered"],atol=3e-15)
            np.testing.assert_array_equal(values["subtraction"],-values["ordered"])
            np.testing.assert_array_equal(values["total"],values["finite"])
            assert np.linalg.norm(values["ordered"])>1e-5
    assert not w.complete_gauged_contractions and not w.longitudinal_envelope_gauged and not w.physical


def test_outside_field_envelope_does_not_remove_a_finite_free_kinetic_current():
    op = operators(replace(regulator(),fermion_modes=2))
    w = ClusterWickCurrentWeights(operators=op,transfer_GeV=(.12,.16),prescription=PRESCRIPTION,component="total")
    assert terms("q",5,reg=op.regulator)==()
    v = w.minus_particle_weight("q",5,np.arange(4),total_k2=9,total_pplus_GeV=1.3,
                               charge="electromagnetic",loop_flavor_charges=(2/3,-1/3))
    np.testing.assert_allclose(v,.011*81/(2*25*1.3**2)*np.array([2/3,2/3,-1/3,-1/3]),atol=2e-18)


def test_gluon_current_resolves_loop_flavors_and_has_no_fictitious_external_charge():
    a = terms("g",2,loops=(1.,-1.))
    assert sum(t.minus for t in a)==pytest.approx(0.,abs=2e-15)
    assert abs(sum(t.minus for t in a if t.loop_flavor==0))>1e-4
    with pytest.raises(ValueError,match="no bare"):
        terms("g",2,e=1.)
    with pytest.raises(ValueError,match="flavor charges"):
        ordered_cluster_wick_terms("g",2,regulator=regulator(),quark_flavors=2,total_k2=9,
            total_pplus_GeV=1.3,transfer_GeV=(.12,.16),external_charge=0.,
            loop_flavor_charges=(1.,),prescription=PRESCRIPTION)
