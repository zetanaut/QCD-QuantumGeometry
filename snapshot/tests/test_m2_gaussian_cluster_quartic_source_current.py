"""All ten physical 1→3 routes: recoil, bare contacts and independent Ward ends."""
from dataclasses import replace
from itertools import product
import numpy as np
import pytest
from deuteron_wigner.quantum.m2_gaussian_cluster_quartic_source_current import (
    PairSourceRoute,quartic_source_geometry,quartic_source_charges,
    quartic_source_coefficients,cluster_quartic_source_current)
from deuteron_wigner.quantum.m2_six_body_quartic import COMPONENTS,direct_routes
from deuteron_wigner.quantum.m2_gaussian_field_regulator import gaussian_shell_integral
from deuteron_wigner.quantum.m2_gaussian_field_vertices import pair_field_orbital,direct_field_orbital
from deuteron_wigner.quantum.m2_quadratic_minus_fock import (
    double_local_orbital,pair_emission_minus_kernel,direct_six_minus_kernel)
from deuteron_wigner.quantum.m2_instantaneous_pair_exit import pair_emission_frame,_route
from deuteron_wigner.quantum.m2_six_body_quartic import double_split_frame
from deuteron_wigner.quantum.m2_coupled_longitudinal_jacobi import jacobi_frame
from deuteron_wigner.quantum.m2_excluded_fock_kinematics import frame as frame5
from deuteron_wigner.quantum.m2_six_body_kinematics import frame as frame6
from test_m2_gaussian_cluster_current import regulator
from deuteron_wigner.quantum.m2_gaussian_cluster_current import PRESCRIPTION

D = (.17,-.11)
MEAN = (.04,.03)
PAIR = [PairSourceRoute(c,(7,1,1),(1,1,1,2,4),0) for c in ("valence_IF","valence_IG")]
PAIR += [PairSourceRoute("valence_sea",(7,1,1),(3,1,1,1,3),0)]
DIRECT = [max(direct_routes(11,c),key=lambda r:abs(r.outgoing[3]-r.outgoing[4]) if c.endswith("_pair")
              else abs(r.outgoing[r.daughters[1]]-r.outgoing[r.daughters[2]]))
          for c in COMPONENTS]
ROUTES = PAIR+DIRECT


def current(route,hs=(1,1,1,-1),*,ci=0,co=1,delta=D,mean=MEAN,r=None,
            e=.7,f=-.2,pp=1.3,order=8):
    return cluster_quartic_source_current(route,ci,co,.4,hs,
        regulator=regulator() if r is None else r,total_pplus_GeV=pp,
        transfer_GeV=delta,mean_total_transverse_GeV=mean,emitter_charge=e,pair_charge=f,
        prescription=PRESCRIPTION,quadrature_order=order)


def coefficients(route,hs,pp=1.3,e=.7,f=-.2):
    return quartic_source_coefficients(route,hs,total_pplus_GeV=pp,emitter_charge=e,pair_charge=f)


def frames(route):
    if isinstance(route,PairSourceRoute):
        return jacobi_frame(route.incoming),_route(route.incoming,route.outgoing,route.emitter,route.sector),pair_emission_frame(
            route.incoming,route.outgoing,route.emitter,sector=route.sector)
    return frame5("qqqg",route.incoming),frame6(route.sector,route.outgoing),double_split_frame(route)


def hs_nonzero(route):
    return next(h for h in product((-1,1),repeat=4) if np.linalg.norm(coefficients(route,h)[0])>1e-10)


def ward_orbital(route,*,ci=0,co=1,delta=D,mean=MEAN,r=None,e=.7,f=-.2):
    r = regulator() if r is None else r
    g = quartic_source_geometry(route,transfer_GeV=delta,mean_total_transverse_GeV=mean)
    charges = quartic_source_charges(route,emitter_charge=e,pair_charge=f)
    B = r.leg_transform(g["modes"])
    result = 0.
    for a,sign in enumerate((-1,-1,-1,1)):
        shift = g["physical_field_shifts"].copy();shift[a] += sign*np.asarray(delta)
        result += sign*charges[a]*gaussian_shell_integral(out_cutoff=co,in_cutoff=ci,
            out_map=g["out_map"],in_map=g["in_map"],in_shift=g["in_shift"],b_GeV=.4,
            field_maps=B@g["physical_field_maps"],field_shifts=B@shift,field_precision=1/r.width_GeV**2)
    return result*g["jacobian"]/(2*1.3)


@pytest.mark.parametrize("route",ROUTES)
def test_actual_global_recoil_preserves_all_spectators_and_all_four_physical_legs(route):
    g = quartic_source_geometry(route,transfer_GeV=D,mean_total_transverse_GeV=MEAN)
    fi,fo,tree = frames(route)
    q = np.random.default_rng(881).normal(size=(len(route.outgoing)-1,2))
    qo,qi = q,g["in_map"]@q+g["in_shift"]
    outgoing = fo.inverse[:,:-1]@qo+fo.fractions[:,None]*(np.array(MEAN)+np.array(D)/2)
    incoming = fi.inverse[:,:-1]@qi+fi.fractions[:,None]*(np.array(MEAN)-np.array(D)/2)
    actual = g["physical_field_maps"]@q+g["physical_field_shifts"]
    np.testing.assert_allclose(actual,np.vstack((outgoing[list(route.daughters)],incoming[route.emitter])),atol=9e-16)
    np.testing.assert_allclose(actual[:3].sum(0)-actual[3],D,atol=7e-16)
    out_spectators = [j for j in range(len(route.outgoing)) if j not in route.daughters]
    in_spectators = [j for j in range(len(route.incoming)) if j!=route.emitter]
    np.testing.assert_allclose(outgoing[out_spectators],incoming[in_spectators],atol=9e-16)
    assert g["modes"]==tuple(route.outgoing[j] for j in route.daughters)+(route.incoming[route.emitter],)
    assert len(g["species"])==4 and g["in_map"].shape==(len(route.incoming)-1,len(route.outgoing)-1)


@pytest.mark.parametrize("route",ROUTES)
@pytest.mark.parametrize("ci,co",[(0,1),(1,0)])
def test_all_physical_source_currents_Ward_endpoints_canonical_zeros_and_path_refinement(route,ci,co):
    hs = hs_nonzero(route)
    a,b = current(route,hs,ci=ci,co=co,order=8),current(route,hs,ci=ci,co=co,order=16)
    h,j = coefficients(route,hs)
    np.testing.assert_allclose(a.total,b.total,atol=3e-15,rtol=3e-12)
    expected = np.einsum("oi,ab->oiab",ward_orbital(route,ci=ci,co=co),h)
    np.testing.assert_allclose(np.einsum("j,joiab->oiab",D,a.total[2:]),expected,atol=4e-16,rtol=3e-11)
    np.testing.assert_array_equal(a.total[0],0.)
    if route.component not in ("valence_IF","gluon_IF_pair"):
        np.testing.assert_array_equal(a.canonical,0.)
    if route.component in ("gluon_A4","gluon_IG_ggg"):
        np.testing.assert_array_equal(a.total,0.)
        assert np.linalg.norm(h)>1e-5
    else:
        assert np.linalg.norm(a.regulator)>1e-6
    assert not a.physical and not a.longitudinal_envelope_gauged


@pytest.mark.parametrize("route",ROUTES)
def test_bare_source_orbitals_and_independent_constrained_current_with_fock_once(route):
    hs = hs_nonzero(route);fi,fo,tree = frames(route)
    a = current(route,hs,ci=1,co=2,r=regulator(1e7),order=2)
    bare = double_local_orbital(fi,route.emitter,fo.fractions[list(route.daughters)],tree,1,2,.4,D)
    if route.component=="valence_IF" and isinstance(route,PairSourceRoute):
        j = .7/np.sqrt(2)*pair_emission_minus_kernel(route.incoming,route.outgoing,route.emitter,hs,total_pplus_GeV=1.3)
    elif route.component in ("valence_IF","gluon_IF_pair"):
        j = (.7 if route.component=="valence_IF" else -.2)*direct_six_minus_kernel(route,hs,total_pplus_GeV=1.3)
    else:
        j = np.zeros_like(coefficients(route,hs)[0])
    np.testing.assert_allclose(a.total[1],np.einsum("oi,ab->oiab",bare,j),atol=3e-15)
    np.testing.assert_allclose(a.regulator,0.,atol=3e-15)


@pytest.mark.parametrize("route",ROUTES)
def test_common_H_conserving_slice_and_both_kinematic_boosts(route):
    hs = hs_nonzero(route)
    a = current(route,hs,ci=1,co=1,delta=(0.,0.))
    if isinstance(route,PairSourceRoute):
        orb = pair_field_orbital(route.incoming,route.outgoing,route.emitter,1,1,.4,
            sector=route.sector,regulator=regulator(),total_pplus_GeV=1.3,total_transverse_GeV=MEAN)
    else:
        orb = direct_field_orbital(route,1,1,.4,regulator=regulator(),total_pplus_GeV=1.3,total_transverse_GeV=MEAN)
    np.testing.assert_allclose(a.canonical[1],np.einsum("oi,ab->oiab",orb,coefficients(route,hs)[1]),atol=3e-16)
    a = current(route,hs,ci=1,co=1)
    beta = np.array([.23,-.16])
    boost = current(route,hs,ci=1,co=1,mean=tuple(np.array(MEAN)+1.3*beta))
    scale = current(route,hs,ci=1,co=1,pp=2.6)
    for name in ("canonical","regulator"):
        aa,bb,ss = [getattr(x,name) for x in (a,boost,scale)]
        np.testing.assert_allclose(bb[2:],aa[2:],atol=3e-15)
        np.testing.assert_allclose(bb[1],aa[1]+np.einsum("j,joiab->oiab",beta,aa[2:]),atol=4e-15)
        np.testing.assert_allclose(ss[1],aa[1]/4,atol=3e-15)
        np.testing.assert_allclose(ss[2:],aa[2:]/2,atol=3e-15)


def test_created_pair_flavor_is_not_valence_flavor_and_envelope_excludes_spectators():
    route = PAIR[-1];hs = hs_nonzero(route)
    a = current(route,hs)
    val,pair = current(route,hs,f=0.),current(route,hs,e=0.)
    np.testing.assert_allclose(a.total,val.total+pair.total,atol=3e-16)
    assert np.linalg.norm(val.regulator)>1e-6 and np.linalg.norm(pair.regulator)>1e-6
    # Spectator valence modes7,7 lie OUTSIDE field envelope5: they are untouched.
    r = PairSourceRoute("valence_sea",(5,7,7),(1,7,7,1,3),0)
    full = current(r,(1,1,1,-1))
    restricted = current(r,(1,1,1,-1),r=replace(regulator(),fermion_modes=3))
    np.testing.assert_allclose(full.total,restricted.total,atol=3e-16)
    np.testing.assert_array_equal(current(r,(1,1,1,-1),r=replace(regulator(),fermion_modes=2)).total,0.)
    # Sum of daughter q/anti modes4 is an internal IG momentum, not a physical gluon.
    np.testing.assert_allclose(current(r,(1,1,1,-1),r=replace(regulator(),boson_modes=1)).total,full.total,atol=3e-16)


def test_invalid_route_charge_definition_and_scales_not_silently_assumed():
    with pytest.raises(ValueError,match="route"):
        quartic_source_geometry(object(),transfer_GeV=D,mean_total_transverse_GeV=MEAN)
    with pytest.raises(ValueError,match="IF"):
        current(replace(PAIR[0],component="anything"))
    with pytest.raises(ValueError):
        current(PAIR[0],e=np.nan)
    with pytest.raises(ValueError):
        current(PAIR[0],pp=0.)
    with pytest.raises(ValueError):
        current(PAIR[0],order=True)


def test_equal_mode_created_pair_current_is_odd_but_not_identically_absent():
    for c in ("gluon_IF_pair","gluon_IG_pair"):
        route = next(r for r in direct_routes(11,c) if r.outgoing[3]==r.outgoing[4])
        hs = hs_nonzero(route)
        np.testing.assert_allclose(current(route,hs,ci=1,co=0).regulator,0.,atol=2e-18)
        assert np.linalg.norm(current(route,hs,ci=0,co=1).regulator)>1e-6


@pytest.mark.parametrize("kind",["valence_gg","valence_sea","gluon_pair"])
def test_four_leg_regulator_minus_and_transverse_against_independent_scalar_derivatives(kind):
    """Direct scalar F, numerical physical-leg derivatives, polar HO quadrature."""
    from math import pi
    from scipy.special import roots_hermite
    from deuteron_wigner.bridge.modes.core import ho_momentum
    from deuteron_wigner.quantum.m2_coupled_longitudinal_jacobi import oscillator_shell
    from deuteron_wigner.quantum.m2_gaussian_cluster_current import cluster_current_integrals
    modes,kinds,charges = {
        "valence_gg":((1,2,4,7),("q","g","g","q"),(.7,0.,0.,.7)),
        "valence_sea":((1,1,3,5),("q","q","qbar","q"),(.7,-.2,.2,.7)),
        "gluon_pair":((1,3,2,6),("q","qbar","g","g"),(-.2,.2,0.,0.)),
    }[kind]
    w = np.array(modes,float);fields = np.array([[.3],[-.13],[.09],[.26]])
    shifts = np.array([[.12,.03],[-.04,.08],[.06,-.02],[-.03,.2]])
    delta = np.array(D);b = .4;reg = regulator();h=.31+.17j
    answer = cluster_current_integrals(regulator=reg,modes=modes,species=kinds,
        outgoing_legs=(True,True,True,False),charges=charges,transfer_GeV=delta,total_k2=11,
        total_pplus_GeV=1.3,prescription=PRESCRIPTION,quadrature_order=16,out_cutoff=1,in_cutoff=1,
        out_map=np.eye(1),in_map=np.eye(1),b_GeV=b,physical_field_maps=fields,
        physical_field_shifts=shifts,hamiltonian_constant=h,hamiltonian_linear=np.zeros((2,1)),
        hamiltonian_leg_derivatives=np.zeros((4,2)),canonical_current_constant=(0.,.21j,0.,0.),
        canonical_current_linear=np.zeros((4,2,1)))
    def amplitude(mom,weights):
        Q = (sum(weights)*np.sum(mom*mom/weights[None,:,None],axis=(1,2))
             -np.sum(np.sum(mom,axis=1)**2,axis=1))/2
        return np.exp(-Q/(2*reg.width_GeV**2))
    nodes,weights = roots_hermite(9)
    ids = np.array(list(product(range(9),repeat=2)));z=nodes[ids]
    B = reg.leg_transform(modes);precision=1/reg.width_GeV**2
    cv = 1/(2/b**2+precision*((B@fields).T@(B@fields)).item())
    def quadrature(s):
        center = -cv*precision*((B@fields).T@(B@s))[0]
        q = np.sqrt(2*cv)*z+center
        measure = 2*cv*np.prod(weights[ids],axis=1)*np.exp(np.sum(z*z,axis=1))/(2*pi)**2
        phi = np.array([(-1)**min(state)*ho_momentum(min(state),state[0]-state[1],q[:,0],q[:,1],b)
                        for state in oscillator_shell(1,1)])
        return fields[None,:,:]*q[:,None,:]+s,measure,phi
    mom,measure,phi = quadrature(shifts)
    canonical = .21j*(phi.conj()*(measure*amplitude(mom,w)))@phi.T
    np.testing.assert_allclose(answer.canonical[1],canonical,atol=3e-15)
    expected = np.zeros_like(answer.regulator)
    ts,tw = np.polynomial.legendre.leggauss(16);eps=2e-4
    for a,(sign,charge) in enumerate(zip((-1,-1,-1,1),charges)):
        if charge==0:continue
        for t,weight in zip((ts+1)/2,tw/2):
            s = shifts.copy();s[a] += sign*t*delta
            mom,measure,phi = quadrature(s)
            for mu in (1,2,3):
                samples=[]
                for step in (-2,-1,1,2):
                    ww,mm=w.copy(),mom.copy()
                    if mu==1:ww[a] += step*eps
                    else:mm[:,a,mu-2] += step*eps
                    samples.append(amplitude(mm,ww))
                derivative=(samples[0]-8*samples[1]+8*samples[2]-samples[3])/(12*eps)
                if mu==1:derivative *= -11/1.3
                expected[mu] += charge*weight*h/(2*1.3)*(phi.conj()*(measure*derivative))@phi.T
    np.testing.assert_allclose(answer.regulator,expected,atol=3e-12,rtol=4e-9)
