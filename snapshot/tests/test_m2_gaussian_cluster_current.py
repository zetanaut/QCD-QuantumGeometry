"""Physical charged legs, canonical spin limits and independent Ward endpoints."""
from dataclasses import replace
from itertools import product
from math import pi,factorial
import numpy as np
import pytest
from scipy.special import roots_hermite
from deuteron_wigner.bridge.modes.core import ho_momentum
from deuteron_wigner.quantum.m2_gaussian_field_regulator import (
    GaussianClusterRegulator, gaussian_shell_integral)
from deuteron_wigner.quantum.m2_gaussian_reference_shape import GaussianReferenceShape
from deuteron_wigner.quantum.m2_gaussian_cluster_current import (
    PRESCRIPTION, cubic_physical_kernel, cubic_canonical_current,
    cluster_current_integrals, cubic_current_geometry, cluster_cubic_current_orbitals)
from deuteron_wigner.quantum.m2_coupled_longitudinal_jacobi import (
    emission_routes, oscillator_shell)
from deuteron_wigner.quantum.m2_excluded_fock_kinematics import split_routes
from deuteron_wigner.quantum.m2_six_body_vertices import split_routes as six_routes
from deuteron_wigner.quantum.m2_transverse_contact_current import contact_route_orbital


def regulator(width=.8):
    return GaussianClusterRegulator(width_GeV=width,boson_modes=8,fermion_modes=8)


KINDS = {
    "q_to_qg":((3,2,5),("q","g","q"),(.7,0.,.7)),
    "anti_to_antig":((3,2,5),("qbar","g","qbar"),(-.7,0.,-.7)),
    "g_to_pair":((1,3,4),("q","qbar","g"),(.7,-.7,0.)),
    "g_to_gg":((2,2,4),("g","g","g"),(0.,0.,0.)),
}
SIX = [("qqqgg","q_to_qg"),("qqqgg","g_to_gg"),("qqqgg","g_to_pair"),
       ("qqqq-qbar","q_to_qg"),("qqqq-qbar","anti_to_antig")]
ROUTES = [emission_routes(9)[-1]]+[split_routes(9,c)[-1] for c in (
    "q_to_qg","g_to_gg","g_to_pair")]+[six_routes(9,s,c)[-1] for s,c in SIX]


@pytest.mark.parametrize("d",[1,2])
def test_cubic_and_quartic_insertions_against_direct_polar_wave_quadrature(d):
    rng = np.random.default_rng(580+d)
    lo,li = np.eye(d),np.linalg.qr(rng.normal(size=(d,d)))[0]
    so,si = rng.normal(size=(d,2))*.07,rng.normal(size=(d,2))*.05
    fields,shifts = rng.normal(size=(3,d)),rng.normal(size=(3,2))*.1
    b,precision = .43,1/.8**2
    terms = {}
    for degree in (0,1,2,3,4):
        for _ in range(4):
            powers = np.bincount(rng.integers(0,2*d,degree),minlength=2*d)
            terms[tuple(powers)] = rng.normal()+1j*rng.normal()
    args = dict(out_cutoff=2,in_cutoff=2,out_map=lo,in_map=li,out_shift=so,in_shift=si,
        b_GeV=b,field_precision=precision,field_maps=fields,field_shifts=shifts,
        constant=0.,polynomial_terms=terms)
    got = gaussian_shell_integral(**args)
    c = (lo.T@lo+li.T@li)/b**2+precision*fields.T@fields
    h = -(lo.T@so+li.T@si)/b**2-precision*fields.T@shifts
    cv = np.linalg.inv(c)
    nodes,weights = roots_hermite(8)
    ids = np.array(list(product(range(8),repeat=2*d)))
    z = nodes[ids]
    q = (np.sqrt(2)*z.reshape(-1,2,d)@np.linalg.cholesky(cv).T+(cv@h).T).transpose(0,2,1)
    measure = np.prod(weights[ids],axis=1)*np.exp(np.sum(z*z,axis=1))*2**d/np.linalg.det(c)/(2*pi)**(2*d)
    qo,qi = np.einsum("ad,ndj->naj",lo,q)+so,np.einsum("ad,ndj->naj",li,q)+si
    legs = np.einsum("ad,ndj->naj",fields,q)+shifts
    field = np.exp(-precision*np.sum(legs**2,axis=(1,2))/2)
    flat = q.transpose(0,2,1).reshape(len(q),-1)
    polynomial = sum(v*np.prod(flat**np.asarray(p),axis=1) for p,v in terms.items())
    def wave(state,points):
        value = np.ones(len(points),complex)
        for j in range(d):
            plus,minus = state[j],state[d+j]
            radial,angular = min(plus,minus),plus-minus
            value *= (-1)**radial*ho_momentum(radial,angular,points[:,j,0],points[:,j,1],b)
        return value
    shell = oscillator_shell(d,2)
    out,inc = np.array([wave(s,qo) for s in shell]),np.array([wave(s,qi) for s in shell])
    expected = (out.conj()*(measure*field*polynomial))@inc.T
    np.testing.assert_allclose(got,expected,atol=3e-14,rtol=5e-13)
    reverse = gaussian_shell_integral(**{**args,"out_map":li,"in_map":lo,
        "out_shift":si,"in_shift":so,"polynomial_terms":{p:v.conjugate() for p,v in terms.items()}})
    np.testing.assert_allclose(got,reverse.conj().T,atol=3e-14)


def test_insertions_add_to_existing_optimized_quadratic_not_replace_it():
    args = dict(out_cutoff=3,in_cutoff=2,out_map=np.eye(1),in_map=np.eye(1),
        b_GeV=.4,field_precision=2.,field_maps=np.array([[.7]]),field_shifts=np.array([[.1,-.2]]))
    terms = {(0,0):.3j,(1,0):.2,(0,1):.1j,(2,0):.4,(1,1):-.7j,(0,2):.2}
    got = gaussian_shell_integral(**args,constant=.1,polynomial_terms=terms)
    expected = gaussian_shell_integral(**args,constant=.1+.3j,linear=[[.2],[.1j]],
        quadratic=[[.8,-.7j],[-.7j,.4]])
    np.testing.assert_allclose(got,expected,atol=3e-16)


@pytest.mark.parametrize("channel",["q_to_qg","anti_to_antig","g_to_pair"])
@pytest.mark.parametrize("shape_degree",[None,2])
def test_all_regulator_components_against_direct_polar_integrals_and_independent_field_derivatives(channel,shape_degree):
    modes,species,charges = KINDS[channel]
    hs = (1,-1,-1) if channel=="g_to_pair" else (1,-1,1)
    w = np.array(modes,float);fields = np.array([[.3],[-.13],[.17]])
    delta = np.array([.17,-.11]);shifts = np.array([[.12,.03],[-.04,.08],[-.09,.22]])
    args = dict(channel=channel,helicities=hs,total_k2=9,total_pplus_GeV=1.3,mass_GeV=.35)
    h0 = cubic_physical_kernel(modes,shifts,**args)
    j0 = cubic_canonical_current(modes,shifts,**args,flavor_charge=.7)
    zero = np.zeros((3,2));c = cubic_physical_kernel(modes,zero,**args);dh = np.zeros((3,2),complex)
    for a in range(3):
        for j in range(2):
            step = zero.copy();step[a,j] = 1.
            dh[a,j] = cubic_physical_kernel(modes,step,**args)-c
    j1 = np.zeros((4,2,1),complex)
    for j in range(2):
        step = shifts.copy();step[:,j] += fields[:,0]
        j1[:,j,0] = cubic_canonical_current(modes,step,**args,flavor_charge=.7)-j0
    reg,b = regulator(),.4
    shape=None if shape_degree is None else GaussianReferenceShape(
        base_width_GeV=reg.width_GeV,reference_width_GeV=1.2,degree=shape_degree)
    answer = cluster_current_integrals(regulator=reg,modes=modes,species=species,
        outgoing_legs=(True,True,False),charges=charges,transfer_GeV=delta,total_k2=9,
        total_pplus_GeV=1.3,prescription=PRESCRIPTION,quadrature_order=16,out_cutoff=1,in_cutoff=1,
        out_map=np.eye(1),in_map=np.eye(1),b_GeV=b,physical_field_maps=fields,
        physical_field_shifts=shifts,hamiltonian_constant=h0,hamiltonian_linear=dh.T@fields,
        hamiltonian_leg_derivatives=dh,canonical_current_constant=j0,canonical_current_linear=j1,
        reference_shape=shape)
    def amplitude(mom,weights):
        # Scalar invariant formula independent of B and insertion recurrences.
        Q = (sum(weights)*np.sum(mom*mom/weights[None,:,None],axis=(1,2))
             -np.sum(np.sum(mom,axis=1)**2,axis=1))/2
        value=np.exp(-Q/(2*reg.width_GeV**2))
        if shape_degree is not None:
            z=(1/(2*reg.width_GeV**2)-1/(2*1.2**2))*Q
            value*=sum(z**m/factorial(m) for m in range(shape_degree+1))
        return value
    nodes,weights = roots_hermite(9)
    ids = np.array(list(product(range(9),repeat=2)));z = nodes[ids]
    B = reg.leg_transform(modes);precision = 1/reg.width_GeV**2
    cv = 1/(2/b**2+precision*((B@fields).T@(B@fields)).item())
    def quadrature(s):
        mean = -cv*precision*((B@fields).T@(B@s))[0]
        q = np.sqrt(2*cv)*z+mean
        measure = 2*cv*np.prod(weights[ids],axis=1)*np.exp(np.sum(z*z,axis=1))/(2*pi)**2
        phi = np.array([(-1)**min(state)*ho_momentum(min(state),state[0]-state[1],q[:,0],q[:,1],b)
                        for state in oscillator_shell(1,1)])
        return q,measure,phi
    q,measure,phi = quadrature(shifts)
    mom = fields[None,:,:]*q[:,None,:]+shifts
    spin = np.array([cubic_canonical_current(modes,p,**args,flavor_charge=.7) for p in mom])
    canonical = np.einsum("on,nm,in,n->moi",phi.conj(),spin,phi,measure*amplitude(mom,w))
    gauged = np.zeros_like(canonical)
    ts,tweights = np.polynomial.legendre.leggauss(16)
    eps = 2e-4
    for a,(sign,charge) in enumerate(zip((-1,-1,1),charges)):
        if not charge:continue
        for t,weight in zip((ts+1)/2,tweights/2):
            s = shifts.copy();s[a] += sign*t*delta
            q,measure,phi = quadrature(s)
            mom = fields[None,:,:]*q[:,None,:]+s
            endpoint = fields[None,:,:]*q[:,None,:]+shifts
            endpoint[:,a] += sign*delta
            vertex = np.array([cubic_physical_kernel(modes,p,**args) for p in endpoint])
            derivative = np.zeros((len(mom),4))
            for mu in (1,2,3):
                samples = []
                for step in (-2,-1,1,2):
                    ww,mm = w.copy(),mom.copy()
                    if mu==1:ww[a] += step*eps
                    else:mm[:,a,mu-2] += step*eps
                    samples.append(amplitude(mm,ww))
                derivative[:,mu] = (samples[0]-8*samples[1]+8*samples[2]-samples[3])/(12*eps)
                if mu==1:derivative[:,mu] *= -9/1.3
            gauged += charge*weight/(2*1.3)*np.einsum("on,nm,in,n->moi",phi.conj(),derivative,phi,measure*vertex)
    np.testing.assert_allclose(answer.canonical,canonical,atol=3e-14,rtol=2e-12)
    np.testing.assert_allclose(answer.regulator,gauged,atol=3e-12,rtol=4e-9)


@pytest.mark.parametrize("channel",KINDS)
def test_independent_physical_leg_Dirac_polynomial_and_canonical_transverse_spin(channel):
    modes,_,charges = KINDS[channel]
    p = np.array([[.2,-.1],[-.07,.3],[.01,.02]])
    for hs in product((-1,1),repeat=3):
        args = dict(channel=channel,helicities=hs,total_k2=9,total_pplus_GeV=1.3,mass_GeV=.35)
        zero = np.zeros((3,2))
        c = cubic_physical_kernel(modes,zero,**args)
        gradient = np.empty((3,2),complex)
        for a in range(3):
            for j in range(2):
                step = zero.copy();step[a,j] = 1.
                gradient[a,j] = cubic_physical_kernel(modes,step,**args)-c
        np.testing.assert_allclose(cubic_physical_kernel(modes,p,**args),c+np.sum(gradient*p),atol=8e-16)
        canonical = cubic_canonical_current(modes,p,**args,flavor_charge=.7)
        np.testing.assert_allclose(np.asarray(charges)@gradient/(2*1.3),canonical[2:],atol=8e-16)


def endpoint_ward_reference(route,hs,delta,mean,reg,ci,co):
    """Full shifted Dirac vertices, independently integrated; no current derivatives."""
    g = cubic_current_geometry(route,transfer_GeV=delta,mean_total_transverse_GeV=mean)
    modes,channel = g["modes"],g["channel"]
    charges = KINDS[channel][2]
    B = reg.leg_transform(modes)
    args = dict(channel=channel,helicities=hs,total_k2=9,total_pplus_GeV=1.3,mass_GeV=.35)
    result = 0.
    for leg,sign in enumerate((-1,-1,1)):
        shifted = g["physical_field_shifts"].copy();shifted[leg] += sign*np.asarray(delta)
        c = cubic_physical_kernel(modes,shifted,**args)
        lin = np.zeros((2,g["out_map"].shape[1]),complex)
        for j in range(2):
            for k in range(lin.shape[1]):
                step = shifted.copy();step[:,j] += g["physical_field_maps"][:,k]
                lin[j,k] = cubic_physical_kernel(modes,step,**args)-c
        result += sign*charges[leg]*gaussian_shell_integral(out_cutoff=co,in_cutoff=ci,
            out_map=g["out_map"],in_map=g["in_map"],in_shift=g["in_shift"],b_GeV=.4,
            field_precision=1/reg.width_GeV**2,field_maps=B@g["physical_field_maps"],
            field_shifts=B@shifted,constant=c,linear=lin)
    return result*g["jacobian"]/(2*1.3)


def route_current(route,hs=(1,-1,1),delta=(.17,-.11),mean=(.03,.07),reg=None,ci=1,co=1,order=12):
    return cluster_cubic_current_orbitals(route,ci,co,.4,.35,hs,
        regulator=regulator() if reg is None else reg,total_pplus_GeV=1.3,
        transfer_GeV=delta,mean_total_transverse_GeV=mean,flavor_charge=.7,
        prescription=PRESCRIPTION,quadrature_order=order)


@pytest.mark.parametrize("route",ROUTES)
@pytest.mark.parametrize("ci,co",[(0,1),(1,0)])
def test_all_actual_cubic_routes_Ward_endpoints_and_path_refinement(route,ci,co):
    hs = (1,-1,-1) if getattr(route,"channel",None)=="g_to_pair" else (1,-1,1)
    delta,mean = np.array([.17,-.11]),np.array([.03,.07])
    a = route_current(route,hs,delta,mean,ci=ci,co=co,order=8)
    b = route_current(route,hs,delta,mean,ci=ci,co=co,order=16)
    np.testing.assert_allclose(a.total,b.total,atol=3e-14,rtol=3e-11)
    ward = endpoint_ward_reference(route,hs,delta,mean,regulator(),ci,co)
    np.testing.assert_allclose(delta@a.total[2:].reshape(2,-1),ward.reshape(-1),atol=4e-15,rtol=3e-11)
    np.testing.assert_array_equal(a.total[0],0.)
    assert not a.physical and not a.complete_gauged_contractions and not a.longitudinal_envelope_gauged
    if getattr(route,"channel",None)!="g_to_gg":
        assert np.linalg.norm(a.regulator)>1e-5
        # Multiplying the old contact by F0 misses a measured Ward contribution.
        assert np.linalg.norm(delta@a.regulator[2:].reshape(2,-1))>1e-7


@pytest.mark.parametrize("route",ROUTES)
def test_actual_cubic_four_current_boost_law_not_only_Ward_projection(route):
    hs = (1,-1,-1) if getattr(route,"channel",None)=="g_to_pair" else (1,-1,1)
    beta,mean = np.array([.23,-.16]),np.array([.03,.07])
    a,b = route_current(route,hs,mean=mean),route_current(route,hs,mean=mean+1.3*beta)
    for old,new in ((a.canonical,b.canonical),(a.regulator,b.regulator)):
        np.testing.assert_allclose(old[2:],new[2:],atol=3e-15,rtol=3e-12)
        np.testing.assert_allclose(new[1],old[1]+np.einsum("j,jab->ab",beta,old[2:]),atol=4e-15,rtol=3e-12)


@pytest.mark.parametrize("delta",[(0.,0.),(1e-10,-2e-10),(.13,.27)])
def test_retained_bare_limit_recovers_canonical_transverse_contact_all_helicities(delta):
    for route in (emission_routes(9)[0],emission_routes(9)[-1]):
        g = cubic_current_geometry(route,transfer_GeV=delta,mean_total_transverse_GeV=(.03,.07))
        orbital = contact_route_orbital(route,1,2,.4,tuple(delta))
        for hs in product((-1,1),repeat=3):
            a = route_current(route,hs,delta,reg=regulator(1e7),ci=1,co=2,order=2)
            spin = cubic_canonical_current(g["modes"],g["physical_field_shifts"],
                channel="q_to_qg",helicities=hs,total_k2=9,total_pplus_GeV=1.3,mass_GeV=.35,flavor_charge=.7)
            np.testing.assert_allclose(a.total[2:],spin[2:,None,None]*orbital,atol=4e-15)
            np.testing.assert_allclose(a.regulator,0.,atol=3e-15)


@pytest.mark.parametrize("delta",[(0.,0.),(1e-10,-2e-10),(.13,.27)])
def test_regulated_small_transfer_has_no_inverse_transfer_singularity(delta):
    a = route_current(ROUTES[0],delta=delta,ci=1,co=2,order=16)
    assert np.all(np.isfinite(a.total)) and np.linalg.norm(a.regulator)>1e-4
    ward = endpoint_ward_reference(ROUTES[0],(1,-1,1),delta,(.03,.07),regulator(),1,2)
    np.testing.assert_allclose(np.asarray(delta)@a.total[2:].reshape(2,-1),ward.reshape(-1),atol=3e-15)


def test_envelope_does_not_delete_spectators_or_gauge_longitudinal_boundary():
    small = replace(regulator(),boson_modes=1,fermion_modes=2)
    allowed = next(r for r in emission_routes(9) if r.outgoing[3]==2 and r.incoming[r.emitter]==3 and max(r.incoming)==5)
    assert np.linalg.norm(route_current(allowed,reg=small).total)>0.
    forbidden = next(r for r in emission_routes(9) if r.incoming[r.emitter]>3)
    np.testing.assert_array_equal(route_current(forbidden,reg=small).total,0.)


def test_auxiliary_or_unspecified_current_prescription_is_rejected():
    with pytest.raises(ValueError,match="actual retained"):
        cubic_current_geometry(object(),transfer_GeV=(.1,0.),mean_total_transverse_GeV=(0.,0.))
    with pytest.raises(ValueError,match="explicit common cluster"):
        cluster_cubic_current_orbitals(ROUTES[0],0,0,.4,.35,(1,1,1),regulator=regulator(),
            total_pplus_GeV=1.3,transfer_GeV=(.1,0.),mean_total_transverse_GeV=(0.,0.),
            flavor_charge=.7,prescription="Ward-quotient",quadrature_order=8)


@pytest.mark.parametrize("route",ROUTES)
def test_all_cubic_bare_minus_limits_against_independent_ladder_moments(route):
    from deuteron_wigner.quantum.m2_one_gluon_minus_contact import one_gluon_minus_orbital
    from deuteron_wigner.quantum.m2_constrained_fermion_current import discrete_gluon_field_factor
    from deuteron_wigner.quantum.m2_coupled_longitudinal_jacobi import jacobi_frame,emission_coordinate_map,EmissionRoute
    from deuteron_wigner.quantum.m2_excluded_fock_kinematics import frame as five_frame,split_frame,SplitRoute
    from deuteron_wigner.quantum.m2_six_body_kinematics import frame as six_frame
    from deuteron_wigner.quantum.m2_six_body_vertices import split_frame as six_split
    if isinstance(route,EmissionRoute):
        fi,fo,parent,children = jacobi_frame(route.incoming),jacobi_frame(route.outgoing),route.emitter,(route.emitter,3)
        rotation,channel = emission_coordinate_map(route),"q_to_qg"
    elif isinstance(route,SplitRoute):
        fi,fo,parent,children = five_frame("qqqg",route.incoming),five_frame(route.sector,route.outgoing),route.parent,route.daughters
        rotation,channel = split_frame(route),route.channel
    else:
        fi,fo,parent,children = five_frame(route.source_sector,route.incoming),six_frame(route.sector,route.outgoing),route.parent,route.daughters
        rotation,channel = six_split(route),route.channel
    if channel=="g_to_gg":
        for hs in product((-1,1),repeat=3):
            np.testing.assert_array_equal(route_current(route,hs,reg=regulator(1e7),ci=1,co=2,order=2).total,0.)
        return
    x,y = fo.fractions[list(children)]
    mode = route.incoming[parent] if channel=="g_to_pair" else route.outgoing[children[1]]
    charge = -.7 if channel=="anti_to_antig" else .7
    for hg in (-1,1):
        old = one_gluon_minus_orbital(fi,parent,x,y,rotation,1,2,(.17,-.11),channel=channel,
            b_GeV=.4,total_pplus_GeV=1.3,mass_GeV=.35,gluon_helicity=hg,mean_total_transverse=(.03,.07))
        for hf,hi in product((-1,1),repeat=2):
            hs = (hf,hi,hg) if channel=="g_to_pair" else (hf,hg,hi)
            a = route_current(route,hs,reg=regulator(1e7),ci=1,co=2,order=2)
            expected = charge*discrete_gluon_field_factor(mode)*old[(hf+1)//2,(hi+1)//2]
            np.testing.assert_allclose(a.total[1],expected,atol=5e-15)
