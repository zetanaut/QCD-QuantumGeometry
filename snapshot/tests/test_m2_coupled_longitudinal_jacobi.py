"""Independent tests of coupled finite-cell/Jacobi kinematics and symmetry."""
from collections import Counter
from itertools import product
from math import comb, pi, sqrt

import numpy as np
import pytest
import sympy as sp
from scipy.special import roots_hermite

from deuteron_wigner.quantum.m2_coupled_longitudinal_jacobi import (
    S3, EmissionRoute, emission_coordinate_map, emission_incidence, emission_routes,
    intrinsic_ground_overlap, intrinsic_ground_precision, jacobi_frame,
    longitudinal_basis, longitudinal_orbital_action, oscillator_rotation,
    oscillator_shell, permutation_jacobi_map, total_sign_project, tree_change,
)
from deuteron_wigner.quantum.m2_c117_pair_current import color_permutation_matrix
from deuteron_wigner.quantum.m2_emitter_tree_recoupling import mass_weighted_tree_matrix
from deuteron_wigner.quantum.m2_spin_flavor_s3_completion import (
    SPIN_FLAVOR_BASIS, build_spin_flavor_s3_algebra,
    build_global_ground_nucleon_spin_flavor_state,
    spin_flavor_permutation_representation,
)


def maxabs(value):
    a = value.data if hasattr(value, "tocsr") else np.asarray(value)
    return float(np.max(np.abs(a))) if np.size(a) else 0.0


@pytest.mark.parametrize("total", [3, 5, 9, 13, 27])
def test_integer_census_and_emission_routes(total):
    q, g = (longitudinal_basis(s, total) for s in ("qqq", "qqqg"))
    n = (total-3)//2
    assert len(q) == comb(n+2, 2)
    assert len(g) == (comb(n+2, 3) if n else 0)
    assert q == tuple(sorted(set(q))) and g == tuple(sorted(set(g)))
    # Independent brute-force finite compositions, not the emission constructor.
    expected = {t for t in product(range(1, total, 2), repeat=3)
                if sum(t) == total}
    assert set(q) == expected
    expected_g = {t+(k,) for t in product(range(1, total, 2), repeat=3)
                  for k in range(2, total, 2) if sum(t)+k == total}
    assert set(g) == expected_g
    routes = emission_routes(total)
    assert len(routes) == len(q)*n == 3*len(g)
    if routes:
        assert set(Counter(r.outgoing for r in routes).values()) == {3}
        assert set(Counter(r.incoming for r in routes).values()) == {n}
    for r in routes:
        assert sum(r.incoming) == sum(r.outgoing) == total
        for i in range(3):
            assert r.incoming[i] == r.outgoing[i] + (r.outgoing[3] if i == r.emitter else 0)
    incidence = sum((emission_incidence(total, i) for i in range(3)))
    assert incidence.nnz == len(routes)
    if len(g):
        assert np.all(np.asarray(incidence.sum(axis=1)) == 3)
    rng = np.random.default_rng(total)
    a, b = rng.normal(size=len(q)), rng.normal(size=len(g))
    assert np.vdot(b, incidence@a) == pytest.approx(np.vdot(incidence.T@b, a))


@pytest.mark.parametrize("sector", ["qqq", "qqqg"])
def test_all_k27_jacobi_frames_invert_metric_cm_and_measure(sector):
    rng = np.random.default_rng(2711)
    for modes in longitudinal_basis(sector, 27):
        f = jacobi_frame(modes)
        n = len(modes)
        x, a, b = f.fractions, f.forward, f.inverse
        np.testing.assert_allclose(a@b, np.eye(n), atol=2e-14)
        np.testing.assert_allclose(b@a, np.eye(n), atol=2e-14)
        np.testing.assert_allclose(a@np.diag(x)@a.T, np.eye(n), atol=2e-14)
        np.testing.assert_allclose(a.T@a, np.diag(1/x), atol=3e-14)
        p = rng.normal(size=(n, 2))
        q = f.to_jacobi(p)
        np.testing.assert_allclose(q[-1], p.sum(axis=0), atol=2e-14)
        np.testing.assert_allclose(f.from_jacobi(q), p, atol=2e-14)
        masses = rng.uniform(.1, 1, size=n)
        direct = np.sum((masses**2+np.sum(p*p, axis=1))/x) - np.sum(p.sum(axis=0)**2)
        reduced = np.sum(masses**2/x)+np.sum(q[:-1]**2)
        assert direct == pytest.approx(reduced, abs=3e-12)
        assert abs(np.linalg.det(a))**-2 == pytest.approx(f.transverse_measure_factor)
        # Intrinsic delta-function measure: choose first n-1 momenta independently.
        c = np.vstack((np.eye(n-1), -np.ones(n-1)))
        assert abs(np.linalg.det(a[:-1]@c))**-2 == pytest.approx(np.prod(x))


@pytest.mark.parametrize("modes", [(1,9,17), (9,9,9), (1,1,1,24), (3,7,9,8)])
def test_independent_exact_intrinsic_measure_determinant(modes):
    n = len(modes)
    c = sp.eye(n-1).col_join(-sp.ones(1,n-1))
    x = [sp.Rational(k,sum(modes)) for k in modes]
    g = c.T*sp.diag(*(1/z for z in x))*c
    assert g.det() == 1/sp.prod(x)
    np.testing.assert_allclose(
        intrinsic_ground_precision(modes,.4), np.asarray(g,dtype=float)/.16, atol=2e-13)


@pytest.mark.parametrize("modes,trees", [
    ((1,9,17), [(0,(1,2)), ((0,1),2), (1,(2,0))]),
    ((1,3,9,14), [((0,3),(1,2)), (0,(1,(2,3))), ((2,0),(3,1))]),
])
def test_arbitrary_binary_tree_changes_preserve_metric_and_compose(modes,trees):
    a,b,c = trees
    ab,bc,ac = tree_change(modes,a,b),tree_change(modes,b,c),tree_change(modes,a,c)
    np.testing.assert_allclose(bc@ab, ac, atol=2e-14)
    np.testing.assert_allclose(ab.T@ab, np.eye(len(modes)-1), atol=2e-14)
    assert abs(np.linalg.det(ab)) == pytest.approx(1)
    np.testing.assert_allclose(jacobi_frame(modes,a).forward[-1], np.ones(len(modes)))


def test_equal_share_exact_m2k_ground_metric_and_tree_maps():
    ref = (0,(1,2))
    for p in S3:
        t = (p[0],(p[1],p[2]))
        np.testing.assert_allclose(tree_change((9,9,9),ref,t),
                                   mass_weighted_tree_matrix(p), atol=2e-14)
    f = jacobi_frame((9,9,9))
    k = np.array([[.1,.2],[.3,-.5],[-.4,.3]])
    expected = np.vstack((sqrt(1.5)*(k[1]-k[2]), 3*k[0]/sqrt(2), [0,0]))
    np.testing.assert_allclose(f.to_jacobi(k), expected, atol=2e-14)
    # b_Qrho=b_Qlambda=.4 means b_rho=sqrt(2/3)*.4 in old local q_rho.
    assert .4/sqrt(1.5) == pytest.approx(sqrt(2/3)*.4)


@pytest.mark.parametrize("sector", ["qqq","qqqg"])
def test_all_k27_fraction_permutation_cocycle(sector):
    for state in longitudinal_basis(sector,27):
        for p in S3:
            out, a = permutation_jacobi_map(state,p)
            np.testing.assert_allclose(a.T@a,np.eye(len(state)-1),atol=2e-14)
            for q in S3:
                final,b = permutation_jacobi_map(out,q)
                composed,c = permutation_jacobi_map(state,tuple(p[i] for i in q))
                assert final == composed
                np.testing.assert_allclose(b@a,c,atol=2e-14)


def test_all_k27_emission_maps_preserve_parent_spectators_and_free_metric():
    rng = np.random.default_rng(53117)
    for route in emission_routes(27):
        o = emission_coordinate_map(route)
        np.testing.assert_allclose(o@o.T,np.eye(3),atol=2e-14)
        out_frame, parent_frame = jacobi_frame(route.outgoing), jacobi_frame(route.incoming)
        q = rng.normal(size=(4,2))
        physical = out_frame.from_jacobi(q)
        parent = physical[:3].copy()
        parent[route.emitter] += physical[3]
        reduced = parent_frame.to_jacobi(parent)
        np.testing.assert_allclose((o@q[:-1])[:2],reduced[:-1],atol=2e-14)
        np.testing.assert_allclose(parent.sum(axis=0),physical.sum(axis=0),atol=2e-14)
        # Check split directly from daughter relative fractions and parent weight.
        i=route.emitter
        x=out_frame.fractions
        xa=x[i]+x[3]
        zi,zg=x[i]/xa,x[3]/xa
        split=(zg*physical[i]-zi*physical[3])/sqrt(xa*zi*zg)
        np.testing.assert_allclose((o@q[:-1])[-1],split,atol=2e-14)
        assert np.sum(q[:-1]**2) == pytest.approx(np.sum((o@q[:-1])**2))


@pytest.mark.parametrize("d", [2,3])
@pytest.mark.parametrize("cutoff", [0,1,2])
def test_complete_oscillator_shell_representation_and_lz(d,cutoff):
    states=oscillator_shell(d,cutoff)
    assert len(states)==comb(cutoff+2*d,2*d)
    rng=np.random.default_rng(200+d)
    a=np.linalg.qr(rng.normal(size=(d,d)))[0]
    b=np.linalg.qr(rng.normal(size=(d,d)))[0]
    ra,rb=oscillator_rotation(a,cutoff),oscillator_rotation(b,cutoff)
    np.testing.assert_allclose(ra.T@ra,np.eye(len(states)),atol=3e-14)
    np.testing.assert_allclose(rb@ra,oscillator_rotation(b@a,cutoff),atol=3e-14)
    angular=np.array([sum(s[:d])-sum(s[d:]) for s in states])
    assert maxabs(ra*(angular[:,None]-angular[None,:]))==0
    for j,s in enumerate(states):
        if sum(s)==2 and max(s)==2:
            source=s.index(2)
            for i,t in enumerate(states):
                # Independent normalized two-boson tensor coefficients.
                occupied=[k for k,v in enumerate(t) for _ in range(v)]
                expected=0.
                if len(occupied)==2:
                    o=np.kron(np.eye(2),a)
                    u,v=occupied
                    expected=o[u,source]*o[v,source]*(1 if u==v else sqrt(2))
                assert ra[i,j]==pytest.approx(expected,abs=3e-14)


@pytest.mark.parametrize("sector", ["qqq","qqqg"])
def test_joint_longitudinal_orbital_action_not_constant_kronecker(sector):
    total,cutoff=9,2
    actions={p:longitudinal_orbital_action(sector,total,cutoff,p) for p in S3}
    identity=actions[(0,1,2)]
    for p in S3:
        assert maxabs(actions[p].T@actions[p]-identity)<4e-14
        for q in S3:
            assert maxabs(actions[q]@actions[p]-actions[tuple(p[i] for i in q)])<4e-14
    tuples=longitudinal_basis(sector,total)
    rotations=[permutation_jacobi_map(k,(1,2,0))[1] for k in tuples]
    assert max(np.linalg.norm(o-rotations[0]) for o in rotations)>.1


@pytest.mark.parametrize("sector", ["qqq","qqqg"])
@pytest.mark.parametrize("cutoff", [0,1])
def test_joint_total_sign_projector_is_hermitian_idempotent_and_sign(sector,cutoff):
    n=len(longitudinal_basis(sector,7))*len(oscillator_shell(2 if sector=="qqq" else 3,cutoff))
    shape=(n,1 if sector=="qqq" else 2,64)
    rng=np.random.default_rng(771)
    x=rng.normal(size=shape)+1j*rng.normal(size=shape)
    y=rng.normal(size=shape)+1j*rng.normal(size=shape)
    px,py=(total_sign_project(v,sector,7,cutoff) for v in (x,y))
    np.testing.assert_allclose(total_sign_project(px,sector,7,cutoff),px,atol=4e-14)
    assert np.vdot(x,py)==pytest.approx(np.vdot(px,y),abs=2e-12)
    assert np.linalg.norm(px)>1
    for p in S3:
        sign=(-1)**sum(p[i]>p[j] for i in range(3) for j in range(i+1,3))
        orbital=longitudinal_orbital_action(sector,7,cutoff,p)
        z=(orbital@px.reshape(n,-1)).reshape(shape)
        z=np.einsum("ab,cs,nbs->nac",color_permutation_matrix(sector,p),
                    spin_flavor_permutation_representation(p),z,optimize=True)
        np.testing.assert_allclose(z,sign*px,atol=4e-14)


@pytest.mark.parametrize("sector", ["qqq","qqqg"])
@pytest.mark.parametrize("target", ["PROTON","NEUTRON"])
@pytest.mark.parametrize("jz2", [-1,1])
def test_projector_preserves_target_jz_isospin_and_baryon(sector,target,jz2):
    d=2 if sector=="qqq" else 3
    orbitals=oscillator_shell(d,1)
    count=len(longitudinal_basis(sector,7))
    lz=np.tile([sum(s[:d])-sum(s[d:]) for s in orbitals],count)
    sf=np.asarray(SPIN_FLAVOR_BASIS)
    iz2=np.sum(np.where(sf<2,1,-1),axis=1)
    sz2=np.sum(2*(sf%2)-1,axis=1)
    target_iz=1 if target=="PROTON" else -1
    iso=build_spin_flavor_s3_algebra().isospin_squared
    p_half=(3.75*np.eye(64)-iso)/3
    for hg in ([0] if sector=="qqq" else [-1,1]):
        mask=(2*lz[:,None]+sz2[None,:]+2*hg==jz2)&(iz2[None,:]==target_iz)
        rng=np.random.default_rng(343)
        z=rng.normal(size=(len(lz),1 if sector=="qqq" else 2,64)).astype(complex)
        z*=mask[:,None,:]
        z=np.einsum("ab,ncb->nca",p_half,z)
        out=total_sign_project(z,sector,7,1)
        assert np.linalg.norm(out)>1e-5
        assert maxabs(out*(~mask[:,None,:]))<2e-14
        np.testing.assert_allclose(np.einsum("ab,ncb->nca",iso,out),.75*out,atol=3e-14)
        # All retained states have exactly three quarks; normalized baryon charge=1.
        charge2=1+iz2  # 2*Q = 1 + 2*Iz for u/d qqq and a neutral gluon
        expected=2 if target=="PROTON" else 0
        assert maxabs(out*(charge2[None,None,:]-expected))<3e-14


@pytest.mark.parametrize("target",["PROTON","NEUTRON"])
@pytest.mark.parametrize("helicity",[-1,1])
def test_equal_share_old_nucleon_ground_is_retained(target,helicity):
    basis=longitudinal_basis("qqq",27)
    state=build_global_ground_nucleon_spin_flavor_state(target,helicity)
    # Existing state object carries its normalized spin-flavor amplitude.
    amplitude=state.vector
    value=np.zeros((len(basis),1,64),complex)
    value[basis.index((9,9,9)),0]=amplitude
    np.testing.assert_allclose(total_sign_project(value,"qqq",27,0),value,atol=3e-14)


@pytest.mark.parametrize("incoming,outgoing", [
    ((9,9,9),(1,9,17)), ((3,7,9,8),(5,5,9,8)),
])
def test_ground_cross_fraction_overlap_against_separate_cartesian_quadrature(incoming,outgoing):
    a,b=intrinsic_ground_precision(incoming,.4),intrinsic_ground_precision(outgoing,.4)
    # Each transverse Cartesian component integrated separately in physical kappa.
    # Weight exp(-sum z_i^2); no use of the Jacobi determinant formula here.
    m=(a+b)/2
    scale=1/np.sqrt(np.diag(m))
    nodes,w=roots_hermite(60)
    mesh=np.meshgrid(*([nodes]*len(a)),indexing="ij")
    z=np.stack([v.ravel() for v in mesh],axis=1)
    weights=np.prod(np.stack(np.meshgrid(*([w]*len(a)),indexing="ij")),axis=0).ravel()
    k=z*scale
    exponent=-np.einsum("bi,ij,bj->b",k,m,k)+np.sum(z*z,axis=1)
    integral=np.prod(scale)*np.sum(weights*np.exp(exponent))
    cartesian=integral*(np.linalg.det(a)*np.linalg.det(b))**.25/pi**(len(a)/2)
    answer=intrinsic_ground_overlap(incoming,outgoing,.4)
    assert 0<answer<1
    assert answer==pytest.approx(cartesian**2,rel=2e-10,abs=2e-10)
    assert answer==pytest.approx(intrinsic_ground_overlap(outgoing,incoming,.4))
    assert answer==pytest.approx(intrinsic_ground_overlap(incoming,outgoing,.8))
    assert intrinsic_ground_overlap(incoming,incoming,.4)==pytest.approx(1)


@pytest.mark.parametrize("call", [
    lambda:longitudinal_basis("qqq",True),
    lambda:longitudinal_basis("qqq",9.0),
    lambda:longitudinal_basis("qqq",8),
    lambda:longitudinal_basis("qq",9),
    lambda:jacobi_frame((1,2,6)),
    lambda:jacobi_frame((1,1,1,3)),
    lambda:jacobi_frame((1,3,5),(0,(1,1))),
    lambda:jacobi_frame((1,3,5),(True,(1,2))),
    lambda:oscillator_shell(2,-1),
    lambda:oscillator_shell(2,1.5),
    lambda:oscillator_rotation(np.diag([2.,1.]),2),
    lambda:intrinsic_ground_precision((9,9,9),0),
    lambda:emission_coordinate_map(EmissionRoute((9,9,9),0,(5,7,9,6))),
    lambda:permutation_jacobi_map((9,9,9),(0,0,1)),
])
def test_invalid_or_inequivalent_domains_rejected(call):
    with pytest.raises((ValueError,TypeError)):
        call()


@pytest.mark.parametrize("sector", ["qqq","qqqg"])
def test_weight_rank_against_independent_dense_projector(sector):
    from deuteron_wigner.quantum.m2_coupled_longitudinal_jacobi import total_sign_weight_rank
    total,cutoff=5,0
    colors=1 if sector=="qqq" else 2
    n=len(longitudinal_basis(sector,total))
    dimension=n*colors*64
    p=np.zeros((dimension,dimension),complex)
    for perm in S3:
        sign=(-1)**sum(perm[i]>perm[j] for i in range(3) for j in range(i+1,3))
        a=longitudinal_orbital_action(sector,total,cutoff,perm).toarray()
        c=color_permutation_matrix(sector,perm)
        s=spin_flavor_permutation_representation(perm)
        p+=sign*np.kron(np.kron(a,c),s)/6
    sf=np.asarray(SPIN_FLAVOR_BASIS)
    iz2=np.where(sf<2,1,-1).sum(axis=1)
    sz2=(2*(sf%2)-1).sum(axis=1)
    iso=build_spin_flavor_s3_algebra().isospin_squared
    ihalf=(3.75*np.eye(64)-iso)/3
    for target in ("PROTON","NEUTRON"):
        for jz2 in (-7,-5,-3,-1,1,3,5,7):
            for hg in ([0] if sector=="qqq" else [-1,1]):
                mask=(iz2==(1 if target=="PROTON" else -1)) & (sz2+2*hg==jz2)
                weight=np.kron(np.eye(n*colors),mask[:,None]*ihalf)
                projected=weight@p
                np.testing.assert_allclose(projected@projected,projected,atol=3e-14)
                eig=np.linalg.eigvalsh(projected)
                count=np.count_nonzero(eig>.5)
                assert total_sign_weight_rank(sector,total,cutoff,target,jz2,hg)==count


@pytest.mark.parametrize("jz", [0, 2, -2, .5, True])
def test_weight_rank_requires_half_integer_not_selected_spin(jz):
    from deuteron_wigner.quantum.m2_coupled_longitudinal_jacobi import total_sign_weight_rank
    with pytest.raises((ValueError, TypeError)):
        total_sign_weight_rank("qqq", 5, 0, "PROTON", jz, 0)


def test_empty_neighboring_weight_is_an_empty_isometry():
    from deuteron_wigner.quantum.m2_coupled_pilot import symmetry_basis
    assert symmetry_basis("qqq", 5, 0, "PROTON", 9, 0).shape == (192, 0)
