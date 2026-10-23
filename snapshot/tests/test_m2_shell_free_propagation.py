import numpy as np
import pytest
from scipy import sparse
from scipy.special import roots_genlaguerre
from math import factorial

from deuteron_wigner.quantum.m2_excluded_fock_action import IntermediateSpace
from deuteron_wigner.quantum.m2_six_body_space import SixBodySpace
from deuteron_wigner.quantum.m2_shell_free_propagation import ShellFreePropagation
from deuteron_wigner.quantum.m2_coupled_pilot import free_orbital
from deuteron_wigner.quantum.m2_coupled_longitudinal_jacobi import oscillator_shell,oscillator_rotation


SECTORS=["qqqgg","qqqq-qbar","qqqggg","qqqq-qbar-g"]

def make(sector,cutoff,target="PROTON"):
    cls=IntermediateSpace if sector in SECTORS[:2] else SixBodySpace
    s=cls(sector,9,cutoff,target,1)
    return ShellFreePropagation(s,b_GeV=.4,quark_mass_GeV=.35,gluon_mass_GeV=.12)

def columns(s,n=2):
    rng=np.random.default_rng(673)
    return (rng.normal(size=(s.dimension,n))+1j*rng.normal(size=(s.dimension,n)))/np.sqrt(s.dimension)


@pytest.mark.parametrize("sector",SECTORS)
@pytest.mark.parametrize("cutoff",[0,1])
def test_previous_diagonal_shells_are_unchanged(sector,cutoff):
    f=make(sector,cutoff);s=f.space;v=columns(s)
    en=s.free_energies(.4,.35,.12)
    np.testing.assert_allclose(f.apply(v),en[:,None]*v,atol=3e-16)
    np.testing.assert_allclose(f.resolve(1.4,v),v/(1.4-en[:,None]),atol=3e-16)


@pytest.mark.parametrize("sector",SECTORS)
def test_second_shell_full_sparse_action_resolvent_and_energy_identity(sector):
    f=make(sector,2);s=f.space;d=4 if sector in SECTORS[:2] else 5
    v=columns(s);orb=free_orbital(d,2,.4)
    # Independent full tensor product and exact packed restriction, not the
    # production angular-block decomposition.
    h=sparse.kron(sparse.eye(len(s.longitudinal)),orb,format="csr")
    h+=sparse.kron(sparse.diags(f.mass_offsets),sparse.eye(len(s.orbitals)),format="csr")
    h=sparse.kron(h,sparse.eye(s.ncolor*s.nspin),format="csr")[s.rows][:,s.rows]
    np.testing.assert_allclose(f.apply(v),h@v,atol=4e-16)
    a=f.resolve(1.4,v)
    np.testing.assert_allclose(1.4*a-h@a,v,atol=8e-16)
    b=f.resolve(1.6,v)
    np.testing.assert_allclose(a-b,.2*f.resolve(1.6,a),atol=8e-16)
    slope=(f.resolve(1.4+1e-5,v)-f.resolve(1.4-1e-5,v))/2e-5
    np.testing.assert_allclose(slope,-f.resolve(1.4,a),atol=3e-11)
    with pytest.raises(ValueError,match="non-diagonal"):
        s.free_energies(.4,.35,.12)


@pytest.mark.parametrize("sector",SECTORS)
def test_free_resolvent_commutes_with_species_permutations_at_shell_two(sector):
    f=make(sector,2);s=f.space;v=columns(s,1)
    if sector=="qqqgg":p=(0,1,2,4,3)
    elif sector=="qqqq-qbar":p=(3,1,2,0,4)
    elif sector=="qqqggg":p=(0,1,2,5,3,4)
    else:p=(3,1,2,0,4,5)
    np.testing.assert_allclose(s.permute(f.resolve(1.4,v),p),
                               f.resolve(1.4,s.permute(v,p)),atol=2e-15)


@pytest.mark.parametrize("d",[4,5])
@pytest.mark.parametrize("cutoff",[2,4])
def test_ground_resolvent_matches_independent_radial_laguerre_quadrature(d,cutoff):
    t=free_orbital(d,cutoff,.4).toarray()
    # Ground radial measure y^(d-1) exp(-y)/Gamma(d), y=Q^2/b^2.
    # Its Gauss-Laguerre Jacobi matrix gives the projected free resolvent.
    nodes,weights=roots_genlaguerre(cutoff//2+1,d-1)
    e=-.7
    ground=np.zeros(len(t));ground[0]=1
    actual=np.linalg.solve(e*np.eye(len(t))-t,ground)[0]
    expected=np.sum(weights/(e-.4**2*nodes))/factorial(d-1)
    assert actual==pytest.approx(expected,abs=8e-15)
    assert abs(actual-1/(e-.4**2*d))>1e-3


@pytest.mark.parametrize("d",[4,5])
def test_free_shell_covariance_under_nontrivial_orthogonal_jacobi_change(d):
    rng=np.random.default_rng(733)
    r,_=np.linalg.qr(rng.normal(size=(d,d)))
    u=oscillator_rotation(r,2);t=free_orbital(d,2,.4).toarray()
    np.testing.assert_allclose(u@t,t@u,atol=2e-15)


def test_poles_invalid_inputs_and_incomplete_shells_are_not_repaired():
    f=make("qqqgg",2);s=f.space;v=columns(s,1)
    for e in [np.nan,np.inf,f.minimum_energy_GeV2]:
        with pytest.raises(ValueError):f.resolve(e,v)
    for a in [np.zeros(s.dimension),np.full((s.dimension,1),np.nan)]:
        with pytest.raises(ValueError):f.apply(a)
    for value in [-1,True,.5]:
        with pytest.raises(ValueError):IntermediateSpace("qqqgg",9,value,"PROTON",1)
        with pytest.raises(ValueError):SixBodySpace("qqqggg",9,value,"PROTON",1)
    s.orbitals=s.orbitals[:-1]
    with pytest.raises(ValueError,match="complete"):
        ShellFreePropagation(s,b_GeV=.4,quark_mass_GeV=.35,gluon_mass_GeV=0)


@pytest.mark.parametrize("sector,p",[
    ("qqqgg",(2,1,0,4,3)),("qqqq-qbar",(3,0,2,1,4))])
@pytest.mark.parametrize("cutoff",[1,2])
def test_five_body_exact_block_permutation_matches_full_tensor(sector,p,cutoff):
    from deuteron_wigner.quantum.m2_excluded_fock_kinematics import color_permutation,permutation_frame
    s=IntermediateSpace(sector,9,cutoff,"PROTON",1);v=columns(s,2)
    c=color_permutation(sector,p)
    spin_lookup=np.arange(s.nspin).reshape(s.spin_dims).transpose(p).ravel()
    long_lookup={state:i for i,state in enumerate(s.longitudinal)}
    expected=np.zeros_like(v)
    for li,state in enumerate(s.longitudinal):
        target,r=permutation_frame(sector,state,p)
        rows=np.flatnonzero(s.coordinates[:,0]==li)
        o,col,spin=s.coordinates[rows,1:].T
        full=np.zeros((len(s.orbitals),s.ncolor,s.nspin,v.shape[1]),complex)
        full[o,col,spin]=v[rows]
        full=np.einsum("oi,ab,ibsk->oask",oscillator_rotation(r,cutoff),c,full,optimize=True)
        s.add(expected,long_lookup[target],np.arange(s.nspin),full[:,:,spin_lookup])
    np.testing.assert_allclose(s.permute(v,p),expected,atol=2e-16)


@pytest.mark.parametrize("sector",["qqqgg","qqqq-qbar"])
def test_third_shell_raw_creation_ladders_verify_new_angular_blocks(sector):
    f=make(sector,3);s=f.space;v=columns(s,1);d=4
    orb_lookup={o:i for i,o in enumerate(s.orbitals)}
    diagonal=f.mass_offsets[s.coordinates[:,0]]+.4**2*np.array(
        [d+sum(o) for o in s.orbitals])[s.coordinates[:,1]]
    # Assemble the action directly from occupation ladders and raw row labels,
    # without the production orbital matrix or Lz block grouping.
    def action(a):
        out=diagonal[:,None]*a
        for oi,o in enumerate(s.orbitals):
            if sum(o)+2>3:continue
            rows=np.flatnonzero(s.coordinates[:,1]==oi)
            li,_,color,spin=s.coordinates[rows].T
            for k in range(d):
                raised=list(o);raised[k]+=1;raised[d+k]+=1
                target_orb=orb_lookup[tuple(raised)]
                raw=((li*len(s.orbitals)+target_orb)*s.ncolor+color)*s.nspin+spin
                target=s.lookup[raw]
                assert np.all(target>=0)
                coefficient=.4**2*np.sqrt((o[k]+1)*(o[d+k]+1))
                out[rows]+=coefficient*a[target]
                out[target]+=coefficient*a[rows]
        return out
    np.testing.assert_allclose(f.apply(v),action(v),atol=3e-16)
    wave=f.resolve(1.4,v)
    np.testing.assert_allclose(1.4*wave-action(wave),v,atol=5e-16)


def test_reference_whitening_preserves_near_degenerate_norm_orthogonality():
    from deuteron_wigner.quantum.m2_matched_h2_reduction import _inverse_sqrt
    rng=np.random.default_rng(691)
    b=rng.normal(size=(9,37))+1j*rng.normal(size=(9,37))
    b/=np.linalg.norm(b)
    for scale in (0.,1e-10,1e-6,.1,2.):
        n=np.eye(37)+scale*b.conj().T@b
        w=_inverse_sqrt(n)
        np.testing.assert_allclose(w,w.conj().T,atol=6e-16)
        np.testing.assert_allclose(w@n@w,np.eye(37),atol=5e-15,rtol=0)


def test_reference_whitening_does_not_clip_nonpositive_norms():
    from deuteron_wigner.quantum.m2_matched_h2_reduction import _inverse_sqrt
    for v in (0.,-.01):
        with pytest.raises(ArithmeticError,match="positive"):
            _inverse_sqrt(np.diag([1.,1.,v]))
