"""Direct algebra, coordinate/Fourier quadrature and Fock tests of the new qq slice."""
from itertools import permutations
from math import pi, sqrt

import numpy as np
import pytest
from scipy.special import roots_laguerre

from deuteron_wigner.bridge.c403_c117_i2_numerical_primitive.spatial import HOMode
from deuteron_wigner.bridge.c404_c117_i2_longitudinal_color_primitive.color_spin import (
    color_spin_validation,
)
from deuteron_wigner.bridge.c406_c117_i2_gluon_normal_order_descendant.normal_order import (
    c151_canonical_one_gluon_factor, dimensionless_descendant_factor_exact,
)
from deuteron_wigner.bridge.c408_c117_i2_weight_routing_closure.i4_q import q_sector_jqjq_scalar
from deuteron_wigner.bridge.c410_c117_i2_retained_aggregation_boundary.aggregate import SOURCE_COEFFICIENT
from deuteron_wigner.bridge.icurrent.core import current_source_manifest, inverse_partial_plus_squared
from deuteron_wigner.bridge.modes.core import ho_coordinate
from deuteron_wigner.microscopic.h0.color import representation_generators
from deuteron_wigner.quantum.m2_c117_pair_current import (
    QUARK_PAIRS, color_pair_matrix, color_permutation_matrix, four_ho_contact,
    instantaneous_pminus_per_gs2, inverse_i_derivative, ordered_quark_mass_factor,
    ordered_quark_pminus_factor, qq_longitudinal_pair_kernel, qqq_longitudinal_basis,
    qqq_longitudinal_permutation, qqq_fixed_width_ground_pair_mass_per_gs2,
    qqq_fixed_width_ground_mass_per_gs2, source_current_square,
)
from deuteron_wigner.quantum.m2_emitter_tree_recoupling import (
    build_emitter_tree_recoupling_audit,
)
from deuteron_wigner.quantum.m2_spin_flavor_s3_completion import (
    build_global_ground_nucleon_spin_flavor_state,
    spin_flavor_permutation_representation,
)


def maxabs(matrix):
    values = matrix.data if hasattr(matrix, "tocsr") else matrix
    return float(np.max(np.abs(values))) if np.size(values) else 0.0


@pytest.mark.parametrize("length", [0.7, 2.0, 7.3])
def test_ordered_square_against_spectral_derivative_and_coordinate_integral(length):
    rng = np.random.default_rng(11706)
    a = rng.normal(size=(3,3)) + 1j*rng.normal(size=(3,3))
    b = rng.normal(size=(3,3)) + 1j*rng.normal(size=(3,3))
    c = rng.normal(size=(3,3)) + 1j*rng.normal(size=(3,3))
    modes = {0: c+c.conj().T, 1:a, -1:a.conj().T, 3:b, -3:b.conj().T}
    points = -length + (2*length/128)*np.arange(128)
    current = sum(np.exp(-1j*pi*n*points/length)[:,None,None]*j
                  for n,j in modes.items())
    omega = 2*pi*np.fft.fftfreq(128, d=2*length/128)
    eigenvalues = np.zeros_like(omega)
    eigenvalues[omega != 0] = -1/omega[omega != 0]
    acted = np.fft.ifft(eigenvalues[:,None,None]*np.fft.fft(current, axis=0), axis=0)
    assert maxabs(current-current.conj().transpose(0,2,1)) < 1e-13
    assert maxabs(acted+acted.conj().transpose(0,2,1)) < 1e-13
    square = (2*length)*np.mean(acted @ acted, axis=0)
    np.testing.assert_allclose(source_current_square(modes,length), square, rtol=2e-13, atol=2e-13)
    expected = length**3/pi**2 * (a@a.conj().T+a.conj().T@a
                                 +(b@b.conj().T+b.conj().T@b)/9)
    actual = instantaneous_pminus_per_gs2(modes,length)
    np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=2e-13)
    assert np.linalg.eigvalsh(actual).min() > 0
    # The zero-mode coefficient is removed, not assigned a physical zero-mode action.
    np.testing.assert_allclose(actual, instantaneous_pminus_per_gs2(
        {n:j for n,j in modes.items() if n}, length))
    assert inverse_i_derivative(0,length) == 0


def test_adjoint_description_and_positive_kernel_scope_are_distinct():
    record = current_source_manifest()
    assert "R=Q0(i partial^+)^-1Q0 is Hermitian" in record["hermitian_rule"]
    assert "pointwise Fock anti-Hermitian" in record["hermitian_rule"]
    assert inverse_partial_plus_squared()["ordered_square_sign_relative_to_positive_kernel"] == -1
    eigenvalues = np.array([inverse_i_derivative(n,2) for n in [-2,-1,1,2]])
    r = np.diag(eigenvalues)
    assert maxabs(r-r.conj().T) == 0
    d_inverse = 1j*r
    assert maxabs(d_inverse+d_inverse.conj().T) == 0


@pytest.mark.parametrize("sector", ["qqq","qqqg"])
def test_complete_color_pair_casimirs_and_s3_covariance(sector):
    size = 1 if sector == "qqq" else 2
    identity = np.eye(size)
    for i in range(3):
        np.testing.assert_allclose(color_pair_matrix(sector,i,i), (4/3)*identity, atol=2e-14)
    qq = sum(color_pair_matrix(sector,*pair) for pair in QUARK_PAIRS)
    np.testing.assert_allclose(qq, (-2 if sector=="qqq" else -.5)*identity, atol=2e-14)
    if sector == "qqqg":
        np.testing.assert_allclose(color_pair_matrix(sector,3,3),3*identity,atol=2e-14)
        np.testing.assert_allclose(sum(color_pair_matrix(sector,i,3) for i in range(3)),
                                   -3*identity,atol=2e-14)
        for i in range(3):
            np.testing.assert_allclose(np.linalg.eigvalsh(color_pair_matrix(sector,i,3)),
                                       [-1.5,-.5],atol=2e-14)
        assert abs(color_pair_matrix(sector,0,3)[0,1]) > .4
    for permutation in permutations(range(3)):
        action = color_permutation_matrix(sector,permutation)
        inverse = tuple(np.argsort(permutation)) + ((3,) if sector=="qqqg" else ())
        np.testing.assert_allclose(action.conj().T@action,identity,atol=2e-14)
        for i in range(3):
            for j in range(3 if sector=="qqq" else 4):
                pair = color_pair_matrix(sector,i,j)
                np.testing.assert_allclose(pair,pair.conj().T,atol=2e-14)
                np.testing.assert_allclose(action@pair@action.conj().T,
                    color_pair_matrix(sector,inverse[i],inverse[j]),atol=2e-14)
    assert color_spin_validation()["pass"]


@pytest.mark.parametrize("bra,ket", [
    (((0,0),(0,0)),((0,0),(0,0))),
    (((1,0),(0,0)),((0,1),(0,-1))),
    (((0,1),(0,-1)),((1,0),(1,0))),
    (((1,2),(1,-1)),((0,0),(2,1))),
    (((0,2),(0,0)),((0,0),(0,0))),
    (((2,-1),(0,2)),((1,1),(1,0))),
])
def test_four_ho_contact_against_independent_coordinate_quadrature(bra,ket):
    b=.4
    bra=tuple(HOMode(*m) for m in bra)
    ket=tuple(HOMode(*m) for m in ket)
    z,w=roots_laguerre(40)
    theta=2*pi*np.arange(64)/64
    r=np.sqrt(z/2)/b
    x,y=r[:,None]*np.cos(theta),r[:,None]*np.sin(theta)
    f=[ho_coordinate(m.n,m.m,x,y,b) for m in (*bra,*ket)]
    integrand=f[0].conj()*f[1].conj()*f[2]*f[3]
    reference=(pi/(2*b*b))*np.sum(w*np.exp(z)*np.mean(integrand,axis=1))
    actual=four_ho_contact(bra,ket,b)
    np.testing.assert_allclose(actual,reference,atol=2e-14,rtol=2e-12)
    np.testing.assert_allclose(actual,np.conj(four_ho_contact(ket,bra,b)),atol=2e-14)
    np.testing.assert_allclose(four_ho_contact(bra,ket,2*b),4*actual,atol=2e-14)


@pytest.mark.parametrize("total", [3,9,27])
def test_longitudinal_pair_actions_and_permutations(total):
    basis=qqq_longitudinal_basis(total)
    n=(total-3)//2
    assert len(basis)==(n+1)*(n+2)//2
    for pair in QUARK_PAIRS:
        matrix=qq_longitudinal_pair_kernel(total,pair)
        assert maxabs(matrix-matrix.getH())==0
        assert np.all(matrix.diagonal()==0)
        spectator=next(i for i in range(3) if i not in pair)
        coo=matrix.tocoo()
        for row,column,value in zip(coo.row,coo.col,coo.data):
            bra,ket=basis[row],basis[column]
            assert bra[spectator]==ket[spectator]
            transfer=(bra[pair[0]]-ket[pair[0]])/2
            assert transfer != 0 and value==1/(transfer**2)
        for p in permutations(range(3)):
            action=qqq_longitudinal_permutation(total,p)
            inv=np.argsort(p)
            other=qq_longitudinal_pair_kernel(total,tuple(inv[i] for i in pair))
            assert maxabs(action@matrix@action.getH()-other)==0


@pytest.mark.parametrize("length", [.25,1.,13.])
def test_canonical_box_factors_and_mass_conversion(length):
    for n in [-3,-1,1,4]:
        # Direct two-current phases use opposite Fourier labels.
        raw=(2*length)/(2*length)**2 * inverse_i_derivative(n,length)*inverse_i_derivative(-n,length)
        np.testing.assert_allclose(ordered_quark_pminus_factor(n,length),-.5*raw,rtol=2e-15)
        np.testing.assert_allclose((pi*27/length)*ordered_quark_pminus_factor(n,length),
                                   ordered_quark_mass_factor(27)/n**2,rtol=2e-15)
    assert ordered_quark_pminus_factor(0,length)==0


def test_existing_quark_shape_has_opposite_sign_before_its_unbound_adapter():
    # This is a source-normalization audit, not a silent alteration of C410 or M2m.
    bqq=q_sector_jqjq_scalar("K9")
    legacy=SOURCE_COEFFICIENT*bqq
    canonical=ordered_quark_mass_factor(9)*bqq
    assert legacy < 0 < canonical
    np.testing.assert_allclose(canonical,(-9/(2*pi))*legacy,rtol=2e-15)


def test_existing_mixed_raw_gluon_normalization_is_not_one_common_scalar():
    ratios=[]
    for bra,ket in [(1,2),(1,4),(2,4)]:
        raw=float(dimensionless_descendant_factor_exact(bra,ket))
        canonical=c151_canonical_one_gluon_factor(bra,ket)
        ratios.append(canonical/raw)
        np.testing.assert_allclose(canonical/raw,1/(2*sqrt(bra*ket)),rtol=2e-15)
    assert np.ptp(ratios) > .1


def test_m2l_ground_contact_and_nonzero_longitudinal_leakage():
    width=.4
    audit=build_emitter_tree_recoupling_audit()
    # q_rho=p_1-p_2 and physical relative momentum=(p_1-p_2)/2.
    contact=(audit.symmetry_adapted_b_rho_GeV/2)**2/pi
    ground=HOMode(0,0)
    np.testing.assert_allclose(contact,four_ho_contact((ground,ground),(ground,ground),
                                                      width/sqrt(3)),rtol=2e-15)
    basis=qqq_longitudinal_basis(27)
    initial=basis.index((9,9,9))
    matrix=qqq_fixed_width_ground_mass_per_gs2(27,width)
    action=matrix[:,initial].toarray().ravel()
    assert action[initial]==0
    assert np.count_nonzero(action)==24
    assert maxabs(matrix-matrix.getH()) < 2e-14
    assert len({tuple(sorted(basis[n])) for n in np.flatnonzero(action)})==4
    prefactor=-(2/3)*(27/(2*pi))*(width*width/(6*pi))
    expected_norm=abs(prefactor)*sqrt(6*sum(1/n**4 for n in range(1,5)))
    np.testing.assert_allclose(np.linalg.norm(action),expected_norm,rtol=3e-14)
    assert expected_norm > .06
    for p in permutations(range(3)):
        permutation=qqq_longitudinal_permutation(27,p)
        assert maxabs(permutation@matrix@permutation.getH()-matrix) < 2e-14
        np.testing.assert_allclose(permutation@action,action,atol=2e-14)
    np.testing.assert_allclose(
        qqq_fixed_width_ground_mass_per_gs2(27,2*width).toarray(),
        4*matrix.toarray(),atol=2e-14)


def _annihilators(count):
    result=[]
    for mode in range(count):
        matrix=np.zeros((1<<count,)*2,complex)
        for state in range(1<<count):
            if (state>>mode)&1:
                matrix[state^(1<<mode),state]=(-1)**bin(state&((1<<mode)-1)).count("1")
        result.append(matrix)
    return result


@pytest.mark.parametrize("target", ["PROTON","NEUTRON"])
@pytest.mark.parametrize("helicity", [-1,1])
def test_leaked_pair_action_retains_total_fermion_sign_and_nucleon_labels(target,helicity):
    sf=build_global_ground_nucleon_spin_flavor_state(target,helicity).vector
    basis=qqq_longitudinal_basis(27)
    vector=qqq_fixed_width_ground_mass_per_gs2(27,.4)[:,basis.index((9,9,9))].toarray().ravel()
    full=np.outer(vector,sf)
    for p in permutations(range(3)):
        sign=(-1)**sum(p[i]>p[j] for i in range(3) for j in range(i+1,3))
        transformed=np.outer(qqq_longitudinal_permutation(27,p)@vector,
                             spin_flavor_permutation_representation(p)@sf)
        transformed*=color_permutation_matrix("qqq",p)[0,0]
        np.testing.assert_allclose(transformed,sign*full,atol=2e-14)


def test_cross_pair_normal_order_sign_against_two_fermion_fock_space():
    # Two longitudinal modes x three colors; no first-quantization exchange sign guessed.
    annihilation=_annihilators(6)
    def second(matrix):
        return sum((matrix[i,j]*annihilation[i].conj().T@annihilation[j]
                    for i in range(6) for j in range(6)),np.zeros((64,64),complex))
    length=1.7
    shift=np.array([[0,0],[1,0]],complex)
    h=np.zeros((64,64),complex)
    self_one=np.zeros((6,6),complex)
    first_pair=np.zeros((36,36),complex)
    for t in representation_generators("3"):
        plus=np.kron(shift,t)
        minus=plus.conj().T
        currents={1:second(plus)/(2*length),-1:second(minus)/(2*length)}
        h+=instantaneous_pminus_per_gs2(currents,length)
        self_one+=length/(4*pi*pi)*(plus@minus+minus@plus)
        first_pair+=length/(2*pi*pi)*(np.kron(plus,minus)+np.kron(minus,plus))
    states=[i for i in range(64) if bin(i).count("1")==2]
    wedge=np.zeros((36,len(states)),complex)
    for column,state in enumerate(states):
        i,j=[k for k in range(6) if (state>>k)&1]
        wedge[6*i+j,column]=1/sqrt(2)
        wedge[6*j+i,column]=-1/sqrt(2)
    actual=(h-second(self_one))[np.ix_(states,states)]
    expected=wedge.conj().T@first_pair@wedge
    assert np.linalg.norm(expected) > .01
    np.testing.assert_allclose(actual,expected,atol=3e-14,rtol=3e-13)


@pytest.mark.parametrize("total", [0,2,4,26,27.5,True])
def test_invalid_total_momentum_rejected(total):
    with pytest.raises((TypeError,ValueError)):
        qqq_longitudinal_basis(total)


def test_invalid_current_inputs_rejected():
    a=np.eye(2,dtype=complex)
    for modes in [{}, {1:a}, {1:a,-1:2*a}, {1:a,-1:np.full((2,2),np.nan)}]:
        with pytest.raises(ValueError):
            source_current_square(modes,1)
    with pytest.raises(TypeError): inverse_i_derivative(.5,1)
    with pytest.raises(ValueError): inverse_i_derivative(1,0)
    with pytest.raises(ValueError): qq_longitudinal_pair_kernel(27,(1,1))
    with pytest.raises(IndexError): color_pair_matrix("qqq",0,3)
