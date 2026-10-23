from math import prod
import numpy as np
import pytest

from deuteron_wigner.quantum.m2_six_body_kinematics import (
    SPECIES,color_decomposition,representation_dimension,longitudinal_states,
    frame,permutation_frame,color_basis,color_emission,color_permutation,
    color_generator_residual,emission_tensor,
)

@pytest.mark.parametrize("sector,count,raw",[("qqqggg",22,13824),("qqqq-qbar-g",8,1944)])
def test_su3_multiplicity_independent_of_source_span(sector,count,raw):
    dec=color_decomposition(SPECIES[sector])
    assert dec[(0,0)]==count
    assert sum(n*representation_dimension(*r) for r,n in dec.items())==raw
    u=color_basis(sector)
    assert u.shape==(raw,count)
    np.testing.assert_allclose(u.conj().T@u,np.eye(count),atol=3e-11)
    assert color_generator_residual(sector)<4e-10
    for i in range(5):
        np.testing.assert_allclose(u@color_emission(sector,i),emission_tensor(sector,i),atol=2e-11)

def test_known_su3_products():
    assert color_decomposition(("q","q","q"))=={(3,0):1,(1,1):2,(0,0):1}
    assert color_decomposition(("g","g"))=={(0,0):1,(1,1):2,(3,0):1,(0,3):1,(2,2):1}
    assert color_decomposition(("q","qbar"))=={(1,1):1,(0,0):1}

@pytest.mark.parametrize("sector,minimum,n9,n11",[("qqqggg",9,1,6),("qqqq-qbar-g",7,6,21)])
def test_longitudinal_modes_and_full_frame(sector,minimum,n9,n11):
    assert longitudinal_states(sector,minimum-2)==()
    assert longitudinal_states(sector,10)==()
    assert len(longitudinal_states(sector,9))==n9
    assert len(longitudinal_states(sector,11))==n11
    for modes in longitudinal_states(sector,11):
        a=frame(sector,modes)
        np.testing.assert_allclose(a.forward@a.inverse,np.eye(6),atol=8e-16)
        metric=np.diag(1/a.fractions)
        np.testing.assert_allclose(a.inverse.T@metric@a.inverse,np.eye(6),atol=8e-16)
        assert sum(modes)==11

@pytest.mark.parametrize("sector,p,q",[
    ("qqqggg",(1,0,2,3,4,5),(0,2,1,3,4,5)),
    ("qqqggg",(0,1,2,4,3,5),(0,1,2,3,5,4)),
    ("qqqq-qbar-g",(1,0,2,3,4,5),(0,2,1,3,4,5)),
])
def test_species_permutation_representation(sector,p,q):
    a,b=color_permutation(sector,p),color_permutation(sector,q)
    np.testing.assert_allclose(a@a,np.eye(len(a)),atol=3e-11)
    np.testing.assert_allclose(a@b@a,b@a@b,atol=3e-11)
    modes=longitudinal_states(sector,11)[1]
    target,r=permutation_frame(sector,modes,p)
    back,s=permutation_frame(sector,target,p)
    assert back==modes
    np.testing.assert_allclose(s@r,np.eye(5),atol=1e-15)

def test_invalid_inputs():
    with pytest.raises(ValueError):longitudinal_states("qqqggg",True)
    with pytest.raises(ValueError):frame("qqqggg",(1,1,1,2,2,1))
    with pytest.raises(ValueError):permutation_frame("qqqggg",(1,1,1,2,2,2),(3,1,2,0,4,5))
    with pytest.raises(ValueError):color_emission("qqqggg",5)
