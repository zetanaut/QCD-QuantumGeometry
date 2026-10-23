import numpy as np
import pytest
from scipy.linalg import expm
from deuteron_wigner.quantum.nonlocal_source import (
    ETA,line_integral,vertex,second_vertex,endpoint_difference,wilson_probe,
)

PATH = np.array([[.1,.2,-.1,.3],[.25,.5,.4,-.1],[.4,-.3,.2,.5]])
Q = np.array([.7,-.3,.2,.8])


@pytest.mark.parametrize('q',[Q,np.zeros(4),[0,.3,0,0]])
def test_endpoint_Ward_limits_and_reversal(q):
    q = np.array(q)
    np.testing.assert_allclose(q@ETA@vertex(PATH,q,2/3),endpoint_difference(PATH,q,2/3),atol=3e-16)
    np.testing.assert_allclose(line_integral(PATH[::-1],q),-line_integral(PATH,q),atol=3e-16)
    np.testing.assert_allclose(line_integral(PATH,-q),line_integral(PATH,q).conj(),atol=3e-16)


def test_path_choice_retains_transverse_information_not_determined_by_Ward():
    difference = vertex(PATH,Q,1)-vertex(PATH[[0,-1]],Q,1)
    assert np.linalg.norm(difference)>.05
    assert abs(Q@ETA@difference)<3e-16


def test_one_and_two_insertions_are_derivatives_of_the_same_Wilson_source():
    k = np.array([-.2,.1,.4,-.3]); e = np.array([.3,.5,.2,-.4]); f = np.array([-.5,.2,.4,.1])
    def source(a,b):return wilson_probe(PATH,[Q,k],[e,f],[a,b],2/3)
    h = 2e-4
    first = (source(h,0)-source(-h,0))/(2*h)
    second = (source(h,h)-source(h,-h)-source(-h,h)+source(-h,-h))/(4*h*h)
    np.testing.assert_allclose(first,e@ETA@vertex(PATH,Q,2/3),atol=2e-9)
    np.testing.assert_allclose(second,e@ETA@second_vertex(PATH,Q,k,2/3)@ETA@f,atol=2e-9)
    np.testing.assert_allclose(Q@ETA@second_vertex(PATH,Q,k,2/3),
                               endpoint_difference(PATH,Q,2/3)*vertex(PATH,k,2/3),atol=3e-16)


@pytest.mark.parametrize('axis',range(3))
def test_nonlocal_four_vertex_Lorentz_transport(axis):
    generator = np.zeros((4,4)); generator[0,axis+1]=generator[axis+1,0]=.37
    transform = expm(generator)
    moved = vertex(PATH@transform.T,transform@Q,-1/3)
    np.testing.assert_allclose(moved,transform@vertex(PATH,Q,-1/3),atol=3e-16)


def test_invalid_path_or_probe_rejected():
    with pytest.raises(ValueError):vertex([[0,0,0,0]],Q,1)
    with pytest.raises(ValueError):vertex(PATH,[1,2],1)
