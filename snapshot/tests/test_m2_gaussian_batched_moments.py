"""Batched exact Gaussian moments retain arbitrarily small nonzero inputs."""
import numpy as np
import pytest
from deuteron_wigner.quantum.m2_gaussian_field_regulator import gaussian_shell_integral


@pytest.mark.parametrize("epsilon",[1e-8,1e-80])
@pytest.mark.parametrize("cutoff",[0,3])
def test_tiny_nonzero_insertions_covariances_and_affine_weights_are_retained(epsilon,cutoff):
    b=.4
    args=dict(out_cutoff=1 if cutoff==0 else 5,in_cutoff=0,out_map=np.eye(1),in_map=np.eye(1),
        b_GeV=b,field_precision=0.,field_maps=np.zeros((0,1)),field_shifts=np.zeros((0,2)),constant=0.)
    affine=gaussian_shell_integral(**args,linear=np.array([[epsilon],[0.]]))
    # Circular one-quantum oscillator, <1+|qx|0>=b/2.
    np.testing.assert_allclose(affine[1,0],epsilon*b/2,atol=0.,rtol=2e-14)
    cubic=gaussian_shell_integral(**args,polynomial_terms={(3,0):epsilon})
    np.testing.assert_allclose(cubic[1,0],epsilon*3*b**3/4,atol=0.,rtol=2e-14)
    quadratic=gaussian_shell_integral(**{**args,"out_cutoff":0},quadratic=np.diag([epsilon,0.]))
    np.testing.assert_allclose(quadratic[0,0],epsilon*b*b/4,atol=0.,rtol=2e-14)
    # A genuinely tiny covariance from a physical field map is also kept.
    # c=A I + vv^T, v=(1,epsilon), det(c)=A(A+1+epsilon²).
    A=2/b**2;det=A*(A+1+epsilon**2)
    mixed=gaussian_shell_integral(out_cutoff=cutoff,in_cutoff=0,out_map=np.eye(2),in_map=np.eye(2),
        b_GeV=b,field_precision=1.,field_maps=np.array([[1.,epsilon]]),field_shifts=np.zeros((1,2)),
        constant=0.,polynomial_terms={(1,1,0,0):1.})
    np.testing.assert_allclose(mixed[0,0],-A*A*epsilon/det**2,atol=0.,rtol=3e-14)
