"""Independent shaped-H minus boundary, not a transverse Ward definition."""
import pytest
import test_m2_smooth_longitudinal_projector as independent
from test_m2_gaussian_cluster_projector_kernel import data as original_data,CASES
from test_m2_gaussian_reference_shape_h2 import shaped
from deuteron_wigner.quantum.m2_gaussian_cluster_projector_kernel import pair_hamiltonian_orbital


@pytest.fixture(params=[1,2])
def shape_data(request,monkeypatch):
    reg=shaped(request.param)
    def data(case):
        _,wi,wf=original_data(case)
        inc,out,pair,si,so,_,_=case
        h=pair_hamiltonian_orbital(inc,out,pair,species=si,out_species=so,
            b_GeV=independent.B,regulator=reg,total_transverse_GeV=(0.,0.))
        assert h.reference_shape is reg.reference_shape
        return h,wi,wf
    # Reuse the independent changed-wavefunction quadrature, not its
    # previously tested unshaped H. pytest restores the helper after each
    # test. No production object or current formula is monkeypatched.
    monkeypatch.setattr(independent,"data",data)


@pytest.mark.parametrize("case",CASES)
def test_shaped_minus_projector_reversal_boost_and_path_refinement(case,shape_data):
    independent.test_exact_physical_pair_minus_boundaries_reverse_and_boost(case)


@pytest.mark.parametrize("case",CASES[:2])
def test_shaped_minus_projector_is_independent_physical_wavefunction_derivative(case,shape_data):
    independent.test_zero_transfer_exact_pair_boundary_is_derivative_of_original_projected_H(case)
