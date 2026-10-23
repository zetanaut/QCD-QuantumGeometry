"""The complete retained finite-H Ward identity with exact Gaussian tails."""
from dataclasses import replace
import numpy as np
import pytest
from deuteron_wigner.quantum.m2_gaussian_cluster_retained_projector import exact_retained_transverse_projector_coefficients
from deuteron_wigner.quantum.m2_gaussian_cluster_current_actions import cluster_cubic_current_between
from deuteron_wigner.quantum.m2_gaussian_cluster_quartic_pair_actions import cluster_quartic_pair_current_between
from deuteron_wigner.quantum.m2_canonical_good_current import good_current_density_between
from deuteron_wigner.quantum.m2_transverse_current_projection import free_transverse_current_between
from deuteron_wigner.quantum.m2_gaussian_cluster_current import PRESCRIPTION
from test_m2_gaussian_cluster_h2 import model

D=np.array([.12,.16])
ARGS=dict(charge="electromagnetic",prescription=PRESCRIPTION,quadrature_order=8)


@pytest.mark.parametrize("target,jz",[("PROTON",1),("NEUTRON",1),("PROTON",-1),("NEUTRON",-1)])
def test_all_actual_PP_orders_close_with_exact_projector_and_reverse(target,jz):
    f,i=model(target=target,jz=jz).pilot,model(target=target).pilot
    k=exact_retained_transverse_projector_coefficients(f,i,D,**ARGS)
    physical={
        0:free_transverse_current_between(f,i,D,total_pplus_GeV=1.3,charge="electromagnetic"),
        1:cluster_cubic_current_between(f,i,D,mean_total_transverse_GeV=(0.,0.),**ARGS).total[2:],
        2:cluster_quartic_pair_current_between(f,i,D,mean_total_transverse_GeV=(0.,0.),**ARGS).total[2:],
    }
    rho=good_current_density_between(f,i,D,charge="electromagnetic")
    def h(p):
        return {0:p.h0,1:p.emission_per_gs,2:p.cross_pairs_per_gs2+p.instantaneous_fermion_per_gs2+p.wick_mass_per_gs2}
    hf,hi=h(f),h(i)
    for n in range(3):
        expected=hf[n]@rho-rho@hi[n]
        np.testing.assert_allclose(2*1.3*np.einsum("j,jab->ab",D,physical[n]+k[n]),expected,atol=2e-14)
    # The uncorrected scalar-loop test had a significant residual here;
    # exact field integration must remove it, not just refine a parent cutoff.
    if jz==1:
        assert np.linalg.norm(2*1.3*np.einsum("j,jab->ab",D,physical[2])-(hf[2]@rho-rho@hi[2]))>1e-4
    reverse=exact_retained_transverse_projector_coefficients(i,f,-D,**ARGS)
    for n in range(3):
        np.testing.assert_allclose(k[n],reverse[n].conj().transpose(0,2,1),atol=2e-14)


def test_exact_retained_binding_rejects_mixed_definition_or_unspecified_pilot():
    p=model().pilot
    with pytest.raises(ValueError,match="pilots"):
        exact_retained_transverse_projector_coefficients(object(),p,D,**ARGS)
    with pytest.raises(ValueError,match="identical cluster"):
        exact_retained_transverse_projector_coefficients(replace(p,total_pplus_GeV=2.),p,D,**ARGS)
