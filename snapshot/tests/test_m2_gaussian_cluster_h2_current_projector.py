"""Actual snapshot binding of finite-parent transverse projector transport."""
from dataclasses import replace
import numpy as np
import pytest
from deuteron_wigner.quantum.m2_gaussian_cluster_h2_current import cluster_transverse_projector_h2_coefficients
from deuteron_wigner.quantum.m2_h2_projector_current import H2ProjectorCurrent
from test_m2_gaussian_cluster_h2 import model
from test_m2_gaussian_cluster_projector_current import qcontrol,parent
from test_m2_interacting_q_loop_regulator import scalar_vectors


@pytest.fixture(scope="module")
def endpoints():
    # Opposite Jz permits a nonzero odd-shell tangent. For equal Jz and
    # ground inner orbitals, the first nonzero Hamiltonian tail is even.
    f,i=model(jz=-1).snapshot(1.39),model().snapshot(1.47)
    q=qcontrol("PROTON",-1,0)
    c=H2ProjectorCurrent(q,parent_outgoing=parent(q.outgoing.pilot),
        parent_incoming=parent(q.incoming.pilot),quadrature_order=16)
    return f,i,c


def test_bound_coefficients_sum_to_existing_complete_transverse_parent_current(endpoints):
    f,i,c=endpoints
    vf,vi=scalar_vectors(432),np.column_stack((scalar_vectors(654),scalar_vectors(976)))
    args=dict(outgoing_vectors=vf,incoming_vectors=vi)
    coefficients=cluster_transverse_projector_h2_coefficients(f,i,c,**args)
    assert set(coefficients)=={"retained_projector","source_projector","Q_projector"}
    total=sum(.3**n*parts[n] for parts in coefficients.values() for n in range(5))
    np.testing.assert_allclose(total,c.current(normalized=False,**args),atol=4e-16)
    normed=cluster_transverse_projector_h2_coefficients(f,i,c,
        outgoing_vectors=f.whitener@vf,incoming_vectors=i.whitener@vi)
    normalized=sum(.3**n*parts[n] for parts in normed.values() for n in range(5))
    np.testing.assert_allclose(normalized,c.current(normalized=True,**args),atol=4e-16)
    assert sum(np.linalg.norm(parts[1]) for parts in coefficients.values())>1e-8


def test_projector_binding_rejects_same_sized_wrong_energy_H_and_norm(endpoints):
    f,i,c=endpoints
    args=dict(outgoing_vectors=scalar_vectors(),incoming_vectors=scalar_vectors(223))
    for changed in (replace(i,energy_GeV2=1.48),
                    replace(i,effective_hamiltonian=i.effective_hamiltonian+.001*np.eye(37)),
                    replace(i,metric=i.metric+.001*np.eye(37))):
        with pytest.raises(ValueError,match="energy|Hamiltonian|norm"):
            cluster_transverse_projector_h2_coefficients(f,changed,c,**args)
    with pytest.raises(ValueError,match="actual P/source/Q"):
        cluster_transverse_projector_h2_coefficients(f,i,None,**args)


def test_projector_binding_does_not_invent_a_minus_or_full_current(endpoints):
    f,i,c=endpoints
    values=cluster_transverse_projector_h2_coefficients(f,i,c,
        outgoing_vectors=scalar_vectors(85),incoming_vectors=scalar_vectors(48))
    for group in values.values():
        assert set(group)==set(range(5))
        assert all(a.shape==(2,1,1) for a in group.values())
    assert not c.complete_physical_four_current
