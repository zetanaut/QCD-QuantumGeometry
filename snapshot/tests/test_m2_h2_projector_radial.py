"""Radial P boundary with non-diagonal Q5 propagation, on actual raw spaces."""
import numpy as np

from deuteron_wigner.quantum.m2_h2_projector_current import H2ProjectorCurrent
from deuteron_wigner.quantum.m2_interacting_q_loop_regulator import InteractingQLoopRegulatorCurrent
from deuteron_wigner.quantum.m2_shell_fock_reference import build_shell_fock_reference
from deuteron_wigner.quantum.m2_shell_canonical_qq import (
    ShellCanonicalQQInsertion, FactorizedCanonicalQQSourceAction)
from deuteron_wigner.quantum.m2_canonical_six_body import CanonicalSixBodyExcursion
from deuteron_wigner.quantum.m2_fourth_order_h2 import FourthOrderH2
from deuteron_wigner.quantum.m2_canonical_good_current import good_current_density_between
from test_m2_interacting_projector_current import pilot
from test_m2_interacting_q_loop_regulator import scalar_vectors


def test_same_helicity_radial_P_boundary_with_mixed_Q_cutoffs():
    ref = build_shell_fock_reference(pilot(0), out_cutoff=3)
    qq = ShellCanonicalQQInsertion(ref, action=FactorizedCanonicalQQSourceAction(ref))
    model = FourthOrderH2(ref, QQ_insertion=qq,
        six_body=CanonicalSixBodyExcursion(ref, out_cutoff=3))
    inner = {"qqqgg": 2, "qqqq-qbar": 2, "qqqggg": 1, "qqqq-qbar-g": 1}
    q = InteractingQLoopRegulatorCurrent(model, model, inner_cutoffs=inner,
        energy_out=1.39, energy_in=1.47, transfer=(.12, .16), charge="electromagnetic",
        total_pplus_GeV=1.3, quadrature_order=16)
    c = H2ProjectorCurrent(q, parent_outgoing=pilot(2), parent_incoming=pilot(2),
                            quadrature_order=16)
    vf = np.column_stack((scalar_vectors(852), scalar_vectors(812)))
    vi = scalar_vectors(630)
    result = c.loop_coefficients(outgoing_vectors=vf, incoming_vectors=vi)
    assert result[2].shape == (2, 2, 1)
    assert np.linalg.norm(result[2]) > 1e-8
    assert np.linalg.norm(result[4]) > 1e-10
    pf, pi = c.embeddings["outgoing"], c.embeddings["incoming"]
    rho = good_current_density_between(c.parents["outgoing"], c.parents["incoming"],
        (.12, .16), charge="electromagnetic")
    rp = pf.conj().T@rho@pi
    hf_rho = c.parent_return_bilinears(endpoint="outgoing", left=pf@vf, right=pf@rp@vi)
    rho_hi = c.parent_return_bilinears(endpoint="incoming", left=pi@rp.conj().T@vf, right=pi@vi)
    parent_f = c.parent_return_bilinears(endpoint="outgoing", left=pf@vf, right=rho@pi@vi)
    parent_i = c.parent_return_bilinears(endpoint="incoming", left=rho.conj().T@pf@vf, right=pi@vi)
    for power in result:
        expected = hf_rho[power]-rho_hi[power]-parent_f[power]+parent_i[power]
        np.testing.assert_allclose(2*1.3*(.12*result[power][0]+.16*result[power][1]),
                                   expected, atol=1e-15, rtol=3e-10)
    # This is a finite-parent Ward check. Q5 outer3/inner2 does not contain
    # every N+2 neighbor and is not claimed to be continuum free-current closure.
    assert not c.common_field_regulator_complete and not c.physical
