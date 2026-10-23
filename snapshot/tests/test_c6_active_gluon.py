from __future__ import annotations

from dataclasses import replace
from math import pi

import numpy as np
import pytest

from deuteron_wigner.formal.diagnostics import ArchitectureError
from deuteron_wigner.formal.gauge_path import (
    ColorRepresentation, StapleOrientation, standard_staple,
)
from deuteron_wigner.pilot.active_gluon.color import (
    ColorChannel, ThreeAdjointColorKernel, antisymmetric_ordered_coupler,
    reject_default_mixture, symmetric_ordered_coupler,
)
from deuteron_wigner.pilot.active_gluon.dynamics import (
    ActiveGluonKernelInput, ActiveGluonRescatteringKernel,
)
from deuteron_wigner.pilot.active_gluon.identity import (
    ActiveGluonOperatorId, OrderedAdjointLinkPair,
)
from deuteron_wigner.pilot.active_gluon.injections import (
    INJECTIONS, detect_injected_violation,
)
from deuteron_wigner.pilot.active_gluon.parent import GluonPolarizationView
from deuteron_wigner.pilot.active_gluon.provenance import (
    C6OverlapLedger, C6Relation, TwoCell, reference_provenance,
)
from deuteron_wigner.pilot.active_gluon.reversal import (
    ActiveGluonProjectedAmplitude, OrderedPairAntiunitaryReversal,
)
from deuteron_wigner.pilot.active_gluon.soft import (
    SoftRoute, SoftRouteSelector, analytic_soft_benchmark,
)
from deuteron_wigner.pilot.active_gluon.status import (
    ActiveGluonResultEnvelope, C6ScientificStatus,
)
from deuteron_wigner.pilot.color import GluonColorSinglet, structure_constants
from deuteron_wigner.pilot.wilson_line.color_guard import symmetric_constants
from deuteron_wigner.pilot.wilson_line.cuts import (
    CutKind, CutLedger, IntermediateStateCut, LFResolventTerm, SpectrumRule,
)
from deuteron_wigner.pilot.wilson_line.identity import (
    BareWilsonSegment, CouplingConvention, FourierConvention,
    MomentumFlowConvention, PathOrdering,
)
from deuteron_wigner.pilot.wilson_line.serialization import deterministic_json


def leg(orientation, label):
    return BareWilsonSegment(
        standard_staple(orientation, ColorRepresentation.ADJOINT),
        "C6:FIBER:F_LEFT_RIGHT", "C6:FIBER:STAPLE_ENDPOINT",
        (1.0, 0.0, 0.0, 1.0), orientation,
        ColorRepresentation.ADJOINT,
        PathOrdering.INCREASING_LAMBDA_RIGHT_TO_LEFT,
        f"C6:CLOSURE:{label}", FourierConvention.EXP_MINUS_I_L_DOT_X,
        CouplingConvention.D_MU_PARTIAL_PLUS_IG_A,
        MomentumFlowConvention.GLUON_INTO_EIKONAL,
        "C6:REG:ANALYTIC_DELTA", stable_id=f"C6:PATH:{label}:{orientation.value}",
    )


def pair(left=StapleOrientation.FUTURE, right=StapleOrientation.FUTURE):
    return OrderedAdjointLinkPair(
        leg(left, "LEFT"), leg(right, "RIGHT"), "C6:TRACE:CLOSED_ADJOINT",
        f"C6:PAIR:{left.value}:{right.value}",
    )


def operator(left=StapleOrientation.FUTURE, right=StapleOrientation.FUTURE):
    return ActiveGluonOperatorId(pair(left, right), "C4:STATE:QQQG:VALIDATION_MEMBER")


def cuts(enabled=True):
    result = CutLedger()
    result.add(IntermediateStateCut(
        "C6:CUT:EIKONAL", CutKind.EIKONAL, "C6:SUPPORT:ACTIVE_GLUON",
        "C6:POLE:PAIR", enabled, 0.65,
    ))
    return result


def kernel_input(left=StapleOrientation.FUTURE, right=StapleOrientation.FUTURE, coupling=0.35, cut=True, oam1=0.4):
    return ActiveGluonKernelInput(
        operator(left, right), cuts(cut),
        LFResolventTerm(
            "C4:STATE:QQQG", "C6:STATE:QQQGG", 1.0, 1.7, 1,
            "C6:VERTEX:ACTIVE_GLUON_RESCATTER", "C6:OP:ACTIVE_GLUON_FF",
            "C6:SUPPORT:ACTIVE_GLUON", SpectrumRule.DECLARED_CONTINUUM_DENSITY,
            "C6:REG:ANALYTIC_DELTA",
        ),
        coupling, 0.25, pi / 3, 1.0, oam1,
    )


def test_c6_a_all_four_ordered_pairs_round_trip_and_remain_distinct():
    words = []
    serial = []
    for left in (StapleOrientation.FUTURE, StapleOrientation.PAST):
        for right in (StapleOrientation.FUTURE, StapleOrientation.PAST):
            value = pair(left, right)
            words.append(value.orientation_word)
            serial.append(deterministic_json(value))
    assert len(set(words)) == len(set(serial)) == 4
    assert pair().swapped().ordered_pair_id != pair().ordered_pair_id
    assert pair().reverse_one("left").orientation_word != pair().orientation_word


@pytest.mark.parametrize(
    "source,expected",
    [
        ((StapleOrientation.FUTURE, StapleOrientation.FUTURE), ("PAST", "PAST")),
        ((StapleOrientation.PAST, StapleOrientation.PAST), ("FUTURE", "FUTURE")),
        ((StapleOrientation.FUTURE, StapleOrientation.PAST), ("PAST", "FUTURE")),
        ((StapleOrientation.PAST, StapleOrientation.FUTURE), ("FUTURE", "PAST")),
    ],
)
def test_c6_a_antiunitary_pair_mapping(source, expected):
    transformed = OrderedPairAntiunitaryReversal().transform_operator(operator(*source))
    assert transformed.link_pair.orientation_word == expected


def test_c6_b_exact_fd_norms_projections_and_reconstruction():
    f, d = structure_constants(), symmetric_constants()
    assert np.vdot(f, f).real == pytest.approx(24)
    assert np.vdot(d, d).real == pytest.approx(40 / 3)
    assert np.vdot(f, d).real == pytest.approx(0, abs=1e-15)
    np.testing.assert_allclose(antisymmetric_ordered_coupler(), 1j * f, atol=1e-15)
    np.testing.assert_allclose(symmetric_ordered_coupler(), d, atol=1e-15)
    color = ThreeAdjointColorKernel.from_ordered_couplers(0.7, -0.25)
    pf, pd, parallel, residual = color.decompose()
    assert pf.amplitude == pytest.approx(0.7)
    assert pd.amplitude == pytest.approx(-0.25)
    assert residual < 3e-15
    np.testing.assert_allclose(parallel, color.tensor, atol=2e-15)
    with pytest.raises(ArchitectureError, match="C6.COLOR.6"):
        reject_default_mixture()


def test_c6_b_orthogonal_color_residual_is_reported_not_clipped():
    tensor = np.zeros((8, 8, 8), complex)
    tensor[0, 0, 0] = 1
    color = ThreeAdjointColorKernel(tensor, "OUTSIDE_FD")
    assert color.decompose()[3] > 0
    with pytest.raises(ArchitectureError, match="C6.COLOR.4"):
        color.require_fd_subspace()


def test_c6_c_active_gluon_parent_and_exact_zero_limits():
    kernel = ActiveGluonRescatteringKernel()
    parent = kernel.evaluate(kernel_input())
    assert parent.tensor.shape == (2, 2, 2, 2, 2, 2, 8, 8, 8)
    assert parent.active_slot_id.endswith("slot:3")
    assert kernel.active_slot()[1] > 0
    for modified in (
        kernel_input(coupling=0), kernel_input(cut=False), kernel_input(oam1=0),
    ):
        result = kernel.evaluate(modified)
        assert np.max(np.abs(result.tensor)) == 0


def test_c6_d_common_parent_projects_independent_fd_and_reconstructs_tensor():
    parent = ActiveGluonRescatteringKernel().evaluate(kernel_input())
    f_matrix = parent.color_projected_matrix(ColorChannel.F_TYPE)
    d_matrix = parent.color_projected_matrix(ColorChannel.D_TYPE)
    assert not np.array_equal(f_matrix, d_matrix)
    for channel in ColorChannel:
        assert parent.reconstruction_residual(channel) < 1e-15
        records = [
            parent.identity_record(channel, view)
            for view in GluonPolarizationView
        ]
        invariant = ("parent_id", "ordered_pair_id", "cut_ledger_id", "color_channel", "state_member_id", "phase_budget_id")
        for key in invariant:
            assert len({record[key] for record in records}) == 1
        assert len({record["polarization_projector"] for record in records}) == 3


def projected(item, channel=ColorChannel.F_TYPE, view=GluonPolarizationView.TRACE):
    parent = ActiveGluonRescatteringKernel().evaluate(item)
    value = parent.polarization_view(channel, view)
    return ActiveGluonProjectedAmplitude(
        complex(value), item.operator_id, channel, view,
        "C6:CUT_LEDGER:ACTIVE_GLUON", item.operator_id.source_state_member_id,
    )


def test_c6_c_and_rev_future_past_opposite_and_zero_odd_limits():
    future = projected(kernel_input())
    past = projected(kernel_input(StapleOrientation.PAST, StapleOrientation.PAST))
    even, odd = OrderedPairAntiunitaryReversal().even_odd(future, past)
    assert abs(even) < 1e-14
    assert abs(odd) > 0
    for item in (kernel_input(coupling=0), kernel_input(cut=False), kernel_input(oam1=0)):
        ref = projected(item)
        partner = projected(replace(
            item, operator_id=OrderedPairAntiunitaryReversal().transform_operator(item.operator_id)
        ))
        assert OrderedPairAntiunitaryReversal().even_odd(ref, partner)[1] == 0


def test_c6_e_soft_overlap_exactly_once_and_rapidity_derivatives():
    soft = analytic_soft_benchmark("F_TYPE", "TRACE")
    assert soft.evaluated(-3) == pytest.approx(soft.evaluated(4), abs=5e-16)
    assert soft.rapidity_derivative(1) == 0
    assert soft.rapidity_derivative(0) == -soft.rapidity_derivative(2)
    assert abs(soft.rapidity_derivative(0)) > 0
    assert soft.to_dict()["uv_finite_matching"] == "UNRESOLVED_NOT_ZERO"


def test_c6_f_soft_routes_are_exclusive_and_joint_route_unimplemented():
    assert SoftRouteSelector.select((SoftRoute.BOUNDARY_ONLY_RESCATTERING,)) == SoftRoute.BOUNDARY_ONLY_RESCATTERING
    with pytest.raises(ArchitectureError, match="C6.ROUTE.1"):
        SoftRouteSelector.select((SoftRoute.BOUNDARY_ONLY_RESCATTERING, SoftRoute.JOINT_MICROSCOPIC_SOFT_SECTOR))
    with pytest.raises(ArchitectureError, match="C6.ROUTE.1"):
        SoftRouteSelector.select((SoftRoute.JOINT_MICROSCOPIC_SOFT_SECTOR,))
    with pytest.raises(ArchitectureError, match="C6.ROUTE.1"):
        SoftRouteSelector.transfer_to_cs_kernel(SoftRoute.BOUNDARY_ONLY_RESCATTERING)


def test_c6_g_ward_closure_fd_and_missing_attachment_failure():
    attachments = {
        "ACTIVE_FIELD": 1 + 2j, "LEFT_LINK": -0.2 - 0.4j,
        "RIGHT_LINK": -0.3 - 0.6j, "SPECTATOR_COLOR": -0.5 - 1j,
    }
    for channel in ("F_TYPE", "D_TYPE"):
        assert ActiveGluonRescatteringKernel.ward_residual(attachments, channel) < 1e-15
    with pytest.raises(ArchitectureError, match="C6.WARD.2"):
        ActiveGluonRescatteringKernel.ward_residual({k:v for k,v in attachments.items() if k != "LEFT_LINK"}, "F_TYPE")
    assert ActiveGluonRescatteringKernel.color_singlet_residual() == pytest.approx(GluonColorSinglet().generator_residual())


def test_c6_g_provenance_two_cells_and_channel_independence():
    graph = reference_provenance()
    assert len(graph["cells"]) == 4
    assert graph["general_provenance_2_complex_complete"] is False
    ledger = C6OverlapLedger()
    cell = TwoCell("x", ("a","b"), "c", C6Relation.OVERLAP_SUBTRACT, "region")
    ledger.add(cell)
    with pytest.raises(ArchitectureError, match="C6.PROV.2"):
        ledger.add(replace(cell, stable_id="y"))
    ledger.require_independent_channels("F_TYPE", "D_TYPE")


def test_c6_status_and_all_downstream_gates_fail_closed():
    envelope = ActiveGluonResultEnvelope("C6:RESULT:REFERENCE", {"value": [0, 1]})
    assert len(envelope.statuses) == len(C6ScientificStatus)
    for method, label in (
        (envelope.require_volume_iv, "Volume IV"),
        (envelope.require_volume_v, "Volume V"),
        (envelope.require_volume_vi, "Volume VI"),
    ):
        with pytest.raises(ArchitectureError, match=label):
            method()
    with pytest.raises(ArchitectureError, match="production"):
        envelope.require_production()


@pytest.mark.parametrize("stable_id,description,diagnostic", INJECTIONS)
def test_all_c6_injections_have_stable_structured_diagnostics(stable_id, description, diagnostic):
    with pytest.raises(ArchitectureError) as caught:
        detect_injected_violation(stable_id)
    assert caught.value.requirement_id == diagnostic


def test_c6_injection_ledger_has_sixty_ordered_entries():
    assert len(INJECTIONS) == 60
    assert [row[0] for row in INJECTIONS] == [f"C6.INJECT.{index:02d}" for index in range(1, 61)]
