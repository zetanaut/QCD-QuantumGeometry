from __future__ import annotations

from dataclasses import replace
from math import pi

import numpy as np
import pytest

from deuteron_wigner.formal.diagnostics import ArchitectureError
from deuteron_wigner.formal.gauge_path import (
    ColorClass, ColorRepresentation, GluonLinkId, StapleOrientation,
    standard_staple,
)
from deuteron_wigner.pilot.states import SpinorOAMState
from deuteron_wigner.pilot.wilson_line.color_guard import (
    color_algebra_report, require_ordered_gluon_identity,
)
from deuteron_wigner.pilot.wilson_line.cuts import (
    CutKind, CutLedger, CutRelation, IntermediateStateCut, LFResolventTerm,
    SpectrumRule,
)
from deuteron_wigner.pilot.wilson_line.distribution import (
    DistributionalPoleEvaluator, compact_bump,
)
from deuteron_wigner.pilot.wilson_line.identity import (
    BareWilsonSegment, CouplingConvention, FourierConvention,
    MomentumFlowConvention, PathOrdering, derived_eikonal_pole,
)
from deuteron_wigner.pilot.wilson_line.injections import (
    INJECTIONS, detect_injected_violation,
)
from deuteron_wigner.pilot.wilson_line.kernel import (
    OneGluonPilotKernel, PilotKernelInput,
)
from deuteron_wigner.pilot.wilson_line.projectors import (
    PilotProjection, PilotSpinBlock, boer_mulders_like_projector,
    sivers_like_projector,
)
from deuteron_wigner.pilot.wilson_line.provenance import graph_dict, require_isolation
from deuteron_wigner.pilot.wilson_line.serialization import (
    deterministic_json, serialized_round_trip,
)
from deuteron_wigner.pilot.wilson_line.status import (
    C5PilotRecord, C5ResultEnvelope, PhaseBudget, ScientificStatus,
)
from deuteron_wigner.pilot.wilson_line.time_reversal import (
    AntiunitaryLinkReversal, reject_raw_link_subtraction,
)


def path(orientation=StapleOrientation.FUTURE, representation=ColorRepresentation.FUNDAMENTAL):
    return BareWilsonSegment(
        standard_staple(orientation, representation),
        "LF:FIBER:0", "+infinity" if orientation == StapleOrientation.FUTURE else "-infinity",
        (1.0, 0.0, 0.0, 1.0), orientation, representation,
        PathOrdering.INCREASING_LAMBDA_RIGHT_TO_LEFT,
        "transverse_at_infinity", FourierConvention.EXP_MINUS_I_L_DOT_X,
        CouplingConvention.D_MU_PARTIAL_PLUS_IG_A,
        MomentumFlowConvention.GLUON_INTO_EIKONAL, "DELTA_ANALYTIC",
    )


def resolvent(rule=SpectrumRule.DECLARED_CONTINUUM_DENSITY):
    return LFResolventTerm(
        "C3:C:SPINOR_OAM", "C5:STATE:QQG", 1.0, 1.6, 1,
        "C5:VERTEX:EIKONAL_ONE_GLUON", "C5:OP:QUARK_GAMMA_PLUS",
        "C5:CUT:ON_SHELL_01", rule, "DELTA_ANALYTIC",
    )


def ledger(enabled=True):
    value = CutLedger()
    value.add(IntermediateStateCut(
        "C5:CUT:EIKONAL", CutKind.EIKONAL, "C5:SUPPORT:01",
        "C5:POLE:FUTURE", enabled, 0.7,
    ))
    return value


def kernel_input(orientation=StapleOrientation.FUTURE, coupling=0.4, enabled=True, amplitudes=(1, 0.4, 0)):
    return PilotKernelInput(
        SpinorOAMState(amplitudes, 0.94), path(orientation), resolvent(),
        ledger(enabled), coupling, 0.3, pi / 3, "QUARK", "u",
        "C3:SLOT:ACTIVE_U",
    )


def test_c5_a_pole_is_derived_and_cut_sign_reverses():
    future, past = path(), path(StapleOrientation.PAST)
    pf, pp = derived_eikonal_pole(future), derived_eikonal_pole(past)
    assert pf.pv_coefficient == pp.pv_coefficient == 1
    assert pf.delta_coefficient_imaginary == -pp.delta_coefficient_imaginary == 1
    with pytest.raises(ArchitectureError, match="C5.POLE.1"):
        derived_eikonal_pole(future, manual_sign=-1)
    assert future.inverted().orientation == StapleOrientation.PAST


def test_c5_a_first_order_path_composition_and_endpoint_guard():
    first = path()
    second = replace(first, start_fiber=first.end_or_infinity_class, end_or_infinity_class="LF:FIBER:XI", stable_id="C5:PATH:SECOND")
    assert second.compose(first) == (second, first)
    with pytest.raises(ArchitectureError, match="C5.PATH.2"):
        first.compose(second)


def test_c5_distributional_pv_cut_and_odd_cancellation():
    evaluator = DistributionalPoleEvaluator()
    even = evaluator.pv_plus_cut(compact_bump, eta=1, support=1)
    past = evaluator.pv_plus_cut(compact_bump, eta=-1, support=1)
    assert abs(even.pv) < 1e-14
    assert even.cut == -past.cut
    odd = evaluator.pv_plus_cut(lambda x: x * compact_bump(x), eta=1, support=1)
    assert abs(odd.cut) < 1e-14
    assert odd.pv > 0
    scaled = evaluator.pv_plus_cut(compact_bump, eta=1, support=1, jacobian=2)
    assert scaled.cut == even.cut / 2


def test_c5_epsilon_is_only_convergence_oracle():
    report = DistributionalPoleEvaluator().epsilon_sequence(
        compact_bump, eta=1, support=1, epsilons=(0.02, 0.005, 0.001),
        points_per_epsilon=300001,
    )
    assert report.epsilon_is_physical is False
    assert abs(report.values[-1] - report.target) < abs(report.values[0] - report.target)
    assert report.final_residual < 0.01


def test_c5_b_discrete_offshell_has_exact_zero_absorption():
    assert resolvent(SpectrumRule.DISCRETE_OFF_SHELL).absorptive_weight() == 0
    assert resolvent().absorptive_weight(0.7) == -0.7


def test_c5_e_cut_ledger_rejects_double_counting_and_counts_equivalence_once():
    cuts = ledger()
    duplicate = IntermediateStateCut(
        "C5:CUT:LF", CutKind.LF_ENERGY, "C5:SUPPORT:01",
        "C5:RESOLVENT:01", True, 0.7,
    )
    with pytest.raises(ArchitectureError, match="C5.CUT.2"):
        cuts.add(duplicate)
    cuts.add(duplicate, CutRelation.EQUIVALENT_COUNT_ONCE, "C5:CUT:EIKONAL")
    assert cuts.active_weight() == pytest.approx(0.7)


def test_equal_numeric_denominators_with_distinct_provenance_are_not_deduplicated():
    cuts = ledger()
    cuts.add(IntermediateStateCut(
        "C5:CUT:DISTINCT", CutKind.LF_ENERGY, "C5:SUPPORT:02",
        "same_float_value_different_physics", True, 0.2,
    ))
    assert cuts.active_weight() == pytest.approx(0.9)


def test_c5_c_future_past_and_zero_limits():
    kernel = OneGluonPilotKernel()
    future = kernel.evaluate(kernel_input())
    past = kernel.evaluate(kernel_input(StapleOrientation.PAST))
    even, odd = AntiunitaryLinkReversal().even_odd(future, past)
    assert even.imag == pytest.approx(0)
    assert odd.real == pytest.approx(0)
    assert odd.imag != 0
    assert kernel.evaluate(kernel_input(coupling=0)).absorptive == 0
    assert kernel.evaluate(kernel_input(enabled=False)).absorptive == 0
    assert kernel.evaluate(kernel_input(amplitudes=(1, 0, 0))).absorptive == 0
    with pytest.raises(ArchitectureError, match="C5.TIME.1"):
        reject_raw_link_subtraction()


def test_c5_projectors_are_distinct_red_maps():
    kernel = OneGluonPilotKernel()
    odd = AntiunitaryLinkReversal().even_odd(
        kernel.evaluate(kernel_input()),
        kernel.evaluate(kernel_input(StapleOrientation.PAST)),
    )[1]
    block = PilotSpinBlock(0.6, -0.25, 0.8)
    sivers, boer = sivers_like_projector(), boer_mulders_like_projector()
    assert sivers.map_class.value == boer.map_class.value == "RED"
    assert sivers.project(odd, block) != boer.project(odd, block)
    with pytest.raises(ArchitectureError, match="C5.QUARK.2"):
        sivers.project(odd, block, route_projection=PilotProjection.BOER_MULDERS_LIKE_PILOT)


def test_c5_d_color_algebra_and_ordered_links():
    report = color_algebra_report()
    assert report["fundamental_casimir_residual"] < 1e-15
    assert abs(report["f_d_inner_product"]) < 1e-14
    f, p = path(representation=ColorRepresentation.ADJOINT), path(StapleOrientation.PAST, ColorRepresentation.ADJOINT)
    expected = GluonLinkId(f.path_id, p.path_id, ColorClass.DIAGONAL_ADJOINT)
    require_ordered_gluon_identity(expected, expected)
    swapped = GluonLinkId(p.path_id, f.path_id, ColorClass.DIAGONAL_ADJOINT)
    with pytest.raises(ArchitectureError, match="C5.GLUON.1"):
        require_ordered_gluon_identity(swapped, expected)


def test_restricted_pilot_ward_closure():
    assert OneGluonPilotKernel.ward_residual(1 + 2j, -0.4 - 0.5j, -0.6 - 1.5j) < 1e-15


def test_status_phase_budget_round_trip_and_fail_closed_gates():
    envelope = C5ResultEnvelope("C5:RESULT:REFERENCE", {"value": [0, 1]}, PhaseBudget(1.0))
    data = envelope.to_dict()
    assert len(data["statuses"]) == len(ScientificStatus)
    assert data["phase_budget"]["soft_overlap_contribution"] == "UNRESOLVED_NOT_ZERO"
    with pytest.raises(ArchitectureError, match="Volume IV"):
        envelope.require_volume_iv()
    with pytest.raises(ArchitectureError, match="Volume V"):
        envelope.require_volume_v()
    with pytest.raises(ArchitectureError, match="production"):
        envelope.require_production()


def test_exported_objects_have_deterministic_lossless_value_records():
    objects = (
        path(), derived_eikonal_pole(path()), resolvent(), ledger(),
        kernel_input(), OneGluonPilotKernel().evaluate(kernel_input()),
        AntiunitaryLinkReversal(), sivers_like_projector(),
        C5PilotRecord(
            "state", "member", "recoil", "overlap", "operator", "path",
            "FUNDAMENTAL", 1, "pole", "intermediate", "ledger",
            ("LZ_0", "LZ_PLUS_1"), "projector", "regulator",
            (("analytic", 1e-12),), complex(0, 1),
        ),
        C5ResultEnvelope("C5:RESULT:SERIALIZE", {"z": complex(1, -2)}, PhaseBudget(0.5)),
    )
    for value in objects:
        first = deterministic_json(value)
        record = serialized_round_trip(value)
        assert first == deterministic_json(value)
        assert isinstance(record, dict)


def test_c5_provenance_is_disjoint_and_honest_about_two_complex():
    graph = graph_dict()
    assert graph["general_provenance_2_complex_complete"] is False
    assert "CUT_EQUIVALENT_COUNT_ONCE" in graph["executable_two_cells"]
    require_isolation({"production:root", "c2:accepted_parent"})
    with pytest.raises(ArchitectureError, match="C5.ISOLATE"):
        require_isolation({"C5:ONE_GLUON_KERNEL"})


@pytest.mark.parametrize("stable_id,description,diagnostic", INJECTIONS)
def test_all_48_injections_have_stable_structured_diagnostics(stable_id, description, diagnostic):
    with pytest.raises(ArchitectureError) as caught:
        detect_injected_violation(stable_id)
    assert caught.value.requirement_id == diagnostic


def test_injection_ledger_is_complete_and_ordered():
    assert len(INJECTIONS) == 48
    assert [row[0] for row in INJECTIONS] == [f"C5.INJECT.{index:02d}" for index in range(1, 49)]
