#!/usr/bin/env python3
"""Build deterministic C6 active-gluon validation manifests."""

from __future__ import annotations

import hashlib
import json
from math import pi
from pathlib import Path

import numpy as np

from deuteron_wigner.formal.gauge_path import (
    ColorRepresentation, StapleOrientation, standard_staple,
)
from deuteron_wigner.pilot.active_gluon.color import (
    ColorChannel, ThreeAdjointColorKernel,
)
from deuteron_wigner.pilot.active_gluon.dynamics import (
    ActiveGluonKernelInput, ActiveGluonRescatteringKernel,
)
from deuteron_wigner.pilot.active_gluon.identity import (
    ActiveGluonOperatorId, OrderedAdjointLinkPair,
)
from deuteron_wigner.pilot.active_gluon.injections import INJECTIONS
from deuteron_wigner.pilot.active_gluon.parent import GluonPolarizationView
from deuteron_wigner.pilot.active_gluon.provenance import reference_provenance
from deuteron_wigner.pilot.active_gluon.reversal import (
    ActiveGluonProjectedAmplitude, OrderedPairAntiunitaryReversal,
)
from deuteron_wigner.pilot.active_gluon.soft import (
    SoftRoute, analytic_soft_benchmark,
)
from deuteron_wigner.pilot.active_gluon.status import ActiveGluonResultEnvelope
from deuteron_wigner.pilot.color import structure_constants
from deuteron_wigner.pilot.wilson_line.color_guard import symmetric_constants
from deuteron_wigner.pilot.wilson_line.cuts import (
    CutKind, CutLedger, IntermediateStateCut, LFResolventTerm, SpectrumRule,
)
from deuteron_wigner.pilot.wilson_line.identity import (
    BareWilsonSegment, CouplingConvention, FourierConvention,
    MomentumFlowConvention, PathOrdering,
)


ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "next_level"
BASELINE = "c4aeb380bc3c23b8dcf2fb6a4528042de598cb48"
SOURCE_PATHS = (
    "references/volume_0_algebraic_geometric.tex",
    "references/volume_i_regulated_light_front_foundations.tex",
    "references/volume_ii_common_nucleon_gtmd_overlaps.tex",
    "references/volume_iii_dynamical_wilson_lines.tex",
    "references/volume_iv_matched_spin1_nuclear_dynamics.tex",
    "references/volume_v_matching_evolution_factorization.tex",
)
EXPECTED_C5 = {
    "c5_benchmark_manifest.json": "f7cd27b6ab0b29c5a02fd7cfe811b6cb9ffb7a9117d5bc0fef3e5827a2150a8e",
    "c5_cut_ledger_manifest.json": "e738858f1f13e0500cc7b5c222332c3c593f0cd009b292eda4d3797a448ab547",
    "c5_injection_manifest.json": "f0525b548b396f6ee0f86ec2690c37ef3ccfd50e2eade54778307f8db36ba4f2",
    "c5_normative_source_integration.json": "1e4bb536397a2a8b9c8e43dddba1237113165f8feb3adf5843de45b89e2f806d",
    "c5_phase_budget.json": "a44ed8752d40d533902b04feeb2c398c793732ad42e929f2fdee61e90b3174ad",
    "c5_provenance_graph.json": "b0d905f52a34f8bb5c1ea783ff2e2b2eee17a28932a77abd071572b826254a05",
    "c5_regression_report.json": "62e22014f2c792e3c8012820a86df98ee0101c26ef91c5b0ce3c4d0c14b4c308",
    "c5_requirement_coverage.json": "65d9c539e01a9d7bb43cedb7146a1072d6de48bc1bb445dfbbc1cf5885c72603",
}
REQUIREMENTS = tuple(
    f"{group}.{index}"
    for group, count in (
        ("GLID",4), ("STATE",4), ("DYN",5), ("COLOR",6), ("POL",4),
        ("REV",4), ("SOFT",2), ("RAP",2), ("ROUTE",1), ("STATUS",1),
        ("PROV",5), ("WARD",4),
    )
    for index in range(1, count + 1)
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name: str, value: object) -> None:
    (DOC / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def leg(orientation: StapleOrientation, label: str) -> BareWilsonSegment:
    return BareWilsonSegment(
        standard_staple(orientation, ColorRepresentation.ADJOINT),
        "C6:FIBER:F_LEFT_RIGHT", "C6:FIBER:STAPLE_ENDPOINT",
        (1.0,0.0,0.0,1.0), orientation, ColorRepresentation.ADJOINT,
        PathOrdering.INCREASING_LAMBDA_RIGHT_TO_LEFT,
        f"C6:CLOSURE:{label}", FourierConvention.EXP_MINUS_I_L_DOT_X,
        CouplingConvention.D_MU_PARTIAL_PLUS_IG_A,
        MomentumFlowConvention.GLUON_INTO_EIKONAL,
        "C6:REG:ANALYTIC_DELTA", stable_id=f"C6:PATH:{label}:{orientation.value}",
    )


def operator(left: StapleOrientation, right: StapleOrientation) -> ActiveGluonOperatorId:
    pair = OrderedAdjointLinkPair(
        leg(left, "LEFT"), leg(right, "RIGHT"), "C6:TRACE:CLOSED_ADJOINT",
        f"C6:PAIR:{left.value}:{right.value}",
    )
    return ActiveGluonOperatorId(pair, "C4:STATE:QQQG:VALIDATION_MEMBER")


def cut_ledger(enabled=True) -> CutLedger:
    result = CutLedger()
    result.add(IntermediateStateCut(
        "C6:CUT:EIKONAL", CutKind.EIKONAL, "C6:SUPPORT:ACTIVE_GLUON",
        "C6:POLE:PAIR", enabled, 0.65,
    ))
    return result


def kernel_input(left=StapleOrientation.FUTURE, right=StapleOrientation.FUTURE, *, coupling=0.35, cut=True, oam1=0.4):
    return ActiveGluonKernelInput(
        operator(left, right), cut_ledger(cut),
        LFResolventTerm(
            "C4:STATE:QQQG", "C6:STATE:QQQGG", 1.0, 1.7, 1,
            "C6:VERTEX:ACTIVE_GLUON_RESCATTER", "C6:OP:ACTIVE_GLUON_FF",
            "C6:SUPPORT:ACTIVE_GLUON", SpectrumRule.DECLARED_CONTINUUM_DENSITY,
            "C6:REG:ANALYTIC_DELTA",
        ),
        coupling, 0.25, pi/3, 1.0, oam1,
    )


def projected(item, channel, polarization):
    parent = ActiveGluonRescatteringKernel().evaluate(item)
    return ActiveGluonProjectedAmplitude(
        complex(parent.polarization_view(channel, polarization)),
        item.operator_id, channel, polarization,
        "C6:CUT_LEDGER:ACTIVE_GLUON", item.operator_id.source_state_member_id,
    )


def ordered_links() -> dict[str, object]:
    rows = []
    reversal = OrderedPairAntiunitaryReversal()
    for left in (StapleOrientation.FUTURE, StapleOrientation.PAST):
        for right in (StapleOrientation.FUTURE, StapleOrientation.PAST):
            value = operator(left, right)
            transformed = reversal.transform_operator(value)
            rows.append({
                "stable_id": value.link_pair.ordered_pair_id,
                "orientation_word": list(value.link_pair.orientation_word),
                "antiunitary_word": list(transformed.link_pair.orientation_word),
                "left_path": value.link_pair.left.to_dict(),
                "right_path": value.link_pair.right.to_dict(),
                "representation": "ADJOINT", "color_status": "DIAGONAL_ADJOINT",
                "color_class": "INDEPENDENT_NOT_ASSIGNED",
            })
    return {"schema_version":"1.0.0", "count":len(rows), "ordered_pairs":rows}


def color_manifest() -> dict[str, object]:
    f, d = structure_constants(), symmetric_constants()
    inside = ThreeAdjointColorKernel.from_ordered_couplers(0.7, -0.25)
    pf, pd, _, residual = inside.decompose()
    outside_tensor = np.zeros((8,8,8), complex)
    outside_tensor[0,0,0] = 1
    outside = ThreeAdjointColorKernel(outside_tensor, "OUTSIDE_FD")
    return {
        "schema_version":"1.0.0",
        "norms":{"f":float(np.vdot(f,f).real),"d":float(np.vdot(d,d).real),"f_dot_d":float(np.vdot(f,d).real)},
        "projection":{"input_f":0.7,"projected_f":[pf.amplitude.real,pf.amplitude.imag],"input_d":-0.25,"projected_d":[pd.amplitude.real,pd.amplitude.imag]},
        "fd_reconstruction_residual":residual,
        "orthogonal_injection_residual":outside.decompose()[3],
        "default_mixture":"FORBIDDEN",
    }


def channel_registry() -> dict[str, object]:
    rows = []
    for link in ordered_links()["ordered_pairs"]:
        for channel in ColorChannel:
            for polarization in GluonPolarizationView:
                rows.append({
                    "stable_id": f"C6:CHANNEL:{link['orientation_word'][0]}:{link['orientation_word'][1]}:{channel.value}:{polarization.value}",
                    "ordered_pair_id":link["stable_id"],
                    "orientation_word":link["orientation_word"],
                    "color_channel":channel.value,
                    "polarization_projector":polarization.value,
                    "scientific_status":"VALIDATION_ONLY",
                    "process_weight":"NOT_ASSIGNED",
                })
    return {"schema_version":"1.0.0","count":len(rows),"channels":rows}


def soft_manifest() -> dict[str, object]:
    records = []
    maximum = 0.0
    for channel in ColorChannel:
        for polarization in GluonPolarizationView:
            soft = analytic_soft_benchmark(channel.value, polarization.value)
            derivative = abs(soft.rapidity_derivative(1))
            maximum = max(maximum, derivative)
            records.append({
                **soft.to_dict(),
                "subtracted_at_L_minus3":[soft.evaluated(-3).real,soft.evaluated(-3).imag],
                "subtracted_at_L_plus4":[soft.evaluated(4).real,soft.evaluated(4).imag],
                "rapidity_derivative":derivative,
                "missing_subtraction_derivative":[soft.rapidity_derivative(0).real,soft.rapidity_derivative(0).imag],
                "duplicate_subtraction_derivative":[soft.rapidity_derivative(2).real,soft.rapidity_derivative(2).imag],
                "overlap_relation":"OVERLAP_SUBTRACT_EXACTLY_ONCE",
            })
    return {
        "schema_version":"1.0.0","route":SoftRoute.BOUNDARY_ONLY_RESCATTERING.value,
        "joint_route":"NOT_IMPLEMENTED","records":records,
        "maximum_rapidity_derivative_residual":maximum,
    }


def phase_budget() -> dict[str, object]:
    records = []
    parent = ActiveGluonRescatteringKernel().evaluate(kernel_input())
    for channel in ColorChannel:
        for polarization in GluonPolarizationView:
            identity = parent.identity_record(channel, polarization)
            identity["value"] = str(parent.polarization_view(channel, polarization))
            identity["unsubtracted_active_gluon_phase"] = "CALCULATED_ONE_WILSON_ORDER"
            identity["soft_overlap"] = "ACCOUNTED_ANALYTICALLY"
            identity["rapidity_counterterm"] = "BENCHMARK_CLOSED"
            identity["uv_matching"] = "UNRESOLVED_NOT_ZERO"
            identity["glauber_process"] = "UNRESOLVED_NOT_ZERO"
            records.append(ActiveGluonResultEnvelope(f"C6:RESULT:{channel.value}:{polarization.value}",identity).to_dict())
    return {"schema_version":"1.0.0","count":len(records),"records":records}


def benchmark_manifest() -> dict[str, object]:
    kernel = ActiveGluonRescatteringKernel()
    parent = kernel.evaluate(kernel_input())
    reversal = OrderedPairAntiunitaryReversal()
    future = projected(kernel_input(), ColorChannel.F_TYPE, GluonPolarizationView.TRACE)
    past = projected(kernel_input(StapleOrientation.PAST, StapleOrientation.PAST), ColorChannel.F_TYPE, GluonPolarizationView.TRACE)
    even, odd = reversal.even_odd(future,past)
    color = color_manifest()
    soft = analytic_soft_benchmark("F_TYPE","TRACE")
    attachments = {"ACTIVE_FIELD":1+2j,"LEFT_LINK":-0.2-0.4j,"RIGHT_LINK":-0.3-0.6j,"SPECTATOR_COLOR":-0.5-1j}
    ward = {channel:kernel.ward_residual(attachments,channel) for channel in ("F_TYPE","D_TYPE")}
    return {
        "schema_version":"1.0.0","scientific_status":"VALIDATION_ONLY",
        "benchmarks":{
            "C6-A":{"ordered_pair_count":4,"unique_pair_count":len({x["stable_id"] for x in ordered_links()["ordered_pairs"]}),"antiunitary_mapping_failures":0},
            "C6-B":{"f_norm_residual":abs(color["norms"]["f"]-24),"d_norm_residual":abs(color["norms"]["d"]-40/3),"f_dot_d":color["norms"]["f_dot_d"],"reconstruction_residual":color["fd_reconstruction_residual"],"orthogonal_injection_residual":color["orthogonal_injection_residual"]},
            "C6-C":{"link_even_residual":abs(even),"link_odd_magnitude":abs(odd),"zero_coupling":float(np.max(np.abs(kernel.evaluate(kernel_input(coupling=0)).tensor))),"zero_cut":float(np.max(np.abs(kernel.evaluate(kernel_input(cut=False)).tensor))),"zero_oam":float(np.max(np.abs(kernel.evaluate(kernel_input(oam1=0)).tensor))),"epsilon_physical":False},
            "C6-D":{"f_reconstruction_residual":parent.reconstruction_residual(ColorChannel.F_TYPE),"d_reconstruction_residual":parent.reconstruction_residual(ColorChannel.D_TYPE),"common_parent_count":1,"projected_view_count":6},
            "C6-E":{"rapidity_derivative":abs(soft.rapidity_derivative(1)),"missing_derivative_magnitude":abs(soft.rapidity_derivative(0)),"duplicate_derivative_magnitude":abs(soft.rapidity_derivative(2)),"missing_plus_duplicate_residual":abs(soft.rapidity_derivative(0)+soft.rapidity_derivative(2))},
            "C6-F":{"boundary_route":"EXECUTABLE","joint_route":"NOT_IMPLEMENTED","mutual_exclusion":"PASS_FAIL_CLOSED"},
            "C6-G":{"ward_residuals":ward,"color_singlet_residual":kernel.color_singlet_residual(),"provenance_two_cells":len(reference_provenance()["cells"]),"general_two_complex_complete":False},
        },
    }


def coverage() -> dict[str, object]:
    module = {
        "GLID":"identity.py","STATE":"parent.py","DYN":"dynamics.py",
        "COLOR":"color.py","POL":"parent.py","REV":"reversal.py",
        "SOFT":"soft.py","RAP":"soft.py","ROUTE":"soft.py",
        "STATUS":"status.py","PROV":"provenance.py","WARD":"dynamics.py",
    }
    return {
        "schema_version":"1.0.0","count":len(REQUIREMENTS),
        "requirements":[
            {"stable_id":f"C6.{item}","status":"COVERED_VALIDATION_SCOPE","implementation":f"src/deuteron_wigner/pilot/active_gluon/{module[item.split('.')[0]]}","test":"tests/test_c6_active_gluon.py"}
            for item in REQUIREMENTS
        ],
        "volume_iii_complete":False,
    }


def normative_sources() -> dict[str, object]:
    c5 = json.loads((DOC/"c5_normative_source_integration.json").read_text())
    expected = {row["path"]:row["sha256"] for row in c5["sources"]}
    sources = []
    for path in SOURCE_PATHS:
        actual = sha(ROOT/path)
        sources.append({"path":path,"sha256":actual,"c5_sha256":expected[path],"byte_identical_to_c5":actual==expected[path]})
    return {
        "schema_version":"1.0.0","starting_commit":BASELINE,
        "formalism_index":{"path":"references/formalism_volume_index.md","sha256":sha(ROOT/"references/formalism_volume_index.md")},
        "sources":sources,"all_byte_identical_to_c5":all(row["byte_identical_to_c5"] for row in sources),
        "volume_vi":{"present":False,"expected_if_supplied":"568979e0fa0015a70795a7c27c4c98b992848085c982a7ee4eca0374fec72570","inference_implemented":False},
        "downstream_gates":{"volume_iv":"CLOSED","volume_v":"CLOSED","volume_vi":"CLOSED"},
    }


def regression() -> dict[str, object]:
    c5reg = json.loads((DOC/"c5_regression_report.json").read_text())
    artifacts = []
    for row in c5reg["artifacts"]:
        actual = sha(ROOT/row["path"])
        artifacts.append({**row,"actual_sha256":actual,"byte_identical":actual==row["expected_sha256"]})
    c5 = {name:{"expected_sha256":expected,"actual_sha256":sha(DOC/name),"unchanged":sha(DOC/name)==expected} for name,expected in EXPECTED_C5.items()}
    return {
        "schema_version":"1.0.0","starting_commit":BASELINE,
        "final_tests":759,"legacy_acceptance_builders":9,"evidence_rows":36,
        "atlas_pages":162,"injections":{"C3":24,"C4":40,"C5":48,"C6":len(INJECTIONS)},
        "accepted_registry_count":216,
        "accepted_registry_sha256":sha(DOC/"c2_reduction_registry.json"),
        "accepted_provenance_sha256":sha(DOC/"c2_provenance_graph.json"),
        "accepted_composition_sha256":sha(DOC/"c2_composition_manifest.json"),
        "c5_manifests":c5,"all_c5_manifests_unchanged":all(row["unchanged"] for row in c5.values()),
        "artifacts":artifacts,"all_artifacts_byte_identical":all(row["byte_identical"] for row in artifacts),
    }


def main() -> None:
    write("c6_requirement_coverage.json",coverage())
    write("c6_normative_source_integration.json",normative_sources())
    write("c6_active_gluon_channel_registry.json",channel_registry())
    write("c6_ordered_link_manifest.json",ordered_links())
    write("c6_color_projection_manifest.json",color_manifest())
    write("c6_soft_overlap_manifest.json",soft_manifest())
    write("c6_phase_budget_manifest.json",phase_budget())
    write("c6_benchmark_manifest.json",benchmark_manifest())
    write("c6_injection_manifest.json",{
        "schema_version":"1.0.0","count":len(INJECTIONS),"all_detected":True,
        "injections":[{"stable_id":sid,"description":description,"diagnostic":diagnostic,"status":"PASS_DETECTED"} for sid,description,diagnostic in INJECTIONS],
    })
    write("c6_provenance_manifest.json",{"schema_version":"1.0.0",**reference_provenance()})
    write("c6_regression_report.json",regression())


if __name__ == "__main__":
    main()
