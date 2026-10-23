#!/usr/bin/env python3
"""Export constituent-resolved canonical quark and gluon nuclear parents."""

from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

from deuteron_wigner.correlator_io import (
    deserialize_gluon_correlator,
    deserialize_quark_correlator,
    gluon_correlator_rows,
    quark_correlator_rows,
)
from deuteron_wigner.gluon_correlator import (
    Spin1GluonCorrelator,
    project_to_allowed_spin1_gluon_basis,
)
from deuteron_wigner.quark_correlator import (
    project_spin1_quark_correlator,
    project_spin1_quark_correlator_at_origin,
)
from deuteron_wigner.resolved_nuclear_parent import (
    ResolvedGluonNuclearParent,
    ResolvedQuarkNuclearParent,
)
from deuteron_wigner.formal.provenance_graph import CompositionPlan, ProvenanceGraph
from deuteron_wigner.formal.accepted_reductions import accepted_reduction_registry


ROOT = Path(__file__).resolve().parents[1]
QLEGACY = ROOT / (
    "outputs/parent_tmds/wp12_multikinematic/"
    "quark_all_tmd_multix_q5.correlators.csv"
)
QCANON = ROOT / (
    "outputs/parent_tmds/wp12_canonical_composed_quark.correlators.csv"
)
GCANON = ROOT / (
    "outputs/parent_tmds/wp12_canonical_composed_gluon.correlators.csv"
)
GLEGACY = str(
    ROOT / "outputs/parent_tmds/wp12_multikinematic/"
    "gluon_x*_q5.correlators.csv"
)
QOUT = ROOT / "outputs/parent_tmds/wp12_resolved_quark_parent.csv"
GOUT = ROOT / "outputs/parent_tmds/wp12_resolved_gluon_parent.csv"
REPORT = ROOT / "outputs/validation/wp12_resolved_nuclear_parent.json"
C2REPORT = ROOT / "outputs/validation/c2_composition_manifest.json"
MASS = 1.87561294257
CLASS = {
    "proton_in_deuteron": "physical_constituent_parent",
    "neutron_in_deuteron": "physical_constituent_parent",
    "nucleon_sum": "derived_constituent_sum",
    "proton_minus_neutron": "derived_isovector_diagnostic",
    "nuclear_correction": "signed_nuclear_correction",
    "canonical_spin1_total": "physical_canonical_total",
}
QRANKS = {
    "f1": 0, "h1perp": 1, "g1": 0, "h1Lperp": 1,
    "f1Tperp": 1, "g1T": 1, "h1": 0, "h1Tperp": 2,
    "f1LL": 0, "h1LLperp": 2, "f1LT": 1, "g1LT": 1,
    "h1LT": 0, "h1LTperp": 2, "f1TT": 2, "g1TT": 2,
    "h1TT": 1, "h1TTperp": 3,
}
GRANKS = {
    "f1": 0, "g1": 0, "h1perp": 2, "h1Lperp": 2,
    "f1Tperp": 1, "g1T": 1, "h1": 1, "h1Tperp": 3,
    "f1LL": 0, "h1LLperp": 2, "f1LT": 1, "g1LT": 1,
    "h1LT": 1, "h1LTperp": 3, "f1TT_minus_h1TTperp": 2,
    "g1TT": 2, "h1TT": 0, "h1TTperpperp": 4,
}


def key(row, gluon=False):
    result = (
        row.species, int(row.flavor), row.gauge_link, float(row.x_N),
        float(row.x_D), float(row.Q_GeV), float(row.k_GeV),
        float(row.azimuth_rad),
    )
    return result + ((row.color_structure,) if gluon else ())


def labels(row, component):
    return {
        "species": row.species, "flavor": int(row.flavor),
        "flavor_label": getattr(row, "flavor_label", "g"),
        "color_structure": getattr(
            row, "color_structure", "not_applicable"
        ),
        "gauge_link": row.gauge_link,
        "x_N": float(row.x_N), "x_D": float(row.x_D),
        "Q_GeV": float(row.Q_GeV), "k_GeV": float(row.k_GeV),
        "azimuth_rad": float(row.azimuth_rad),
        "component": component,
        "component_class": CLASS[component],
        "combine_policy": (
            "constituents retained simultaneously; canonical total is a "
            "projection, never a replacement for proton/neutron dynamics"
        ),
    }


def main() -> None:
    # C2 native semantic gate: validate the accepted plan before arrays combine.
    graph_payload = json.loads(
        (ROOT / "docs/next_level/c2_provenance_graph.json").read_text()
    )
    plan_payload = json.loads(
        (ROOT / "docs/next_level/c2_composition_manifest.json").read_text()
    )
    c2_graph = ProvenanceGraph.from_dict(graph_payload)
    c2_plan = CompositionPlan.from_dict(plan_payload["default_plan"])
    c2_plan.validate(c2_graph)
    native_reductions = accepted_reduction_registry()
    qlegacy = pd.read_csv(QLEGACY)
    qlegacy = qlegacy[
        qlegacy.mechanism.isin(["proton_impulse", "neutron_impulse"])
    ]
    qcanon = pd.read_csv(QCANON)
    qkeys = [
        "species", "flavor", "flavor_label", "gauge_link", "x_N", "x_D",
        "Q_GeV", "k_GeV", "azimuth_rad",
    ]
    qsource = {
        key(block.iloc[0]): (
            block.iloc[0],
            {
                mechanism: deserialize_quark_correlator(part)
                for mechanism, part in block.groupby("mechanism")
            },
        )
        for _, block in qlegacy.groupby(qkeys, sort=True)
    }
    qtotals = {
        key(block.iloc[0]): deserialize_quark_correlator(block)
        for _, block in qcanon.groupby(qkeys, sort=True)
    }
    qrows, qmatrix, qclosure = [], [], []
    for identity, (first, source) in qsource.items():
        resolved = ResolvedQuarkNuclearParent(
            source["proton_impulse"], source["neutron_impulse"],
            qtotals[identity],
        )
        momentum = (
            first.k_GeV*np.cos(first.azimuth_rad),
            first.k_GeV*np.sin(first.azimuth_rad),
        )
        for component, parent in resolved.components().items():
            projected = (
                project_spin1_quark_correlator_at_origin(parent, MASS)
                if np.isclose(first.k_GeV, 0.0)
                else project_spin1_quark_correlator(parent, momentum, MASS)
            )
            common = labels(first, component)
            qrows.extend({
                **common, "tmd": name, "rank": QRANKS[name],
                "F_GeV-2": native_reductions.get(
                    f"RED:{first.species}:{first.flavor_label}:{name}:"
                    f"{first.gauge_link}:NOT_APPLICABLE"
                )(value),
            } for name, value in projected.items())
            qmatrix.extend(quark_correlator_rows(parent, common))
        qclosure.append(resolved.closure_residual())

    glegacy = pd.concat(
        [pd.read_csv(path) for path in sorted(glob.glob(GLEGACY))]
        + [pd.read_csv(
            ROOT / "outputs/parent_tmds/"
            "gluon_av18_canonical_lfwf_todd.correlators.csv"
        )],
        ignore_index=True,
    )
    glegacy = glegacy[
        glegacy.mechanism.isin(["proton_impulse", "neutron_impulse"])
    ]
    gcanon = pd.read_csv(GCANON)
    gkeys = [
        "species", "flavor", "color_structure", "gauge_link", "x_N",
        "x_D", "Q_GeV", "k_GeV", "azimuth_rad",
    ]
    gsource = {
        key(block.iloc[0], True): (
            block.iloc[0],
            {
                mechanism: Spin1GluonCorrelator(
                    deserialize_gluon_correlator(part)
                )
                for mechanism, part in block.groupby("mechanism")
            },
        )
        for _, block in glegacy.groupby(gkeys, sort=True)
    }
    gtotals = {
        key(block.iloc[0], True): Spin1GluonCorrelator(
            deserialize_gluon_correlator(block)
        )
        for _, block in gcanon.groupby(gkeys, sort=True)
    }
    grows, gmatrix, gclosure = [], [], []
    for identity, (first, source) in gsource.items():
        resolved = ResolvedGluonNuclearParent(
            source["proton_impulse"], source["neutron_impulse"],
            gtotals[identity],
        )
        momentum = (
            first.k_GeV*np.cos(first.azimuth_rad),
            first.k_GeV*np.sin(first.azimuth_rad),
        )
        for component, parent in resolved.components().items():
            _, projected, residual = project_to_allowed_spin1_gluon_basis(
                parent.values, momentum, MASS
            )
            common = labels(first, component)
            color_id = {
                "f_type_antisymmetric": "F_TYPE",
                "d_type_symmetric": "D_TYPE",
            }[first.color_structure]
            grows.extend({
                **common, "tmd": name, "rank": GRANKS[name],
                "F_GeV-2": native_reductions.get(
                    f"RED:g:NOT_APPLICABLE:{name}:{first.gauge_link}:{color_id}"
                )(value),
                "basis_residual": residual,
            } for name, value in projected.items())
            gmatrix.extend(gluon_correlator_rows(parent.values, common))
        gclosure.append(resolved.closure_residual())

    QOUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(qrows).to_csv(QOUT, index=False)
    pd.DataFrame(qmatrix).to_csv(QOUT.with_suffix(".correlators.csv"), index=False)
    pd.DataFrame(grows).to_csv(GOUT, index=False)
    pd.DataFrame(gmatrix).to_csv(GOUT.with_suffix(".correlators.csv"), index=False)
    report = {
        "status": "pass",
        "components": list(CLASS),
        "quark_rows": len(qrows), "gluon_rows": len(grows),
        "maximum_quark_closure_residual": max(qclosure),
        "maximum_gluon_closure_residual": max(gclosure),
        "interpretation": (
            "The complete model retains proton and neutron parents and their "
            "isovector difference. The spin-1 nuclear total is one derived "
            "projection and does not replace constituent dynamics."
        ),
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2)+"\n")
    C2REPORT.write_text(json.dumps({
        "schema_version": "1.0.0",
        "requirement_id": "C2.COMPOSE",
        "production_builder": "scripts/build_wp12_resolved_nuclear_parent.py",
        "semantic_plan": c2_plan.dry_run(c2_graph),
        "numerical_order_preserved": True,
        "authoritative_outputs_modified_by_metadata": False,
    }, indent=2, sort_keys=True)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
